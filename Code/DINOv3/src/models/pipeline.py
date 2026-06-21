"""Production model entry point for the seismic denoising pipeline.

This is the only model file used by training and evaluation. The legacy
root-level dinov3_denoiser.py (week-1 prototype, no LoRA/2.5D) has been
removed. Import DINOv3Denoiser from here for all experiments.
"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

# Ensure src/ is on the path regardless of how this file is invoked.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from models.decoder import PaperDecoder

_PATCH_EMB_INITS = {"mixed", "random_outer", "all_mean", "center_preserve"}
_CENTRAL_PRETRAINED_INITS = {"mixed", "random_outer", "center_preserve"}
_SFM_DECODER_PREFIXES = (
    "mask_token",
    "decoder_pos_embed",
    "decoder_embed.",
    "decoder_blocks.",
    "decoder_norm.",
    "decoder_pred.",
)


def _validate_patch_embed_request(
    in_chans: int,
    init: str,
    pretrained_in_chans: int = 3,
) -> None:
    """Raise ValueError if the requested channel-expansion init is impossible.

    Center-preserving inits (mixed, random_outer, center_preserve) require that
    in_chans > pretrained_in_chans and the surplus is even so pretrained weights
    can be placed symmetrically around the center channels.
    """
    if in_chans < 1:
        raise ValueError(f"in_chans must be >= 1, got {in_chans}")
    if pretrained_in_chans < 1:
        raise ValueError(
            f"pretrained_in_chans must be >= 1, got {pretrained_in_chans}"
        )
    if init not in _PATCH_EMB_INITS:
        raise ValueError(
            f"patch_emb_init must be one of {sorted(_PATCH_EMB_INITS)}, got {init!r}"
        )
    if in_chans != pretrained_in_chans and init in _CENTRAL_PRETRAINED_INITS:
        if in_chans < pretrained_in_chans or (in_chans - pretrained_in_chans) % 2:
            raise ValueError(
                f"patch_emb_init={init!r} requires centered pretrained "
                f"{pretrained_in_chans}-channel weights, got in_chans={in_chans}"
            )


def _trusted_torch_load(path: Path, map_location: str | torch.device = "cpu") -> object:
    """Load trusted local checkpoints, including Linux-saved SFM checkpoints."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as exc:
        if os.name != "nt" or "PosixPath" not in str(exc):
            raise
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[assignment]
            return torch.load(path, map_location=map_location, weights_only=False)
        finally:
            pathlib.PosixPath = original_posix_path  # type: ignore[assignment]


def _state_dict_from_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    ckpt = _trusted_torch_load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(ckpt)!r}")
    state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict-like object at {path}, got {type(state)!r}")

    clean_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module."):]
        clean_state[clean_key] = value
    return clean_state


class _SFMMLP(nn.Module):
    """Two-layer GELU MLP matching the feed-forward block in the SFM ViT encoder."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _SFMAttention(nn.Module):
    """Vanilla multi-head self-attention matching the SFM encoder checkpoint.

    Uses standard scaled dot-product attention (no Flash Attention). LoRA targets
    qkv and proj on this class during backbone-comparison runs.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_tokens, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, n_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(bsz, n_tokens, channels)
        return self.proj(x)


class _SFMBlock(nn.Module):
    """Pre-norm transformer block (LayerNorm before attention and MLP)."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = _SFMAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = _SFMMLP(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _SFMPatchEmbed(nn.Module):
    """Patchify conv for the 1-channel SFM input (grayscale seismic slices)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class SFMViTBackbone(nn.Module):
    """Minimal MAE-style ViT encoder matching the local SFM-Base checkpoint."""

    pretrained_in_chans = 1

    def __init__(
        self,
        weights_path: Path,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.num_features = embed_dim
        self.patch_size = patch_size
        self.patch_embed = _SFMPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=self.pretrained_in_chans,
            embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [_SFMBlock(embed_dim, num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        state = _state_dict_from_checkpoint(weights_path)
        encoder_state = {
            key: value
            for key, value in state.items()
            if not key.startswith(_SFM_DECODER_PREFIXES)
        }
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        unexpected = [
            key
            for key in unexpected
            if not key.startswith(_SFM_DECODER_PREFIXES)
            and key not in {"head.weight", "head.bias"}
        ]
        if missing or unexpected:
            raise RuntimeError(
                "SFM encoder checkpoint load mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

    def replace_input_projection(self, new_proj: nn.Conv2d) -> None:
        """Swap the patch-embedding conv to support multi-channel (2.5D) input."""
        self.patch_embed.proj = new_proj

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return {"x_norm_patchtokens": x[:, 1:, :]}


class TorchvisionSwinV2Backbone(nn.Module):
    """SwinV2 adapter exposing the stride-16 stage as a 14x14 feature map."""

    pretrained_in_chans = 3
    _ARCH_FEATURE_DIMS = {
        "swin_v2_t": 384,
        "swin_v2_s": 384,
        "swin_v2_b": 512,
    }

    def __init__(self, model_name: str, weights_path: Path):
        super().__init__()
        try:
            from torchvision import models as tv_models
        except ImportError as exc:
            raise ImportError("torchvision is required for SwinV2 backbones") from exc

        if model_name not in self._ARCH_FEATURE_DIMS:
            raise ValueError(
                f"Unsupported SwinV2 model {model_name!r}; "
                f"expected one of {sorted(self._ARCH_FEATURE_DIMS)}"
            )
        model_fn = getattr(tv_models, model_name)
        self.model = model_fn(weights=None)
        self.num_features = self._ARCH_FEATURE_DIMS[model_name]
        if weights_path:
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"SwinV2 weights not found: {weights_path}. "
                    "Download the matching torchvision checkpoint before training."
                )
            state = _state_dict_from_checkpoint(weights_path)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            missing = [key for key in missing if not key.startswith("head.")]
            unexpected = [key for key in unexpected if not key.startswith("head.")]
            if missing or unexpected:
                raise RuntimeError(
                    "SwinV2 checkpoint load mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )

    @property
    def input_projection(self) -> nn.Conv2d:
        return self.model.features[0][0]

    def replace_input_projection(self, new_proj: nn.Conv2d) -> None:
        self.model.features[0][0] = new_proj

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx == 5:
                break
        return x.permute(0, 3, 1, 2).contiguous()


def _unwrap_peft(model: nn.Module) -> nn.Module:
    """Return the base model, stripping any PEFT wrapper if present."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()  # type: ignore[no-any-return]
    return model


def _attention_lora_targets(model: nn.Module, requested: list[str]) -> list[str]:
    """Resolve logical LoRA targets {qkv, proj} to concrete module paths.

    DINOv3 accepts the logical names directly. For SFM/SwinV2 backbones the
    attribute paths differ, so this function walks named_modules and returns
    the actual dotted paths that end in .attn.qkv or .attn.proj.
    """
    if set(requested) != {"qkv", "proj"}:
        return requested
    matches = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith(".attn.qkv") or name.endswith(".attn.proj"):
            matches.append(name)
    return matches or requested


def _new_input_projection(
    old_proj: nn.Conv2d,
    in_chans: int,
    pretrained_weight: torch.Tensor,
    pretrained_bias: torch.Tensor | None,
    init: str,
) -> nn.Conv2d:
    """Build a new patch-embedding Conv2d that accepts in_chans input channels.

    init strategies:
      mixed / all_mean    — fill all new channels with the per-channel mean of the
                            pretrained weights, then overwrite center channels.
      random_outer        — keep outer channels randomly initialised (default
                            nn.Conv2d init), copy pretrained weights to center.
      center_preserve     — zero-init all channels, copy pretrained to center.
    The pretrained (3-channel) weights are always placed at the symmetric center
    of the new projection so spatial alignment is preserved.
    """
    pretrained_in_chans = int(pretrained_weight.shape[1])
    _validate_patch_embed_request(in_chans, init, pretrained_in_chans)
    out_chans, _, patch_h, patch_w = pretrained_weight.shape
    new_proj = nn.Conv2d(
        in_chans,
        out_chans,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        dilation=old_proj.dilation,
        groups=old_proj.groups,
        bias=pretrained_bias is not None,
        padding_mode=old_proj.padding_mode,
    )
    if tuple(new_proj.weight.shape[-2:]) != (patch_h, patch_w):
        raise ValueError(
            f"New projection kernel shape {tuple(new_proj.weight.shape[-2:])} "
            f"does not match pretrained shape {(patch_h, patch_w)}"
        )

    start = (in_chans - pretrained_in_chans) // 2
    with torch.no_grad():
        if init == "center_preserve":
            new_proj.weight.data.zero_()
        elif init == "random_outer":
            pass
        else:
            mean_w = pretrained_weight.mean(dim=1, keepdim=True)
            new_proj.weight.data[:] = mean_w.expand(-1, in_chans, -1, -1)

        if init in _CENTRAL_PRETRAINED_INITS:
            new_proj.weight.data[:, start:start + pretrained_in_chans] = pretrained_weight

        if pretrained_bias is not None:
            new_proj.bias.data.copy_(pretrained_bias)
    return new_proj


class DINOv3Denoiser(nn.Module):
    """Foundation-model encoder plus lightweight CNN decoder for denoising.

    DINOv3 remains the default/primary path. SFM and SwinV2 are optional
    backbone-comparison adapters that expose the same 14x14 spatial feature map
    interface to the decoder.
    """

    def __init__(
        self,
        repo_dir: Path,
        weights_path: Path,
        model_name: str = "dinov3_vits16",
        in_chans: int = 3,
        lora_rank: int = 16,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        lora_targets: list[str] | None = None,
        patch_emb_init: str = "mixed",
        full_finetune: bool = False,
    ):
        super().__init__()

        if lora_targets is None:
            lora_targets = ["qkv", "proj"]
        in_chans = int(in_chans)
        patch_emb_init = str(patch_emb_init)
        self.model_name = model_name

        backbone, hidden_dim, pretrained_in_chans = self._build_backbone(
            repo_dir=repo_dir,
            weights_path=weights_path,
            model_name=model_name,
        )
        _validate_patch_embed_request(in_chans, patch_emb_init, pretrained_in_chans)
        input_proj = self._input_projection(backbone)
        pretrained_weight = input_proj.weight.data.clone()
        pretrained_bias = (
            input_proj.bias.data.clone() if input_proj.bias is not None else None
        )

        if not full_finetune:
            for p in backbone.parameters():
                p.requires_grad = False

        if lora_rank > 0 and not full_finetune:
            resolved_targets = (
                lora_targets
                if model_name.startswith("dinov3")
                else _attention_lora_targets(backbone, lora_targets)
            )
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=resolved_targets,
            )
            self.backbone = get_peft_model(backbone, lora_cfg)
        else:
            self.backbone = backbone

        # Replace input projection after PEFT wrapping so LoRA never targets it.
        if in_chans != pretrained_in_chans:
            self._replace_input_projection(
                in_chans,
                pretrained_weight,
                pretrained_bias,
                patch_emb_init,
                input_proj,
            )

        self.decoder = PaperDecoder(hidden_dim=hidden_dim)

    def _build_backbone(
        self,
        repo_dir: Path,
        weights_path: Path,
        model_name: str,
    ) -> tuple[nn.Module, int, int]:
        if model_name.startswith("dinov3"):
            backbone = torch.hub.load(
                str(repo_dir),
                model_name,
                source="local",
                weights=str(weights_path),
            )
            return backbone, int(backbone.num_features), 3
        if model_name == "sfm_vit_base_patch16":
            backbone = SFMViTBackbone(weights_path=weights_path)
            return backbone, int(backbone.num_features), backbone.pretrained_in_chans
        if model_name in TorchvisionSwinV2Backbone._ARCH_FEATURE_DIMS:
            backbone = TorchvisionSwinV2Backbone(model_name, weights_path)
            return backbone, int(backbone.num_features), backbone.pretrained_in_chans
        raise ValueError(
            f"Unsupported model.name={model_name!r}. Expected DINOv3, "
            "sfm_vit_base_patch16, or a supported SwinV2 model."
        )

    def _input_projection(self, backbone: nn.Module) -> nn.Conv2d:
        if hasattr(backbone, "input_projection"):
            return getattr(backbone, "input_projection")
        if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "proj"):
            return backbone.patch_embed.proj
        raise AttributeError(f"Cannot locate input projection for {type(backbone)!r}")

    def _set_input_projection(self, backbone: nn.Module, new_proj: nn.Conv2d) -> None:
        if hasattr(backbone, "replace_input_projection"):
            backbone.replace_input_projection(new_proj)
            return
        if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "proj"):
            backbone.patch_embed.proj = new_proj
            return
        raise AttributeError(f"Cannot replace input projection for {type(backbone)!r}")

    def _replace_input_projection(
        self,
        in_chans: int,
        pretrained_weight: torch.Tensor,
        pretrained_bias: torch.Tensor | None,
        init: str = "mixed",
        old_proj_template: nn.Conv2d | None = None,
    ) -> None:
        base = _unwrap_peft(self.backbone)
        old_proj = old_proj_template or self._input_projection(base)
        new_proj = _new_input_projection(
            old_proj=old_proj,
            in_chans=int(in_chans),
            pretrained_weight=pretrained_weight,
            pretrained_bias=pretrained_bias,
            init=str(init),
        )
        self._set_input_projection(base, new_proj)
        for p in new_proj.parameters():
            p.requires_grad = True

    def _extract_spatial_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, hidden_dim, H/16, W/16]."""
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            patch_tokens = out["x_norm_patchtokens"]
            bsz, n_tokens, channels = patch_tokens.shape
            h = x.shape[-2] // 16
            w = x.shape[-1] // 16
            if h * w != n_tokens:
                raise ValueError(f"Cannot reshape {n_tokens} tokens to {(h, w)}")
            return patch_tokens.transpose(1, 2).reshape(bsz, channels, h, w)
        if out.ndim == 4:
            return out
        if out.ndim == 3:
            bsz, n_tokens, channels = out.shape
            h = w = int(n_tokens ** 0.5)
            if h * w != n_tokens:
                raise ValueError(f"Cannot reshape {n_tokens} tokens into a square map")
            return out.transpose(1, 2).reshape(bsz, channels, h, w)
        raise TypeError(f"Unsupported forward_features output type/shape: {type(out)!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, 1, H, W]."""
        features = self._extract_spatial_features(x)
        return self.decoder(features)


def _count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    repo_dir = project / "external" / "dinov3"
    weights = project / "weights" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

    def _smoke(label: str, in_chans: int, lora_rank: int) -> None:
        m = DINOv3Denoiser(
            repo_dir=repo_dir,
            weights_path=weights,
            in_chans=in_chans,
            lora_rank=lora_rank,
        )
        m.eval()
        x = torch.randn(2, in_chans, 224, 224)
        with torch.no_grad():
            y = m(x)
        total, trainable = _count_params(m)
        print(f"\n[{label}]")
        print(f"  in_chans={in_chans}  lora_rank={lora_rank}")
        print(f"  Input {tuple(x.shape)} -> Output {tuple(y.shape)}")
        print(f"  Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")
        assert y.shape == (2, 1, 224, 224), f"Bad output shape: {y.shape}"
        print("  PASSED")

    _smoke("2D / 3ch  (LoRA)", in_chans=3, lora_rank=16)
    _smoke("5ch-A     (no LoRA)", in_chans=5, lora_rank=0)
    _smoke("5ch-B     (LoRA r=16)", in_chans=5, lora_rank=16)
    print("\nAll smoke tests passed.")
