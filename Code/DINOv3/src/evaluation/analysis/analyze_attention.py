"""Visualize DINOv3 self-attention, per-head behaviour, and neighbour-channel
saliency for the 2D-1ch, 2.5D-3ch, and 2.5D-5ch denoising models.

This produces the appendix attention-analysis figure set. It is a runtime-only
diagnostic: it monkey-patches DINOv3 SelfAttention so attention weights are
materialized during inference, without modifying the external/dinov3 source tree.

Three things to know about what these figures can and cannot show:

  1. Self-attention is purely SPATIAL (patch-to-patch). The neighbouring slices
     of a 2.5D input are fused at the patch-embedding conv BEFORE the transformer
     runs, so there is no per-neighbour attention to extract. Figures F1/F2 below
     therefore compare how the SPATIAL attention differs across variants and heads.

  2. To answer "how much does each neighbour slice contribute", we use a
     different technique: per-channel input-gradient saliency (F3). This is input
     attribution, not attention. The causal evidence for neighbour use remains the
     context counterfactuals; this is a qualitative appendix illustration.

  3. Everything here is qualitative / descriptive. Do not build claims on the
     marginal entropy differences between variants.

Outputs (under --out-dir):
  attn_ds{D}_ts{T}_s{idx}_r{rank}_comparison.png   head-averaged attention, 2D/3ch/5ch
  attn_ds{D}_ts{T}_s{idx}_r{rank}_heads_{variant}.png   per-head maps for one variant
  attn_ds{D}_ts{T}_s{idx}_r{rank}_saliency_{variant}.png   per-channel saliency maps
  attn_ds{D}_ts{T}_s{idx}_r{rank}_saliency_bars.png   per-neighbour contribution bars
  attention_spread_summary.png                       entropy / attention radius per variant
  attention_summary.csv                              per-query attention stats
  attention_spread.csv                               per-(sample,variant) spread stats

Usage:
    python evaluation/analyze_attention.py \
        --project-root C:/UNI/Y3/RP \
        --data-seed 101 --training-seed 42 \
        --out-dir C:/UNI/Y3/RP/experiments/summaries/mechanism_analysis/attention_maps
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import ensure_external_dinov3_on_path, ensure_src_on_path, project_root as default_project_root
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXTERNAL_DINOV3 = ensure_external_dinov3_on_path(__file__)

from dinov3.layers.attention import SelfAttention

from evaluation.figures.generate_comparison_panels import (
    _VARIANTS,
    _build_shared_dataset,
    _build_variant_dataset,
    _load_model,
    _load_sample_for_mode,
    _resolve_checkpoint,
)

# Canonical display labels for the appendix (see CLAUDE.md / writing-style note).
_DISPLAY_LABEL = {"2D": "2D-1ch", "3ch": "2.5D-3ch", "5ch": "2.5D-5ch"}
_VARIANT_ORDER = ["2D", "3ch", "5ch"]


def _patch_self_attention() -> None:
    """Patch SelfAttention.compute_attention to store attention weights."""
    if getattr(SelfAttention.compute_attention, "_captures_attention", False):
        return

    original = SelfAttention.compute_attention

    def patched_compute_attention(self, qkv, attn_bias=None, rope=None):
        assert attn_bias is None
        batch, n_tokens, _ = qkv.shape
        channels = self.qkv.in_features

        qkv = qkv.reshape(
            batch,
            n_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        self.captured_attn_weights = weights.detach().float().cpu()

        x = torch.matmul(weights, v)
        x = x.transpose(1, 2)
        return x.reshape(batch, n_tokens, channels)

    patched_compute_attention._captures_attention = True
    patched_compute_attention._original_compute_attention = original
    SelfAttention.compute_attention = patched_compute_attention


def _parse_xy(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected coordinate as 'x,y'")
    return int(parts[0]), int(parts[1])


def _find_attention_modules(model: torch.nn.Module) -> list[tuple[str, SelfAttention]]:
    modules: list[tuple[str, SelfAttention]] = []
    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            modules.append((name, module))
    return modules


# ---------------------------------------------------------------------------
# Sample and query selection
# ---------------------------------------------------------------------------
def _read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _select_samples(
    metadata_path: Path,
    ranks: list[int] | None,
    explicit_indices: list[int] | None,
) -> list[dict[str, Any]]:
    """Return a list of selected samples with provenance from the (mid-filtered) CSV.

    Each entry: {sample_idx, rank_by_gap, vol_id, slice_t, mid_slice_pct,
                 delta_5ch_vs_2d}. When the metadata CSV is unavailable we fall
    back to explicit indices (or sample 0 with a loud warning).
    """
    rows = _read_metadata(metadata_path)

    if explicit_indices:
        out = []
        by_idx = {int(r["sample_idx"]): r for r in rows}
        for idx in explicit_indices:
            r = by_idx.get(idx)
            out.append({
                "sample_idx": idx,
                "rank_by_gap": int(r["rank_by_gap"]) if r else -1,
                "vol_id": int(r["vol_id"]) if r else None,
                "slice_t": int(r["slice_t"]) if r else None,
                "mid_slice_pct": float(r["mid_slice_pct"]) if r else None,
                "delta_5ch_vs_2d": float(r["delta_5ch_vs_2d"]) if r else None,
            })
        return out

    if not rows:
        print(
            "WARNING: no comparison_metadata.csv found for this data seed; "
            "falling back to sample index 0 (likely a noise-only edge slice). "
            "Run generate_comparison_panels.py --pool-data-seeds first."
        )
        return [{"sample_idx": 0, "rank_by_gap": -1, "vol_id": None,
                 "slice_t": None, "mid_slice_pct": None, "delta_5ch_vs_2d": None}]

    # rows are sorted by rank_by_gap ascending (rank 0 = largest 5ch-over-2D gain)
    rows_by_rank = sorted(rows, key=lambda r: int(r["rank_by_gap"]))
    if ranks is None:
        # Headline (largest gain) + a median-gap sample, to avoid cherry-picking.
        ranks = [0, len(rows_by_rank) // 2]
    out = []
    for rk in ranks:
        rk = max(0, min(rk, len(rows_by_rank) - 1))
        r = rows_by_rank[rk]
        out.append({
            "sample_idx": int(r["sample_idx"]),
            "rank_by_gap": int(r["rank_by_gap"]),
            "vol_id": int(r["vol_id"]),
            "slice_t": int(r["slice_t"]),
            "mid_slice_pct": float(r["mid_slice_pct"]),
            "delta_5ch_vs_2d": float(r["delta_5ch_vs_2d"]),
        })
    return out


def _auto_query_points(
    clean: np.ndarray,
    margin: int = 32,
    patch: int = 16,
    min_separation: int = 80,
) -> list[tuple[str, tuple[int, int]]]:
    """Pick a high-amplitude and a low-amplitude query point from the clean target.

    High-amplitude regions correspond to strong reflectivity (clear reflectors);
    low-amplitude regions are weak-reflectivity / quiet zones. We smooth the
    absolute amplitude so single-pixel spikes are not chosen, restrict to the
    image interior so the query patch is well inside the crop, and snap to the
    centre of a 16x16 patch. The low-amplitude point is forced at least
    ``min_separation`` pixels away from the high-amplitude point so the two
    figure rows contrast a reflector against a genuinely different quiet zone
    rather than landing on adjacent patches.
    """
    h, w = clean.shape
    amp = np.abs(clean.astype(np.float32))
    # Box-smooth |amplitude| so single-pixel spikes are not chosen as queries.
    smooth = _box_smooth(amp, k=9)

    interior = np.full_like(smooth, np.nan)
    interior[margin:h - margin, margin:w - margin] = smooth[margin:h - margin, margin:w - margin]

    def _snap(yx: tuple[int, int]) -> tuple[int, int]:
        y, x = yx
        py = min((y // patch) * patch + patch // 2, h - 1)
        px = min((x // patch) * patch + patch // 2, w - 1)
        return int(px), int(py)  # return as (x, y)

    hi_yx = np.unravel_index(np.nanargmax(interior), interior.shape)
    # Mask out a neighbourhood around the high-amplitude point before picking the
    # low-amplitude one, so the two queries are spatially well separated.
    yy, xx = np.ogrid[:h, :w]
    far = (yy - hi_yx[0]) ** 2 + (xx - hi_yx[1]) ** 2 >= min_separation ** 2
    low_field = np.where(far, interior, np.nan)
    if np.all(np.isnan(low_field)):  # fall back if separation leaves nothing
        low_field = interior
    lo_yx = np.unravel_index(np.nanargmin(low_field), low_field.shape)
    return [
        ("high-amplitude region", _snap(hi_yx)),
        ("low-amplitude region", _snap(lo_yx)),
    ]


# ---------------------------------------------------------------------------
# Attention capture and per-query maps
# ---------------------------------------------------------------------------
def _capture_attention(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    layer_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    modules = _find_attention_modules(model)
    if not modules:
        raise RuntimeError("No DINOv3 SelfAttention modules found in model")
    if layer_index < 0:
        layer_index = len(modules) + layer_index
    layer_index = max(0, min(layer_index, len(modules) - 1))
    layer_name, layer = modules[layer_index]
    for _, module in modules:
        if hasattr(module, "captured_attn_weights"):
            delattr(module, "captured_attn_weights")

    with torch.no_grad():
        _ = model.backbone.forward_features(noisy.unsqueeze(0).to(device))

    if not hasattr(layer, "captured_attn_weights"):
        raise RuntimeError(f"Layer did not capture attention: {layer_name}")
    return layer.captured_attn_weights, layer_name


def _grid_dims(n_tokens: int, image_hw: tuple[int, int]) -> tuple[int, int, int]:
    image_h, image_w = image_hw
    grid_h = image_h // 16
    grid_w = image_w // 16
    n_patches = grid_h * grid_w
    prefix = n_tokens - n_patches
    if prefix < 0:
        raise RuntimeError(
            f"Attention sequence has {n_tokens} tokens but expected {n_patches} patches"
        )
    return grid_h, grid_w, prefix


def _query_attention_map(
    attn: torch.Tensor,
    query_xy: tuple[int, int],
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return averaged and per-head upsampled maps for one query pixel."""
    _, n_heads, n_tokens, _ = attn.shape
    image_h, image_w = image_hw
    grid_h, grid_w, prefix = _grid_dims(n_tokens, image_hw)
    n_patches = grid_h * grid_w

    x = max(0, min(int(query_xy[0]), image_w - 1))
    y = max(0, min(int(query_xy[1]), image_h - 1))
    patch_x = max(0, min(x // 16, grid_w - 1))
    patch_y = max(0, min(y // 16, grid_h - 1))
    patch_idx = patch_y * grid_w + patch_x
    query_token = prefix + patch_idx

    patch_weights = attn[0, :, query_token, prefix:prefix + n_patches]
    if patch_weights.shape[-1] != n_patches:
        raise RuntimeError(
            f"Expected {n_patches} patch weights, got {patch_weights.shape[-1]}"
        )
    patch_weights = patch_weights.reshape(n_heads, grid_h, grid_w)
    upsampled = F.interpolate(
        patch_weights.unsqueeze(1),
        size=(image_h, image_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    mean_map = upsampled.mean(dim=0).numpy()
    head_maps = upsampled.numpy()

    p = patch_weights.reshape(n_heads, -1)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = (-(p * p.clamp_min(1e-12).log2()).sum(dim=1)).mean().item()
    stats = {
        "n_heads": n_heads,
        "n_tokens": n_tokens,
        "token_prefix": prefix,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "query_x": x,
        "query_y": y,
        "patch_x": patch_x,
        "patch_y": patch_y,
        "query_token": query_token,
        "mean_entropy_bits": entropy,
        "max_attention": float(patch_weights.mean(dim=0).max().item()),
    }
    return mean_map, head_maps, stats


def _attention_spread_stats(
    attn: torch.Tensor,
    image_hw: tuple[int, int],
) -> dict[str, float]:
    """Global descriptors of how localised the spatial attention is.

    Averaged over ALL patch-query tokens and all heads (one forward pass), so it
    needs no query-point selection:
      mean_entropy_bits  — higher = attention is spread over more patches.
      mean_radius_patches — expected distance (in patch units) from a query patch
                            to the patches it attends to; higher = longer-range.
    """
    n_tokens = attn.shape[2]
    grid_h, grid_w, prefix = _grid_dims(n_tokens, image_hw)
    n_patches = grid_h * grid_w

    # Patch-to-patch block, renormalised over keys after dropping prefix columns.
    w = attn[0, :, prefix:prefix + n_patches, prefix:prefix + n_patches]  # [H, P, P]
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Patch coordinates and pairwise distance matrix (patch units).
    ys, xs = np.divmod(np.arange(n_patches), grid_w)
    coords = np.stack([ys, xs], axis=1).astype(np.float32)  # [P, 2]
    dist = np.sqrt(
        ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    )  # [P, P]
    dist_t = torch.from_numpy(dist)

    radius = (w * dist_t).sum(dim=-1).mean().item()
    entropy = (-(w * w.clamp_min(1e-12).log2()).sum(dim=-1)).mean().item()
    return {
        "mean_entropy_bits": float(entropy),
        "mean_radius_patches": float(radius),
        "max_radius_patches": float(np.sqrt((grid_h - 1) ** 2 + (grid_w - 1) ** 2)),
    }


def _channel_saliency(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    device: torch.device,
    target: str = "energy",
    clean: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-input-channel gradient saliency of the decoded output.

    Answers "how much does each input slice drive the prediction" by
    differentiating a scalar output target w.r.t. the input channels. Returns:
      per_channel_mean : [C]      mean |grad| per channel (sensitivity)
      saliency_maps    : [C,H,W]  spatial |grad| per channel
    target='energy' uses mean(output^2) (no clean reference needed); target='mse'
    uses MSE to the clean target.
    """
    x = noisy.clone().to(device).requires_grad_(True)
    out = model(x.unsqueeze(0))  # [1, 1, H, W]
    if target == "mse":
        if clean is None:
            raise ValueError("target='mse' requires a clean target")
        ref = clean.to(device).unsqueeze(0)
        scalar = F.mse_loss(out, ref)
    else:
        scalar = (out ** 2).mean()
    model.zero_grad(set_to_none=True)
    scalar.backward()
    grad = x.grad.detach().abs().cpu().numpy()  # [C, H, W]
    per_channel_mean = grad.mean(axis=(1, 2))
    return per_channel_mean, grad


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    denom = float(x.max())
    return x / denom if denom > 1e-12 else x


def _box_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Fast separable box blur via integral image (no scipy dependency)."""
    x = x.astype(np.float32)
    h, w = x.shape
    pad = k // 2
    padded = np.pad(x, pad, mode="reflect")
    csum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    csum = np.pad(csum, ((1, 0), (1, 0)), mode="constant")
    out = csum[k:, k:] - csum[:-k, k:] - csum[k:, :-k] + csum[:-k, :-k]
    return (out / (k * k))[:h, :w]


def _draw_cross(ax, xy: tuple[int, int]) -> None:
    x, y = xy
    ax.plot([x], [y], marker="+", markersize=12, markeredgewidth=2.0, color="cyan")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save_comparison_grid(
    out_path: Path,
    noisy_center: np.ndarray,
    clean: np.ndarray,
    query_results: dict[str, dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]],
    queries: list[tuple[str, tuple[int, int]]],
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(queries), 5, figsize=(18, 4.2 * len(queries)))
    if len(queries) == 1:
        axes = np.asarray([axes])

    for row, (query_label, xy) in enumerate(queries):
        axes[row, 0].imshow(noisy_center, cmap="seismic", vmin=-2, vmax=2, aspect="auto")
        _draw_cross(axes[row, 0], xy)
        axes[row, 0].set_title(f"Noisy input\n{query_label}")
        axes[row, 0].axis("off")

        for col, key in enumerate(_VARIANT_ORDER, start=1):
            mean_map = _normalize_map(query_results[query_label][key][0])
            axes[row, col].imshow(noisy_center, cmap="gray", aspect="auto")
            axes[row, col].imshow(mean_map, cmap="magma", alpha=0.55, aspect="auto")
            _draw_cross(axes[row, col], xy)
            ent = query_results[query_label][key][2]["mean_entropy_bits"]
            axes[row, col].set_title(f"{_DISPLAY_LABEL[key]}\nattention (entropy {ent:.2f} bits)")
            axes[row, col].axis("off")

        axes[row, 4].imshow(clean, cmap="seismic", vmin=-2, vmax=2, aspect="auto")
        _draw_cross(axes[row, 4], xy)
        axes[row, 4].set_title("Clean target")
        axes[row, 4].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_head_breakdown(
    out_path: Path,
    noisy_center: np.ndarray,
    query_results: dict[str, dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]],
    queries: list[tuple[str, tuple[int, int]]],
    variant_key: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_heads = query_results[queries[0][0]][variant_key][1].shape[0]
    fig, axes = plt.subplots(len(queries), n_heads, figsize=(3.1 * n_heads, 3.2 * len(queries)))
    if len(queries) == 1:
        axes = np.asarray([axes])

    for row, (query_label, xy) in enumerate(queries):
        head_maps = query_results[query_label][variant_key][1]
        for head in range(n_heads):
            ax = axes[row, head]
            ax.imshow(noisy_center, cmap="gray", aspect="auto")
            ax.imshow(_normalize_map(head_maps[head]), cmap="magma", alpha=0.55, aspect="auto")
            _draw_cross(ax, xy)
            ax.set_title(f"{query_label}\nhead {head}")
            ax.axis("off")

    fig.suptitle(f"{_DISPLAY_LABEL[variant_key]} per-head attention (final block)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_saliency_maps(
    out_path: Path,
    input_slices: np.ndarray,
    saliency_maps: np.ndarray,
    offsets: list[int],
    variant_key: str,
) -> None:
    """Top row: the input neighbour slices. Bottom row: per-channel |grad| saliency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = saliency_maps.shape[0]
    center = c // 2
    # Smooth the spiky raw gradients and share one colour scale across channels so
    # the maps show both WHERE and the RELATIVE magnitude of each neighbour's
    # contribution. Clip to the 99th percentile so single-pixel spikes don't
    # wash everything else out.
    smoothed = np.stack([_box_smooth(saliency_maps[ch], k=5) for ch in range(c)], axis=0)
    vmax = float(np.percentile(smoothed, 99.0))
    vmax = vmax if vmax > 1e-12 else float(smoothed.max() or 1.0)

    fig, axes = plt.subplots(2, c, figsize=(3.3 * c, 6.6))
    for ch in range(c):
        off = offsets[ch]
        tag = "t" if off == 0 else (f"t{off:+d}")
        marker = " (centre)" if off == 0 else ""
        axes[0, ch].imshow(input_slices[ch], cmap="seismic", vmin=-2, vmax=2, aspect="auto")
        axes[0, ch].set_title(f"input {tag}{marker}")
        axes[0, ch].axis("off")
        axes[1, ch].imshow(smoothed[ch], cmap="magma", vmin=0.0, vmax=vmax, aspect="auto")
        axes[1, ch].set_title(f"|gradient| {tag}")
        axes[1, ch].axis("off")
        if ch == center:
            for r in (0, 1):
                for spine in axes[r, ch].spines.values():
                    spine.set_visible(True)
                    spine.set_color("cyan")
                    spine.set_linewidth(2.5)
                axes[r, ch].axis("on")
                axes[r, ch].set_xticks([])
                axes[r, ch].set_yticks([])

    fig.suptitle(
        f"{_DISPLAY_LABEL[variant_key]} input-channel saliency "
        f"(how much each neighbour slice drives the output)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_saliency_bars(
    out_path: Path,
    per_channel: dict[str, tuple[np.ndarray, list[int]]],
) -> None:
    """Grouped bars: relative per-channel saliency for each 2.5D variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [k for k in _VARIANT_ORDER if k in per_channel]
    fig, axes = plt.subplots(1, len(keys), figsize=(5.5 * len(keys), 4.2), squeeze=False)
    for col, key in enumerate(keys):
        vals, offsets = per_channel[key]
        frac = vals / max(vals.sum(), 1e-12)
        labels = ["t" if o == 0 else f"t{o:+d}" for o in offsets]
        colors = ["#d62728" if o == 0 else "#1f77b4" for o in offsets]
        ax = axes[0, col]
        bars = ax.bar(labels, frac, color=colors)
        ax.set_title(f"{_DISPLAY_LABEL[key]}")
        ax.set_ylabel("relative saliency (fraction)")
        ax.set_ylim(0, max(frac) * 1.2)
        for b, fv in zip(bars, frac):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{fv:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Per-neighbour input saliency (red = centre slice)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _save_spread_summary(out_path: Path, spread_rows: list[dict[str, Any]]) -> None:
    """Bar charts: mean attention entropy and mean attention radius per variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg: dict[str, dict[str, list[float]]] = {}
    for row in spread_rows:
        agg.setdefault(row["variant_key"], {"entropy": [], "radius": []})
        agg[row["variant_key"]]["entropy"].append(row["mean_entropy_bits"])
        agg[row["variant_key"]]["radius"].append(row["mean_radius_patches"])

    keys = [k for k in _VARIANT_ORDER if k in agg]
    labels = [_DISPLAY_LABEL[k] for k in keys]
    ent = [float(np.mean(agg[k]["entropy"])) for k in keys]
    rad = [float(np.mean(agg[k]["radius"])) for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(labels, ent, color="#4c72b0")
    axes[0].set_title("Mean spatial attention entropy")
    axes[0].set_ylabel("bits")
    axes[1].bar(labels, rad, color="#55a868")
    axes[1].set_title("Mean attention radius")
    axes[1].set_ylabel("patches (16 px each)")
    for ax, vals in zip(axes, (ent, rad)):
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Spatial attention spread (final block, averaged over query patches)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _build_datasets(project_root: Path, stride: int, data_seed: int):
    ds_2d = _build_shared_dataset(project_root, stride, data_seed=data_seed)
    return {
        "2D": ds_2d,
        "3ch": _build_variant_dataset(ds_2d, "2.5d_3ch"),
        "5ch": _build_variant_dataset(ds_2d, "2.5d_5ch"),
    }


def _load_models(project_root: Path, training_seed: int, data_seed: int, device: torch.device):
    models = {}
    for variant in _VARIANTS:
        ckpt = _resolve_checkpoint(
            project_root, training_seed, variant, data_seed=data_seed, strict=True
        )
        models[variant["key"]] = _load_model(ckpt, variant, device)
    missing = {"2D", "3ch", "5ch"} - set(models)
    if missing:
        raise RuntimeError(f"Missing required model(s): {sorted(missing)}")
    return models


def _archive_legacy_outputs(out_dir: Path) -> None:
    """Move pre-existing fixed-split figures into a _legacy_fixed_split subfolder.

    Only the old `attention_sample*` naming is archived. New-protocol outputs
    (`attn_ds*`, `attention_spread*`) are never touched, so re-running does not
    re-archive freshly generated figures.
    """
    if not out_dir.exists():
        return
    legacy = out_dir / "_legacy_fixed_split"
    stale = [p for p in out_dir.glob("attention_sample*.png") if p.is_file()]
    if not stale:
        return
    legacy.mkdir(exist_ok=True)
    for p in stale:
        target = legacy / p.name
        if target.exists():
            target.unlink()
        shutil.move(str(p), str(target))
    print(f"Archived {len(stale)} stale fixed-split file(s) -> {legacy}")


# ---------------------------------------------------------------------------
# Multi-run aggregation (9-run mean +/- std for the thesis tables)
# ---------------------------------------------------------------------------
def _aggregate_saliency_and_spread(
    project_root: Path,
    out_dir: Path,
    data_seeds: list[int],
    training_seeds: list[int],
    stride: int,
    n_per_run: int,
    layer_index: int,
    saliency_target: str,
    device: torch.device,
) -> None:
    """Aggregate per-channel saliency and attention spread over many runs.

    For every (data_seed, training_seed) run we take a uniform held-out subsample
    of that data seed's comparison_metadata.csv (the same CSV the figures use),
    compute per-sample saliency fractions (3ch, 5ch) and attention spread (all
    variants), and reduce to one mean value per run. We then report mean +/- std
    ACROSS the runs, matching the thesis "mean +/- std over nine runs" convention.
    The single-slice figures are illustrations; these CSVs carry the numbers.
    """
    # Per-run records.
    sal_runs: dict[str, list[dict[int, float]]] = {"3ch": [], "5ch": []}
    spread_runs: dict[str, dict[str, list[float]]] = {
        k: {"entropy": [], "radius": []} for k in _VARIANT_ORDER
    }
    n_runs = 0

    for data_seed in data_seeds:
        metadata = (
            project_root / "experiments" / "summaries" / "comparison_panels"
            / f"ds{data_seed}" / "comparison_metadata.csv"
        )
        rows = _read_metadata(metadata)
        if not rows:
            print(f"  [agg] ds{data_seed}: no comparison_metadata.csv; skipping data seed.")
            continue
        rows_sorted = sorted(rows, key=lambda r: int(r["rank_by_gap"]))
        step = max(1, len(rows_sorted) // n_per_run)
        sel_rows = rows_sorted[::step][:n_per_run]

        datasets = _build_datasets(project_root, stride, data_seed)
        anchor_ds = datasets["5ch"]

        for training_seed in training_seeds:
            try:
                models = _load_models(project_root, training_seed, data_seed, device)
            except RuntimeError as exc:
                print(f"  [agg] skip ds{data_seed} ts{training_seed}: {exc}")
                continue

            per_offset_frac: dict[str, dict[int, list[float]]] = {
                "3ch": defaultdict(list), "5ch": defaultdict(list)
            }
            per_variant_spread: dict[str, dict[str, list[float]]] = {
                k: {"entropy": [], "radius": []} for k in _VARIANT_ORDER
            }

            for r in sel_rows:
                sample_idx = max(0, min(int(r["sample_idx"]), len(anchor_ds) - 1))
                noisy_path, clean_path, ax, t, oh, ow = anchor_ds.samples[sample_idx]
                oh = oh or 0
                ow = ow or 0
                vol_id = int(noisy_path.stem.split("_")[-1])
                # Same index sanity check as the figure path: the slice must match.
                if vol_id != int(r["vol_id"]) or t != int(r["slice_t"]):
                    raise RuntimeError(
                        f"[agg] Index mismatch ds{data_seed} sample {sample_idx}: dataset "
                        f"gives vol={vol_id}, slice={t} but metadata says vol={r['vol_id']}, "
                        f"slice={r['slice_t']}. Check data_seed/stride consistency."
                    )

                for variant in _VARIANTS:
                    key = variant["key"]
                    noisy, clean = _load_sample_for_mode(
                        datasets[key], noisy_path, clean_path, ax, t, oh, ow
                    )
                    image_hw = (int(noisy.shape[-2]), int(noisy.shape[-1]))
                    with torch.no_grad():
                        attn, _ = _capture_attention(models[key], noisy, layer_index, device)
                    sp = _attention_spread_stats(attn, image_hw)
                    per_variant_spread[key]["entropy"].append(sp["mean_entropy_bits"])
                    per_variant_spread[key]["radius"].append(sp["mean_radius_patches"])
                    if key in ("3ch", "5ch"):
                        pcm, _ = _channel_saliency(
                            models[key], noisy, device,
                            target=saliency_target, clean=clean,
                        )
                        frac = pcm / max(pcm.sum(), 1e-12)
                        for o, fv in zip(datasets[key].offsets, frac):
                            per_offset_frac[key][int(o)].append(float(fv))

            n_runs += 1
            for key in ("3ch", "5ch"):
                sal_runs[key].append(
                    {o: float(np.mean(v)) for o, v in per_offset_frac[key].items()}
                )
            for key in _VARIANT_ORDER:
                spread_runs[key]["entropy"].append(
                    float(np.mean(per_variant_spread[key]["entropy"]))
                )
                spread_runs[key]["radius"].append(
                    float(np.mean(per_variant_spread[key]["radius"]))
                )
            print(f"  [agg] ds{data_seed} ts{training_seed}: "
                  f"{len(sel_rows)} samples done (run {n_runs})")

    if n_runs == 0:
        print("  [agg] No runs aggregated; nothing written.")
        return

    def _ms(arr: list[float]) -> tuple[float, float]:
        a = np.asarray(arr, dtype=np.float64)
        std = float(a.std(ddof=1)) if a.size > 1 else 0.0
        return float(a.mean()), std

    # saliency_aggregate.csv
    sal_path = out_dir / "saliency_aggregate.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with sal_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant_key", "offset", "mean_fraction", "std_fraction",
                         "n_runs", "n_samples_per_run"])
        for key in ("3ch", "5ch"):
            offsets = sorted({o for run in sal_runs[key] for o in run})
            for o in offsets:
                per_run = [run[o] for run in sal_runs[key] if o in run]
                m, s = _ms(per_run)
                writer.writerow([_DISPLAY_LABEL[key], o, f"{m:.4f}", f"{s:.4f}",
                                 len(per_run), n_per_run])
    print(f"Saved {sal_path}")

    # spread_aggregate.csv
    spread_path = out_dir / "spread_aggregate.csv"
    with spread_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["variant_key", "mean_entropy_bits", "std_entropy_bits",
                         "mean_radius_patches", "std_radius_patches",
                         "n_runs", "n_samples_per_run"])
        for key in _VARIANT_ORDER:
            em, es = _ms(spread_runs[key]["entropy"])
            rm, rs = _ms(spread_runs[key]["radius"])
            writer.writerow([_DISPLAY_LABEL[key], f"{em:.4f}", f"{es:.4f}",
                             f"{rm:.4f}", f"{rs:.4f}", n_runs, n_per_run])
    print(f"Saved {spread_path}")

    # Console summary.
    print(f"\n=== Aggregated over {n_runs} run(s), {n_per_run} samples/run ===")
    print("Per-neighbour saliency fraction (mean +/- std over runs):")
    for key in ("3ch", "5ch"):
        offsets = sorted({o for run in sal_runs[key] for o in run})
        parts = []
        for o in offsets:
            per_run = [run[o] for run in sal_runs[key] if o in run]
            m, s = _ms(per_run)
            lbl = "t" if o == 0 else f"t{o:+d}"
            parts.append(f"{lbl}={m:.3f}+/-{s:.3f}")
        print(f"  {_DISPLAY_LABEL[key]}: " + ", ".join(parts))
    print("Attention spread (mean +/- std over runs):")
    for key in _VARIANT_ORDER:
        em, es = _ms(spread_runs[key]["entropy"])
        rm, rs = _ms(spread_runs[key]["radius"])
        print(f"  {_DISPLAY_LABEL[key]}: entropy={em:.3f}+/-{es:.3f} bits, "
              f"radius={rm:.3f}+/-{rs:.3f} patches")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-seed", type=int, default=101,
                        help="Data split seed (new main protocol: 101/202/303). The "
                             "dataset is built at this seed so the visualized slice is "
                             "held out for the matching checkpoints.")
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--sample-index", type=int, nargs="+", default=None,
                        help="Explicit anchor sample index/indices (overrides ranks).")
    parser.add_argument("--ranks", type=int, nargs="+", default=None,
                        help="rank_by_gap rows to render (default: headline 0 + median).")
    parser.add_argument("--layer-index", type=int, default=-1,
                        help="Attention block index (default -1 = final block 11).")
    parser.add_argument("--query-xy", type=_parse_xy, nargs="+", default=None,
                        help="Manual query points 'x,y'; default auto-picks high/low amplitude.")
    parser.add_argument("--saliency-target", choices=["energy", "mse"], default="energy")
    parser.add_argument("--no-head-breakdown", action="store_true")
    parser.add_argument("--no-saliency", action="store_true")
    parser.add_argument("--no-archive", action="store_true",
                        help="Do not move existing fixed-split figures aside.")
    parser.add_argument("--aggregate-saliency", type=int, default=0,
                        help="If > 0, also run a multi-run aggregation pass over "
                             "--data-seeds x --training-seeds, using this many held-out "
                             "samples per run, and write saliency_aggregate.csv and "
                             "spread_aggregate.csv (mean +/- std over the runs). "
                             "0 disables (default); the single-slice figures are unaffected.")
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[101, 202, 303],
                        help="Data seeds for the aggregation pass (default 101 202 303).")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[42, 43, 44],
                        help="Training seeds for the aggregation pass (default 42 43 44).")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (
        project_root / "experiments" / "summaries" / "mechanism_analysis" / "attention_maps"
    )
    metadata = (
        project_root / "experiments" / "summaries" / "comparison_panels"
        / f"ds{args.data_seed}" / "comparison_metadata.csv"
    )

    _patch_self_attention()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.aggregate_saliency > 0:
        print(f"Aggregation pass: data_seeds={args.data_seeds}, "
              f"training_seeds={args.training_seeds}, "
              f"{args.aggregate_saliency} samples/run, stride={args.stride}")
        _aggregate_saliency_and_spread(
            project_root, out_dir, args.data_seeds, args.training_seeds,
            args.stride, args.aggregate_saliency, args.layer_index,
            args.saliency_target, device,
        )
        return

    print(f"Protocol: data_seed={args.data_seed}, training_seed={args.training_seed}, "
          f"stride={args.stride}, layer_index={args.layer_index}")

    if not args.no_archive:
        _archive_legacy_outputs(out_dir)

    selected = _select_samples(metadata, args.ranks, args.sample_index)
    print(f"Selected {len(selected)} sample(s): "
          + ", ".join(f"idx={s['sample_idx']}(rank {s['rank_by_gap']})" for s in selected))

    datasets = _build_datasets(project_root, args.stride, args.data_seed)
    anchor_ds = datasets["5ch"]
    models = _load_models(project_root, args.training_seed, args.data_seed, device)

    summary_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []
    layer_names: dict[str, str] = {}
    prefix = f"attn_ds{args.data_seed}_ts{args.training_seed}"

    for s_i, sel in enumerate(selected):
        sample_idx = max(0, min(sel["sample_idx"], len(anchor_ds) - 1))
        noisy_path, clean_path, ax, t, oh, ow = anchor_ds.samples[sample_idx]
        oh = oh or 0
        ow = ow or 0
        vol_id = int(noisy_path.stem.split("_")[-1])

        # Index sanity check: the slice must match what the metadata recorded,
        # otherwise the split/stride assumptions are wrong and figures are bogus.
        if sel["vol_id"] is not None:
            if vol_id != sel["vol_id"] or t != sel["slice_t"]:
                raise RuntimeError(
                    f"Index mismatch for sample {sample_idx}: dataset gives "
                    f"vol={vol_id}, slice={t} but metadata says vol={sel['vol_id']}, "
                    f"slice={sel['slice_t']}. Check data_seed/stride consistency."
                )
        is_headline = (s_i == 0)
        rank_tag = f"r{sel['rank_by_gap']:03d}" if sel["rank_by_gap"] >= 0 else "rNA"
        tag = f"{prefix}_s{sample_idx:04d}_{rank_tag}"
        print(f"\n=== Sample {sample_idx} | vol={vol_id} slice={t} "
              f"mid={sel['mid_slice_pct']}% rank_by_gap={sel['rank_by_gap']} ===")

        # Load the shared slice for every variant (same t/oh/ow anchor).
        per_variant_noisy: dict[str, torch.Tensor] = {}
        clean_t: torch.Tensor | None = None
        noisy_center_np: np.ndarray | None = None
        for variant in _VARIANTS:
            key = variant["key"]
            noisy, clean = _load_sample_for_mode(datasets[key], noisy_path, clean_path, ax, t, oh, ow)
            per_variant_noisy[key] = noisy
            if clean_t is None:
                clean_t = clean
                noisy_center_np = noisy[variant["in_chans"] // 2].numpy()
        assert clean_t is not None and noisy_center_np is not None
        clean_np = clean_t[0].numpy()

        # Query points (auto from clean target unless overridden).
        if args.query_xy:
            queries = [(f"query {i+1}", xy) for i, xy in enumerate(args.query_xy)]
        else:
            queries = _auto_query_points(clean_np)
        print("  Queries: " + ", ".join(f"{lbl}={xy}" for lbl, xy in queries))

        query_results: dict[str, dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]] = {
            label: {} for label, _ in queries
        }
        layer_names: dict[str, str] = {}

        for variant in _VARIANTS:
            key = variant["key"]
            noisy = per_variant_noisy[key]
            attn, layer_name = _capture_attention(models[key], noisy, args.layer_index, device)
            layer_names[key] = layer_name
            image_hw = (int(noisy.shape[-2]), int(noisy.shape[-1]))

            spread = _attention_spread_stats(attn, image_hw)
            spread_rows.append({
                "sample_idx": sample_idx, "vol_id": vol_id, "slice_t": t,
                "variant_key": key, "layer_name": layer_name, **spread,
            })

            for query_label, xy in queries:
                mean_map, head_maps, stats = _query_attention_map(attn, xy, image_hw)
                query_results[query_label][key] = (mean_map, head_maps, stats)
                summary_rows.append({
                    "sample_idx": sample_idx, "vol_id": vol_id, "slice_t": t,
                    "mid_slice_pct": sel["mid_slice_pct"], "rank_by_gap": sel["rank_by_gap"],
                    "variant_key": key, "layer_name": layer_name,
                    "query_label": query_label, **stats,
                })

        title = (
            f"Attention maps | data_seed={args.data_seed}, training_seed={args.training_seed} | "
            f"vol={vol_id}, slice={t} ({sel['mid_slice_pct']}% depth), "
            f"rank_by_gap={sel['rank_by_gap']}, block={args.layer_index}"
        )
        _save_comparison_grid(
            out_dir / f"{tag}_comparison.png",
            noisy_center_np, clean_np, query_results, queries, title,
        )

        if is_headline and not args.no_head_breakdown:
            for key in _VARIANT_ORDER:
                _save_head_breakdown(
                    out_dir / f"{tag}_heads_{key}.png",
                    noisy_center_np, query_results, queries, variant_key=key,
                )

        if is_headline and not args.no_saliency:
            per_channel: dict[str, tuple[np.ndarray, list[int]]] = {}
            for key in ("3ch", "5ch"):
                noisy = per_variant_noisy[key]
                offsets = datasets[key].offsets
                pcm, smaps = _channel_saliency(
                    models[key], noisy, device,
                    target=args.saliency_target, clean=clean_t,
                )
                per_channel[key] = (pcm, offsets)
                _save_saliency_maps(
                    out_dir / f"{tag}_saliency_{key}.png",
                    noisy.numpy(), smaps, offsets, variant_key=key,
                )
                frac = pcm / max(pcm.sum(), 1e-12)
                desc = ", ".join(
                    f"t{o:+d}={f:.2f}" if o else f"t={f:.2f}"
                    for o, f in zip(offsets, frac)
                )
                print(f"  [{key}] relative neighbour saliency: {desc}")
            _save_saliency_bars(out_dir / f"{tag}_saliency_bars.png", per_channel)

    # Spread summary figure across all rendered samples.
    if spread_rows:
        _save_spread_summary(out_dir / "attention_spread_summary.png", spread_rows)

    # CSVs
    summary_path = out_dir / "attention_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved {summary_path}")

    spread_path = out_dir / "attention_spread.csv"
    with spread_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(spread_rows[0].keys()))
        writer.writeheader()
        writer.writerows(spread_rows)
    print(f"Saved {spread_path}")

    print("\nLayer captured for each variant (last sample):")
    for key, name in layer_names.items():
        print(f"  {key}: {name}")


if __name__ == "__main__":
    main()
