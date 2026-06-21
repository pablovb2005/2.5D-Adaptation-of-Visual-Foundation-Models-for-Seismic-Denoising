"""Stitched full-section evaluation for Image Impeccable.

Runs Hann-weighted overlap stitching over the full 300×300 section rather than
only the center 224×224 crop. Uses existing center-crop-trained checkpoints —
no retraining required.

Two metric sets are written per sample:
  center  — 224×224 center region, directly comparable to evaluate.py results.
  full    — full HxW section, evaluates border reconstruction.

Normalization: each 224×224 patch is z-scored independently before model input,
matching the per-crop z-score used during training. The clean target uses
section-level z-score as a consistent amplitude reference. MS-SSIM and PSNR
both apply _to_unit (min-max → [0,1]) before comparison, making them robust to
the amplitude-frame mismatch across stitched patches. MSE values are NOT
directly comparable to main evaluate.py results.

Usage:
    python evaluation/evaluate_stitched.py \\
        --config configs/dinov3_vits_2d.yaml \\
        --checkpoint path/to/best.pt

To refresh only the qualitative example after metrics already exist:
    python evaluation/evaluate_stitched.py \\
        --config configs/dinov3_vits_2d.yaml \\
        --checkpoint path/to/best.pt \\
        --example-only
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from data.image_impeccable import ThinkOnwardDataset
from data.input_modes import make_offsets
from models.pipeline import DINOv3Denoiser
from evaluation.common.metrics import (
    compute_ms_ssim,
    compute_ms_ssim_r,
    compute_mse,
    compute_psnr,
)

_PATCH_SIZE = 224


def _resolve(cfg_dir: Path, rel: str) -> Path:
    return (cfg_dir / rel).resolve()


def _load_model(cfg: dict, cfg_dir: Path, ckpt_path: Path, device: torch.device):
    project_root = SRC.parent
    repo_dir = project_root / "external" / "dinov3"
    weights = _resolve(cfg_dir, cfg["model"]["weights"])
    m_cfg = cfg["model"]
    model = DINOv3Denoiser(
        repo_dir=repo_dir,
        weights_path=weights,
        model_name=m_cfg["name"],
        in_chans=int(m_cfg.get("in_chans", 3)),
        lora_rank=m_cfg["lora_rank"],
        lora_alpha=m_cfg["lora_alpha"],
        lora_dropout=m_cfg["lora_dropout"],
        lora_targets=m_cfg["lora_targets"],
        patch_emb_init=str(m_cfg.get("patch_emb_init", "mixed")),
        full_finetune=bool(m_cfg.get("full_finetune", False)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    val_score = ckpt.get("val_ms_ssim")
    val_text = f"{float(val_score):.4f}" if isinstance(val_score, (int, float)) else "N/A"
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')} (val MS-SSIM={val_text})")
    return model


def _full_section(vol: np.ndarray, ax: int, t: int) -> np.ndarray:
    if ax == 0:
        return vol[t]
    if ax == 1:
        return vol[:, t, :]
    return vol[:, :, t]


def _zscore_section(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    return (x - x.mean()) / std if std > 1e-8 else x - x.mean()


def _patch_positions(n: int, ps: int = _PATCH_SIZE) -> list[int]:
    """Return edge-aligned patch start positions for a spatial axis of length n."""
    if n <= ps:
        return [0]
    return sorted({0, n - ps})


def _hann_window(ps: int = _PATCH_SIZE) -> np.ndarray:
    # np.hanning endpoints are 0; shift by ±1 to avoid zero-weight corners.
    w = np.hanning(ps + 2)[1:-1].astype(np.float32)
    return np.outer(w, w)


def _stitch_section(
    noisy_channels: list[np.ndarray],
    model: DINOv3Denoiser,
    device: torch.device,
) -> np.ndarray:
    """Return stitched prediction using Hann-weighted overlap averaging.

    Args:
        noisy_channels: per-channel raw float32 full sections, each [H, W].
            Each 224×224 crop is z-scored independently inside this function,
            matching the per-crop normalization used during training.
        model: trained denoiser — takes [1, C, ps, ps], returns [1, 1, ps, ps].
        device: torch device.

    Returns:
        Stitched prediction [H, W], same shape as input sections.
    """
    H, W = noisy_channels[0].shape
    ps = _PATCH_SIZE
    hann_w = _hann_window(ps)

    accum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for roff in _patch_positions(H, ps):
        for coff in _patch_positions(W, ps):
            # Per-crop z-score matching training normalization.
            crops = [
                _zscore_section(ch[roff:roff + ps, coff:coff + ps])
                for ch in noisy_channels
            ]
            patches = np.stack(crops, axis=0)  # [C, ps, ps]
            x = torch.from_numpy(patches[None]).to(device)  # [1, C, ps, ps]
            with torch.no_grad():
                pred = model(x).cpu().numpy()[0, 0]  # [ps, ps]
            accum[roff:roff + ps, coff:coff + ps] += pred * hann_w
            weight_sum[roff:roff + ps, coff:coff + ps] += hann_w

    return accum / np.maximum(weight_sum, 1e-8)


def _save_figure(noisy: np.ndarray, pred: np.ndarray, clean: np.ndarray, path: Path) -> None:
    residual = noisy - pred
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, title, img in zip(
        axes,
        ["Noisy input", "Denoised (stitched)", "Clean (GT)", "Residual"],
        [noisy, pred, clean, residual],
    ):
        ax.imshow(img, cmap="seismic", aspect="auto", vmin=-2, vmax=2)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_example_index(
    samples: list[tuple],
    example_policy: str,
    example_slice: int | None,
) -> int:
    """Pick the sample used for the saved qualitative example.

    The legacy behavior saved sample 0, which is a boundary slice for all main
    variants. Boundary slices in Image Impeccable can have very low physical
    amplitude and look like speckle after z-scoring, so the default now selects
    a mid-volume slice.
    """
    if not samples:
        raise RuntimeError("Cannot select an example from an empty sample list.")
    if example_policy == "first" and example_slice is None:
        return 0
    if example_policy != "middle":
        raise ValueError(f"Unknown example_policy={example_policy!r}")

    shape_cache: dict[Path, tuple[int, ...]] = {}
    best_idx = 0
    best_rank: tuple[int, int] | None = None
    for idx, (noisy_path, _clean_path, ax, t, *_rest) in enumerate(samples):
        noisy_path = Path(noisy_path)
        if noisy_path not in shape_cache:
            vol = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
            shape_cache[noisy_path] = tuple(vol.shape)
        n_slices = shape_cache[noisy_path][ax]
        target_t = n_slices // 2 if example_slice is None else int(example_slice)
        rank = (abs(int(t) - target_t), idx)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_idx = idx
    return best_idx


def _metrics_for_region(
    pred: np.ndarray,
    clean: np.ndarray,
    noisy_center: np.ndarray,
) -> dict[str, float]:
    """Compute MS-SSIM, MS-SSIM-R, MSE, PSNR for a [H, W] region."""
    p = torch.from_numpy(pred[None, None])
    c = torch.from_numpy(clean[None, None])
    n = torch.from_numpy(noisy_center[None, None])
    return {
        "ms_ssim": compute_ms_ssim(p, c),
        "ms_ssim_r": compute_ms_ssim_r(p, n),
        "mse": compute_mse(p, c),
        "psnr": compute_psnr(p, c),
    }


def _process_sample(
    sample: tuple,
    offsets: list[int],
    center_ch: int,
    model: DINOv3Denoiser,
    device: torch.device,
) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    noisy_path, clean_path, ax, t, *_ = sample
    noisy_vol = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
    clean_vol = np.load(clean_path, mmap_mode="r", allow_pickle=False)

    # Load raw float32 sections; per-crop z-score is applied inside _stitch_section.
    noisy_channels: list[np.ndarray] = []
    for dt in offsets:
        sec = _full_section(noisy_vol, ax, t + dt).astype(np.float32)
        noisy_channels.append(sec)

    clean_section = _full_section(clean_vol, ax, t).astype(np.float32)
    # Section-level z-score for a consistent full-section target reference.
    clean_norm = _zscore_section(clean_section)
    # Noisy center channel (section-level z-score) for the qualitative figure
    # and MS-SSIM-R reference; _to_unit in metrics absorbs the amplitude frame.
    noisy_center_norm = _zscore_section(noisy_channels[center_ch])

    H, W = noisy_center_norm.shape

    # Run stitched inference.
    pred_full = _stitch_section(noisy_channels, model, device)

    # Verify output shape.
    assert pred_full.shape == (H, W), f"Expected ({H},{W}), got {pred_full.shape}"

    # Full-section metrics.
    full_m = _metrics_for_region(pred_full, clean_norm, noisy_center_norm)

    # Center-region metrics: 224x224 crop at the center of the full section.
    oh = (H - _PATCH_SIZE) // 2
    ow = (W - _PATCH_SIZE) // 2
    center_m = _metrics_for_region(
        pred_full[oh:oh + _PATCH_SIZE, ow:ow + _PATCH_SIZE],
        clean_norm[oh:oh + _PATCH_SIZE, ow:ow + _PATCH_SIZE],
        noisy_center_norm[oh:oh + _PATCH_SIZE, ow:ow + _PATCH_SIZE],
    )

    vol_id = noisy_path.stem.split("_")[-1]
    row = {
        "vol_id": vol_id,
        "slice_t": t,
        **{f"center_{k}": v for k, v in center_m.items()},
        **{f"full_{k}": v for k, v in full_m.items()},
    }
    meta = {
        "vol_id": vol_id,
        "slice_t": int(t),
        "slice_axis": int(ax),
        "n_slices": int(noisy_vol.shape[ax]),
        "height": int(H),
        "width": int(W),
    }
    return row, (noisy_center_norm, pred_full, clean_norm), meta


def _save_example_meta(
    row: dict,
    meta: dict,
    out_dir: Path,
    sample_idx: int,
    sample_count: int,
    example_policy: str,
    example_slice: int | None,
) -> None:
    payload = {
        **meta,
        "sample_idx": int(sample_idx),
        "sample_count": int(sample_count),
        "example_policy": example_policy,
        "requested_example_slice": example_slice,
        "center_ms_ssim": float(row["center_ms_ssim"]),
        "full_ms_ssim": float(row["full_ms_ssim"]),
        "center_psnr": float(row["center_psnr"]),
        "full_psnr": float(row["full_psnr"]),
    }
    (out_dir / "stitched_example_meta.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def evaluate_stitched(
    cfg: dict,
    cfg_path: Path,
    ckpt_path: Path,
    example_policy: str = "middle",
    example_slice: int | None = None,
    example_only: bool = False,
) -> None:
    cfg_dir = cfg_path.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _load_model(cfg, cfg_dir, ckpt_path, device)

    data_cfg = cfg["data"]
    ii_root = _resolve(cfg_dir, data_cfg["root_dir"])
    ii_mode = str(data_cfg["mode"])
    neighbor_stride = int(data_cfg.get("neighbor_stride", 1))
    offsets = make_offsets(ii_mode, neighbor_stride)
    center_ch = len(offsets) // 2

    base_stride = int(data_cfg.get("slice_stride", 5))
    eval_stride = int(data_cfg.get("eval_slice_stride", base_stride))
    test_stride = int(data_cfg.get("test_slice_stride", eval_stride))

    # Build the test sample index using the standard split logic.
    # crop_mode="center" gives one sample per (vol, t) pair, which is what we want.
    split_ds = ThinkOnwardDataset(
        root_dir=ii_root,
        mode=ii_mode,
        split="test",
        n_train=int(data_cfg.get("n_train", 20)),
        n_val=int(data_cfg.get("n_val", 5)),
        n_test=int(data_cfg.get("n_test", 5)),
        slice_stride=test_stride,
        crop_size=_PATCH_SIZE,
        seed=int(data_cfg["seed"]),
        neighbor_stride=neighbor_stride,
        crop_mode="center",
    )

    out_dir = ckpt_path.parent / "stitched_eval_results"
    out_dir.mkdir(exist_ok=True)

    example_idx = _select_example_index(split_ds.samples, example_policy, example_slice)
    example_note = "" if example_slice is None else f" (requested slice {example_slice})"
    print(
        f"Qualitative example policy: {example_policy}{example_note}; "
        f"sample_idx={example_idx}"
    )

    if example_only:
        row, (noisy_img, pred_img, clean_img), meta = _process_sample(
            split_ds.samples[example_idx], offsets, center_ch, model, device
        )
        _save_figure(noisy_img, pred_img, clean_img, out_dir / "stitched_example_full.png")
        _save_example_meta(
            row, meta, out_dir, example_idx, len(split_ds.samples),
            example_policy, example_slice,
        )
        print(
            "Example refreshed: "
            f"vol={row['vol_id']} slice_t={row['slice_t']} "
            f"full_ms_ssim={row['full_ms_ssim']:.4f}"
        )
        print(f"Example saved to {out_dir}/")
        return

    rows: list[dict] = []
    agg: dict[str, float] = {}
    n_samples = 0

    for noisy_path, clean_path, ax, t, *_ in split_ds.samples:
        noisy_vol = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
        clean_vol = np.load(clean_path, mmap_mode="r", allow_pickle=False)

        # Load raw float32 sections; per-crop z-score is applied inside _stitch_section.
        noisy_channels: list[np.ndarray] = []
        for dt in offsets:
            sec = _full_section(noisy_vol, ax, t + dt).astype(np.float32)
            noisy_channels.append(sec)

        clean_section = _full_section(clean_vol, ax, t).astype(np.float32)
        # Section-level z-score for a consistent full-section target reference.
        clean_norm = _zscore_section(clean_section)
        # Noisy center channel (section-level z-score) for the qualitative figure
        # and MS-SSIM-R reference; _to_unit in metrics absorbs the amplitude frame.
        noisy_center_norm = _zscore_section(noisy_channels[center_ch])

        H, W = noisy_center_norm.shape

        # Run stitched inference.
        pred_full = _stitch_section(noisy_channels, model, device)

        # Verify output shape.
        assert pred_full.shape == (H, W), f"Expected ({H},{W}), got {pred_full.shape}"

        # Full-section metrics.
        full_m = _metrics_for_region(pred_full, clean_norm, noisy_center_norm)

        # Center-region metrics: 224×224 crop at the center of the full section.
        oh = (H - _PATCH_SIZE) // 2
        ow = (W - _PATCH_SIZE) // 2
        center_m = _metrics_for_region(
            pred_full[oh:oh + _PATCH_SIZE, ow:ow + _PATCH_SIZE],
            clean_norm[oh:oh + _PATCH_SIZE, ow:ow + _PATCH_SIZE],
            noisy_center_norm[oh:oh + _PATCH_SIZE, ow:ow + _PATCH_SIZE],
        )

        vol_id = noisy_path.stem.split("_")[-1]
        row = {
            "vol_id": vol_id,
            "slice_t": t,
            **{f"center_{k}": v for k, v in center_m.items()},
            **{f"full_{k}": v for k, v in full_m.items()},
        }
        rows.append(row)

        for k, v in {**{f"center_{k}": v for k, v in center_m.items()},
                     **{f"full_{k}": v for k, v in full_m.items()}}.items():
            agg[k] = agg.get(k, 0.0) + v
        n_samples += 1

        if n_samples - 1 == example_idx:
            _save_figure(
                noisy_center_norm, pred_full, clean_norm,
                out_dir / "stitched_example_full.png",
            )
            _save_example_meta(
                row,
                {
                    "vol_id": vol_id,
                    "slice_t": int(t),
                    "slice_axis": int(ax),
                    "n_slices": int(noisy_vol.shape[ax]),
                    "height": int(H),
                    "width": int(W),
                },
                out_dir,
                example_idx,
                len(split_ds.samples),
                example_policy,
                example_slice,
            )

    n = max(1, n_samples)
    agg = {k: v / n for k, v in agg.items()}

    print(f"\n--- Stitched test results ({n_samples} samples) ---")
    print(f"  Center {_PATCH_SIZE}×{_PATCH_SIZE}  MS-SSIM:   {agg['center_ms_ssim']:.4f}")
    print(f"  Center {_PATCH_SIZE}×{_PATCH_SIZE}  MSE:       {agg['center_mse']:.6f}")
    print(f"  Center {_PATCH_SIZE}×{_PATCH_SIZE}  PSNR:      {agg['center_psnr']:.2f} dB")
    print(f"  Full   section   MS-SSIM:   {agg['full_ms_ssim']:.4f}")
    print(f"  Full   section   MSE:       {agg['full_mse']:.6f}")
    print(f"  Full   section   PSNR:      {agg['full_psnr']:.2f} dB")

    fieldnames = [
        "vol_id", "slice_t",
        "center_ms_ssim", "center_ms_ssim_r", "center_mse", "center_psnr",
        "full_ms_ssim", "full_ms_ssim_r", "full_mse", "full_psnr",
    ]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--example-policy",
        choices=["middle", "first"],
        default="middle",
        help="Which sample to save as stitched_example_full.png.",
    )
    parser.add_argument(
        "--example-slice",
        type=int,
        default=None,
        help="Optional central-slice index to use for the saved example; nearest valid sample is used.",
    )
    parser.add_argument(
        "--example-only",
        action="store_true",
        help="Refresh only stitched_example_full.png and stitched_example_meta.json.",
    )
    args = parser.parse_args()

    cfg_path = (SRC / args.config).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    evaluate_stitched(
        cfg,
        cfg_path,
        Path(args.checkpoint).resolve(),
        example_policy=args.example_policy,
        example_slice=args.example_slice,
        example_only=args.example_only,
    )
