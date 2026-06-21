"""Zero-shot robustness evaluation on the Netherlands F3 field dataset.

F3 is unlabelled field data, so this script reports no-reference diagnostics
only. It never reports accuracy metrics such as MS-SSIM, MSE, or PSNR for F3.

Usage:
    python evaluation/evaluate_robustness.py \
        --config configs/<config>_daic.yaml \
        --checkpoint /path/to/best.pt \
        --dataset f3 \
        --data-root /path/to/Dataset/F3 \
        --out-dir experiments/runs/robustness/f3/2d/.../seed42_run01 \
        [--sample-count all --common-context-radius 2] \
        [--max-samples 2]
"""

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from data.robustness import F3FieldDataset
from evaluation.common.metrics import compute_ms_ssim_r
from models.pipeline import DINOv3Denoiser


# ---------------------------------------------------------------------------
# Model loading (mirrors evaluate.py)
# ---------------------------------------------------------------------------

def _resolve_cfg(cfg_dir: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    return (cfg_dir / rel).resolve()


def _load_model(cfg: dict, cfg_dir: Path, ckpt_path: Path, device: torch.device) -> DINOv3Denoiser:
    project_root = SRC.parent
    repo_dir = project_root / "external" / "dinov3"
    weights = _resolve_cfg(cfg_dir, cfg["model"]["weights"])
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
    print(
        f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
        f"(val MS-SSIM={ckpt.get('val_ms_ssim', float('nan')):.4f})"
    )
    return model


# ---------------------------------------------------------------------------
# F3 diagnostics (no clean reference)
# ---------------------------------------------------------------------------

def _residual_energy_fraction(denoised: torch.Tensor, noisy_center: torch.Tensor) -> float:
    residual = noisy_center - denoised
    numerator = float((residual ** 2).sum())
    denominator = float((noisy_center ** 2).sum())
    return numerator / denominator if denominator > 1e-12 else 0.0


def _residual_input_corr(denoised: torch.Tensor, noisy_center: torch.Tensor) -> float:
    residual = (noisy_center - denoised).flatten()
    noisy = noisy_center.flatten()
    residual = residual - residual.mean()
    noisy = noisy - noisy.mean()
    numerator = float((residual * noisy).sum())
    denominator = float(residual.norm() * noisy.norm())
    return numerator / denominator if denominator > 1e-12 else 0.0


def _amplitude_ratio(denoised: torch.Tensor, noisy_center: torch.Tensor) -> float:
    denoised_std = float(denoised.std())
    noisy_std = float(noisy_center.std())
    return denoised_std / noisy_std if noisy_std > 1e-12 else 0.0


def _low_freq_energy_frac(arr: torch.Tensor) -> float:
    """Return the residual energy fraction in the central half of the 2D FFT grid."""
    x = arr.squeeze().float().numpy()
    fft = np.abs(np.fft.fftshift(np.fft.fft2(x)))
    h, w = fft.shape
    h4, w4 = h // 4, w // 4
    low_energy = float((fft[h4:h - h4, w4:w - w4] ** 2).sum())
    total_energy = float((fft ** 2).sum())
    return low_energy / total_energy if total_energy > 1e-12 else 0.0


def _f3_metrics(denoised: torch.Tensor, noisy: torch.Tensor) -> dict:
    noisy_center = noisy[:, noisy.shape[1] // 2:noisy.shape[1] // 2 + 1]
    return {
        "ms_ssim_r": compute_ms_ssim_r(denoised, noisy),
        "residual_energy_frac": _residual_energy_fraction(denoised, noisy_center),
        "residual_input_corr": _residual_input_corr(denoised, noisy_center),
        "denoised_input_amplitude_ratio": _amplitude_ratio(denoised, noisy_center),
        "low_freq_residual_energy_frac": _low_freq_energy_frac((noisy_center - denoised)[0]),
    }


def _save_f3_panel(noisy_center: np.ndarray, denoised: np.ndarray, path: Path) -> None:
    residual = noisy_center - denoised
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [("Noisy input", noisy_center), ("Denoised", denoised), ("Residual", residual)]
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img.squeeze(), cmap="seismic", aspect="auto", vmin=-2, vmax=2)
        ax.set_title(title)
        ax.axis("off")
    plt.suptitle("F3 Field Transfer - No Ground Truth", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

PANEL_SECTION_FRACTION = [0.0, 0.33, 0.67, 1.0]


def _parse_sample_count(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"sample count must be a positive integer or 'all', got {value!r}"
        ) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"sample count must be a positive integer or 'all', got {value!r}"
        )
    return parsed


def _normalise_meta_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _validate_model_data_channels(in_chans: int, dataset: object) -> None:
    offsets = getattr(dataset, "offsets", None)
    if offsets is None:
        return
    offsets = list(offsets)
    if in_chans != len(offsets):
        mode = getattr(dataset, "mode", "<unknown>")
        raise ValueError(
            f"model.in_chans={in_chans} does not match F3 data.mode={mode!r}, "
            f"which produces {len(offsets)} channel(s) with offsets {offsets}."
        )


def run_f3(
    model: DINOv3Denoiser,
    cfg: dict,
    data_root: Path,
    out_dir: Path,
    device: torch.device,
    orientation: str,
    sample_count: int | None,
    common_context_radius: int | None,
    section_min: int | None,
    section_max: int | None,
    max_samples: int | None,
    batch_size: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = data_root / "processed" / "f3_original.npy"
    meta_path = data_root / "processed" / "f3_meta.json"

    if not npy_path.exists():
        sys.exit(
            f"F3 canonical volume not found at {npy_path}\n"
            "Run: python data/prepare_f3.py --input <f3.segy> --output <data_root>/processed/"
        )

    mode = str(cfg["data"]["mode"])
    neighbor_stride = int(cfg["data"].get("neighbor_stride", 1))

    ds = F3FieldDataset(
        npy_path=npy_path,
        mode=mode,
        orientation=orientation,
        sample_count=sample_count,
        crop_size=224,
        neighbor_stride=neighbor_stride,
        common_context_radius=common_context_radius,
        meta_path=meta_path if meta_path.exists() else None,
        section_min=section_min,
        section_max=section_max,
    )
    _validate_model_data_channels(int(cfg["model"].get("in_chans", 3)), ds)

    if max_samples is not None:
        from torch.utils.data import Subset

        ds = Subset(ds, list(range(min(max_samples, len(ds)))))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    n = len(ds)
    panel_indices = sorted(set(int(f * (n - 1)) for f in PANEL_SECTION_FRACTION))

    rows: list[dict] = []
    panel_batch_items: dict[int, tuple] = {}
    sample_counter = 0

    print(f"\n--- F3 field transfer ({orientation}) ---")
    with torch.no_grad():
        for noisy_batch, meta_batch in loader:
            noisy_batch = noisy_batch.to(device)
            pred_batch = model(noisy_batch).cpu()
            noisy_batch = noisy_batch.cpu()

            for i in range(noisy_batch.shape[0]):
                noisy_i = noisy_batch[i:i + 1]
                pred_i = pred_batch[i:i + 1]
                metrics = _f3_metrics(pred_i, noisy_i)
                meta_i = {k: _normalise_meta_value(v[i]) for k, v in meta_batch.items()}
                rows.append({**meta_i, **metrics})

                if sample_counter in panel_indices:
                    panel_batch_items[sample_counter] = (noisy_i, pred_i, meta_i)
                sample_counter += 1

    in_chans = int(cfg["model"].get("in_chans", 3))
    center_channel = in_chans // 2
    for sample_idx, (noisy_i, pred_i, meta_i) in panel_batch_items.items():
        panel_name = f"f3_panel_{meta_i.get('section_global_id', sample_idx)}.png"
        _save_f3_panel(
            noisy_i[0, center_channel].numpy(),
            pred_i[0, 0].numpy(),
            out_dir / panel_name,
        )

    if rows:
        fieldnames = list(rows[0].keys())
        csv_path = out_dir / "f3_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        agg_keys = [
            "ms_ssim_r",
            "residual_energy_frac",
            "residual_input_corr",
            "denoised_input_amplitude_ratio",
            "low_freq_residual_energy_frac",
        ]
        for key in agg_keys:
            vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
            if vals:
                print(f"  {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
        print(f"  Results: {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation on Netherlands F3.")
    parser.add_argument("--config", required=True, help="Config YAML (relative to src/ or absolute).")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt.")
    parser.add_argument("--dataset", required=True, choices=["f3"])
    parser.add_argument("--data-root", required=True, help="Root of F3 dataset.")
    parser.add_argument("--out-dir", required=True, help="Output directory for this run.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (smoke testing).")
    parser.add_argument(
        "--orientation",
        default="both",
        choices=["inline", "crossline", "both", "timeslice"],
        help=(
            "F3 section orientation (default: both). 'timeslice' takes horizontal "
            "inline x crossline planes at fixed time samples, matching the "
            "Image Impeccable training slice orientation."
        ),
    )
    parser.add_argument(
        "--section-min",
        type=int,
        default=None,
        help=(
            "Lowest valid center index (inclusive). For 'timeslice', use to skip "
            "the shallow no-data zone of the F3 survey, e.g. --section-min 50."
        ),
    )
    parser.add_argument(
        "--section-max",
        type=int,
        default=None,
        help="Highest valid center index (exclusive upper bound). Default: full range.",
    )
    parser.add_argument(
        "--sample-count",
        type=_parse_sample_count,
        default=32,
        help="Sections per orientation to sample, or 'all' for every valid center (default: 32).",
    )
    parser.add_argument(
        "--common-context-radius",
        type=int,
        default=None,
        help=(
            "Optional shared valid-center radius. Use 2 to evaluate 2D, 3ch, "
            "and 5ch on the same F3 section IDs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.common_context_radius is not None and args.common_context_radius < 0:
        parser.error("--common-context-radius must be >= 0")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (SRC / cfg_path).resolve()
    else:
        cfg_path = cfg_path.resolve()

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Output: {out_dir}")
    print(f"Orientation: {args.orientation}")
    print(f"Sample count: {'all' if args.sample_count is None else args.sample_count}")
    print(f"Common context radius: {args.common_context_radius}")
    print(f"Section window: [{args.section_min}, {args.section_max})")

    model = _load_model(cfg, cfg_path.parent, ckpt_path, device)
    run_f3(
        model=model,
        cfg=cfg,
        data_root=data_root,
        out_dir=out_dir,
        device=device,
        orientation=args.orientation,
        sample_count=args.sample_count,
        common_context_radius=args.common_context_radius,
        section_min=args.section_min,
        section_max=args.section_max,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )

    print("\nRobustness evaluation complete.")


if __name__ == "__main__":
    main()
