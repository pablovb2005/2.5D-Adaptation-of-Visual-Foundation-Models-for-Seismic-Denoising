"""Baseline MS-SSIM between raw noisy and clean slices (no model).

Samples volumes with the same protocol as the main experiments:
  - longest-axis slicing
  - stride=5
  - centre crop 224x224
  - per-image min-max normalisation to [0,1] before MS-SSIM (matches metrics.py)

Outputs go to experiments/summaries/baselines/.

Usage:
    python Code/DINOv3/src/evaluation/noisy_clean_baseline.py
    python Code/DINOv3/src/evaluation/noisy_clean_baseline.py \
        --dataset_root Code/Dataset/ThinkOnwards/training_data/extracted \
        --out_dir experiments/summaries/baselines
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root

import numpy as np
import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

SLICE_STRIDE = 5
CROP_SIZE = 224

_ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)


def _to_unit(x: np.ndarray) -> torch.Tensor:
    mn, mx = float(x.min()), float(x.max())
    denom = mx - mn
    if denom < 1e-8:
        return torch.zeros(1, 1, *x.shape)
    return torch.tensor((x - mn) / denom, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def _crop_slice(vol: np.ndarray, ax: int, t: int, oh: int, ow: int, cs: int) -> np.ndarray:
    if ax == 0:
        return vol[t, oh:oh + cs, ow:ow + cs]
    if ax == 1:
        return vol[oh:oh + cs, t, ow:ow + cs]
    return vol[oh:oh + cs, ow:ow + cs, t]


def compute_volume_baseline(noisy_path: Path, clean_path: Path) -> dict:
    noisy_vol = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
    clean_vol = np.load(clean_path, mmap_mode="r", allow_pickle=False)

    ax = int(np.argmax(noisy_vol.shape))
    n_slices = noisy_vol.shape[ax]
    spatial = [noisy_vol.shape[i] for i in range(3) if i != ax]
    oh = (spatial[0] - CROP_SIZE) // 2
    ow = (spatial[1] - CROP_SIZE) // 2

    scores: list[float] = []
    for t in range(0, n_slices, SLICE_STRIDE):
        ns = _crop_slice(noisy_vol, ax, t, oh, ow, CROP_SIZE).astype(np.float32)
        cs_arr = _crop_slice(clean_vol, ax, t, oh, ow, CROP_SIZE).astype(np.float32)
        score = _ms_ssim(_to_unit(ns), _to_unit(cs_arr)).item()
        scores.append(score)

    return {
        "vol_id": noisy_path.stem.split("_")[-1],
        "n_slices": len(scores),
        "ms_ssim_mean": float(np.mean(scores)),
        "ms_ssim_std": float(np.std(scores)),
        "ms_ssim_min": float(np.min(scores)),
        "ms_ssim_max": float(np.max(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute noisy-vs-clean baseline MS-SSIM.")
    parser.add_argument(
        "--dataset_root",
        default="Code/Dataset/ThinkOnwards/training_data/extracted",
    )
    parser.add_argument("--out_dir", default="experiments/summaries/baselines")
    args = parser.parse_args()

    repo_root = default_project_root(__file__)
    dataset_root = (
        Path(args.dataset_root)
        if Path(args.dataset_root).is_absolute()
        else repo_root / args.dataset_root
    )
    out_dir = (
        Path(args.out_dir)
        if Path(args.out_dir).is_absolute()
        else repo_root / args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    noisy_files = sorted(dataset_root.rglob("seismic_w_noise_vol_*.npy"))
    if not noisy_files:
        print(f"ERROR: no noisy volumes found under {dataset_root}", file=sys.stderr)
        sys.exit(1)

    pairs: list[tuple[Path, Path]] = []
    for nf in noisy_files:
        vol_id = nf.stem.split("_")[-1]
        cf = nf.parent / f"seismicCubes_RFC_fullstack_2024.{vol_id}.npy"
        if cf.exists():
            pairs.append((nf, cf))
        else:
            print(f"WARNING: clean file missing for volume {vol_id}, skipping.")

    print(f"Found {len(pairs)} complete pairs. Computing baseline MS-SSIM...")

    rows: list[dict] = []
    for i, (nf, cf) in enumerate(pairs, 1):
        vol_id = nf.stem.split("_")[-1]
        print(f"  [{i:02d}/{len(pairs)}] volume {vol_id} ...", end="", flush=True)
        row = compute_volume_baseline(nf, cf)
        rows.append(row)
        print(f"  ms_ssim = {row['ms_ssim_mean']:.4f} ± {row['ms_ssim_std']:.4f}")

    all_means = [r["ms_ssim_mean"] for r in rows]
    aggregate = {
        "n_volumes": len(rows),
        "ms_ssim_mean": float(np.mean(all_means)),
        "ms_ssim_std": float(np.std(all_means)),
        "ms_ssim_min": float(np.min(all_means)),
        "ms_ssim_max": float(np.max(all_means)),
        "slice_stride": SLICE_STRIDE,
        "crop_size": CROP_SIZE,
        "note": "noisy vs clean, no model — lower bound for improvement",
    }

    csv_path = out_dir / "noisy_clean_baseline_mssim.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_dir / "noisy_clean_baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    print(f"\nPer-volume CSV  -> {csv_path}")
    print(f"Aggregate JSON  -> {summary_path}")
    print(f"\n=== BASELINE (noisy vs clean, no model) ===")
    print(f"Volumes:      {aggregate['n_volumes']}")
    print(f"MS-SSIM:      {aggregate['ms_ssim_mean']:.4f} ± {aggregate['ms_ssim_std']:.4f}")
    print(f"Range:        [{aggregate['ms_ssim_min']:.4f}, {aggregate['ms_ssim_max']:.4f}]")
    print(f"Protocol:     stride={SLICE_STRIDE}, crop={CROP_SIZE}x{CROP_SIZE}, longest-axis slicing")


if __name__ == "__main__":
    main()
