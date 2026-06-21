"""Add input-quality columns to comparison_metadata.csv without running model inference.

Computes input_psnr, input_ms_ssim, and input_mse (noisy center slice vs clean target)
for every sample in the test split, then joins with the existing metadata CSV on
sample_idx.  The updated CSV is written in-place (or to --out-csv if specified).

Usage:
    python evaluation/add_input_quality_columns.py --project-root C:/UNI/Y3/RP
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

import numpy as np
import torch

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from data.image_impeccable import ThinkOnwardDataset
from evaluation.common.metrics import compute_mse, compute_ms_ssim, compute_psnr


def _find_dataset_root(project_root: Path) -> Path:
    candidates = [
        project_root / "Code" / "Dataset" / "ThinkOnwards" / "training_data" / "extracted",
        project_root / "Dataset" / "ThinkOnwards" / "training_data" / "extracted",
    ]
    root = next((c for c in candidates if c.exists()), None)
    if root is None:
        raise FileNotFoundError(
            f"Image Impeccable dataset not found. Checked: {[str(c) for c in candidates]}"
        )
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description="Add input-quality columns to comparison_metadata.csv.")
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to comparison_metadata.csv (default: experiments/summaries/comparison_panels/comparison_metadata.csv)",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Output path (default: overwrite --metadata)")
    parser.add_argument("--seed", type=int, default=42, help="Dataset split seed (must match original panel run)")
    parser.add_argument("--stride", type=int, default=5, help="Test slice stride")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    meta_path = args.metadata or project_root / "experiments" / "summaries" / "comparison_panels" / "comparison_metadata.csv"
    out_path = args.out_csv or meta_path

    print(f"Project root : {project_root}")
    print(f"Metadata     : {meta_path}")
    print(f"Output       : {out_path}")
    print(f"Split seed   : {args.seed}, stride: {args.stride}")

    # Load existing CSV
    with meta_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {meta_path}")

    existing_cols = list(rows[0].keys())
    print(f"Existing columns: {existing_cols}")

    if all(c in existing_cols for c in ("input_psnr", "input_ms_ssim", "input_mse")):
        print("All input-quality columns already present. Nothing to do.")
        return

    # Build test dataset (2d mode, same seed/stride as original)
    root = _find_dataset_root(project_root)
    ds = ThinkOnwardDataset(
        root_dir=root,
        mode="2d",
        split="test",
        slice_stride=args.stride,
        crop_mode="center",
        seed=args.seed,
    )
    n = len(ds)
    print(f"Test split: {n} samples")

    # Compute input quality for every sample
    print("Computing input quality metrics...")
    quality: dict[int, dict[str, float]] = {}
    for idx in range(n):
        noisy, clean = ds[idx]
        # For mode="2d" all channels are the same center repeat; use channel 0
        noisy_center = noisy[0].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        clean_b = clean.unsqueeze(0)                        # [1,1,H,W]
        quality[idx] = {
            "input_ms_ssim": float(compute_ms_ssim(noisy_center, clean_b)),
            "input_mse":     float(compute_mse(noisy_center, clean_b)),
            "input_psnr":    float(compute_psnr(noisy_center, clean_b)),
        }
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{n}", end="\r")
    print(f"  {n}/{n} done.   ")

    # Merge and write
    new_cols = [c for c in ("input_ms_ssim", "input_mse", "input_psnr") if c not in existing_cols]
    # Reorder: rank_by_gap, vol_id, slice_t, sample_idx, [input cols], rest
    priority = ["rank_by_gap", "vol_id", "slice_t", "sample_idx", "input_ms_ssim", "input_mse", "input_psnr"]
    remaining = [c for c in existing_cols if c not in priority]
    fieldnames = [c for c in priority if c in existing_cols or c in new_cols] + remaining

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            idx = int(row["sample_idx"])
            q = quality.get(idx)
            if q:
                for col in new_cols:
                    row[col] = f"{q[col]:.6f}"
            writer.writerow(row)

    print(f"Written: {out_path}")
    print(f"Added columns: {new_cols}")


if __name__ == "__main__":
    main()
