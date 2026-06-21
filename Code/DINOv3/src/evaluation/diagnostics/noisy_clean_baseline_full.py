"""Extended noisy-vs-clean baseline: MS-SSIM, MSE, and PSNR on the test split.

Uses ThinkOnwardDataset directly so normalization (z-score per channel) matches
evaluate.py exactly.  Metrics are computed via the same metrics.py functions
used for model evaluation.

Usage:
    python Code/DINOv3/src/evaluation/noisy_clean_baseline_full.py
    python Code/DINOv3/src/evaluation/noisy_clean_baseline_full.py --data_seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

import numpy as np
import torch
from torch.utils.data import DataLoader

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from data.image_impeccable import ThinkOnwardDataset
from evaluation.common.metrics import compute_ms_ssim, compute_mse, compute_psnr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="Code/Dataset/ThinkOnwards/training_data/extracted",
    )
    parser.add_argument("--out_dir", default="experiments/summaries/baselines")
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=20)
    parser.add_argument("--n_val", type=int, default=5)
    parser.add_argument("--n_test", type=int, default=5)
    args = parser.parse_args()

    repo_root = default_project_root(__file__)
    dataset_root = (
        Path(args.dataset_root) if Path(args.dataset_root).is_absolute()
        else repo_root / args.dataset_root
    )
    out_dir = (
        Path(args.out_dir) if Path(args.out_dir).is_absolute()
        else repo_root / args.out_dir
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load test split with same parameters as main experiments.
    # mode='2d' returns [3, 224, 224] noisy (center slice repeated) + [1, 224, 224] clean,
    # both z-score normalised — matching evaluate.py exactly.
    test_ds = ThinkOnwardDataset(
        root_dir=dataset_root,
        mode="2d",
        split="test",
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        slice_stride=5,
        crop_size=224,
        seed=args.data_seed,
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    print(f"Test split (data_seed={args.data_seed}): {len(test_ds)} slices")

    ms_ssim_sum = mse_sum = psnr_sum = 0.0
    n_batches = 0
    for noisy, clean in loader:
        # noisy: [B, 3, H, W] z-score — all 3 channels identical (center repeated)
        # Use channel 1 (center) as single-channel noisy input: [B, 1, H, W]
        noisy_center = noisy[:, 1:2, :, :]
        ms_ssim_sum += compute_ms_ssim(noisy_center, clean)
        mse_sum += compute_mse(noisy_center, clean)
        psnr_sum += compute_psnr(noisy_center, clean)
        n_batches += 1

    ms_ssim_mean = ms_ssim_sum / n_batches
    mse_mean = mse_sum / n_batches
    psnr_mean = psnr_sum / n_batches

    summary = {
        "data_seed": args.data_seed,
        "n_test_slices": len(test_ds),
        "ms_ssim_mean": ms_ssim_mean,
        "mse_mean": mse_mean,
        "psnr_mean": psnr_mean,
        "note": (
            "noisy input vs clean target, no model, test split only. "
            "Same normalisation (z-score + per-image min-max) as evaluate.py."
        ),
    }

    out_path = out_dir / f"noisy_clean_baseline_full_seed{args.data_seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== BASELINE (noisy input, test split, data_seed={args.data_seed}) ===")
    print(f"Slices:   {len(test_ds)}")
    print(f"MS-SSIM:  {ms_ssim_mean:.4f}")
    print(f"MSE:      {mse_mean:.4f}")
    print(f"PSNR:     {psnr_mean:.2f} dB")
    print(f"Saved ->  {out_path}")


if __name__ == "__main__":
    main()
