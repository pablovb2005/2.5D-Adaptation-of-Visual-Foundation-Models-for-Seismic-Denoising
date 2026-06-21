"""Test simple clean-target orientation transforms for Image Impeccable pairs.

This diagnostic is for suspicious raw noisy-center-vs-clean similarity gaps.
It compares the same noisy center slice against clean targets under identity,
in-plane transpose, and flips. A large jump for one transform indicates that the
canonical dataset may have a clean-target orientation problem.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path
from typing import Any

import torch

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.diagnostics.audit_raw_image_impeccable import (  # noqa: E402
    _audit_split,
    _dataset_from_args,
    _fallback_source,
    _load_config,
    _pearson_batch,
    _source_map,
    _volume_id_from_sample,
    _write_csv,
)
from evaluation.common.metrics import compute_ms_ssim  # noqa: E402


TRANSFORMS = (
    "identity",
    "swap_hw",
    "flip_h",
    "flip_w",
    "flip_hw",
    "swap_hw_flip_h",
    "swap_hw_flip_w",
    "swap_hw_flip_hw",
)


def _clean_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "identity":
        return x
    if name == "swap_hw":
        return x.transpose(-1, -2)
    if name == "flip_h":
        return torch.flip(x, dims=(-2,))
    if name == "flip_w":
        return torch.flip(x, dims=(-1,))
    if name == "flip_hw":
        return torch.flip(x, dims=(-2, -1))
    if name == "swap_hw_flip_h":
        return torch.flip(x.transpose(-1, -2), dims=(-2,))
    if name == "swap_hw_flip_w":
        return torch.flip(x.transpose(-1, -2), dims=(-1,))
    if name == "swap_hw_flip_hw":
        return torch.flip(x.transpose(-1, -2), dims=(-2, -1))
    raise ValueError(f"Unknown transform: {name}")


def _audit_orientation_split(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    cfg_dir: Path,
    split: str,
) -> list[dict[str, Any]]:
    ds = _dataset_from_args(args, cfg, cfg_dir, split)
    source_by_id = _source_map(ds.root_dir)
    indices_by_volume: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(ds.samples):
        indices_by_volume[_volume_id_from_sample(sample)].append(idx)

    volume_ids = sorted(indices_by_volume, key=lambda v: int(v) if v.isdigit() else v)
    if args.max_volumes is not None:
        volume_ids = volume_ids[: int(args.max_volumes)]

    rows: list[dict[str, Any]] = []
    batch_size = max(1, int(args.batch_size))
    transforms = tuple(args.transforms)
    for volume_id in volume_ids:
        indices = indices_by_volume[volume_id]
        sums = {name: {"raw_ms_ssim": 0.0, "raw_pearson": 0.0} for name in transforms}
        n_samples = 0
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            noisy_items = []
            clean_items = []
            for sample_idx in batch_indices:
                noisy, clean = ds[sample_idx]
                noisy_items.append(noisy)
                clean_items.append(clean)
            noisy_batch = torch.stack(noisy_items, dim=0)
            clean_batch = torch.stack(clean_items, dim=0)
            center = noisy_batch[:, noisy_batch.shape[1] // 2 : noisy_batch.shape[1] // 2 + 1]
            batch_n = len(batch_indices)
            n_samples += batch_n
            for name in transforms:
                transformed_clean = _clean_transform(clean_batch, name)
                sums[name]["raw_ms_ssim"] += compute_ms_ssim(center, transformed_clean) * batch_n
                sums[name]["raw_pearson"] += _pearson_batch(center, transformed_clean) * batch_n

        source_part = source_by_id.get(volume_id, _fallback_source(volume_id))
        for name in transforms:
            row = {
                "split": split,
                "mode": ds.mode,
                "volume_id": volume_id,
                "source_part": source_part,
                "clean_transform": name,
                "n_samples": n_samples,
                "raw_ms_ssim_mean": sums[name]["raw_ms_ssim"] / n_samples if n_samples else math.nan,
                "raw_pearson_mean": sums[name]["raw_pearson"] / n_samples if n_samples else math.nan,
            }
            rows.append(row)

        best = max((r for r in rows if r["split"] == split and r["volume_id"] == volume_id), key=lambda r: r["raw_ms_ssim_mean"])
        print(
            f"{split} {ds.mode} volume={volume_id} source={source_part} "
            f"best={best['clean_transform']} raw_MS-SSIM={best['raw_ms_ssim_mean']:.4f}"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", default="2d")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--n-val", type=int)
    parser.add_argument("--n-test", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--slice-stride", type=int)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--neighbor-stride", type=int)
    parser.add_argument("--crop-mode", choices=["center", "random", "grid4"])
    parser.add_argument("--crop-seed", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-volumes", type=int)
    parser.add_argument("--cache-volumes", action="store_true")
    parser.add_argument("--transforms", nargs="+", default=list(TRANSFORMS), choices=list(TRANSFORMS))
    args = parser.parse_args()

    cfg, cfg_dir = _load_config(args.config)
    rows: list[dict[str, Any]] = []
    for split in args.splits:
        rows.extend(_audit_orientation_split(args, cfg, cfg_dir, split))

    fields = [
        "split",
        "mode",
        "volume_id",
        "source_part",
        "clean_transform",
        "n_samples",
        "raw_ms_ssim_mean",
        "raw_pearson_mean",
    ]
    _write_csv(args.output_dir / "raw_orientation_similarity_by_volume.csv", rows, fields)

    best_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["volume_id"]))].append(row)
    for (_split, _volume_id), items in sorted(grouped.items()):
        best_rows.append(max(items, key=lambda r: float(r["raw_ms_ssim_mean"])))
    _write_csv(args.output_dir / "raw_orientation_similarity_best_by_volume.csv", best_rows, fields)
    print(f"Wrote orientation audit to {args.output_dir}")


if __name__ == "__main__":
    main()
