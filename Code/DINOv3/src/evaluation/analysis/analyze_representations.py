"""Analyze final DINOv3 patch-token representations for 2D, 3ch, and 5ch.

The script compares noisy-input backbone tokens with clean-input backbone tokens
for the same adapted model. It reports:
  - mean patch-token cosine similarity to clean-input tokens,
  - horizontal and vertical spatial autocorrelation decay,
  - singular-value entropy of the final patch-token matrix.

Outputs are written under:
  experiments/summaries/mechanism_analysis/representation_analysis/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.image_impeccable import _zscore
from evaluation.figures.generate_comparison_panels import (
    _VARIANTS,
    _build_shared_dataset,
    _build_variant_dataset,
    _load_model,
    _load_sample_for_mode,
    _resolve_checkpoint,
)


def _load_models(project_root: Path, seed: int, device: torch.device):
    models = {}
    for variant in _VARIANTS:
        ckpt = _resolve_checkpoint(project_root, seed, variant)
        if ckpt is None:
            print(f"[{variant['key']}] no checkpoint found, skipping")
            continue
        models[variant["key"]] = _load_model(ckpt, variant, device)
    missing = {"2D", "3ch", "5ch"} - set(models)
    if missing:
        raise RuntimeError(f"Missing required model(s): {sorted(missing)}")
    return models


def _build_datasets(project_root: Path, stride: int):
    ds_2d = _build_shared_dataset(project_root, stride)
    return {
        "2D": ds_2d,
        "3ch": _build_variant_dataset(ds_2d, "2.5d_3ch"),
        "5ch": _build_variant_dataset(ds_2d, "2.5d_5ch"),
    }


def _load_clean_input_for_mode(
    ds,
    clean_path: Path,
    ax: int,
    t: int,
    oh: int,
    ow: int,
) -> torch.Tensor:
    cs = ds.crop_size
    clean_vol = np.load(clean_path, mmap_mode="r", allow_pickle=True)
    clean_slices = [
        np.take(clean_vol, t + dt, axis=ax)[oh:oh + cs, ow:ow + cs].astype(np.float32)
        for dt in ds.offsets
    ]
    clean_slices = [_zscore(s) for s in clean_slices]
    return torch.from_numpy(np.stack(clean_slices, axis=0))


def _extract_tokens(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        out = model.backbone.forward_features(x.unsqueeze(0).to(device))
    return out["x_norm_patchtokens"][0].detach().float().cpu()


def _clean_similarity(noisy_tokens: torch.Tensor, clean_tokens: torch.Tensor) -> float:
    return float(F.cosine_similarity(noisy_tokens, clean_tokens, dim=1).mean().item())


def _token_grid(tokens: torch.Tensor, image_hw: tuple[int, int]) -> torch.Tensor:
    image_h, image_w = image_hw
    grid_h = image_h // 16
    grid_w = image_w // 16
    expected = grid_h * grid_w
    if tokens.shape[0] != expected:
        side = int(round(tokens.shape[0] ** 0.5))
        if side * side != tokens.shape[0]:
            raise RuntimeError(f"Cannot infer spatial grid for {tokens.shape[0]} tokens")
        grid_h = grid_w = side
    return tokens.reshape(grid_h, grid_w, tokens.shape[1])


def _autocorr_decay(
    tokens: torch.Tensor,
    image_hw: tuple[int, int],
    max_offset: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    grid = _token_grid(tokens, image_hw)
    grid = F.normalize(grid, dim=-1)
    rows: list[dict[str, Any]] = []
    horiz_vals = []
    vert_vals = []

    for offset in range(1, max_offset + 1):
        if offset < grid.shape[1]:
            h = (grid[:, :-offset, :] * grid[:, offset:, :]).sum(dim=-1).mean().item()
            horiz_vals.append(h)
            rows.append({"axis": "horizontal", "offset": offset, "cosine": float(h)})
        if offset < grid.shape[0]:
            v = (grid[:-offset, :, :] * grid[offset:, :, :]).sum(dim=-1).mean().item()
            vert_vals.append(v)
            rows.append({"axis": "vertical", "offset": offset, "cosine": float(v)})

    h_mean = float(np.mean(horiz_vals)) if horiz_vals else float("nan")
    v_mean = float(np.mean(vert_vals)) if vert_vals else float("nan")
    metrics = {
        "horiz_autocorr_mean": h_mean,
        "vert_autocorr_mean": v_mean,
        "autocorr_anisotropy": h_mean - v_mean,
    }
    return metrics, rows


def _svd_entropy(tokens: torch.Tensor) -> float:
    centered = tokens - tokens.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    p = singular / singular.sum().clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log2()).sum()
    return float(entropy.item())


def _read_top_gap_indices(metadata_path: Path, limit: int) -> list[int]:
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if limit <= 0:
        return [int(row["sample_idx"]) for row in rows]
    return [int(row["sample_idx"]) for row in rows[:limit]]


def _select_indices(
    n_samples: int,
    max_samples: int,
    selection: str,
    metadata_path: Path,
) -> list[int]:
    if selection == "top-gap":
        indices = _read_top_gap_indices(metadata_path, max_samples)
        if indices:
            return [i for i in indices if 0 <= i < n_samples]

    if max_samples <= 0 or max_samples >= n_samples:
        return list(range(n_samples))
    return sorted({int(i) for i in np.linspace(0, n_samples - 1, max_samples)})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Number of shared test samples to process. Use 0 for the full test split.",
    )
    parser.add_argument(
        "--selection",
        choices=["uniform", "top-gap"],
        default="uniform",
        help="Uniform covers the split; top-gap follows comparison_metadata rank order.",
    )
    parser.add_argument("--max-offset", type=int, default=5)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (
        project_root
        / "experiments"
        / "summaries"
        / "mechanism_analysis"
        / "representation_analysis"
    )
    metadata = args.metadata or (
        project_root / "experiments" / "summaries" / "comparison_panels" / "comparison_metadata.csv"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    datasets = _build_datasets(project_root, args.stride)
    anchor_ds = datasets["5ch"]
    indices = _select_indices(len(anchor_ds), args.max_samples, args.selection, metadata)
    print(f"Processing {len(indices)} sample(s) from {len(anchor_ds)} shared test samples")
    models = _load_models(project_root, args.seed, device)

    metric_rows: list[dict[str, Any]] = []
    decay_rows: list[dict[str, Any]] = []

    for pos, sample_idx in enumerate(indices, start=1):
        noisy_path, clean_path, ax, t, oh, ow = anchor_ds.samples[sample_idx]
        oh = oh or 0
        ow = ow or 0
        vol_id = int(noisy_path.stem.split("_")[-1])

        for variant in _VARIANTS:
            key = variant["key"]
            ds = datasets[key]
            noisy, _ = _load_sample_for_mode(ds, noisy_path, clean_path, ax, t, oh, ow)
            clean_input = _load_clean_input_for_mode(ds, clean_path, ax, t, oh, ow)
            image_hw = (int(noisy.shape[-2]), int(noisy.shape[-1]))

            noisy_tokens = _extract_tokens(models[key], noisy, device)
            clean_tokens = _extract_tokens(models[key], clean_input, device)

            sim = _clean_similarity(noisy_tokens, clean_tokens)
            autocorr_metrics, per_offset = _autocorr_decay(
                noisy_tokens,
                image_hw,
                max_offset=args.max_offset,
            )
            entropy = _svd_entropy(noisy_tokens)

            metric_rows.append({
                "sample_idx": sample_idx,
                "vol_id": vol_id,
                "slice_t": t,
                "variant_key": key,
                "clean_similarity": sim,
                "horiz_autocorr_mean": autocorr_metrics["horiz_autocorr_mean"],
                "vert_autocorr_mean": autocorr_metrics["vert_autocorr_mean"],
                "autocorr_anisotropy": autocorr_metrics["autocorr_anisotropy"],
                "svd_entropy": entropy,
            })
            for row in per_offset:
                decay_rows.append({
                    "sample_idx": sample_idx,
                    "vol_id": vol_id,
                    "slice_t": t,
                    "variant_key": key,
                    **row,
                })

        if pos == 1 or pos % 10 == 0 or pos == len(indices):
            print(f"Processed {pos}/{len(indices)} samples")

    _write_csv(out_dir / "representation_metrics.csv", metric_rows)
    _write_csv(out_dir / "representation_autocorr_decay.csv", decay_rows)


if __name__ == "__main__":
    main()
