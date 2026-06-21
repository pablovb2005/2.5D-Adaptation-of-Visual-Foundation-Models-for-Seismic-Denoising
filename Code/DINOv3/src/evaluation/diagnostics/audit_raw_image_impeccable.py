"""Audit raw noisy-center versus clean-target similarity for Image Impeccable.

This is a dataset sanity check, not a model evaluation. It answers whether the
held-out noisy input is already structurally close to the clean target, grouped
by volume and source part when manifest files are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path
from typing import Any

import torch
import yaml

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data.image_impeccable import ThinkOnwardDataset
from evaluation.common.metrics import compute_mse, compute_ms_ssim, compute_psnr


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _safe_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    if path is None:
        return {}, Path.cwd()
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}, path.parent


def _source_map(root_dir: Path) -> dict[str, str]:
    """Map volume_id -> source part from canonical builder manifests."""
    mapping: dict[str, str] = {}
    manifest_dir = root_dir / "_manifests"
    if not manifest_dir.is_dir():
        return mapping
    for path in sorted(manifest_dir.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        part = str(payload.get("part") or path.stem)
        for volume_id in payload.get("volume_ids", []) or []:
            mapping[str(volume_id)] = part
        for row in payload.get("pairs", []) or []:
            if "volume_id" in row:
                mapping[str(row["volume_id"])] = part
    return mapping


def _fallback_source(volume_id: str) -> str:
    try:
        numeric = int(volume_id)
    except ValueError:
        return "unknown"
    if numeric < 70_000_000:
        return "unknown_parts_01_02_range"
    return "unknown_official_parts_range"


def _volume_id_from_sample(sample: tuple[Any, ...]) -> str:
    noisy_path = Path(sample[0])
    return noisy_path.stem.split("_")[-1]


def _pearson_batch(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten(start_dim=1).float()
    b_flat = b.flatten(start_dim=1).float()
    a_flat = a_flat - a_flat.mean(dim=1, keepdim=True)
    b_flat = b_flat - b_flat.mean(dim=1, keepdim=True)
    denom = torch.sqrt((a_flat.square().sum(dim=1) * b_flat.square().sum(dim=1)).clamp(min=1e-12))
    corr = (a_flat * b_flat).sum(dim=1) / denom
    return float(corr.mean().item())


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _std(values: list[float]) -> float | None:
    vals = [v for v in values if math.isfinite(v)]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_from_args(args: argparse.Namespace, cfg: dict[str, Any], cfg_dir: Path, split: str) -> ThinkOnwardDataset:
    data_cfg = cfg.get("data", {})
    root_dir = args.root_dir or _resolve(cfg_dir, data_cfg.get("root_dir"))
    if root_dir is None:
        raise ValueError("Provide --root-dir or --config with data.root_dir")
    mode = args.mode or str(data_cfg.get("mode", "2d"))
    return ThinkOnwardDataset(
        root_dir=root_dir,
        mode=mode,
        split=split,
        n_train=int(args.n_train if args.n_train is not None else data_cfg.get("n_train", 20)),
        n_val=int(args.n_val if args.n_val is not None else data_cfg.get("n_val", 5)),
        n_test=int(args.n_test if args.n_test is not None else data_cfg.get("n_test", 5)),
        slice_stride=int(args.slice_stride if args.slice_stride is not None else data_cfg.get("slice_stride", 5)),
        crop_size=int(args.crop_size if args.crop_size is not None else data_cfg.get("crop_size", 224)),
        seed=int(args.seed if args.seed is not None else data_cfg.get("seed", 42)),
        neighbor_stride=int(args.neighbor_stride if args.neighbor_stride is not None else data_cfg.get("neighbor_stride", 1)),
        crop_mode=str(args.crop_mode or data_cfg.get("eval_crop_mode", data_cfg.get("crop_mode", "center"))),
        crop_seed=int(args.crop_seed if args.crop_seed is not None else data_cfg.get("crop_seed", data_cfg.get("seed", 42))),
        cache_volumes=bool(args.cache_volumes),
    )


def _audit_split(args: argparse.Namespace, cfg: dict[str, Any], cfg_dir: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ds = _dataset_from_args(args, cfg, cfg_dir, split)
    source_by_id = _source_map(ds.root_dir)
    indices_by_volume: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(ds.samples):
        indices_by_volume[_volume_id_from_sample(sample)].append(idx)

    volume_ids = sorted(indices_by_volume, key=lambda v: int(v) if v.isdigit() else v)
    if args.max_volumes is not None:
        volume_ids = volume_ids[: int(args.max_volumes)]

    batch_size = max(1, int(args.batch_size))
    batch_rows: list[dict[str, Any]] = []
    volume_rows: list[dict[str, Any]] = []

    for volume_id in volume_ids:
        indices = indices_by_volume[volume_id]
        weighted: dict[str, list[float]] = defaultdict(list)
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
            metrics = {
                "raw_ms_ssim": compute_ms_ssim(center, clean_batch),
                "raw_mse": compute_mse(center, clean_batch),
                "raw_psnr": compute_psnr(center, clean_batch),
                "raw_pearson": _pearson_batch(center, clean_batch),
            }
            n_samples += batch_n
            for key, value in metrics.items():
                weighted[key].append(float(value) * batch_n)
            batch_rows.append(
                {
                    "split": split,
                    "mode": ds.mode,
                    "volume_id": volume_id,
                    "source_part": source_by_id.get(volume_id, _fallback_source(volume_id)),
                    "batch_start_sample": batch_start,
                    "batch_n": batch_n,
                    **metrics,
                }
            )

        row = {
            "split": split,
            "mode": ds.mode,
            "volume_id": volume_id,
            "source_part": source_by_id.get(volume_id, _fallback_source(volume_id)),
            "n_samples": n_samples,
            "n_batches": math.ceil(n_samples / batch_size),
        }
        for key in ("raw_ms_ssim", "raw_mse", "raw_psnr", "raw_pearson"):
            row[f"{key}_mean"] = sum(weighted[key]) / n_samples if n_samples else None
        volume_rows.append(row)
        print(
            f"{split} {ds.mode} volume={volume_id} source={row['source_part']} "
            f"n={n_samples} raw_MS-SSIM={row['raw_ms_ssim_mean']:.4f} "
            f"pearson={row['raw_pearson_mean']:.4f}"
        )
    return batch_rows, volume_rows


def _source_rows(volume_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in volume_rows:
        grouped[(str(row["split"]), str(row["mode"]), str(row["source_part"]))].append(row)
    rows = []
    for (split, mode, source_part), items in sorted(grouped.items()):
        out = {
            "split": split,
            "mode": mode,
            "source_part": source_part,
            "n_volumes": len(items),
            "n_samples": sum(int(item["n_samples"]) for item in items),
        }
        for key in ("raw_ms_ssim_mean", "raw_mse_mean", "raw_psnr_mean", "raw_pearson_mean"):
            vals = [_safe_float(item.get(key)) for item in items]
            vals2 = [v for v in vals if v is not None]
            out[key] = _mean(vals2)
            out[key.replace("_mean", "_std_across_volumes")] = _std(vals2)
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional run config to read data settings from")
    parser.add_argument("--root-dir", type=Path, help="Image Impeccable extracted root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", help="Input mode used only to choose sampled center-slice indices")
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
    parser.add_argument("--max-volumes", type=int, help="Debug limit")
    parser.add_argument("--cache-volumes", action="store_true")
    args = parser.parse_args()

    cfg, cfg_dir = _load_config(args.config)
    all_batch_rows: list[dict[str, Any]] = []
    all_volume_rows: list[dict[str, Any]] = []
    for split in args.splits:
        batch_rows, volume_rows = _audit_split(args, cfg, cfg_dir, split)
        all_batch_rows.extend(batch_rows)
        all_volume_rows.extend(volume_rows)

    source_rows = _source_rows(all_volume_rows)
    out = args.output_dir
    _write_csv(
        out / "raw_noisy_center_similarity_batches.csv",
        all_batch_rows,
        [
            "split",
            "mode",
            "volume_id",
            "source_part",
            "batch_start_sample",
            "batch_n",
            "raw_ms_ssim",
            "raw_mse",
            "raw_psnr",
            "raw_pearson",
        ],
    )
    _write_csv(
        out / "raw_noisy_center_similarity_by_volume.csv",
        all_volume_rows,
        [
            "split",
            "mode",
            "volume_id",
            "source_part",
            "n_samples",
            "n_batches",
            "raw_ms_ssim_mean",
            "raw_mse_mean",
            "raw_psnr_mean",
            "raw_pearson_mean",
        ],
    )
    _write_csv(
        out / "raw_noisy_center_similarity_by_source.csv",
        source_rows,
        [
            "split",
            "mode",
            "source_part",
            "n_volumes",
            "n_samples",
            "raw_ms_ssim_mean",
            "raw_ms_ssim_std_across_volumes",
            "raw_mse_mean",
            "raw_mse_std_across_volumes",
            "raw_psnr_mean",
            "raw_psnr_std_across_volumes",
            "raw_pearson_mean",
            "raw_pearson_std_across_volumes",
        ],
    )
    print(f"Wrote raw similarity audit to {out}")


if __name__ == "__main__":
    main()
