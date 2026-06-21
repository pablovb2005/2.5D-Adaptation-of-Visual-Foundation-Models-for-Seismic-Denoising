"""Counterfactual test for whether 2.5D models use aligned neighboring context.

No training is performed. Existing 2D, 3ch, and 5ch checkpoints are evaluated
on the same center slices while neighboring channels are perturbed at test time:

  aligned             normal input, with correct adjacent slices
  repeated_center     all channels replaced by the center noisy slice
  shuffled_neighbors  center slice kept; neighbor channels copied from another sample
  distant_neighbors   center slice kept; neighbors replaced by farther slices

The main quantities are the drops from aligned to each counterfactual condition
in output quality and clean-token similarity.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root
from statistics import mean, stdev
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
from evaluation.common.metrics import compute_all


CONDITIONS_BY_VARIANT = {
    "2D": ["aligned"],
    "3ch": ["aligned", "repeated_center", "shuffled_neighbors", "distant_neighbors"],
    "5ch": ["aligned", "repeated_center", "shuffled_neighbors", "distant_neighbors"],
}

HIGHER_IS_BETTER = ["ms_ssim", "psnr", "clean_similarity"]
LOWER_IS_BETTER = ["mse", "ms_ssim_r"]
VARIANT_ORDER = ["2D", "3ch", "5ch"]
CONDITION_ORDER = ["aligned", "repeated_center", "shuffled_neighbors", "distant_neighbors"]
COLORS = {
    "aligned": "#4C78A8",
    "repeated_center": "#F58518",
    "shuffled_neighbors": "#E45756",
    "distant_neighbors": "#72B7B2",
}


def _load_models(project_root: Path, seed: int, device: torch.device, data_seed: int = 101,
                  strict: bool = False):
    models = {}
    for variant in _VARIANTS:
        ckpt = _resolve_checkpoint(project_root, seed, variant, data_seed=data_seed, strict=strict)
        if ckpt is None:
            print(f"[{variant['key']}] no checkpoint found, skipping")
            continue
        models[variant["key"]] = _load_model(ckpt, variant, device)
    missing = set(VARIANT_ORDER) - set(models)
    if missing:
        raise RuntimeError(f"Missing required model(s): {sorted(missing)}")
    return models


def _build_datasets(project_root: Path, stride: int, data_seed: int = 42):
    ds_2d = _build_shared_dataset(project_root, stride, data_seed=data_seed)
    return {
        "2D": ds_2d,
        "3ch": _build_variant_dataset(ds_2d, "2.5d_3ch"),
        "5ch": _build_variant_dataset(ds_2d, "2.5d_5ch"),
    }


def _read_top_gap_indices(metadata_path: Path, limit: int) -> list[int]:
    if not metadata_path.exists():
        return []
    with metadata_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if limit <= 0:
        return [int(row["sample_idx"]) for row in rows]
    return [int(row["sample_idx"]) for row in rows[:limit]]


def _select_indices(n_samples: int, max_samples: int, selection: str, metadata_path: Path) -> list[int]:
    if selection == "top-gap":
        indices = _read_top_gap_indices(metadata_path, max_samples)
        if indices:
            return [i for i in indices if 0 <= i < n_samples]
    if max_samples <= 0 or max_samples >= n_samples:
        return list(range(n_samples))
    return sorted({int(i) for i in np.linspace(0, n_samples - 1, max_samples)})


def _load_volume_slice(
    path: Path,
    axis: int,
    t: int,
    oh: int,
    ow: int,
    crop_size: int,
) -> np.ndarray:
    vol = np.load(path, mmap_mode="r", allow_pickle=True)
    t = max(0, min(int(t), vol.shape[axis] - 1))
    return np.take(vol, t, axis=axis)[oh:oh + crop_size, ow:ow + crop_size].astype(np.float32)


def _donor_sample(anchor_ds, sample_idx: int):
    current = anchor_ds.samples[sample_idx]
    n = len(anchor_ds)
    for step in range(1, n):
        donor_idx = (sample_idx + step * 137) % n
        donor = anchor_ds.samples[donor_idx]
        if donor[0] != current[0] or donor[3] != current[3]:
            return donor
    return anchor_ds.samples[(sample_idx + 1) % n]


def _counterfactual_input(
    ds,
    condition: str,
    noisy_path: Path,
    axis: int,
    t: int,
    oh: int,
    ow: int,
    donor_sample: tuple,
    distant_stride: int,
) -> torch.Tensor:
    cs = ds.crop_size
    if condition == "aligned":
        channels = [
            _zscore(_load_volume_slice(noisy_path, axis, t + offset, oh, ow, cs))
            for offset in ds.offsets
        ]
        return torch.from_numpy(np.stack(channels, axis=0))

    center = _zscore(_load_volume_slice(noisy_path, axis, t, oh, ow, cs))
    donor_noisy, _, donor_axis, donor_t, donor_oh, donor_ow = donor_sample
    donor_oh = donor_oh or 0
    donor_ow = donor_ow or 0

    channels = []
    for offset in ds.offsets:
        if offset == 0 or condition == "repeated_center":
            channels.append(center)
            continue
        if condition == "shuffled_neighbors":
            source = _load_volume_slice(
                donor_noisy,
                donor_axis,
                donor_t + offset,
                donor_oh,
                donor_ow,
                cs,
            )
        elif condition == "distant_neighbors":
            source = _load_volume_slice(
                noisy_path,
                axis,
                t + offset * distant_stride,
                oh,
                ow,
                cs,
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")
        channels.append(_zscore(source))
    return torch.from_numpy(np.stack(channels, axis=0))


def _clean_aligned_input(ds, clean_path: Path, axis: int, t: int, oh: int, ow: int) -> torch.Tensor:
    cs = ds.crop_size
    channels = [
        _zscore(_load_volume_slice(clean_path, axis, t + offset, oh, ow, cs))
        for offset in ds.offsets
    ]
    return torch.from_numpy(np.stack(channels, axis=0))


def _forward_with_tokens(model: torch.nn.Module, x: torch.Tensor, device: torch.device):
    with torch.no_grad():
        xb = x.unsqueeze(0).to(device)
        out = model.backbone.forward_features(xb)
        tokens = out["x_norm_patchtokens"]
        batch, n_tokens, channels = tokens.shape
        h = x.shape[-2] // 16
        w = x.shape[-1] // 16
        if h * w != n_tokens:
            side = int(round(n_tokens ** 0.5))
            if side * side != n_tokens:
                raise RuntimeError(f"Cannot reshape {n_tokens} patch tokens to a spatial grid")
            h = w = side
        features = tokens.transpose(1, 2).reshape(batch, channels, h, w)
        pred = model.decoder(features)
    return pred.detach().cpu(), tokens[0].detach().float().cpu()


def _clean_similarity(tokens: torch.Tensor, clean_tokens: torch.Tensor) -> float:
    return float(F.cosine_similarity(tokens, clean_tokens, dim=1).mean().item())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")


def _finite(values: list[float]) -> list[float]:
    return [float(v) for v in values if np.isfinite(v)]


def _mean_std(values: list[float]) -> tuple[float, float]:
    values = _finite(values)
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant_key"], row["condition"])].append(row)

    aligned_means: dict[str, dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        aligned = grouped.get((variant, "aligned"), [])
        aligned_means[variant] = {}
        for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
            aligned_means[variant][metric] = _mean_std([float(r[metric]) for r in aligned])[0]

    summary = []
    for variant in VARIANT_ORDER:
        for condition in CONDITION_ORDER:
            group = grouped.get((variant, condition), [])
            if not group:
                continue
            out = {
                "variant_key": variant,
                "condition": condition,
                "n": len(group),
            }
            for metric in HIGHER_IS_BETTER + LOWER_IS_BETTER:
                mu, sd = _mean_std([float(r[metric]) for r in group])
                out[f"{metric}_mean"] = mu
                out[f"{metric}_std"] = sd
                out[f"{metric}_delta_vs_aligned"] = mu - aligned_means[variant][metric]
            summary.append(out)
    return summary


def _assign_difficulty(rows: list[dict[str, Any]]) -> dict[int, str]:
    baseline = {
        int(row["sample_idx"]): float(row["ms_ssim"])
        for row in rows
        if row["variant_key"] == "2D" and row["condition"] == "aligned"
    }
    if not baseline:
        return {}
    values = np.asarray(list(baseline.values()), dtype=float)
    q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
    labels = {}
    for sample_idx, value in baseline.items():
        if value <= q25:
            labels[sample_idx] = "Q1 hardest"
        elif value <= q50:
            labels[sample_idx] = "Q2 medium-hard"
        elif value <= q75:
            labels[sample_idx] = "Q3 medium-easy"
        else:
            labels[sample_idx] = "Q4 easiest"
    return labels


def _difficulty_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    difficulty = _assign_difficulty(rows)
    aligned_by_sample_variant = {
        (int(row["sample_idx"]), row["variant_key"]): row
        for row in rows
        if row["condition"] == "aligned"
    }
    grouped: dict[tuple[str, str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "aligned" or row["variant_key"] == "2D":
            continue
        sample_idx = int(row["sample_idx"])
        aligned = aligned_by_sample_variant.get((sample_idx, row["variant_key"]))
        label = difficulty.get(sample_idx)
        if aligned is None or label is None:
            continue
        grouped[(label, row["variant_key"], row["condition"])].append({
            "ms_ssim_drop": float(aligned["ms_ssim"]) - float(row["ms_ssim"]),
            "clean_similarity_drop": float(aligned["clean_similarity"]) - float(row["clean_similarity"]),
            "psnr_drop": float(aligned["psnr"]) - float(row["psnr"]),
            "mse_increase": float(row["mse"]) - float(aligned["mse"]),
        })

    order = ["Q1 hardest", "Q2 medium-hard", "Q3 medium-easy", "Q4 easiest"]
    output = []
    for label in order:
        for variant in ["3ch", "5ch"]:
            for condition in CONDITION_ORDER[1:]:
                group = grouped.get((label, variant, condition), [])
                if not group:
                    continue
                out = {
                    "difficulty_bin": label,
                    "variant_key": variant,
                    "condition": condition,
                    "n": len(group),
                }
                for metric in ["ms_ssim_drop", "clean_similarity_drop", "psnr_drop", "mse_increase"]:
                    mu, sd = _mean_std([float(r[metric]) for r in group])
                    out[f"{metric}_mean"] = mu
                    out[f"{metric}_std"] = sd
                output.append(out)
    return output


def _plot_bars(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    metrics = [("ms_ssim_mean", "MS-SSIM", "higher is better"), ("clean_similarity_mean", "Clean-token similarity", "higher is better")]
    x = np.arange(len(CONDITION_ORDER))
    width = 0.34

    for ax, (metric, title, subtitle) in zip(axes, metrics):
        for idx, variant in enumerate(["3ch", "5ch"]):
            vals = []
            for condition in CONDITION_ORDER:
                row = next(
                    (r for r in summary_rows if r["variant_key"] == variant and r["condition"] == condition),
                    None,
                )
                vals.append(float(row[metric]) if row else np.nan)
            ax.bar(x + (idx - 0.5) * width, vals, width=width, label=variant, alpha=0.82)
        baseline = next(
            (r for r in summary_rows if r["variant_key"] == "2D" and r["condition"] == "aligned"),
            None,
        )
        if baseline:
            ax.axhline(float(baseline[metric]), color="black", linestyle="--", linewidth=1.2, label="2D aligned")
        ax.set_title(title)
        ax.set_xlabel(subtitle)
        ax.set_xticks(x)
        ax.set_xticklabels(["aligned", "repeat", "shuffle", "distant"], rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Context counterfactuals: aligned neighbors vs broken context", fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_by_difficulty(diff_rows: list[dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["Q1 hardest", "Q2 medium-hard", "Q3 medium-easy", "Q4 easiest"]
    short = ["Q1", "Q2", "Q3", "Q4"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, variant in zip(axes, ["3ch", "5ch"]):
        for condition in CONDITION_ORDER[1:]:
            vals = []
            for label in labels:
                row = next(
                    (
                        r for r in diff_rows
                        if r["variant_key"] == variant
                        and r["condition"] == condition
                        and r["difficulty_bin"] == label
                    ),
                    None,
                )
                vals.append(float(row["ms_ssim_drop_mean"]) if row else np.nan)
            ax.plot(short, vals, marker="o", label=condition, color=COLORS[condition])
        ax.set_title(f"{variant}: MS-SSIM drop from aligned")
        ax.set_xlabel("Difficulty bin from 2D aligned MS-SSIM")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Aligned - counterfactual MS-SSIM")
    fig.suptitle("Context reliance by sample difficulty", fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Context Counterfactual Report",
        "",
        "No new training was used. Existing checkpoints were evaluated with broken neighboring context.",
        "",
        "| Variant | Condition | n | MS-SSIM | Delta MS-SSIM vs aligned | Clean-token similarity | Delta similarity vs aligned |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        for condition in CONDITION_ORDER:
            row = next(
                (r for r in summary_rows if r["variant_key"] == variant and r["condition"] == condition),
                None,
            )
            if row is None:
                continue
            lines.append(
                f"| {variant} | {condition} | {row['n']} | "
                f"{float(row['ms_ssim_mean']):.6f} | {float(row['ms_ssim_delta_vs_aligned']):+.6f} | "
                f"{float(row['clean_similarity_mean']):.6f} | {float(row['clean_similarity_delta_vs_aligned']):+.6f} |"
            )
    lines.extend([
        "",
        "Interpretation rule: a large negative delta for shuffled or distant neighbors means the trained",
        "2.5D model depends on correctly aligned neighboring slices, not merely on having extra channels.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def _run_cf_loop(
    project_root: Path,
    training_seed: int,
    data_seed: int,
    stride: int,
    max_samples: int,
    selection: str,
    metadata_path: Path,
    distant_stride: int,
    device: torch.device,
    strict: bool = False,
) -> list:
    """Run the full counterfactual inference loop for one (data_seed, training_seed) pair.

    Returns the raw per-sample rows (one row per sample × variant × condition).
    Models are freed from memory before returning.
    """
    datasets = _build_datasets(project_root, stride, data_seed=data_seed)
    anchor_ds = datasets["5ch"]
    indices = _select_indices(len(anchor_ds), max_samples, selection, metadata_path)
    print(f"  Processing {len(indices)} sample(s) from {len(anchor_ds)} test samples "
          f"(data_seed={data_seed}, training_seed={training_seed})")
    models = _load_models(project_root, training_seed, device, data_seed=data_seed, strict=strict)

    rows: list = []
    for pos, sample_idx in enumerate(indices, start=1):
        noisy_path, clean_path, axis, t, oh, ow = anchor_ds.samples[sample_idx]
        oh = oh or 0
        ow = ow or 0
        vol_id = int(noisy_path.stem.split("_")[-1])
        donor_sample = _donor_sample(anchor_ds, sample_idx)

        for variant in _VARIANTS:
            key = variant["key"]
            ds = datasets[key]
            _, clean_target = _load_sample_for_mode(ds, noisy_path, clean_path, axis, t, oh, ow)
            clean_aligned = _clean_aligned_input(ds, clean_path, axis, t, oh, ow)
            _, clean_tokens = _forward_with_tokens(models[key], clean_aligned, device)

            for condition in CONDITIONS_BY_VARIANT[key]:
                if condition == "aligned":
                    noisy_input, _ = _load_sample_for_mode(ds, noisy_path, clean_path, axis, t, oh, ow)
                else:
                    noisy_input = _counterfactual_input(
                        ds, condition, noisy_path, axis, t, oh, ow, donor_sample, distant_stride,
                    )

                pred, tokens = _forward_with_tokens(models[key], noisy_input, device)
                metrics = compute_all(
                    pred.cpu(),
                    clean_target.unsqueeze(0).cpu(),
                    noisy_input.unsqueeze(0).cpu(),
                )
                rows.append({
                    "sample_idx": sample_idx,
                    "vol_id": vol_id,
                    "slice_t": t,
                    "variant_key": key,
                    "condition": condition,
                    "n_channels": int(noisy_input.shape[0]),
                    "distant_stride": distant_stride,
                    **metrics,
                    "clean_similarity": _clean_similarity(tokens, clean_tokens),
                })

        if pos == 1 or pos % 10 == 0 or pos == len(indices):
            print(f"  Processed {pos}/{len(indices)} samples")

    del models
    return rows


def _average_cf_rows(per_seed_rows_list: list[list]) -> list:
    """Average numeric metric values across training seeds, grouped by (sample_idx, variant, condition)."""
    if len(per_seed_rows_list) == 1:
        return per_seed_rows_list[0]

    numeric_keys = ["ms_ssim", "psnr", "mse", "ms_ssim_r", "clean_similarity"]

    from collections import defaultdict as _dd
    grouped: dict = _dd(list)
    for seed_rows in per_seed_rows_list:
        for row in seed_rows:
            key = (int(row["sample_idx"]), row["variant_key"], row["condition"])
            grouped[key].append(row)

    averaged = []
    for (sample_idx, variant_key, condition), group in grouped.items():
        ref = group[0]
        avg_row: dict = {
            "sample_idx":    sample_idx,
            "vol_id":        ref["vol_id"],
            "slice_t":       ref["slice_t"],
            "variant_key":   variant_key,
            "condition":     condition,
            "n_channels":    ref["n_channels"],
            "distant_stride": ref["distant_stride"],
        }
        for field in numeric_keys:
            vals = [float(r[field]) for r in group if field in r and np.isfinite(float(r[field]))]
            avg_row[field] = float(np.mean(vals)) if vals else float("nan")
        averaged.append(avg_row)
    return averaged


def _write_outputs(rows: list, out_dir: Path) -> None:
    summary_rows = _summarize(rows)
    difficulty_rows = _difficulty_summary(rows)
    _write_csv(out_dir / "context_counterfactual_metrics.csv", rows)
    _write_csv(out_dir / "context_counterfactual_summary.csv", summary_rows)
    _write_csv(out_dir / "context_counterfactual_by_difficulty.csv", difficulty_rows)
    _plot_bars(summary_rows, out_dir / "context_counterfactual_bars.png")
    _plot_by_difficulty(difficulty_rows, out_dir / "context_counterfactual_by_difficulty.png")
    _write_report(out_dir / "context_counterfactual_report.md", summary_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=None,
                        help="Multiple training seeds to average over (activates multi-seed mode).")
    parser.add_argument("--pool-data-seeds", action="store_true",
                        help="Run all three data seeds (101, 202, 303), averaging training seeds "
                             "per split and pooling across splits.")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="Data split seed; selects both the test split and the model checkpoint.")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=64, help="Use 0 for the full shared test split.")
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run on all test samples (overrides --max-samples). Output goes to context_counterfactuals_full/.",
    )
    parser.add_argument("--selection", choices=["uniform", "top-gap"], default="uniform")
    parser.add_argument("--distant-stride", type=int, default=5)
    args = parser.parse_args()

    if args.full_test:
        args.max_samples = 0

    project_root = args.project_root.resolve()

    # --- Determine active training seeds ---
    training_seeds = args.training_seeds if args.training_seeds else [args.seed]

    # --- Multi-seed / pooled mode -----------------------------------------------
    from evaluation.figures.generate_comparison_panels import _MAIN_DATA_SEEDS as _MSEEDS
    if args.pool_data_seeds or len(training_seeds) > 1:
        _cf_subdir = "context_counterfactuals_full" if args.full_test else "context_counterfactuals"
        _ts_tag = "_".join(str(ts) for ts in training_seeds)
        base_dir = args.out_dir or (
            project_root / "experiments" / "summaries" / "mechanism_analysis" / _cf_subdir
        )
        data_seeds_to_run = _MSEEDS if args.pool_data_seeds else [args.data_seed]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        print(f"Multi-seed CF: data_seeds={data_seeds_to_run}, training_seeds={training_seeds}")

        all_pooled_rows: list = []
        global_offset = 0

        for ds in data_seeds_to_run:
            # Metadata path for top-gap sample selection within this data split
            meta_for_ds = (
                project_root / "experiments" / "summaries" / "comparison_panels"
                / f"ds{ds}" / "comparison_metadata.csv"
            )
            if not meta_for_ds.exists():
                meta_for_ds = (
                    project_root / "experiments" / "summaries" / "comparison_panels"
                    / "comparison_metadata.csv"
                )

            print(f"\n=== Data seed {ds} (averaging {len(training_seeds)} training seeds) ===")
            per_seed_rows_list = []
            for ts in training_seeds:
                print(f"\n  Training seed {ts}...")
                seed_rows = _run_cf_loop(
                    project_root=project_root,
                    training_seed=ts,
                    data_seed=ds,
                    stride=args.stride,
                    max_samples=args.max_samples,
                    selection=args.selection,
                    metadata_path=meta_for_ds,
                    distant_stride=args.distant_stride,
                    device=device,
                    strict=True,
                )
                per_seed_rows_list.append(seed_rows)

            averaged_rows = _average_cf_rows(per_seed_rows_list)

            # Per-split outputs
            ds_out_dir = base_dir / f"ds{ds}_ts{_ts_tag}"
            _write_outputs(averaged_rows, ds_out_dir)

            # Collect for pooled output with globally unique sample_idx + data_seed column
            local_max_idx = max((int(r["sample_idx"]) for r in averaged_rows), default=0)
            for row in averaged_rows:
                pooled_row = dict(row)
                pooled_row["sample_idx"] = int(row["sample_idx"]) + global_offset
                pooled_row["data_seed"] = ds
                all_pooled_rows.append(pooled_row)
            global_offset += local_max_idx + 1

        # Pooled outputs
        pooled_out_dir = base_dir / f"pooled_ts{_ts_tag}"
        print(f"\nWriting pooled outputs ({len(all_pooled_rows)} rows) -> {pooled_out_dir}")
        _write_outputs(all_pooled_rows, pooled_out_dir)
        return

    # --- Single-seed mode (legacy, backward-compatible) -------------------------
    data_seed = args.data_seed
    _cf_subdir = "context_counterfactuals_full" if args.full_test else "context_counterfactuals"
    _run_tag = f"ds{data_seed}_ts{training_seeds[0]}"
    out_dir = args.out_dir or (
        project_root / "experiments" / "summaries" / "mechanism_analysis"
        / _cf_subdir / _run_tag
    )
    metadata = args.metadata or (
        project_root / "experiments" / "summaries" / "comparison_panels"
        / f"ds{data_seed}" / "comparison_metadata.csv"
    )
    if not metadata.exists():
        metadata = (
            project_root / "experiments" / "summaries" / "comparison_panels"
            / "comparison_metadata.csv"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  data_seed={data_seed}  training_seed={training_seeds[0]}")

    rows = _run_cf_loop(
        project_root=project_root,
        training_seed=training_seeds[0],
        data_seed=data_seed,
        stride=args.stride,
        max_samples=args.max_samples,
        selection=args.selection,
        metadata_path=metadata,
        distant_stride=args.distant_stride,
        device=device,
        strict=False,
    )
    _write_outputs(rows, out_dir)


if __name__ == "__main__":
    main()
