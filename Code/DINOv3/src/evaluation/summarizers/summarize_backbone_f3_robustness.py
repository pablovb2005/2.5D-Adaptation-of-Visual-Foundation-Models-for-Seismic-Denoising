"""Aggregate F3 field-transfer robustness results for backbone comparison runs.

Discovers:
    experiments/runs/robustness/f3/backbone_comparison/
        <backbone>/<variant_dir>/<data_seed_dir>/<run_id>/f3_metrics.csv

Outputs (under --output-dir):
    backbone_f3_raw.csv               - per-run no-reference diagnostics with labels
    backbone_f3_replicate_summary.csv - mean±std per (backbone, variant) over all 9 runs
    backbone_f3_bars.png              - bar chart: MS-SSIM-R and amplitude ratio by backbone/variant

Usage:
    python evaluation/summarize_backbone_f3_robustness.py \\
        --robustness-root experiments/runs/robustness/f3/backbone_comparison \\
        --output-dir experiments/summaries/f3_backbone_robustness
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

F3_METRICS = [
    "ms_ssim_r",
    "residual_energy_frac",
    "residual_input_corr",
    "denoised_input_amplitude_ratio",
    "low_freq_residual_energy_frac",
]

BACKBONE_DISPLAY = {
    "sfm_vit_base_patch16": "SFM-Base ViT-B/16",
    "swin_v2_t": "SwinV2-T",
}

BACKBONE_ORDER = ["sfm_vit_base_patch16", "swin_v2_t"]
VARIANT_ORDER = ["2D", "3ch", "5ch"]


def _variant_label(variant_dir: str) -> str | None:
    d = variant_dir.lower()
    if "neighbors5" in d:
        return "5ch"
    if "neighbors3" in d:
        return "3ch"
    if "2d" in d:
        return "2D"
    return None


def _read_f3_metrics(csv_path: Path) -> dict[str, float] | None:
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        result: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            for key in F3_METRICS:
                if key in row and row[key] not in ("", "nan"):
                    try:
                        result[key].append(float(row[key]))
                    except ValueError:
                        pass
        return {k: float(np.mean(v)) for k, v in result.items() if v}
    except Exception as exc:
        print(f"  [WARN] Could not read {csv_path}: {exc}", file=sys.stderr)
        return None


def discover_results(rob_root: Path) -> list[dict]:
    records: list[dict] = []
    for backbone_dir in sorted(rob_root.iterdir()):
        if not backbone_dir.is_dir():
            continue
        backbone = backbone_dir.name
        for variant_dir in sorted(backbone_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = _variant_label(variant_dir.name)
            if variant is None:
                print(f"  [SKIP] Unrecognised variant dir: {variant_dir.name}", file=sys.stderr)
                continue
            for data_seed_dir in sorted(variant_dir.iterdir()):
                if not data_seed_dir.is_dir():
                    continue
                data_seed = data_seed_dir.name  # e.g. data_seed101
                for run_dir in sorted(data_seed_dir.iterdir()):
                    if not run_dir.is_dir():
                        continue
                    metrics_csv = run_dir / "f3_metrics.csv"
                    if not metrics_csv.exists():
                        continue
                    metrics = _read_f3_metrics(metrics_csv)
                    if metrics is None:
                        continue
                    records.append({
                        "backbone": backbone,
                        "variant": variant,
                        "data_seed": data_seed,
                        "run_id": run_dir.name,
                        **metrics,
                    })
    return records


def aggregate_by_cell(records: list[dict]) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        cells[(r["backbone"], r["variant"])].append(r)

    rows = []
    for (backbone, variant), cell_records in sorted(cells.items()):
        row: dict = {"backbone": backbone, "variant": variant, "n": len(cell_records)}
        for metric in F3_METRICS:
            vals = [r[metric] for r in cell_records if metric in r]
            if vals:
                row[f"{metric}_mean"] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  [SKIP] No rows to write to {path}", file=sys.stderr)
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows to {path}")


def plot_bars(agg_rows: list[dict], output_dir: Path) -> None:
    if not _HAS_MPL:
        print("  [SKIP] matplotlib not available; skipping bar chart.", file=sys.stderr)
        return

    import matplotlib.pyplot as plt  # noqa: PLC0415

    cells: dict[tuple, dict] = {}
    for row in agg_rows:
        cells[(row["backbone"], row["variant"])] = row

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics_to_plot = [
        ("ms_ssim_r", "MS-SSIM-R (lower = more noise removed)", True),
        ("denoised_input_amplitude_ratio", "Amplitude ratio (higher = better preservation)", False),
    ]

    variant_colors = {"2D": "#888888", "3ch": "#3a86ff", "5ch": "#06d6a0"}
    x = np.arange(len(BACKBONE_ORDER))
    width = 0.25

    for ax, (metric, ylabel, lower_better) in zip(axes, metrics_to_plot):
        for vi, variant in enumerate(VARIANT_ORDER):
            means, errs = [], []
            for backbone in BACKBONE_ORDER:
                cell = cells.get((backbone, variant), {})
                means.append(cell.get(f"{metric}_mean", float("nan")))
                errs.append(cell.get(f"{metric}_std", 0.0))
            offset = (vi - 1) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                label=variant,
                color=variant_colors[variant],
                yerr=errs,
                capsize=3,
                error_kw={"linewidth": 1},
            )
            for bar, val in zip(bars, means):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (errs[0] if errs else 0) + 0.005,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [BACKBONE_DISPLAY.get(b, b) for b in BACKBONE_ORDER],
            fontsize=9,
        )
        ax.legend(fontsize=8)
        ax.set_title(f"{'Lower is better' if lower_better else 'Higher is better'}", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Backbone comparison: F3 field-transfer diagnostics (mean ± 1 std, 9 runs per cell)",
        fontsize=10,
    )
    plt.tight_layout()
    out_path = output_dir / "backbone_f3_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-root",
        default="experiments/runs/robustness/f3/backbone_comparison",
        help="Root directory containing backbone comparison F3 results.",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/summaries/f3_backbone_robustness",
        help="Directory to write summary CSVs and figures.",
    )
    args = parser.parse_args()

    rob_root = Path(args.robustness_root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not rob_root.exists():
        sys.exit(f"ERROR: robustness root not found: {rob_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering results under: {rob_root}")
    records = discover_results(rob_root)
    print(f"Found {len(records)} runs with F3 metrics.")

    if not records:
        print("No results found. Check that F3 eval jobs have completed.")
        return

    write_csv(output_dir / "backbone_f3_raw.csv", records)

    agg_rows = aggregate_by_cell(records)
    write_csv(output_dir / "backbone_f3_replicate_summary.csv", agg_rows)

    print("\n=== Backbone F3 Summary ===")
    for row in agg_rows:
        ms = row.get("ms_ssim_r_mean", float("nan"))
        ms_s = row.get("ms_ssim_r_std", float("nan"))
        amp = row.get("denoised_input_amplitude_ratio_mean", float("nan"))
        amp_s = row.get("denoised_input_amplitude_ratio_std", float("nan"))
        n = row.get("n", "?")
        print(
            f"  {row['backbone']:30s}  {row['variant']:4s}  "
            f"MS-SSIM-R={ms:.4f}±{ms_s:.4f}  amp={amp:.4f}±{amp_s:.4f}  n={n}"
        )

    plot_bars(agg_rows, output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
