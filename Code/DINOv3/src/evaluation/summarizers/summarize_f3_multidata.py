"""Aggregate F3 robustness results from all 27 main_multidata PEFT runs.

After submit_peft_f3_batch.sh completes, each run produces:
    experiments/runs/robustness/f3_allsections/main_multidata/
        {family}/{variant_dir}/data_seed{N}/{run_id}/f3_metrics.csv

This script aggregates across all 9 (data_seed × training_seed) combinations
per variant and writes a summary table plus plots.

Usage:
    python evaluation/summarize_f3_multidata.py [--project-root PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root
from typing import Any

import numpy as np

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_SEEDS = [101, 202, 303]
TRAINING_SEEDS = [42, 43, 44]
SEED_TO_RUN = {42: "seed42_run01", 43: "seed43_run02", 44: "seed44_run03"}

VARIANTS = {
    "2D": {"family": "2d", "variant_dir": "impeccable_repeated_stride5_lora_r16"},
    "3ch": {"family": "3ch", "variant_dir": "impeccable_neighbors3_stride5_lora_r16"},
    "5ch": {"family": "5ch", "variant_dir": "impeccable_neighbors5_stride5_patch_emb_lora_r16"},
}
VARIANT_ORDER = ["2D", "3ch", "5ch"]

F3_METRIC_KEYS = [
    "ms_ssim_r",
    "residual_energy_frac",
    "residual_input_corr",
    "denoised_input_amplitude_ratio",
    "low_freq_residual_energy_frac",
]

F3_METRIC_LABELS = {
    "ms_ssim_r": "MS-SSIM-R",
    "residual_energy_frac": "Residual energy fraction",
    "residual_input_corr": "Residual-input correlation",
    "denoised_input_amplitude_ratio": "Denoised/input amplitude",
    "low_freq_residual_energy_frac": "Low-frequency residual energy",
}


def _mean_std(vals: list[float]) -> tuple[float, float]:
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    mu = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mu, sd


def _read_run_metrics(f3_csv: Path) -> dict[str, float] | None:
    if not f3_csv.exists():
        return None
    with f3_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    result: dict[str, float] = {}
    for key in F3_METRIC_KEYS:
        vals = [float(r[key]) for r in rows if r.get(key, "") != ""]
        result[key] = float(np.mean(vals)) if vals else float("nan")
    result["n_samples"] = len(rows)
    return result


def _collect_variant(rob_root: Path, variant_key: str) -> dict[str, Any]:
    cfg = VARIANTS[variant_key]
    family, variant_dir = cfg["family"], cfg["variant_dir"]
    per_run: list[dict[str, float]] = []
    missing: list[str] = []

    for ds in DATA_SEEDS:
        for ts in TRAINING_SEEDS:
            run_id = SEED_TO_RUN[ts]
            f3_csv = rob_root / family / variant_dir / f"data_seed{ds}" / run_id / "f3_metrics.csv"
            metrics = _read_run_metrics(f3_csv)
            tag = f"ds{ds}_ts{ts}"
            if metrics is None:
                missing.append(tag)
            else:
                per_run.append(metrics)

    agg: dict[str, Any] = {
        "variant_key": variant_key,
        "n_complete": len(per_run),
        "n_total": 9,
        "missing": missing,
    }
    for key in F3_METRIC_KEYS:
        vals = [r[key] for r in per_run if np.isfinite(r.get(key, float("nan")))]
        mu, sd = _mean_std(vals)
        agg[f"{key}_mean"] = mu
        agg[f"{key}_std"] = sd
    if per_run:
        agg["n_samples_per_run"] = float(np.mean([r.get("n_samples", 0) for r in per_run]))
    return agg


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")


def _write_report(agg_rows: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# F3 Robustness Report — PEFT Main Multidata (3x3 Protocol)",
        "",
        "Aggregate across 9 runs per variant (3 data seeds × 3 training seeds).",
        "No clean ground truth: lower residual metrics are preferred; amplitude ratio is a preservation check.",
        "",
        "| Variant | Runs | MS-SSIM-R | Residual energy | Residual-input corr | Amplitude ratio | Low-freq residual |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def _fmt2(mean: Any, std: Any) -> str:
        try:
            return f"{float(mean):.4f} ± {float(std):.4f}"
        except (TypeError, ValueError):
            return "-"

    for row in agg_rows:
        lines.append(
            f"| {row['variant_key']} | {row['n_complete']}/9 "
            f"| {_fmt2(row.get('ms_ssim_r_mean'), row.get('ms_ssim_r_std'))} "
            f"| {_fmt2(row.get('residual_energy_frac_mean'), row.get('residual_energy_frac_std'))} "
            f"| {_fmt2(row.get('residual_input_corr_mean'), row.get('residual_input_corr_std'))} "
            f"| {_fmt2(row.get('denoised_input_amplitude_ratio_mean'), row.get('denoised_input_amplitude_ratio_std'))} "
            f"| {_fmt2(row.get('low_freq_residual_energy_frac_mean'), row.get('low_freq_residual_energy_frac_std'))} |"
        )

    lines.extend([
        "",
        "## Claim Boundaries",
        "",
        "- F3 results measure unlabelled field-domain transfer only.",
        "- No accuracy metrics are reported (no clean ground truth).",
        "- Models were selected on Image Impeccable validation, not F3.",
        "",
        "## Missing Runs",
        "",
    ])
    any_missing = False
    for row in agg_rows:
        if row["missing"]:
            any_missing = True
            lines.append(f"- **{row['variant_key']}**: missing {row['missing']}")
    if not any_missing:
        lines.append("All 9 runs complete for all variants.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def _plot_bars(agg_rows: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot")
        return

    metrics = ["ms_ssim_r", "residual_energy_frac", "residual_input_corr",
               "denoised_input_amplitude_ratio", "low_freq_residual_energy_frac"]
    variants = [r["variant_key"] for r in agg_rows]
    x = np.arange(len(variants))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, metric in zip(axes.flat, metrics):
        means = [float(r.get(f"{metric}_mean") or 0.0) for r in agg_rows]
        stds = [float(r.get(f"{metric}_std") or 0.0) for r in agg_rows]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(variants)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_title(F3_METRIC_LABELS[metric])
        ax.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle(
        "F3 field-transfer diagnostics — PEFT 3x3 multidata (9 runs per variant)\n"
        "No clean target; lower residual diagnostics are better"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--rob-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    rob_root = (
        args.rob_root
        or project_root / "experiments" / "runs" / "robustness" / "f3_allsections" / "main_multidata"
    ).resolve()
    out_dir = (
        args.out_dir
        or project_root / "experiments" / "summaries" / "f3_allsections_robustness_multidata"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Robustness root: {rob_root}")
    print(f"Output dir:      {out_dir}")

    agg_rows: list[dict[str, Any]] = []
    for variant_key in VARIANT_ORDER:
        agg = _collect_variant(rob_root, variant_key)
        n = agg["n_complete"]
        ms = agg.get("ms_ssim_r_mean", float("nan"))
        ms_s = agg.get("ms_ssim_r_std", float("nan"))
        print(f"  {variant_key}: {n}/9 complete  MS-SSIM-R: {ms:.4f} ± {ms_s:.4f}")
        if agg["missing"]:
            print(f"    missing: {agg['missing'][:4]}")
        agg_rows.append(agg)

    if not any(r["n_complete"] > 0 for r in agg_rows):
        sys.exit("No F3 results found. Run submit_peft_f3_batch.sh first.")

    _write_csv(agg_rows, out_dir / "f3_multidata_summary.csv")
    _write_report(agg_rows, out_dir / "f3_multidata_report.md")
    _plot_bars(agg_rows, out_dir / "f3_multidata_bars.png")
    print(f"\nF3 multidata summary saved to: {out_dir}")


if __name__ == "__main__":
    main()
