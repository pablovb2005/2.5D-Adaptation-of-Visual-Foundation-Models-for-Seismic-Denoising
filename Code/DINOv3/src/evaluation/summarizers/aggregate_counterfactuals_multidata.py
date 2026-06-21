"""Aggregate context counterfactual results from all 9 multidata runs.

After submit_counterfactuals_multidata.sh completes, each (data_seed, training_seed)
run produces CSVs under:

    experiments/summaries/mechanism_analysis/context_counterfactuals_full/ds{ds}_ts{ts}/
        context_counterfactual_summary.csv   (per-variant × per-condition aggregate)
        context_counterfactual_metrics.csv   (per-sample rows)

This script reads the per-run summary CSVs, averages the MS-SSIM and
clean_similarity deltas across all 9 runs, and writes a combined report.

Usage:
    python evaluation/aggregate_counterfactuals_multidata.py [--project-root PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root
from typing import Any

import numpy as np

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_SEEDS = [101, 202, 303]
TRAINING_SEEDS = [42, 43, 44]
VARIANT_ORDER = ["2D", "3ch", "5ch"]
CONDITION_ORDER = ["aligned", "repeated_center", "shuffled_neighbors", "distant_neighbors"]

FLOAT_COLS = [
    "ms_ssim_mean", "ms_ssim_std", "ms_ssim_delta_vs_aligned",
    "clean_similarity_mean", "clean_similarity_std", "clean_similarity_delta_vs_aligned",
    "mse_mean", "mse_std", "mse_delta_vs_aligned",
    "psnr_mean", "psnr_std", "psnr_delta_vs_aligned",
]


def _read_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [
            {k: (float(v) if k in FLOAT_COLS else v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _collect_all_runs(cf_root: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Returns {(variant_key, condition): [row_from_each_run]}."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    n_found = 0
    for ds in DATA_SEEDS:
        for ts in TRAINING_SEEDS:
            run_dir = cf_root / f"ds{ds}_ts{ts}"
            summary_path = run_dir / "context_counterfactual_summary.csv"
            rows = _read_summary(summary_path)
            if not rows:
                print(f"  [MISSING] {summary_path}")
                continue
            n_found += 1
            for row in rows:
                grouped[(row["variant_key"], row["condition"])].append(row)
    print(f"Loaded {n_found}/9 per-run summaries")
    return grouped


def _aggregate_grouped(grouped: dict[tuple[str, str], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        for condition in CONDITION_ORDER:
            rows = grouped.get((variant, condition), [])
            if not rows:
                continue
            agg: dict[str, Any] = {
                "variant_key": variant,
                "condition": condition,
                "n_runs": len(rows),
            }
            for col in FLOAT_COLS:
                vals = [r[col] for r in rows if col in r and np.isfinite(r[col])]
                agg[f"{col}_agg_mean"] = float(np.mean(vals)) if vals else float("nan")
                agg[f"{col}_agg_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out.append(agg)
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")


def _write_report(agg_rows: list[dict[str, Any]], n_runs: int, out_path: Path) -> None:
    lines = [
        "# Context Counterfactual Report (3x3 Multidata Protocol)",
        "",
        f"Aggregated across {n_runs}/9 runs (3 data seeds × 3 training seeds).",
        "No new training was used. Existing checkpoints were evaluated with broken neighboring context.",
        "",
        "| Variant | Condition | n runs | MS-SSIM | Δ MS-SSIM vs aligned | Clean-token sim | Δ sim vs aligned |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in agg_rows:
        ms = row.get("ms_ssim_mean_agg_mean", float("nan"))
        ms_delta = row.get("ms_ssim_delta_vs_aligned_agg_mean", float("nan"))
        sim = row.get("clean_similarity_mean_agg_mean", float("nan"))
        sim_delta = row.get("clean_similarity_delta_vs_aligned_agg_mean", float("nan"))
        lines.append(
            f"| {row['variant_key']} | {row['condition']} | {row['n_runs']} | "
            f"{ms:.6f} | {ms_delta:+.6f} | {sim:.6f} | {sim_delta:+.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A large negative Δ MS-SSIM vs aligned for shuffled or distant neighbors means the 2.5D model",
        "depends on correctly aligned neighboring slices, not merely on having extra channels.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--cf-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cf_root = (
        args.cf_root
        or project_root / "experiments" / "summaries" / "mechanism_analysis" / "context_counterfactuals_full"
    ).resolve()
    out_dir = (
        args.out_dir
        or project_root / "experiments" / "summaries" / "mechanism_analysis" / "context_counterfactuals_multidata"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading per-run summaries from: {cf_root}")
    grouped = _collect_all_runs(cf_root)

    if not grouped:
        sys.exit("No counterfactual summaries found. Run submit_counterfactuals_multidata.sh first.")

    agg_rows = _aggregate_grouped(grouped)
    n_runs = max((r["n_runs"] for r in agg_rows), default=0)
    _write_csv(agg_rows, out_dir / "context_counterfactual_multidata_summary.csv")
    _write_report(agg_rows, n_runs, out_dir / "context_counterfactual_multidata_report.md")
    print(f"\nAggregated counterfactual results saved to: {out_dir}")


if __name__ == "__main__":
    main()
