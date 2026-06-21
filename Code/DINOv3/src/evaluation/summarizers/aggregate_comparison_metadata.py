"""Aggregate comparison_metadata CSVs from all 9 multidata runs and run mechanism analysis.

After submit_generate_panels_multidata.sh completes, each (data_seed, training_seed)
combination produces a comparison_metadata.csv under:

    experiments/summaries/comparison_panels/ds{data_seed}/comparison_metadata.csv

This script concatenates all 9 (or however many are present) into a single CSV
and feeds it to analyze_mechanism.py to produce the stratification outputs.

Usage:
    python evaluation/aggregate_comparison_metadata.py [--project-root PATH]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_SEEDS = [101, 202, 303]
STRATIFY_METRICS = ["ms_ssim_2d", "input_psnr", "input_ms_ssim"]


def _collect_rows(panels_root: Path) -> list[dict]:
    all_rows: list[dict] = []
    found = 0
    for ds in DATA_SEEDS:
        meta_path = panels_root / f"ds{ds}" / "comparison_metadata.csv"
        if not meta_path.exists():
            print(f"  [MISSING] {meta_path}")
            continue
        with meta_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        # Tag each row with its data_seed for traceability
        for row in rows:
            row["data_seed"] = str(ds)
        all_rows.extend(rows)
        found += 1
        print(f"  [OK] ds{ds}: {len(rows)} rows")
    print(f"Total: {len(all_rows)} rows from {found}/3 data seeds")
    return all_rows


def _write_combined(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved combined metadata: {out_path} ({len(rows)} rows)")


def _run_mechanism_analysis(
    project_root: Path,
    metadata_path: Path,
    out_root: Path,
    stratify_metric: str,
) -> None:
    out_dir = out_root / f"by_{stratify_metric}"
    cmd = [
        sys.executable,
        str(SRC / "evaluation" / "analyze_mechanism.py"),
        "--project-root", str(project_root),
        "--metadata", str(metadata_path),
        "--out-dir", str(out_dir),
        "--stratify-metric", stratify_metric,
    ]
    print(f"\n--- analyze_mechanism: stratify_metric={stratify_metric} -> {out_dir} ---")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  [WARN] analyze_mechanism.py returned non-zero for {stratify_metric}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--panels-root", type=Path, default=None,
        help="Root containing ds{N}/comparison_metadata.csv files. "
             "Default: <project-root>/experiments/summaries/comparison_panels"
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    panels_root = (args.panels_root or project_root / "experiments" / "summaries" / "comparison_panels").resolve()
    out_dir = (args.out_dir or project_root / "experiments" / "summaries" / "mechanism_analysis" / "multidata").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting metadata from: {panels_root}")
    rows = _collect_rows(panels_root)
    if not rows:
        sys.exit("No comparison_metadata.csv files found. Run submit_generate_panels_multidata.sh first.")

    combined_path = out_dir / "comparison_metadata_combined.csv"
    _write_combined(rows, combined_path)

    print("\nRunning mechanism analysis for each stratify metric...")
    for metric in STRATIFY_METRICS:
        _run_mechanism_analysis(project_root, combined_path, out_dir, metric)

    print(f"\nAll mechanism analyses written to: {out_dir}")


if __name__ == "__main__":
    main()
