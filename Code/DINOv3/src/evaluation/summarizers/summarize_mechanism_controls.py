"""Summarize mechanism control training runs for thesis comparison.

Reads eval_results/results.csv from each control run and computes 3-seed
mean ± std, then writes a comparison table against the existing baselines.

Controls:
  3ch_shuffled         — 3ch architecture, shuffled neighbors during training
  5ch_repeated_center  — 5ch architecture, [t,t,t,t,t] capacity control
  5ch_shuffled         — 5ch architecture, shuffled neighbors during training

Usage:
    python evaluation/summarize_mechanism_controls.py [--project-root PATH]
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

STUDENT_BULK = Path("/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal")

CONTROLS = {
    "3ch_shuffled": {
        "label": "3ch_shuffled (train shuffled neighbors)",
        "seeds": {
            "seed42_run01": "3ch_shuffled/impeccable_shuffled3_stride5_lora_r16/seed42_run01",
            "seed43_run02": "3ch_shuffled/impeccable_shuffled3_stride5_lora_r16/seed43_run02",
            "seed44_run03": "3ch_shuffled/impeccable_shuffled3_stride5_lora_r16/seed44_run03",
        },
    },
    "5ch_repeated_center": {
        "label": "5ch_repeated_center ([t,t,t,t,t] capacity control)",
        "seeds": {
            "seed42_run01": "5ch_repeated_center/impeccable_repeated_center_stride5_patch_emb_lora_r16/seed42_run01",
            "seed43_run02": "5ch_repeated_center/impeccable_repeated_center_stride5_patch_emb_lora_r16/seed43_run02",
            "seed44_run03": "5ch_repeated_center/impeccable_repeated_center_stride5_patch_emb_lora_r16/seed44_run03",
        },
    },
    "5ch_shuffled": {
        "label": "5ch_shuffled (train shuffled neighbors)",
        "seeds": {
            "seed42_run01": "5ch_shuffled/impeccable_shuffled5_stride5_patch_emb_lora_r16/seed42_run01",
            "seed43_run02": "5ch_shuffled/impeccable_shuffled5_stride5_patch_emb_lora_r16/seed43_run02",
            "seed44_run03": "5ch_shuffled/impeccable_shuffled5_stride5_patch_emb_lora_r16/seed44_run03",
        },
    },
}

BASELINES = {
    "2D": {
        "label": "2D (repeated center, train aligned)",
        "seeds": {
            "seed42_run01": "2d/impeccable_repeated_stride5_lora_r16/seed42_run01",
            "seed43_run02": "2d/impeccable_repeated_stride5_lora_r16/seed43_run02",
            "seed44_run03": "2d/impeccable_repeated_stride5_lora_r16/seed44_run03",
        },
    },
    "3ch": {
        "label": "3ch (aligned neighbors, train aligned)",
        "seeds": {
            "seed42_run01": "3ch/impeccable_neighbors3_stride5_lora_r16/seed42_run01",
            "seed43_run02": "3ch/impeccable_neighbors3_stride5_lora_r16/seed43_run02",
            "seed44_run03": "3ch/impeccable_neighbors3_stride5_lora_r16/seed44_run03",
        },
    },
    "5ch": {
        "label": "5ch (aligned neighbors, train aligned)",
        "seeds": {
            "seed42_run01": "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed42_run01",
            "seed43_run02": "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed43_run02",
            "seed44_run03": "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed44_run03",
        },
    },
}

METRICS = ["ms_ssim", "mse", "psnr"]


def _read_mean_from_csv(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test":
                rows.append(row)
    if not rows:
        return None
    result = {}
    for m in METRICS:
        vals = [float(r[m]) for r in rows if r.get(m, "") != ""]
        result[m] = float(np.mean(vals)) if vals else float("nan")
    return result


def _load_group(runs_root: Path, seed_paths: dict[str, str]) -> dict[str, dict[str, float] | None]:
    results = {}
    for seed_key, rel_path in seed_paths.items():
        csv_path = runs_root / rel_path / "eval_results" / "results.csv"
        results[seed_key] = _read_mean_from_csv(csv_path)
    return results


def _aggregate(seed_results: dict[str, dict[str, float] | None]) -> dict[str, Any]:
    complete = {k: v for k, v in seed_results.items() if v is not None}
    missing = [k for k, v in seed_results.items() if v is None]
    out: dict[str, Any] = {"n_complete": len(complete), "missing_seeds": missing}
    if not complete:
        return out
    for m in METRICS:
        vals = [v[m] for v in complete.values() if not np.isnan(v[m])]
        out[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _write_report(summary: dict[str, dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Mechanism Control Training Summary",
        "",
        "Comparison between aligned-trained baselines and trained mechanism controls.",
        "All results: test MS-SSIM mean ± std over 3 seeds (where available).",
        "",
        "| Variant | Label | n seeds | Test MS-SSIM mean ± std | Test MSE mean ± std | Test PSNR mean ± std |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def _fmt_row(key: str, label: str, agg: dict[str, Any]) -> str:
        n = agg.get("n_complete", 0)
        if n == 0:
            return f"| {key} | {label} | 0/3 | — | — | — |"
        ms = agg.get("ms_ssim_mean", float("nan"))
        ms_s = agg.get("ms_ssim_std", 0.0)
        mse = agg.get("mse_mean", float("nan"))
        mse_s = agg.get("mse_std", 0.0)
        psnr = agg.get("psnr_mean", float("nan"))
        psnr_s = agg.get("psnr_std", 0.0)
        return (
            f"| {key} | {label} | {n}/3 "
            f"| {ms:.4f} ± {ms_s:.4f} "
            f"| {mse:.4f} ± {mse_s:.4f} "
            f"| {psnr:.2f} ± {psnr_s:.2f} |"
        )

    for key, agg in summary.items():
        label = BASELINES[key]["label"] if key in BASELINES else CONTROLS[key]["label"]
        lines.append(_fmt_row(key, label, agg))

    lines.extend([
        "",
        "## Interpretation",
        "",
        "**3ch_shuffled trained → test aligned vs 2D baseline (0.8079):**",
        "If 3ch_shuffled ≈ 2D, training requires correct neighbor alignment.",
        "",
        "**5ch_repeated_center → test aligned vs 2D baseline (0.8079):**",
        "If 5ch_repeated_center ≈ 2D, extra capacity alone does not explain the 5ch gain.",
        "",
        "**5ch_shuffled → test aligned vs 2D baseline (0.8079):**",
        "Consistent check with 3ch_shuffled; should also land near 2D if alignment matters.",
        "",
        "## Missing runs",
        "",
    ])
    any_missing = False
    for key, agg in summary.items():
        missing = agg.get("missing_seeds", [])
        if missing:
            any_missing = True
            lines.append(f"- **{key}**: missing seeds {missing}")
    if not any_missing:
        lines.append("All runs complete.")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def _write_csv(summary: dict[str, dict[str, Any]], out_path: Path) -> None:
    rows = []
    for key, agg in summary.items():
        row: dict[str, Any] = {"variant_key": key, "n_complete": agg.get("n_complete", 0)}
        for m in METRICS:
            row[f"{m}_mean"] = agg.get(f"{m}_mean", float("nan"))
            row[f"{m}_std"] = agg.get(f"{m}_std", float("nan"))
        row["missing_seeds"] = ";".join(agg.get("missing_seeds", []))
        rows.append(row)
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {out_path} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_project_root(__file__),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runs_root = project_root / "experiments" / "runs"
    out_dir = args.out_dir or (
        project_root / "experiments" / "summaries" / "mechanism_analysis" / "trained_controls"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, Any]] = {}

    print("Loading baseline results...")
    for key, cfg in BASELINES.items():
        seed_results = _load_group(runs_root, cfg["seeds"])
        agg = _aggregate(seed_results)
        summary[key] = agg
        status = f"{agg['n_complete']}/3 seeds complete"
        if agg["missing_seeds"]:
            status += f" (missing: {agg['missing_seeds']})"
        print(f"  {key}: {status}")
        if agg["n_complete"] > 0:
            print(f"    MS-SSIM: {agg.get('ms_ssim_mean', float('nan')):.4f} ± {agg.get('ms_ssim_std', 0.0):.4f}")

    print("\nLoading control results...")
    for key, cfg in CONTROLS.items():
        seed_results = _load_group(runs_root, cfg["seeds"])
        agg = _aggregate(seed_results)
        summary[key] = agg
        status = f"{agg['n_complete']}/3 seeds complete"
        if agg["missing_seeds"]:
            status += f" (missing: {agg['missing_seeds']})"
        print(f"  {key}: {status}")
        if agg["n_complete"] > 0:
            print(f"    MS-SSIM: {agg.get('ms_ssim_mean', float('nan')):.4f} ± {agg.get('ms_ssim_std', 0.0):.4f}")

    _write_csv(summary, out_dir / "trained_controls_summary.csv")
    _write_report(summary, out_dir / "trained_controls_report.md")

    print("\nSummary saved to:", out_dir)
    print("\nTo run full-test counterfactuals on the full test set (inference-time upgrade):")
    print("  python evaluation/analyze_context_counterfactuals.py --full-test")


if __name__ == "__main__":
    main()
