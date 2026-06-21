"""Aggregate filtered-reference F3 evaluation results into CSVs, plots, and a report.

Reads per-section f3_filtered_ref_metrics.csv files produced by
evaluate_filtered_reference.py and writes aggregate summaries.

Usage:
    python evaluation/summarize_filtered_reference.py \
        --robustness-root experiments/runs/robustness \
        --output-dir experiments/summaries/f3_filtered_ref_robustness

Discovers:
    <robustness-root>/f3_filtered_ref/<family>/<variant>/<run_id>/f3_filtered_ref_metrics.csv

Outputs:
    f3_filtered_ref_summary.csv              - per-run metrics (both pseudo-paired and no-ref)
    f3_filtered_ref_main_summary.csv         - main 2D/3ch/5ch aggregate across seeds
    f3_filtered_ref_paired_bars.png          - pseudo-paired metric bars (MS-SSIM, PSNR)
    f3_filtered_ref_noref_bars.png           - no-reference diagnostic bars
    f3_filtered_ref_report.md               - markdown report with claim boundaries

CLAIM FRAMING: pseudo-paired results are "filtered-reference agreement", never
"real field ground-truth accuracy" or "clean field accuracy".
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


NOREF_METRIC_KEYS = [
    "ms_ssim_r",
    "residual_energy_frac",
    "residual_input_corr",
    "denoised_input_amplitude_ratio",
    "low_freq_residual_energy_frac",
]

PAIRED_METRIC_KEYS = [
    "ms_ssim_ref",
    "mse_ref",
    "psnr_ref",
]

ALL_METRIC_KEYS = PAIRED_METRIC_KEYS + NOREF_METRIC_KEYS

METRIC_LABELS = {
    "ms_ssim_ref": "MS-SSIM vs filtered ref",
    "mse_ref": "MSE vs filtered ref",
    "psnr_ref": "PSNR vs filtered ref (dB)",
    "ms_ssim_r": "MS-SSIM-R (no-ref)",
    "residual_energy_frac": "Residual energy fraction",
    "residual_input_corr": "Residual-input correlation",
    "denoised_input_amplitude_ratio": "Denoised/input amplitude",
    "low_freq_residual_energy_frac": "Low-frequency residual energy",
}

VARIANT_ORDER = {"2D": 0, "3ch": 1, "5ch": 2}
MAIN_VARIANTS = {"2D", "3ch", "5ch"}
MAIN_RUN_IDS = {"seed42_run01", "seed43_run02", "seed44_run03"}
DISPLAY_VARIANT_LABELS = {
    "2D": "2D-1",
    "3ch": "2.5D-3",
    "5ch": "2.5D-5",
}

RESULT_DATASET = "f3_filtered_ref"
METRICS_CSV_NAME = "f3_filtered_ref_metrics.csv"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _variant_key(family: str, variant_dir: str) -> str | None:
    text = f"{family}/{variant_dir}".lower()
    if "patch_emb_head" in text or "5ch_a" in text or "_a_" in text:
        return None
    if family == "2d" or text.startswith("2d/"):
        return "2D"
    for channels in (3, 5, 7, 9):
        label = f"{channels}ch"
        if family == label or text.startswith(f"{label}/"):
            return label
    return None


def _n_vols_from_variant(variant_dir: str) -> int:
    match = re.search(r"_n(\d+)vols", variant_dir)
    if match:
        return int(match.group(1))
    return 20


def _sort_key(row: dict[str, Any]) -> tuple:
    return (
        VARIANT_ORDER.get(str(row.get("variant_key")), 9),
        str(row.get("run_id") or ""),
    )


def _discover_runs(robustness_root: Path, result_dataset: str = RESULT_DATASET) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    result_root = robustness_root / result_dataset
    if not result_root.exists():
        return records

    for csv_path in sorted(result_root.rglob(METRICS_CSV_NAME)):
        parts = csv_path.relative_to(result_root).parts
        if len(parts) < 4:
            continue
        family, variant_dir, run_id = parts[0], parts[1], parts[2]
        variant_key = _variant_key(family, variant_dir)
        if variant_key is None:
            continue
        run_group = "main" if run_id in MAIN_RUN_IDS else "other"
        records.append(
            {
                "family": family,
                "variant_dir": variant_dir,
                "variant_key": variant_key,
                "run_id": run_id,
                "run_group": run_group,
                "n_vols": _n_vols_from_variant(variant_dir),
                "csv_path": csv_path,
            }
        )
    return sorted(records, key=_sort_key)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(v: object) -> float | None:
    try:
        value = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return None, None
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return mean, None
    return mean, float(np.std(vals, ddof=1))


def _protocol_value(data: list[dict[str, str]], key: str, default: Any = None) -> Any:
    vals = sorted({str(row.get(key, "")).strip() for row in data if str(row.get(key, "")).strip()})
    if not vals:
        return default
    return vals[0] if len(vals) == 1 else ",".join(vals)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def build_run_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        data = _read_csv(rec["csv_path"])
        row: dict[str, Any] = {
            "variant_key": rec["variant_key"],
            "run_group": rec["run_group"],
            "run_id": rec["run_id"],
            "family": rec["family"],
            "variant_dir": rec["variant_dir"],
            "n_samples": len(data),
            "orientations": ",".join(
                sorted({str(r.get("orientation", "")).strip() for r in data if r.get("orientation")})
            ),
            "mode": _protocol_value(data, "mode", ""),
            "neighbor_stride": _protocol_value(data, "neighbor_stride", ""),
            "sample_count_request": _protocol_value(data, "sample_count_request", ""),
            "common_context_radius": _protocol_value(data, "common_context_radius", ""),
        }
        for metric in ALL_METRIC_KEYS:
            vals = [_float(d.get(metric)) for d in data]
            vals = [v for v in vals if v is not None]
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return sorted(rows, key=_sort_key)


def build_main_summary(run_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in run_summary:
        if (
            row.get("run_group") == "main"
            and row.get("run_id") in MAIN_RUN_IDS
            and row.get("variant_key") in MAIN_VARIANTS
        ):
            grouped[str(row["variant_key"])].append(row)

    rows: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(grouped.items(), key=lambda item: VARIANT_ORDER.get(item[0], 9)):
        out: dict[str, Any] = {
            "variant_key": variant,
            "display_label": DISPLAY_VARIANT_LABELS.get(variant, variant),
            "n_runs": len(variant_rows),
            "seeds": ",".join(
                r["run_id"].split("_")[0].replace("seed", "")
                for r in sorted(variant_rows, key=_sort_key)
            ),
            "samples_per_run": ",".join(
                str(r.get("n_samples")) for r in sorted(variant_rows, key=_sort_key)
            ),
        }
        for metric in ALL_METRIC_KEYS:
            vals = [_float(r.get(f"{metric}_mean")) for r in variant_rows]
            vals = [v for v in vals if v is not None]
            mean, std = _mean_std(vals)
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        rows.append(out)
    return rows


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _require_mpl() -> None:
    if not _HAS_MPL:
        raise RuntimeError("matplotlib is required for plotting.")


def _load_pyplot():
    _require_mpl()
    import matplotlib.pyplot as plt
    return plt


def _fmt(v: Any, decimals: int = 4) -> str:
    value = _float(v)
    return f"{value:.{decimals}f}" if value is not None else "-"


def _mean_std_text(row: dict[str, Any], metric: str, decimals: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None:
        return "-"
    if std is None:
        return _fmt(mean, decimals)
    return f"{_fmt(mean, decimals)} +/- {_fmt(std, decimals)}"


def plot_paired_metrics(main_summary: list[dict[str, Any]], out_path: Path) -> None:
    if not main_summary:
        return
    plt = _load_pyplot()
    variants = [DISPLAY_VARIANT_LABELS.get(r["variant_key"], r["variant_key"]) for r in main_summary]
    x = np.arange(len(variants))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, metric in zip(axes, PAIRED_METRIC_KEYS):
        means = [float(r.get(f"{metric}_mean") or 0.0) for r in main_summary]
        stds = [float(r.get(f"{metric}_std") or 0.0) for r in main_summary]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(variants)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle(
        "F3 filtered-reference agreement across main seeds\n"
        "(reference = dip-steered median filter; not ground-truth accuracy)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_noref_metrics(main_summary: list[dict[str, Any]], out_path: Path) -> None:
    if not main_summary:
        return
    plt = _load_pyplot()
    variants = [DISPLAY_VARIANT_LABELS.get(r["variant_key"], r["variant_key"]) for r in main_summary]
    x = np.arange(len(variants))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    metrics = [k for k in NOREF_METRIC_KEYS if k != "ms_ssim_r"]  # ms_ssim_r is less informative here
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        means = [float(r.get(f"{metric}_mean") or 0.0) for r in main_summary]
        stds = [float(r.get(f"{metric}_std") or 0.0) for r in main_summary]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(variants)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("F3 no-reference diagnostics (filtered-reference evaluation runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Report and CSV writers
# ---------------------------------------------------------------------------

def build_report(run_summary: list[dict[str, Any]], main_summary: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# F3 Filtered-Reference Evaluation Report",
        "",
        "> Generated by `summarize_filtered_reference.py`.",
        "> Do not edit manually; re-run the summarizer after adding new results.",
        "",
        "## Claim Boundaries",
        "",
        "- Results measure agreement with an algorithmically filtered pseudo-clean reference,",
        "  NOT real field ground-truth accuracy.",
        "- The reference is the dip-steered median-filter volume from F3 Demo 2023.",
        "  It is a teacher output from a classical filter, not a clean seismic recording.",
        "- Use safe wording: 'filtered-reference agreement', 'pseudo-paired agreement',",
        "  or 'agreement with an algorithmically filtered field-data reference'.",
        "- Do NOT write: 'real field accuracy', 'clean field target', 'ground-truth denoising accuracy'.",
        "- `best.pt` checkpoints were selected only on Image Impeccable validation.",
        "  F3 is used only at evaluation time; no fine-tuning on F3.",
        "- `ms_ssim_ref`, `mse_ref`, `psnr_ref`: pseudo-paired metrics vs filtered reference.",
        "- No-reference diagnostics (`ms_ssim_r`, `residual_energy_frac`, etc.) are identical",
        "  to the unlabelled F3 robustness experiment and reported here for completeness.",
        "",
        "## Main Filtered-Reference Agreement (Pseudo-Paired)",
        "",
        "Higher MS-SSIM and PSNR, lower MSE indicate better agreement with the filtered reference.",
        "These are teacher-reference agreement scores, not accuracy claims.",
        "",
    ]

    if main_summary:
        lines.append(
            "| Variant | Display | Runs | Seeds | Samples/run "
            "| MS-SSIM-ref | MSE-ref | PSNR-ref (dB) |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in main_summary:
            lines.append(
                f"| {r['variant_key']} | {r['display_label']} | {r['n_runs']} | {r['seeds']} "
                f"| {r.get('samples_per_run', '-')} "
                f"| {_mean_std_text(r, 'ms_ssim_ref')} "
                f"| {_mean_std_text(r, 'mse_ref')} "
                f"| {_mean_std_text(r, 'psnr_ref')} |"
            )
    else:
        lines.append("No main results found.")

    lines += [
        "",
        "## Main No-Reference Diagnostics",
        "",
        "These match the unlabelled F3 robustness experiment since the same checkpoints",
        "and noisy input are used. Lower residual diagnostics indicate more filtering.",
        "",
    ]

    if main_summary:
        lines.append(
            "| Variant | MS-SSIM-R | Residual energy | Residual-input corr "
            "| Amplitude ratio | Low-freq residual |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in main_summary:
            lines.append(
                f"| {r['variant_key']} "
                f"| {_mean_std_text(r, 'ms_ssim_r')} "
                f"| {_mean_std_text(r, 'residual_energy_frac')} "
                f"| {_mean_std_text(r, 'residual_input_corr')} "
                f"| {_mean_std_text(r, 'denoised_input_amplitude_ratio')} "
                f"| {_mean_std_text(r, 'low_freq_residual_energy_frac')} |"
            )

    lines += [
        "",
        "## Per-Run Results",
        "",
    ]

    if run_summary:
        lines.append(
            "| Variant | Group | Run | Samples | MS-SSIM-ref | MSE-ref | PSNR-ref |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in run_summary:
            lines.append(
                f"| {r['variant_key']} | {r['run_group']} | {r['run_id']} "
                f"| {r['n_samples']} "
                f"| {_fmt(r.get('ms_ssim_ref_mean'))} "
                f"| {_fmt(r.get('mse_ref_mean'))} "
                f"| {_fmt(r.get('psnr_ref_mean'))} |"
            )

    lines += [
        "",
        "## Generated Figures",
        "",
        "- `f3_filtered_ref_paired_bars.png`: pseudo-paired agreement metrics for 2D, 3ch, 5ch.",
        "- `f3_filtered_ref_noref_bars.png`: no-reference diagnostics for completeness.",
        "",
        "## Thesis Usage",
        "",
        "Safe wording template:",
        "",
        "> On the F3 field volume, the 2.5D checkpoints show stronger agreement with",
        "> an algorithmically filtered reference (dip-steered median filter) than the",
        "> 2D control, consistent with the paired Image Impeccable results.",
        "> Because the reference is a teacher output rather than clean ground truth,",
        "> these results are supplementary filtered-reference diagnostics, not accuracy claims.",
        "",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved: {out_path}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        print(f"  No data to write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate filtered-reference F3 evaluation results."
    )
    parser.add_argument("--robustness-root", required=True, help="Root of experiments/runs/robustness/.")
    parser.add_argument(
        "--result-dataset",
        default=RESULT_DATASET,
        help=(
            "Result folder under robustness-root to summarize "
            f"(default: {RESULT_DATASET}). Use 'f3_filtered_ref_horizontal' for "
            "the horizontal time-slice re-evaluation."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Where to write summary files.")
    args = parser.parse_args()

    rob_root = Path(args.robustness_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Robustness root: {rob_root}")
    print(f"Result dataset: {args.result_dataset}")
    print(f"Output dir: {out_dir}")

    records = _discover_runs(rob_root, args.result_dataset)
    print(f"Found {len(records)} filtered-reference run(s).")

    if not records:
        sys.exit(
            f"No filtered-reference results found under {rob_root / args.result_dataset}. "
            "Run evaluate_filtered_reference.py first."
        )

    run_summary = build_run_summary(records)
    main_summary = build_main_summary(run_summary)
    _write_csv(run_summary, out_dir / "f3_filtered_ref_summary.csv")
    _write_csv(main_summary, out_dir / "f3_filtered_ref_main_summary.csv")

    if _HAS_MPL:
        plot_paired_metrics(main_summary, out_dir / "f3_filtered_ref_paired_bars.png")
        plot_noref_metrics(main_summary, out_dir / "f3_filtered_ref_noref_bars.png")
    else:
        print("matplotlib not installed; skipping plots.")

    build_report(run_summary, main_summary, out_dir / "f3_filtered_ref_report.md")
    print("\nSummarization complete.")


if __name__ == "__main__":
    main()
