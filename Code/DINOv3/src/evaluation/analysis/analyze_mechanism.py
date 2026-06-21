"""Analyze when 2.5D context helps on Image Impeccable test samples.

This script consumes the per-sample metrics written by
``generate_comparison_panels.py`` and produces the mechanism-analysis outputs
used for the thesis results narrative:

    difficulty_stratification_summary.csv  - difficulty/input-quality quartiles
    difficulty_correlation_summary.csv     - correlations between stratification metric and gains
    difficulty_volume_summary.csv          - volume-level concentration check
    difficulty_top_examples.csv            - largest-gain samples for figure selection
    difficulty_gain_by_quartile.png        - quartile gain plot
    difficulty_delta_scatter.png           - sample-level gain vs 2D quality
    difficulty_volume_delta.png            - per-volume mean gain plot
    mechanism_analysis_report.md           - short interpretation with claim boundaries
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("matplotlib is required. Install it with: pip install matplotlib") from exc


REQUIRED_COLUMNS = [
    "vol_id",
    "slice_t",
    "sample_idx",
    "ms_ssim_2d",
    "ms_ssim_3ch",
    "ms_ssim_5ch",
    "delta_3ch_vs_2d",
    "delta_5ch_vs_2d",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "input_ms_ssim",
    "input_mse",
    "input_psnr",
]

STRATIFY_METRICS = {
    "ms_ssim_2d": {
        "label": "2D MS-SSIM",
        "higher_is_better": True,
        "description": "2D baseline MS-SSIM",
        "hardness": "lower 2D MS-SSIM means the sample is harder for the center-slice-only baseline",
        "folder": "stratification/by_2d_ms_ssim",
    },
    "input_psnr": {
        "label": "Noisy-input PSNR",
        "higher_is_better": True,
        "description": "PSNR between the noisy center slice and the clean target",
        "hardness": "lower noisy-input PSNR means the input slice is more degraded before any model is applied",
        "folder": "stratification/by_input_psnr",
    },
    "input_ms_ssim": {
        "label": "Noisy-input MS-SSIM",
        "higher_is_better": True,
        "description": "MS-SSIM between the noisy center slice and the clean target",
        "hardness": "lower noisy-input MS-SSIM means the input slice is more degraded before any model is applied",
        "folder": "stratification/by_input_ms_ssim",
        "title": "2.5D gain by noisy-input MS-SSIM quartile",
    },
    "input_mse": {
        "label": "Noisy-input MSE",
        "higher_is_better": False,
        "description": "MSE between the noisy center slice and the clean target",
        "hardness": "higher noisy-input MSE means the input slice is more degraded before any model is applied",
        "folder": "stratification/by_input_mse",
    },
}

QUARTILE_LABELS = [
    "Q1 (Hardest)",
    "Q2 (Medium-Hard)",
    "Q3 (Medium-Easy)",
    "Q4 (Easiest)",
]


def _project_root() -> Path:
    return default_project_root(__file__)


def _default_out_dir(project_root: Path, stratify_metric: str) -> Path:
    folder = STRATIFY_METRICS.get(stratify_metric, {}).get("folder", f"stratification/by_{stratify_metric}")
    return project_root / "experiments" / "summaries" / "mechanism_analysis" / folder


def _metric_info(stratify_metric: str) -> dict[str, Any]:
    return STRATIFY_METRICS.get(
        stratify_metric,
        {
            "label": stratify_metric,
            "higher_is_better": True,
            "description": stratify_metric,
            "hardness": f"lower {stratify_metric} means the sample is harder",
            "folder": f"stratification/by_{stratify_metric}",
        },
    )


def _read_rows(path: Path, stratify_metric: str) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    required = list(REQUIRED_COLUMNS)
    if stratify_metric not in required:
        required.append(stratify_metric)
    missing = [col for col in required if col not in rows[0]]
    if missing:
        hint = ""
        if any(col in OPTIONAL_NUMERIC_COLUMNS for col in missing):
            hint = " Regenerate comparison_metadata.csv with generate_comparison_panels.py to add input-quality columns."
        raise KeyError(f"{path} is missing required column(s): {missing}.{hint}")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed_row = {
            "vol_id": str(row["vol_id"]),
            "slice_t": int(row["slice_t"]),
            "sample_idx": int(row["sample_idx"]),
            "ms_ssim_2d": float(row["ms_ssim_2d"]),
            "ms_ssim_3ch": float(row["ms_ssim_3ch"]),
            "ms_ssim_5ch": float(row["ms_ssim_5ch"]),
            "delta_3ch_vs_2d": float(row["delta_3ch_vs_2d"]),
            "delta_5ch_vs_2d": float(row["delta_5ch_vs_2d"]),
        }
        for col in sorted(set(OPTIONAL_NUMERIC_COLUMNS + [stratify_metric])):
            if col in parsed_row:
                continue
            if col in row and row[col] not in ("", None):
                parsed_row[col] = float(row[col])
        parsed.append(parsed_row)
    return parsed


def _mean(values: list[float]) -> float:
    return float(np.mean(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks for Spearman correlation, handling ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_ranks(x), _ranks(y))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def build_quartile_summary(rows: list[dict[str, Any]], stratify_metric: str) -> list[dict[str, Any]]:
    metric_info = _metric_info(stratify_metric)
    sorted_rows = sorted(
        rows,
        key=lambda r: r[stratify_metric],
        reverse=not bool(metric_info["higher_is_better"]),
    )
    n = len(sorted_rows)
    summary: list[dict[str, Any]] = []

    for idx, label in enumerate(QUARTILE_LABELS):
        start = int(np.floor(idx * n / 4))
        end = int(np.floor((idx + 1) * n / 4))
        group = sorted_rows[start:end]
        if not group:
            continue
        summary.append(
            {
                "quartile": label,
                "stratify_metric": stratify_metric,
                "count": len(group),
                "mean_stratify_metric": _mean([r[stratify_metric] for r in group]),
                "mean_2d": _mean([r["ms_ssim_2d"] for r in group]),
                "mean_3ch": _mean([r["ms_ssim_3ch"] for r in group]),
                "mean_5ch": _mean([r["ms_ssim_5ch"] for r in group]),
                "mean_delta_3ch": _mean([r["delta_3ch_vs_2d"] for r in group]),
                "mean_delta_5ch": _mean([r["delta_5ch_vs_2d"] for r in group]),
                "std_delta_3ch": _std([r["delta_3ch_vs_2d"] for r in group]),
                "std_delta_5ch": _std([r["delta_5ch_vs_2d"] for r in group]),
            }
        )
    return summary


def build_correlation_summary(rows: list[dict[str, Any]], stratify_metric: str) -> list[dict[str, Any]]:
    x = np.array([r[stratify_metric] for r in rows], dtype=float)
    metrics = [
        ("delta_3ch_vs_2d", "3ch gain vs 2D"),
        ("delta_5ch_vs_2d", "5ch gain vs 2D"),
        ("ms_ssim_3ch", "3ch absolute MS-SSIM"),
        ("ms_ssim_5ch", "5ch absolute MS-SSIM"),
    ]
    out: list[dict[str, Any]] = []
    for key, label in metrics:
        y = np.array([r[key] for r in rows], dtype=float)
        out.append(
            {
                "x_metric": stratify_metric,
                "y_metric": key,
                "description": label,
                "n_samples": len(rows),
                "pearson_r": _pearson(x, y),
                "spearman_r": _spearman(x, y),
            }
        )
    return out


def build_volume_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["vol_id"]].append(row)

    out: list[dict[str, Any]] = []
    for vol_id, group in sorted(grouped.items()):
        out.append(
            {
                "vol_id": vol_id,
                "count": len(group),
                "mean_2d": _mean([r["ms_ssim_2d"] for r in group]),
                "mean_delta_3ch": _mean([r["delta_3ch_vs_2d"] for r in group]),
                "mean_delta_5ch": _mean([r["delta_5ch_vs_2d"] for r in group]),
                "max_delta_5ch": max(r["delta_5ch_vs_2d"] for r in group),
                "top_gap_count": 0,
            }
        )

    top_n = min(50, len(rows))
    top_rows = sorted(rows, key=lambda r: r["delta_5ch_vs_2d"], reverse=True)[:top_n]
    top_counts: dict[str, int] = defaultdict(int)
    for row in top_rows:
        top_counts[row["vol_id"]] += 1
    for row in out:
        row["top_gap_count"] = top_counts[row["vol_id"]]

    return sorted(out, key=lambda r: float(r["mean_delta_5ch"]), reverse=True)


def build_top_examples(rows: list[dict[str, Any]], n_examples: int) -> list[dict[str, Any]]:
    top_rows = sorted(rows, key=lambda r: r["delta_5ch_vs_2d"], reverse=True)[:n_examples]
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(top_rows, start=1):
        out.append(
            {
                "rank": rank,
                "vol_id": row["vol_id"],
                "slice_t": row["slice_t"],
                "sample_idx": row["sample_idx"],
                "ms_ssim_2d": row["ms_ssim_2d"],
                "ms_ssim_3ch": row["ms_ssim_3ch"],
                "ms_ssim_5ch": row["ms_ssim_5ch"],
                "delta_3ch_vs_2d": row["delta_3ch_vs_2d"],
                "delta_5ch_vs_2d": row["delta_5ch_vs_2d"],
            }
        )
    return out


def plot_quartile_gain(rows: list[dict[str, Any]], stratify_metric: str, out_path: Path) -> None:
    metric_info = _metric_info(stratify_metric)
    labels = ["Q1\nHardest", "Q2\nMed-hard", "Q3\nMed-easy", "Q4\nEasiest"][: len(rows)]
    x = np.arange(len(rows)) * 1.32
    width = 0.46
    gain_3ch = [float(r["mean_delta_3ch"]) for r in rows]
    gain_5ch = [float(r["mean_delta_5ch"]) for r in rows]

    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    bars_3 = ax.bar(x - width / 2, gain_3ch, width, label="2.5D-3ch − 2D", color="#008C95", alpha=0.92)
    bars_5 = ax.bar(x + width / 2, gain_5ch, width, label="2.5D-5ch − 2D", color="#D48210", alpha=0.92)
    ax.axhline(0, color="#333333", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.set_ylabel("Mean MS-SSIM gain over 2D", fontsize=17, fontweight="bold")
    ax.set_title(
        metric_info.get("title", f"2.5D gain by {metric_info['label']} quartile"),
        fontsize=21,
        fontweight="bold",
        pad=18,
    )
    ax.grid(True, axis="y", alpha=0.25, linewidth=1.0)
    ax.legend(frameon=False, prop={"weight": "bold", "size": 15}, loc="upper right")

    y_max = max(gain_3ch + gain_5ch) if rows else 0.0
    ax.set_ylim(0.0, max(0.02, y_max * 1.32))
    ax.margins(x=0.08)
    for bars in (bars_3, bars_5):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(0.0015, y_max * 0.035),
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_delta_scatter(
    rows: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    stratify_metric: str,
    out_path: Path,
) -> None:
    metric_info = _metric_info(stratify_metric)
    x = np.array([r[stratify_metric] for r in rows], dtype=float)
    y3 = np.array([r["delta_3ch_vs_2d"] for r in rows], dtype=float)
    y5 = np.array([r["delta_5ch_vs_2d"] for r in rows], dtype=float)
    corr_by_metric = {r["y_metric"]: r for r in correlations}

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(x, y3, s=13, alpha=0.35, color="#008C95", label="2.5D-3ch − 2D")
    ax.scatter(x, y5, s=13, alpha=0.35, color="#D48210", label="2.5D-5ch − 2D")
    for y, color in [(y3, "#008C95"), (y5, "#D48210")]:
        slope, intercept = np.polyfit(x, y, deg=1)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=2.0)

    r3 = corr_by_metric["delta_3ch_vs_2d"]["pearson_r"]
    r5 = corr_by_metric["delta_5ch_vs_2d"]["pearson_r"]
    ax.text(
        0.03,
        0.94,
        f"Pearson r: 2.5D-3ch={r3:.3f}, 2.5D-5ch={r5:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.9},
    )
    ax.axhline(0, color="#333333", linewidth=0.8)
    hardness_side = "left" if bool(metric_info["higher_is_better"]) else "right"
    ax.set_xlabel(f"{metric_info['label']} (harder samples on the {hardness_side})")
    ax.set_ylabel("MS-SSIM gain over 2D")
    ax.set_title(f"Sample-level gains versus {metric_info['label']}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_volume_delta(rows: list[dict[str, Any]], out_path: Path) -> None:
    # Keep every volume in the plot. There are only five test volumes in the current split.
    sorted_rows = sorted(rows, key=lambda r: str(r["vol_id"]))
    labels = [str(r["vol_id"]) for r in sorted_rows]
    x = np.arange(len(sorted_rows))
    gain_3ch = [float(r["mean_delta_3ch"]) for r in sorted_rows]
    gain_5ch = [float(r["mean_delta_5ch"]) for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.plot(x, gain_3ch, marker="o", linewidth=2.2, color="#008C95", label="2.5D-3ch − 2D")
    ax.plot(x, gain_5ch, marker="o", linewidth=2.2, color="#D48210", label="2.5D-5ch − 2D")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Mean MS-SSIM gain over 2D")
    ax.set_xlabel("Test volume ID")
    ax.set_title("Volume-level check: gains should not depend on one sample only")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _gain_hardness_interpretation(stratify_metric: str, pearson_r: float) -> str:
    higher_is_better = bool(_metric_info(stratify_metric)["higher_is_better"])
    if np.isnan(pearson_r):
        return "not interpretable"
    harder_when_metric_decreases = higher_is_better
    larger_on_harder = pearson_r < 0 if harder_when_metric_decreases else pearson_r > 0
    return "larger gains on harder samples" if larger_on_harder else "larger gains on easier samples"


def write_report(
    out_path: Path,
    n_samples: int,
    stratify_metric: str,
    quartiles: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    volumes: list[dict[str, Any]],
    top_examples: list[dict[str, Any]],
) -> None:
    metric_info = _metric_info(stratify_metric)
    corr_lookup = {row["y_metric"]: row for row in correlations}
    q1 = quartiles[0]
    q4 = quartiles[-1]
    gain_ratio = float(q1["mean_delta_5ch"]) / max(float(q4["mean_delta_5ch"]), 1e-12)
    gain_trend = _gain_hardness_interpretation(
        stratify_metric,
        float(corr_lookup["delta_5ch_vs_2d"]["pearson_r"]),
    )
    include_2d_column = stratify_metric != "ms_ssim_2d"

    lines = [
        "# Mechanism Analysis: When 2.5D Context Helps",
        "",
        "> Generated by `analyze_mechanism.py` from `comparison_metadata.csv`.",
        "",
        "## Question",
        "",
        "The main experiment shows that 5ch performs best on average. This analysis asks whether that gain is uniform, "
        "or whether neighboring slices help more in particular kinds of test samples.",
        "",
        "## Difficulty Stratification",
        "",
        f"The analysis uses {n_samples} shared test samples and stratifies them by {metric_info['description']}. "
        f"In this stratification, {metric_info['hardness']}. Q1 contains the hardest samples and Q4 the easiest samples.",
        "",
    ]
    header = ["Quartile", "Count", str(metric_info["label"])]
    aligns = ["---", "---:", "---:"]
    if include_2d_column:
        header.append("2D MS-SSIM")
        aligns.append("---:")
    header += ["3ch gain", "5ch gain"]
    aligns += ["---:", "---:"]
    lines += [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for row in quartiles:
        values = [
            str(row["quartile"]),
            str(row["count"]),
            _fmt(float(row["mean_stratify_metric"])),
        ]
        if include_2d_column:
            values.append(_fmt(float(row["mean_2d"])))
        values += [
            _fmt(float(row["mean_delta_3ch"])),
            _fmt(float(row["mean_delta_5ch"])),
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines += [
        "",
        "## Correlations",
        "",
        "| Relationship | Pearson r | Spearman r | Interpretation |",
        "| --- | ---: | ---: | --- |",
    ]
    for key in ("delta_3ch_vs_2d", "delta_5ch_vs_2d"):
        row = corr_lookup[key]
        sign = _gain_hardness_interpretation(stratify_metric, float(row["pearson_r"]))
        lines.append(
            f"| {metric_info['label']} vs {row['description']} | "
            f"{_fmt(float(row['pearson_r']))} | {_fmt(float(row['spearman_r']))} | {sign} |"
        )

    lines += [
        "",
        "## Volume-Level Check",
        "",
        "The volume summary checks whether the effect is concentrated in a single test volume. "
        "A broad positive gain across volumes is stronger evidence than one isolated volume.",
        "",
        "| Volume | Samples | Mean 2D | Mean 5ch gain | Top-50 gap samples |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(volumes, key=lambda r: str(r["vol_id"])):
        lines.append(
            f"| {row['vol_id']} | {row['count']} | {_fmt(float(row['mean_2d']))} | "
            f"{_fmt(float(row['mean_delta_5ch']))} | {row['top_gap_count']} |"
        )

    lines += [
        "",
        "## Largest-Gain Examples",
        "",
        "| Rank | Volume | Slice | 2D | 3ch | 5ch | 5ch gain |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_examples[:8]:
        lines.append(
            f"| {row['rank']} | {row['vol_id']} | {row['slice_t']} | "
            f"{_fmt(float(row['ms_ssim_2d']))} | {_fmt(float(row['ms_ssim_3ch']))} | "
            f"{_fmt(float(row['ms_ssim_5ch']))} | {_fmt(float(row['delta_5ch_vs_2d']))} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        f"- The 5ch gain is {_fmt(float(q1['mean_delta_5ch']))} in Q1 and "
        f"{_fmt(float(q4['mean_delta_5ch']))} in Q4, a Q1/Q4 ratio of about {gain_ratio:.1f}.",
        f"- The Pearson correlation between {metric_info['label']} and 5ch gain is "
        f"{_fmt(float(corr_lookup['delta_5ch_vs_2d']['pearson_r']))}, corresponding to {gain_trend}.",
        "- For `ms_ssim_2d`, this is a baseline-conditioned mechanism diagnostic, not a model-independent noisiness split.",
        "- For `input_psnr`, `input_ms_ssim`, or `input_mse`, this is a model-independent input-quality check because the bins are defined before any denoising model is applied.",
        "- This is metric-level evidence, not proof of a specific attention mechanism inside DINOv3.",
        "",
        "## Generated Figures",
        "",
        f"- `difficulty_gain_by_quartile.png`: mean 3ch/5ch gain by {metric_info['label']} quartile.",
        f"- `difficulty_delta_scatter.png`: per-sample gain versus {metric_info['label']}.",
        "- `difficulty_volume_delta.png`: mean gain by held-out test volume.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 2.5D mechanism from comparison metadata.")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--stratify-metric",
        default="ms_ssim_2d",
        help=(
            "Metadata column used to define Q1-Q4 hardness bins. "
            "Use ms_ssim_2d for the baseline-conditioned mechanism diagnostic, "
            "or input_psnr/input_ms_ssim/input_mse for model-independent input-quality bins."
        ),
    )
    parser.add_argument("--top-examples", type=int, default=20)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    stratify_metric = args.stratify_metric
    metadata_path = (
        args.metadata.resolve()
        if args.metadata is not None
        else project_root / "experiments" / "summaries" / "comparison_panels" / "comparison_metadata.csv"
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir is not None
        else _default_out_dir(project_root, stratify_metric)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Metadata: {metadata_path}")
    print(f"Stratify metric: {stratify_metric}")
    print(f"Output dir: {out_dir}")

    rows = _read_rows(metadata_path, stratify_metric)
    quartiles = build_quartile_summary(rows, stratify_metric)
    correlations = build_correlation_summary(rows, stratify_metric)
    volumes = build_volume_summary(rows)
    top_examples = build_top_examples(rows, args.top_examples)

    _write_csv(out_dir / "difficulty_stratification_summary.csv", quartiles)
    _write_csv(out_dir / "difficulty_correlation_summary.csv", correlations)
    _write_csv(out_dir / "difficulty_volume_summary.csv", volumes)
    _write_csv(out_dir / "difficulty_top_examples.csv", top_examples)

    plot_quartile_gain(quartiles, stratify_metric, out_dir / "difficulty_gain_by_quartile.png")
    plot_delta_scatter(rows, correlations, stratify_metric, out_dir / "difficulty_delta_scatter.png")
    plot_volume_delta(volumes, out_dir / "difficulty_volume_delta.png")

    write_report(
        out_dir / "mechanism_analysis_report.md",
        len(rows),
        stratify_metric,
        quartiles,
        correlations,
        volumes,
        top_examples,
    )
    print("Mechanism analysis complete.")


if __name__ == "__main__":
    main()
