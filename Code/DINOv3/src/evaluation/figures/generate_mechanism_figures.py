"""Generate thesis/poster figures from mechanism-analysis CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root
from statistics import mean, stdev
from typing import Any

import numpy as np


VARIANT_ORDER = ["2D", "3ch", "5ch"]
COLORS = {"2D": "#4C78A8", "3ch": "#F58518", "5ch": "#54A24B"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


def _summary_stats(values: list[float]) -> tuple[int, float, float]:
    clean = [v for v in values if np.isfinite(v)]
    if not clean:
        return 0, float("nan"), float("nan")
    if len(clean) == 1:
        return 1, clean[0], 0.0
    return len(clean), mean(clean), stdev(clean)


def _plot_metric_boxplots(metric_rows: list[dict[str, str]], out_path: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    metrics = [
        ("clean_similarity", "Similarity to clean tokens", "higher is better"),
        ("svd_entropy", "SVD entropy", "lower can indicate more structured tokens"),
        ("autocorr_anisotropy", "Horizontal - vertical autocorr", "larger = more directional"),
    ]
    summary_rows: list[dict[str, Any]] = []

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    for ax, (metric, title, subtitle) in zip(axes, metrics):
        grouped = []
        for variant in VARIANT_ORDER:
            vals = [
                _to_float(row[metric])
                for row in metric_rows
                if row.get("variant_key") == variant
            ]
            vals = [v for v in vals if np.isfinite(v)]
            grouped.append(vals)
            n, mu, sd = _summary_stats(vals)
            summary_rows.append({
                "metric": metric,
                "variant_key": variant,
                "n": n,
                "mean": mu,
                "std": sd,
            })

        try:
            box = ax.boxplot(grouped, tick_labels=VARIANT_ORDER, patch_artist=True, showmeans=True)
        except TypeError:  # matplotlib < 3.9
            box = ax.boxplot(grouped, labels=VARIANT_ORDER, patch_artist=True, showmeans=True)
        for patch, variant in zip(box["boxes"], VARIANT_ORDER):
            patch.set_facecolor(COLORS[variant])
            patch.set_alpha(0.65)
        ax.set_title(title)
        ax.set_xlabel(subtitle)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("DINOv3 patch-token representation diagnostics", fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return summary_rows


def _plot_autocorr_decay(decay_rows: list[dict[str, str]], out_path: Path) -> list[dict[str, Any]]:
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in decay_rows:
        key = (row["variant_key"], row["axis"], int(row["offset"]))
        grouped[key].append(_to_float(row["cosine"]))

    summary_rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in VARIANT_ORDER:
        for axis, linestyle in [("horizontal", "-"), ("vertical", "--")]:
            offsets = sorted({
                offset
                for (v, a, offset) in grouped
                if v == variant and a == axis
            })
            means = []
            stds = []
            for offset in offsets:
                values = [v for v in grouped[(variant, axis, offset)] if np.isfinite(v)]
                n, mu, sd = _summary_stats(values)
                means.append(mu)
                stds.append(sd)
                summary_rows.append({
                    "variant_key": variant,
                    "axis": axis,
                    "offset": offset,
                    "n": n,
                    "mean_cosine": mu,
                    "std_cosine": sd,
                })
            if offsets:
                label = f"{variant} {axis}"
                ax.plot(offsets, means, marker="o", linestyle=linestyle, color=COLORS[variant], label=label)
                lower = np.asarray(means) - np.asarray(stds)
                upper = np.asarray(means) + np.asarray(stds)
                ax.fill_between(offsets, lower, upper, color=COLORS[variant], alpha=0.12)

    ax.set_title("Spatial autocorrelation decay of patch tokens")
    ax.set_xlabel("Patch offset")
    ax.set_ylabel("Mean cosine similarity")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return summary_rows


def _plot_attention_entropy(attn_rows: list[dict[str, str]], out_path: Path) -> None:
    if not attn_rows:
        return
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in attn_rows:
        grouped[(row["query_label"], row["variant_key"])].append(_to_float(row["mean_entropy_bits"]))
    queries = sorted({q for q, _ in grouped})
    if not queries:
        return

    width = 0.25
    x = np.arange(len(queries))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, variant in enumerate(VARIANT_ORDER):
        vals = []
        for query in queries:
            values = [v for v in grouped[(query, variant)] if np.isfinite(v)]
            vals.append(mean(values) if values else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=variant, color=COLORS[variant], alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(queries)
    ax.set_ylabel("Attention entropy, bits")
    ax.set_title("Attention spread by query type")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    by_metric = defaultdict(list)
    for row in summary_rows:
        by_metric[row["metric"]].append(row)
    lines = [
        "# Mechanism Representation Summary",
        "",
        "Generated from `representation_metrics.csv`.",
        "",
    ]
    for metric, rows in by_metric.items():
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Variant | n | mean | std |")
        lines.append("|---|---:|---:|---:|")
        for variant in VARIANT_ORDER:
            row = next((r for r in rows if r["variant_key"] == variant), None)
            if row is None:
                continue
            lines.append(
                f"| {variant} | {row['n']} | {float(row['mean']):.6f} | {float(row['std']):.6f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--representation-dir", type=Path, default=None)
    parser.add_argument("--attention-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    mechanism_root = project_root / "experiments" / "summaries" / "mechanism_analysis"
    representation_dir = args.representation_dir or (mechanism_root / "representation_analysis")
    attention_dir = args.attention_dir or (mechanism_root / "attention_maps")
    out_dir = args.out_dir or (mechanism_root / "mechanism_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = _read_csv(representation_dir / "representation_metrics.csv")
    decay_rows = _read_csv(representation_dir / "representation_autocorr_decay.csv")
    attn_rows = _read_csv(attention_dir / "attention_summary.csv")

    if not metric_rows:
        raise FileNotFoundError(f"Missing representation metrics: {representation_dir / 'representation_metrics.csv'}")
    if not decay_rows:
        raise FileNotFoundError(
            f"Missing autocorrelation metrics: {representation_dir / 'representation_autocorr_decay.csv'}"
        )

    summary_rows = _plot_metric_boxplots(metric_rows, out_dir / "representation_metric_boxplots.png")
    decay_summary = _plot_autocorr_decay(decay_rows, out_dir / "representation_autocorr_decay.png")
    _plot_attention_entropy(attn_rows, out_dir / "attention_entropy.png")
    _write_csv(out_dir / "representation_metric_summary.csv", summary_rows)
    _write_csv(out_dir / "representation_autocorr_summary.csv", decay_summary)
    _write_markdown_summary(out_dir / "mechanism_figure_summary.md", summary_rows)


if __name__ == "__main__":
    main()
