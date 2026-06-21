"""Plot trained mechanism control results against aligned baselines.

Reads trained_controls_summary.csv and produces a grouped bar chart.

Usage:
    python evaluation/plot_mechanism_controls.py [--project-root PATH]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Display order and labels
VARIANTS = [
    ("2D",                  "2D\n(baseline)",              "baseline"),
    ("3ch",                 "3ch\n(aligned train)",        "baseline"),
    ("3ch_shuffled",        "3ch\n(shuffled train)",       "control"),
    ("5ch",                 "5ch\n(aligned train)",        "baseline"),
    ("5ch_repeated_center", "5ch\n(repeated center)",      "control"),
    ("5ch_shuffled",        "5ch\n(shuffled train)",       "control"),
]

COLORS = {
    "baseline": "#2196F3",   # blue
    "control":  "#FF9800",   # orange
}


def _load_summary(csv_path: Path) -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = row["variant_key"]
            data[key] = {
                "ms_ssim_mean": float(row["ms_ssim_mean"]),
                "ms_ssim_std":  float(row["ms_ssim_std"]),
                "mse_mean":     float(row["mse_mean"]),
                "mse_std":      float(row["mse_std"]),
                "psnr_mean":    float(row["psnr_mean"]),
                "psnr_std":     float(row["psnr_std"]),
                "n_complete":   float(row["n_complete"]),
            }
    return data


def _make_bar_chart(data: dict[str, dict[str, float]], out_path: Path) -> None:
    keys    = [v[0] for v in VARIANTS]
    labels  = [v[1] for v in VARIANTS]
    kinds   = [v[2] for v in VARIANTS]

    means = [data[k]["ms_ssim_mean"] if k in data else float("nan") for k in keys]
    stds  = [data[k]["ms_ssim_std"]  if k in data else 0.0           for k in keys]
    colors = [COLORS[kd] for kd in kinds]

    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 1.5, "ecolor": "#444"})

    # Reference line at 2D baseline
    two_d_mean = data.get("2D", {}).get("ms_ssim_mean", 0.8079)
    ax.axhline(two_d_mean, color="#555", linestyle="--", linewidth=1.2,
               label=f"2D baseline ({two_d_mean:.4f})")

    # Value labels on top of bars
    for bar, mean, std in zip(bars, means, stds):
        if not np.isnan(mean):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.003,
                f"{mean:.4f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Test MS-SSIM (mean ± std, 3 seeds)", fontsize=10)
    ax.set_title("Trained Mechanism Controls vs Aligned Baselines", fontsize=11, fontweight="bold")
    ax.set_ylim(0.72, 0.92)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=COLORS["baseline"], label="Aligned training (main result)"),
        mpatches.Patch(color=COLORS["control"],  label="Mechanism control"),
        mlines.Line2D([0], [0], color="#555", linestyle="--", linewidth=1.2, label=f"2D baseline ({two_d_mean:.4f})"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _print_table(data: dict[str, dict[str, float]]) -> None:
    two_d = data.get("2D", {}).get("ms_ssim_mean", 0.8079)
    print("\n=== Trained Mechanism Controls: MS-SSIM Summary ===\n")
    print(f"{'Variant':<24} {'MS-SSIM':>10}  {'±std':>7}  {'vs 2D':>8}  {'n'}")
    print("-" * 62)
    for key, label, _ in VARIANTS:
        if key not in data:
            print(f"{label:<24} {'—':>10}")
            continue
        d = data[key]
        mean = d["ms_ssim_mean"]
        std  = d["ms_ssim_std"]
        delta = mean - two_d
        n = int(d["n_complete"])
        print(f"{key:<24} {mean:>10.4f}  {std:>7.4f}  {delta:>+8.4f}  {n}/3")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=default_project_root(__file__))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    summary_dir  = project_root / "experiments" / "summaries" / "mechanism_analysis" / "trained_controls"
    csv_path     = summary_dir / "trained_controls_summary.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run summarize_mechanism_controls.py first.")
        sys.exit(1)

    data = _load_summary(csv_path)
    _print_table(data)
    _make_bar_chart(data, summary_dir / "trained_controls_bar.png")


if __name__ == "__main__":
    main()
