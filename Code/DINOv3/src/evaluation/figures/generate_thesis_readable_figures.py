"""Generate large-text thesis figure exports from cached summaries.

This script is intentionally lightweight: it redraws the thesis-facing figures
from existing PNG caches and CSV summaries without running model inference.

Outputs are written both to their summary folders and to
``Deliverables/Thesis/draftv9/figures`` by default.

Usage:
    cd C:/UNI/Y3/RP/Code/DINOv3/src
    python -m evaluation.figures.generate_thesis_readable_figures --project-root C:/UNI/Y3/RP
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from evaluation.figures.plot_f3_comparison_grid_with_median import (
    build_cache as build_f3_cache,
    load_cache as load_f3_cache,
    assemble_figure as assemble_f3_figure,
)


QUALITATIVE_DEFAULT_MS = {
    "2D": "0.8956",
    "3ch": "0.9276",
    "5ch": "0.9317",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _copy_to(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"Copied {src} -> {dest}")


def _runs(active: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                out.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_len:
        out.append((start, len(active)))
    return out


def _extract_image_runs(img: np.ndarray, expected_rows: int = 2) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract the image tiles from a cached 2-row qualitative panel.

    The source panel already contains small text. We isolate only the red/blue
    seismic image regions and redraw all labels at thesis-readable size.
    """
    spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    mask = (spread > 25) & (img.min(axis=2) < 245)

    row_active = mask.sum(axis=1) > max(20, img.shape[1] * 0.01)
    row_runs = _runs(row_active, min_len=120)
    if len(row_runs) < expected_rows:
        raise RuntimeError("Could not identify the two image rows in the qualitative source panel.")
    row_runs = sorted(row_runs, key=lambda r: r[0])[:expected_rows]

    row_crops: list[list[np.ndarray]] = []
    for y0, y1 in row_runs:
        row_mask = mask[y0:y1, :]
        col_active = row_mask.sum(axis=0) > max(20, (y1 - y0) * 0.04)
        col_runs = _runs(col_active, min_len=120)
        if not col_runs:
            raise RuntimeError("Could not identify image columns in the qualitative source panel.")
        row_crops.append([img[y0:y1, x0:x1, :] for x0, x1 in col_runs])

    top = row_crops[0]
    bottom = row_crops[1]
    if len(top) < 5 or len(bottom) < 3:
        raise RuntimeError(
            f"Expected at least 5 top-row and 3 residual tiles, got {len(top)} and {len(bottom)}."
        )
    return top[:5], bottom[:3]


def plot_qualitative(source_path: Path, out_path: Path, ms_values: dict[str, str]) -> None:
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    img = np.array(Image.open(source_path).convert("RGB"))
    top_tiles, residual_tiles = _extract_image_runs(img)
    H_top = top_tiles[0].shape[0]
    W_res = residual_tiles[0].shape[1]

    top_labels = [
        "Noisy\ninput",
        f"2D\nMS-SSIM\n{ms_values['2D']}",
        f"3ch\nMS-SSIM\n{ms_values['3ch']}",
        f"5ch\nMS-SSIM\n{ms_values['5ch']}",
        "Clean\ntarget",
    ]
    residual_labels = ["Residual 2D", "Residual 3ch", "Residual 5ch"]

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(21.0, 9.6),
        gridspec_kw={"height_ratios": [1.0, 1.0], "wspace": 0.035, "hspace": 0.18},
    )
    fig.suptitle(
        "Denoising Results on the Image Impeccable Dataset",
        fontsize=28,
        fontweight="bold",
        y=0.985,
    )

    for i, (ax, tile, label) in enumerate(zip(axes[0], top_tiles, top_labels)):
        ax.imshow(tile, aspect="auto")
        ax.set_title(label, fontsize=21, fontweight="bold", pad=9)
        if i == 0:
            # The cached Image Impeccable panel comes from vol[t, :, :] after
            # canonicalization; the visible crop axes are the two in-plane
            # spatial dimensions.
            ax.set_yticks([0, H_top // 2, H_top - 1])
            ax.set_yticklabels(["0", "112", "223"], fontsize=13, fontweight="bold")
            ax.set_xticks([])
            for spine in ("top", "right", "bottom"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="y", length=3, pad=2)
            ax.set_ylabel("Inline position", fontsize=15, fontweight="bold")
        else:
            ax.axis("off")

    axes[1, 0].axis("off")
    axes[1, 4].axis("off")
    for j, (ax, tile, label) in enumerate(zip(axes[1, 1:4], residual_tiles, residual_labels)):
        ax.imshow(tile, aspect="auto")
        ax.set_title(label, fontsize=22, fontweight="bold", pad=10)
        if j == 0:
            # x-axis ticks on the first residual panel only (224-position crop).
            ax.set_xticks([0, W_res // 2, W_res - 1])
            ax.set_xticklabels(["0", "112", "223"], fontsize=13, fontweight="bold")
            ax.set_yticks([])
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.tick_params(axis="x", length=3, pad=2)
            ax.set_xlabel("Crossline position", fontsize=15, fontweight="bold")
        else:
            ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.06, top=0.76, wspace=0.035, hspace=0.45)

    # Single shared colorbar spanning both rows (amplitude and residual use the same ±2 scale)
    pos_right = axes[0, 4].get_position()
    pos_res   = axes[1, 3].get_position()
    cbar_left = pos_right.x1 + 0.015
    cbar_w    = 0.018
    cbar_bot  = pos_res.y0
    cbar_h    = (pos_right.y0 + pos_right.height) - pos_res.y0

    shared_cax = fig.add_axes((cbar_left, cbar_bot, cbar_w, cbar_h))
    shared_sm = ScalarMappable(cmap="seismic", norm=Normalize(vmin=-2, vmax=2))
    shared_sm.set_array([])
    cb = fig.colorbar(shared_sm, cax=shared_cax)
    cb.set_label("Amplitude\n(z-score, ±2σ)", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=12)

    fig.savefig(out_path, dpi=190, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_difficulty(csv_path: Path, out_path: Path) -> None:
    rows = _read_csv(csv_path)
    labels = ["Q1\nHardest", "Q2\nMed-hard", "Q3\nMed-easy", "Q4\nEasiest"][: len(rows)]
    gain_3ch = [float(row["mean_delta_3ch"]) for row in rows]
    gain_5ch = [float(row["mean_delta_5ch"]) for row in rows]

    x = np.arange(len(rows)) * 1.32
    bar_width = 0.30
    bar_offset = 0.23
    y_max = max(gain_3ch + gain_5ch)

    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    bars_3 = ax.bar(x - bar_offset, gain_3ch, bar_width, color="#008C95", label="2.5D-3ch - 2D")
    bars_5 = ax.bar(x + bar_offset, gain_5ch, bar_width, color="#D48210", label="2.5D-5ch - 2D")
    ax.axhline(0, color="#333333", linewidth=1.2)

    ax.set_title("2.5D gain by noisy-input quartile", fontsize=21, fontweight="bold", pad=15)
    ax.set_ylabel("Mean MS-SSIM gain over 2D", fontsize=17, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, axis="y", alpha=0.25, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, y_max * 1.34)
    ax.margins(x=0.08)
    ax.legend(loc="upper right", frameon=False, prop={"weight": "bold", "size": 15})

    for bars in (bars_3, bars_5):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_max * 0.035,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=15,
                fontweight="bold",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out_path}")


def _summary_by_variant(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    return {row["variant_key"]: row for row in rows}


def plot_trained_controls(csv_path: Path, out_path: Path) -> None:
    rows = _summary_by_variant(_read_csv(csv_path))
    order = ["2D", "3ch", "3ch_shuffled", "5ch", "5ch_repeated_center", "5ch_shuffled"]
    labels = [
        "2D-1ch",
        "2.5D-3ch",
        "2.5D-3ch\nshuffled",
        "2.5D-5ch",
        "2.5D-5ch\nrepeated center",
        "2.5D-5ch\nshuffled",
    ]
    roles = {
        "2D": "baseline",
        "3ch": "baseline",
        "3ch_shuffled": "control",
        "5ch": "baseline",
        "5ch_repeated_center": "control",
        "5ch_shuffled": "control",
    }
    colors = {"baseline": "#2F6FB0", "control": "#D9782D"}

    means = [float(rows[key]["ms_ssim_mean"]) for key in order]
    stds = [float(rows[key]["ms_ssim_std"]) for key in order]
    x = np.arange(len(order))
    y_range = max(m + s for m, s in zip(means, stds)) - min(m - s for m, s in zip(means, stds))

    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=6,
        color=[colors[roles[key]] for key in order],
        edgecolor="white",
        linewidth=1.0,
        error_kw={"elinewidth": 2.0, "ecolor": "#333333", "capthick": 2.0},
    )

    two_d = means[0]
    ax.axhline(two_d, color="#444444", linestyle="--", linewidth=1.8)
    ax.set_title("Trained controls require aligned context", fontsize=21, fontweight="bold", pad=14)
    ax.set_ylabel("Test MS-SSIM", fontsize=17, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15, fontweight="bold")
    ax.tick_params(axis="y", labelsize=15)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")
    ax.grid(True, axis="y", alpha=0.25, linewidth=1.0)
    ax.set_axisbelow(True)
    y_bottom = min(m - s for m, s in zip(means, stds)) - y_range * 0.15
    y_top = max(m + s for m, s in zip(means, stds)) + y_range * 0.40
    ax.set_ylim(y_bottom, y_top)
    ax.margins(x=0.08)

    for bar, mean, std in zip(bars, means, stds):
        text_y = mean + std + y_range * 0.03
        if text_y < two_d + y_range * 0.05:
            text_y = two_d + y_range * 0.05
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            text_y,
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    from matplotlib import lines as mlines
    from matplotlib import patches as mpatches

    handles = [
        mpatches.Patch(color=colors["baseline"], label="Aligned-training baseline"),
        mpatches.Patch(color=colors["control"], label="Trained control"),
        mlines.Line2D([0], [0], color="#444444", linestyle="--", linewidth=1.8, label="2D-1ch baseline"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        prop={"weight": "bold", "size": 13},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_f3_grid(project_root: Path, out_path: Path, inline_idx: int = 419) -> None:
    summaries = project_root / "experiments" / "summaries"
    rob_root = project_root / "experiments" / "runs" / "robustness" / "f3"
    f3_npy = project_root / "Code" / "Dataset" / "F3" / "processed" / "f3_original.npy"
    filt_npy = project_root / "Code" / "Dataset" / "F3" / "processed" / "f3_filtered_ref.npy"

    crops = load_f3_cache(summaries, inline_idx)
    if crops is None:
        build_f3_cache(rob_root, f3_npy, filt_npy, summaries, inline_idx)
        crops = load_f3_cache(summaries, inline_idx)
    if crops is None:
        raise RuntimeError("Could not load or build the F3 crop cache.")
    assemble_f3_figure(crops, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--thesis-fig-dir", type=Path, default=None)
    parser.add_argument("--qualitative-source", type=Path, default=None)
    parser.add_argument("--ms-2d", default=QUALITATIVE_DEFAULT_MS["2D"])
    parser.add_argument("--ms-3ch", default=QUALITATIVE_DEFAULT_MS["3ch"])
    parser.add_argument("--ms-5ch", default=QUALITATIVE_DEFAULT_MS["5ch"])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    summaries = project_root / "experiments" / "summaries"
    thesis_fig_dir = (
        args.thesis_fig_dir.resolve()
        if args.thesis_fig_dir is not None
        else project_root / "Deliverables" / "Thesis" / "draftv9" / "figures"
    )
    thesis_fig_dir.mkdir(parents=True, exist_ok=True)

    thesis_summary_dir = summaries / "thesis_figures"
    qualitative_source = (
        args.qualitative_source.resolve()
        if args.qualitative_source is not None
        else thesis_summary_dir / "sources" / "qualitative_median_5ch_source.png"
    )
    if not qualitative_source.exists():
        fallback = summaries / "comparison_panels" / "by_median" / "panel_00.png"
        print(f"Qualitative source missing: {qualitative_source}")
        print(f"Falling back to current median panel: {fallback}")
        qualitative_source = fallback

    ms_values = {"2D": args.ms_2d, "3ch": args.ms_3ch, "5ch": args.ms_5ch}

    qualitative_out = thesis_summary_dir / "qualitative_median_5ch.png"
    plot_qualitative(qualitative_source, qualitative_out, ms_values)
    _copy_to(qualitative_out, thesis_fig_dir / "qualitative_median_5ch.png")

    difficulty_dir = (
        summaries
        / "mechanism_analysis"
        / "stratification_multidata_nofilter"
        / "pooled"
        / "by_input_ms_ssim"
    )
    difficulty_out = difficulty_dir / "difficulty_gain_by_quartile.png"
    plot_difficulty(difficulty_dir / "difficulty_stratification_summary.csv", difficulty_out)
    _copy_to(difficulty_out, thesis_fig_dir / "difficulty_gain_by_quartile.png")

    controls_dir = summaries / "mechanism_analysis" / "trained_controls_multidata"
    controls_out = controls_dir / "trained_controls_multidata_bar.png"
    plot_trained_controls(controls_dir / "trained_controls_multidata_summary.csv", controls_out)
    _copy_to(controls_out, thesis_fig_dir / "trained_controls_bar.png")

    f3_out = summaries / "f3_robustness" / "f3_comparison_grid.png"
    plot_f3_grid(project_root, f3_out)
    _copy_to(f3_out, thesis_fig_dir / "f3_comparison_grid.png")

    print(f"\nReadable thesis figures exported to: {thesis_fig_dir}")


if __name__ == "__main__":
    main()
