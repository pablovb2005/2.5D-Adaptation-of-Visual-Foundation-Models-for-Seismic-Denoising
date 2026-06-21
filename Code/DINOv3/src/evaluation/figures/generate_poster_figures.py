"""Generate poster-optimized result figures from existing experiment summaries.

The regular summary plots are meant for inspection and reports. These figures are
more compact, with larger labels and fewer panels, for the midterm poster.
"""

from __future__ import annotations

import argparse
import csv
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

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("Pillow is required. Install it with: pip install Pillow") from exc


COLORS = {
    "2D": "#62727A",
    "3ch": "#008C95",
    "5ch": "#D48210",
}


def _project_root() -> Path:
    return default_project_root(__file__)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _variant_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"2D": 0, "3ch": 1, "5ch": 2}
    return sorted(rows, key=lambda r: order.get(str(r.get("variant_key")), 99))


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.22)
    ax.tick_params(axis="both", labelsize=11)


def _annotate_bars(ax, bars, values: list[float], fmt: str = "{:.3f}") -> None:
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * 0.018
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )


def poster_main_result(summary_dir: Path, out_path: Path) -> None:
    rows = _variant_order(_read_csv(summary_dir / "main_replicate_summary.csv"))
    labels = [r["variant_key"] for r in rows]
    colors = [COLORS[l] for l in labels]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    specs = [
        ("test_ms_ssim", "MS-SSIM", "higher is better", (0.78, 0.875), "{:.3f}"),
        ("test_psnr", "PSNR (dB)", "higher is better", (22.0, 24.0), "{:.2f}"),
    ]
    for ax, (metric, title, subtitle, ylim, fmt) in zip(axes, specs):
        means = [_f(r, f"{metric}_mean") for r in rows]
        stds = [_f(r, f"{metric}_std") for r in rows]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.92)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_title(f"{title}\n{subtitle}", fontsize=14, fontweight="bold")
        _style_axes(ax)
        _annotate_bars(ax, bars, means, fmt)

    ms_gain = _f(rows[2], "test_ms_ssim_mean") - _f(rows[0], "test_ms_ssim_mean")
    psnr_gain = _f(rows[2], "test_psnr_mean") - _f(rows[0], "test_psnr_mean")
    fig.suptitle(
        f"Main study: 5ch improves over 2D by +{ms_gain:.3f} MS-SSIM and +{psnr_gain:.2f} dB",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def poster_data_efficiency(data_efficiency_dir: Path, main_summary_dir: Path, out_path: Path) -> None:
    rows = _read_csv(data_efficiency_dir / "data_efficiency_summary.csv")
    main_rows = {r["variant_key"]: r for r in _read_csv(main_summary_dir / "main_comparison.csv")
                 if r.get("training_seed") == "42"}
    by_variant: dict[str, list[tuple[int, float]]] = {}
    for r in rows:
        by_variant.setdefault(r["variant_key"], []).append((int(r["n_vols"]), _f(r, "test_ms_ssim")))
    for variant, row in main_rows.items():
        by_variant.setdefault(variant, []).append((20, _f(row, "test_ms_ssim")))

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for variant in ("2D", "3ch", "5ch"):
        points = sorted(set(by_variant.get(variant, [])))
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.6,
            label=variant,
            color=COLORS[variant],
        )
        ax.text(xs[-1] + 0.25, ys[-1], variant, color=COLORS[variant], fontsize=11, fontweight="bold")

    ax.set_title("Dataset efficiency: context helps more with more volumes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training volumes", fontsize=12)
    ax.set_ylabel("Test MS-SSIM (higher is better)", fontsize=12)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_xlim(4.3, 21.0)
    ax.set_ylim(0.74, 0.875)
    _style_axes(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def poster_f3_robustness(robustness_summary_dir: Path, out_path: Path) -> None:
    rows = _variant_order(_read_csv(robustness_summary_dir / "f3_main_replicate_summary.csv"))
    labels = [r["variant_key"] for r in rows]
    colors = [COLORS[l] for l in labels]
    x = np.arange(len(rows))

    metrics = [
        ("ms_ssim_r", "MS-SSIM-R", (0.42, 0.84)),
        ("residual_energy_frac", "Residual energy", (0.35, 0.86)),
        ("residual_input_corr", "Residual-input corr.", (0.58, 0.94)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    for ax, (metric, title, ylim) in zip(axes, metrics):
        means = [_f(r, f"{metric}_mean") for r in rows]
        stds = [_f(r, f"{metric}_std") for r in rows]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.92)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.set_title(f"{title}\nlower is better", fontsize=13, fontweight="bold")
        _style_axes(ax)
        _annotate_bars(ax, bars, means, "{:.3f}")

    fig.suptitle("F3 unlabelled field transfer: 5ch has cleaner residual diagnostics", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def poster_final_results(summary_dir: Path, robustness_summary_dir: Path, out_path: Path) -> None:
    main = {r["variant_key"]: r for r in _read_csv(summary_dir / "main_replicate_summary.csv")}
    f3 = {r["variant_key"]: r for r in _read_csv(robustness_summary_dir / "f3_main_replicate_summary.csv")}

    paired = [
        ("MS-SSIM", _f(main["5ch"], "test_ms_ssim_mean") - _f(main["2D"], "test_ms_ssim_mean"), "+{:.3f}", "higher"),
        ("PSNR", _f(main["5ch"], "test_psnr_mean") - _f(main["2D"], "test_psnr_mean"), "+{:.2f} dB", "higher"),
        (
            "MSE",
            100.0 * (_f(main["2D"], "test_mse_mean") - _f(main["5ch"], "test_mse_mean")) / _f(main["2D"], "test_mse_mean"),
            "-{:.0f}%",
            "lower",
        ),
    ]
    field = [
        (
            "F3 residual energy",
            100.0 * (_f(f3["2D"], "residual_energy_frac_mean") - _f(f3["5ch"], "residual_energy_frac_mean"))
            / _f(f3["2D"], "residual_energy_frac_mean"),
            "-{:.0f}%",
            "lower",
        ),
        (
            "F3 residual corr.",
            100.0 * (_f(f3["2D"], "residual_input_corr_mean") - _f(f3["5ch"], "residual_input_corr_mean"))
            / _f(f3["2D"], "residual_input_corr_mean"),
            "-{:.0f}%",
            "lower",
        ),
    ]
    cards = paired + field
    display = [r[2].format(r[1]) for r in cards]

    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    ax.axis("off")
    ax.text(
        0.5,
        0.95,
        "Final result: 5ch is strongest in paired testing and F3 transfer",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    card_w = 0.18
    gap = 0.017
    start_x = (1 - (5 * card_w + 4 * gap)) / 2
    for i, ((label, _value, _fmt_str, direction), value_text) in enumerate(zip(cards, display)):
        x0 = start_x + i * (card_w + gap)
        y0 = 0.30
        color = COLORS["5ch"] if i < 3 else COLORS["3ch"]
        rect = plt.Rectangle((x0, y0), card_w, 0.43, transform=ax.transAxes,
                             facecolor=color, alpha=0.15, edgecolor=color, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(
            x0 + card_w / 2,
            y0 + 0.285,
            value_text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x0 + card_w / 2,
            y0 + 0.165,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color="#111820",
        )
        ax.text(
            x0 + card_w / 2,
            y0 + 0.070,
            f"{direction} is better",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.5,
            color="#4A555C",
        )

    ax.text(
        0.5,
        0.12,
        "Values are 5ch improvement over 2D. F3 has no clean target; diagnostics are no-reference only.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4A555C",
    )
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _crop_top_row_tiles(panel_path: Path) -> list[Image.Image]:
    img = Image.open(panel_path).convert("RGB")
    w, h = img.size

    # Generated panel has 5 equal-width image tiles in the top row. Crop just
    # the seismic images, not the old titles, so labels can be rewritten.
    y0 = int(h * 0.098)
    y1 = int(h * 0.520)
    margin = int(w * 0.005)
    gap = int(w * 0.007)
    tile_w = int((w - 2 * margin - 4 * gap) / 5)
    tiles: list[Image.Image] = []
    for i in range(5):
        x0 = margin + i * (tile_w + gap)
        x1 = x0 + tile_w
        tiles.append(img.crop((x0, y0, x1, y1)))
    return tiles


def poster_qualitative_example(comparison_root: Path, out_path: Path) -> None:
    panel_path = comparison_root / "by_gap" / "panel_00.png"
    metadata_path = comparison_root / "comparison_metadata.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing qualitative panel: {panel_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing comparison metadata: {metadata_path}")

    meta = _read_csv(metadata_path)[0]
    tiles = _crop_top_row_tiles(panel_path)
    ms_5ch = meta["ms_ssim_5ch"]
    delta_5ch = meta["delta_5ch_vs_2d"]
    if ms_5ch is None or delta_5ch is None:
        raise KeyError("comparison_metadata.csv must contain 5ch MS-SSIM and delta columns.")
    labels = [
        "Noisy",
        f"2D\nMS-SSIM {float(meta['ms_ssim_2d']):.3f}",
        f"3ch\nMS-SSIM {float(meta['ms_ssim_3ch']):.3f}",
        f"5ch\nMS-SSIM {float(ms_5ch):.3f}",
        "Clean target",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(13.2, 3.8))
    for ax, tile, label in zip(axes, tiles, labels):
        ax.imshow(tile)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.axis("off")

    delta = float(delta_5ch)
    fig.suptitle(
        f"Hard test slice example: 5ch gains +{delta:.3f} MS-SSIM over 2D",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poster-optimized result figures.")
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (args.out_dir or (project_root / "experiments" / "summaries" / "poster_figures")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_root = project_root / "experiments" / "summaries"
    summary_dir = summary_root / "main_experiment"
    data_efficiency_dir = summary_root / "data_efficiency"
    robustness_summary_dir = summary_root / "f3_robustness"
    comparison_root = summary_root / "comparison_panels"

    outputs = {
        "poster_main_result.png": lambda p: poster_main_result(summary_dir, p),
        "poster_data_efficiency.png": lambda p: poster_data_efficiency(data_efficiency_dir, summary_dir, p),
        "poster_f3_robustness.png": lambda p: poster_f3_robustness(robustness_summary_dir, p),
        "poster_final_results.png": lambda p: poster_final_results(summary_dir, robustness_summary_dir, p),
        "poster_qualitative_example.png": lambda p: poster_qualitative_example(comparison_root, p),
    }
    for name, fn in outputs.items():
        path = out_dir / name
        fn(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
