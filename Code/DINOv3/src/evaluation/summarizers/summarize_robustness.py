"""Aggregate F3 field-transfer robustness results into CSVs, plots, and a report.

Usage:
    python evaluation/summarize_robustness.py \
        --robustness-root experiments/runs/robustness \
        --output-dir experiments/summaries/f3_robustness

    python evaluation/summarize_robustness.py \
        --robustness-root experiments/runs/robustness \
        --result-dataset f3_allsections \
        --output-dir experiments/summaries/f3_allsections_robustness

Discovers:
    <robustness-root>/<result-dataset>/<family>/<variant>/<run_id>/f3_metrics.csv

Outputs:
    f3_summary.csv                  - per-run no-reference F3 diagnostics
    f3_main_replicate_summary.csv   - main 2D/3ch/5ch aggregate across seeds
    f3_main_metrics_bars.png        - poster-ready main diagnostic bars
    f3_main_seed_metrics.png        - per-seed stability plot for core diagnostics
    f3_data_efficiency.png          - F3 diagnostics vs training volumes
    f3_metric_tradeoff.png          - residual energy vs amplitude preservation
    f3_comparison_grid.png          - shared F3 panel for 2D/3ch/5ch
    robustness_report.md            - markdown report with claim boundaries
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

VARIANT_ORDER = {"2D": 0, "3ch": 1, "5ch": 2, "7ch": 3, "9ch": 4}
MAIN_VARIANTS = {"2D", "3ch", "5ch"}
MAIN_RUN_IDS = {"seed42_run01", "seed43_run02", "seed44_run03"}
DISPLAY_VARIANT_LABELS = {
    "2D": "2D-1",
    "3ch": "2.5D-3",
    "5ch": "2.5D-5",
}


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


def _run_group(variant_dir: str, run_id: str) -> str:
    if re.search(r"_n\d+vols", variant_dir):
        return "data_efficiency"
    if run_id in MAIN_RUN_IDS:
        return "main"
    return "other"


def _sort_key(row: dict[str, Any]) -> tuple:
    group_order = {"main": 0, "data_efficiency": 1, "other": 2}
    return (
        str(row.get("dataset_key") or ""),
        group_order.get(str(row.get("run_group")), 9),
        VARIANT_ORDER.get(str(row.get("variant_key")), 9),
        int(row.get("n_vols") or 999),
        str(row.get("run_id") or ""),
    )


_KNOWN_FAMILIES = {"2d", "3ch", "5ch", "7ch", "9ch"}


def _parse_path_parts(
    parts: tuple[str, ...],
) -> tuple[str, str, str] | None:
    """Return (family, variant_dir, run_id) from relative path parts, or None to skip.

    Handles two layouts:
      Old: <family>/<variant_dir>/<run_id>/f3_metrics.csv           (4 parts total)
      New: <exp_set>/<family>/<variant_dir>/<data_seed>/<run_id>/…  (6 parts total)
    """
    if len(parts) < 4:
        return None
    if parts[0].lower() in _KNOWN_FAMILIES:
        # Old layout
        return parts[0], parts[1], parts[2]
    if len(parts) >= 6 and parts[1].lower() in _KNOWN_FAMILIES:
        # New layout: exp_set / family / variant_dir / data_seed / run_id / file
        return parts[1], parts[2], parts[4]
    return None


def _discover_f3(
    root: Path,
    result_dataset: str,
    experiment_sets: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return discovered F3 result records, excluding non-main control variants.

    If *experiment_sets* is given, only walks those named subdirectories under
    the result-dataset root (e.g. ["main_multidata", "full_ft_multidata"]).
    This prevents old fixed-split result trees that sit directly under the
    result-dataset root (e.g. f3_allsections/2d/, f3_allsections/3ch/) from
    being mixed into the aggregate.
    """
    records: list[dict[str, Any]] = []
    f3_root = root / result_dataset
    if not f3_root.exists():
        return records

    if experiment_sets:
        csv_paths: list[Path] = []
        for exp_set in experiment_sets:
            exp_dir = f3_root / exp_set
            if exp_dir.exists():
                csv_paths.extend(sorted(exp_dir.rglob("f3_metrics.csv")))
            else:
                print(f"  [SKIP] experiment set not found: {exp_dir}", file=sys.stderr)
    else:
        csv_paths = sorted(f3_root.rglob("f3_metrics.csv"))

    for csv_path in csv_paths:
        parts = csv_path.relative_to(f3_root).parts
        parsed = _parse_path_parts(parts)
        if parsed is None:
            continue
        family, variant_dir, run_id = parsed
        variant_key = _variant_key(family, variant_dir)
        if variant_key is None:
            continue
        records.append(
            {
                "family": family,
                "variant_dir": variant_dir,
                "variant_key": variant_key,
                "dataset_key": result_dataset,
                "run_id": run_id,
                "run_group": _run_group(variant_dir, run_id),
                "n_vols": _n_vols_from_variant(variant_dir),
                "label": f"{variant_key} {run_id}",
                "csv_path": csv_path,
                "panel_dir": csv_path.parent,
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


def _orientations(data: list[dict[str, str]]) -> str:
    vals = sorted({str(row.get("orientation", "")).strip() for row in data if str(row.get("orientation", "")).strip()})
    return ",".join(vals) if vals else "-"


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def build_f3_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        data = _read_csv(rec["csv_path"])
        row: dict[str, Any] = {
            "dataset_key": rec["dataset_key"],
            "variant_key": rec["variant_key"],
            "run_group": rec["run_group"],
            "n_vols": rec["n_vols"],
            "run_id": rec["run_id"],
            "family": rec["family"],
            "variant_dir": rec["variant_dir"],
            "n_samples": len(data),
            "orientations": _orientations(data),
            "sample_count_request": _protocol_value(data, "sample_count_request", ""),
            "common_context_radius": _protocol_value(data, "common_context_radius", ""),
            "effective_context_radius": _protocol_value(data, "effective_context_radius", ""),
            "mode": _protocol_value(data, "mode", ""),
            "neighbor_stride": _protocol_value(data, "neighbor_stride", ""),
        }
        for metric in F3_METRIC_KEYS:
            vals = [_float(d.get(metric)) for d in data]
            vals = [v for v in vals if v is not None]
            mean, std = _mean_std(vals)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return sorted(rows, key=_sort_key)


def build_main_replicate_summary(f3_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in f3_summary:
        if (
            row.get("run_group") == "main"
            and row.get("run_id") in MAIN_RUN_IDS
            and row.get("variant_key") in MAIN_VARIANTS
        ):
            grouped[(str(row.get("dataset_key") or "f3"), str(row["variant_key"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (dataset_key, variant), variant_rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], VARIANT_ORDER.get(item[0][1], 9)),
    ):
        out: dict[str, Any] = {
            "dataset_key": dataset_key,
            "variant_key": variant,
            "n_runs": len(variant_rows),
            "seeds": ",".join(r["run_id"].split("_")[0].replace("seed", "") for r in sorted(variant_rows, key=_sort_key)),
            "samples_per_run": ",".join(str(r.get("n_samples")) for r in sorted(variant_rows, key=_sort_key)),
            "orientations": _protocol_value(variant_rows, "orientations", ""),
            "sample_count_request": _protocol_value(variant_rows, "sample_count_request", ""),
            "common_context_radius": _protocol_value(variant_rows, "common_context_radius", ""),
            "effective_context_radius": _protocol_value(variant_rows, "effective_context_radius", ""),
        }
        for metric in F3_METRIC_KEYS:
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
        raise RuntimeError("matplotlib is required for plotting. Install it with: pip install matplotlib")


def _load_pyplot():
    _require_mpl()
    import matplotlib.pyplot as plt
    return plt


def _main_rows(f3_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in f3_summary
        if r.get("run_group") == "main"
        and r.get("run_id") in MAIN_RUN_IDS
        and r.get("variant_key") in MAIN_VARIANTS
    ]


def plot_f3_main_metrics(main_summary: list[dict[str, Any]], out_path: Path) -> None:
    if not main_summary:
        return
    plt = _load_pyplot()
    metrics = F3_METRIC_KEYS
    variants = [r["variant_key"] for r in main_summary]
    x = np.arange(len(variants))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for ax, metric in zip(axes.flat, metrics):
        means = [float(r.get(f"{metric}_mean") or 0.0) for r in main_summary]
        stds = [float(r.get(f"{metric}_std") or 0.0) for r in main_summary]
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors[:len(variants)], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_title(F3_METRIC_LABELS[metric])
        ax.grid(True, axis="y", alpha=0.25)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("F3 field-transfer diagnostics across main seeds\nNo clean target: lower residual diagnostics are better")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f3_main_seed_metrics(f3_summary: list[dict[str, Any]], out_path: Path) -> None:
    rows = _main_rows(f3_summary)
    if not rows:
        return
    plt = _load_pyplot()
    metrics = ["ms_ssim_r", "residual_energy_frac", "residual_input_corr"]
    variants = sorted({r["variant_key"] for r in rows}, key=lambda v: VARIANT_ORDER.get(v, 9))
    x_base = np.arange(len(variants))
    offsets = {"seed42_run01": -0.18, "seed43_run02": 0.0, "seed44_run03": 0.18}
    colors = {"seed42_run01": "#4c78a8", "seed43_run02": "#f58518", "seed44_run03": "#54a24b"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, metric in zip(axes, metrics):
        for run_id in sorted(MAIN_RUN_IDS):
            vals = []
            xs = []
            for i, variant in enumerate(variants):
                row = next((r for r in rows if r["variant_key"] == variant and r["run_id"] == run_id), None)
                if row and row.get(f"{metric}_mean") is not None:
                    xs.append(x_base[i] + offsets[run_id])
                    vals.append(float(row[f"{metric}_mean"]))
            if vals:
                ax.scatter(xs, vals, s=60, label=run_id.replace("_run0", " run"), color=colors[run_id], alpha=0.9)
        ax.set_xticks(x_base)
        ax.set_xticklabels(variants)
        ax.set_title(F3_METRIC_LABELS[metric])
        ax.grid(True, axis="y", alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle("F3 main-run seed stability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f3_data_efficiency(f3_summary: list[dict[str, Any]], out_path: Path) -> None:
    rows = [
        r for r in f3_summary
        if r.get("run_group") == "data_efficiency"
        or (r.get("run_group") == "main" and r.get("run_id") == "seed42_run01")
    ]
    if not rows:
        return
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["variant_key"])].append(row)
    by_variant = {k: v for k, v in by_variant.items() if len({r["n_vols"] for r in v}) >= 2}
    if not by_variant:
        return

    plt = _load_pyplot()
    metrics = [
        "ms_ssim_r",
        "residual_energy_frac",
        "residual_input_corr",
        "low_freq_residual_energy_frac",
    ]
    colors = {"2D": "#4c78a8", "3ch": "#f58518", "5ch": "#54a24b"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, metrics):
        for variant, vrows in sorted(by_variant.items(), key=lambda item: VARIANT_ORDER.get(item[0], 9)):
            points = sorted(vrows, key=lambda r: int(r["n_vols"]))
            ax.plot(
                [int(r["n_vols"]) for r in points],
                [float(r[f"{metric}_mean"]) for r in points],
                marker="o",
                linewidth=1.8,
                label=variant,
                color=colors.get(variant),
            )
        ax.set_title(F3_METRIC_LABELS[metric])
        ax.set_xlabel("Training volumes")
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(True, alpha=0.25)
    axes.flat[0].legend(fontsize=9)
    fig.suptitle("F3 field-transfer diagnostics vs Image Impeccable training volumes")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f3_metric_tradeoff(f3_summary: list[dict[str, Any]], out_path: Path) -> None:
    rows = [r for r in f3_summary if r.get("residual_energy_frac_mean") is not None]
    if not rows:
        return
    plt = _load_pyplot()
    colors = {"2D": "#4c78a8", "3ch": "#f58518", "5ch": "#54a24b"}
    markers = {"main": "o", "data_efficiency": "s", "other": "^"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for row in rows:
        variant = str(row["variant_key"])
        group = str(row["run_group"])
        size = 45 + int(row.get("n_vols") or 20) * 3
        ax.scatter(
            float(row["residual_energy_frac_mean"]),
            float(row["denoised_input_amplitude_ratio_mean"]),
            s=size,
            color=colors.get(variant, "#777777"),
            marker=markers.get(group, "o"),
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
        )
        if group == "main":
            ax.text(
                float(row["residual_energy_frac_mean"]) + 0.006,
                float(row["denoised_input_amplitude_ratio_mean"]) + 0.004,
                f"{variant} {row['run_id'].split('_')[0]}",
                fontsize=7,
            )
    ax.set_xlabel("Residual energy fraction (lower is better)")
    ax.set_ylabel("Denoised/input amplitude ratio")
    ax.set_title("F3 residual removal vs amplitude preservation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f3_comparison_grid(records: list[dict[str, Any]], out_path: Path) -> None:
    selected: list[dict[str, Any]] = []
    for variant in ("2D", "3ch", "5ch"):
        row = next((r for r in records if r["variant_key"] == variant and r["run_id"] == "seed42_run01" and r["run_group"] == "main"), None)
        if row:
            selected.append(row)
    if len(selected) < 2:
        selected = [next(iter(v), None) for _, v in _group_records_by_variant(records).items()]
        selected = [r for r in selected if r is not None]
    if not selected:
        return

    try:
        from PIL import Image  # type: ignore[import]
    except ImportError:
        print("  Pillow not available; skipping F3 comparison grid. Install it with: pip install Pillow")
        return

    def panel_score(path: Path) -> tuple[int, int, str]:
        match = re.search(r"f3_panel_(inline|crossline|timeslice)_(\d+)\.png$", path.name)
        if not match:
            return (9, 999999, path.name)
        orientation, idx_raw = match.group(1), match.group(2)
        targets = {"inline": 419, "crossline": 306, "timeslice": 231}
        order = {"inline": 0, "crossline": 1, "timeslice": 2}
        return (order[orientation], abs(int(idx_raw) - targets[orientation]), path.name)

    panel_paths: list[Path] = []
    common = set.intersection(
        *(set(p.name for p in rec["panel_dir"].glob("f3_panel_*.png")) for rec in selected)
    )
    if common:
        panel_name = next((p for p in sorted(common) if "inline_0419" in p), sorted(common)[0])
        panel_paths = [rec["panel_dir"] / panel_name for rec in selected]
    else:
        for rec in selected:
            panels = sorted(rec["panel_dir"].glob("f3_panel_*.png"), key=panel_score)
            if not panels:
                print(f"  No F3 panels found for {rec['label']}; skipping comparison grid.")
                return
            panel_paths.append(panels[0])

    def _runs(active: np.ndarray, min_len: int) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for idx, is_active in enumerate(active.tolist()):
            if is_active and start is None:
                start = idx
            elif not is_active and start is not None:
                if idx - start >= min_len:
                    runs.append((start, idx))
                start = None
        if start is not None and len(active) - start >= min_len:
            runs.append((start, len(active)))
        return runs

    def _extract_panel_crops(path: Path) -> list[np.ndarray]:
        img = np.array(Image.open(path).convert("RGB"))
        # The saved panels contain title text and white margins. Use color spread
        # to isolate the red/blue seismic image boxes while ignoring black text.
        color_spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
        color_mask = (color_spread > 25) & (img.min(axis=2) < 245)

        row_counts = color_mask.sum(axis=1)
        col_counts = color_mask.sum(axis=0)
        row_active = row_counts > max(8, img.shape[1] * 0.01)
        col_active = col_counts > max(8, img.shape[0] * 0.04)

        row_runs = _runs(row_active, min_len=80)
        col_runs = _runs(col_active, min_len=120)
        if not row_runs or len(col_runs) < 3:
            h, w = img.shape[:2]
            body = img[int(round(h * 0.22)):, :, :]
            third = w // 3
            return [body[:, i * third:(i + 1) * third, :] for i in range(3)]

        y0 = min(start for start, _ in row_runs)
        y1 = max(end for _, end in row_runs)
        return [img[y0:y1, x0:x1, :] for x0, x1 in col_runs[:3]]

    plt = _load_pyplot()
    row_crops = [_extract_panel_crops(path) for path in panel_paths]
    variant_labels = [DISPLAY_VARIANT_LABELS.get(rec["variant_key"], rec["variant_key"]) for rec in selected]

    n = len(row_crops)
    fig = plt.figure(figsize=(11.6, 8.6))
    grid = fig.add_gridspec(
        nrows=n,
        ncols=4,
        width_ratios=[0.36, 1.0, 1.0, 1.0],
        wspace=0.035,
        hspace=0.065,
        left=0.035,
        right=0.995,
        bottom=0.035,
        top=0.915,
    )

    col_labels = ["Noisy input", "Denoised output", "Residual"]
    for row_idx, (crops, label) in enumerate(zip(row_crops, variant_labels)):
        label_ax = fig.add_subplot(grid[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(
            0.98,
            0.5,
            label,
            ha="right",
            va="center",
            fontsize=12.5,
            fontweight="bold",
        )

        for col_idx, crop in enumerate(crops):
            ax = fig.add_subplot(grid[row_idx, col_idx + 1])
            ax.imshow(crop, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=12, pad=4)

    fig.suptitle("Denoising Results on the F3 Field Volume", fontsize=15, fontweight="bold", y=0.985)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _group_records_by_variant(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec["variant_key"])].append(rec)
    return dict(sorted(grouped.items(), key=lambda item: VARIANT_ORDER.get(item[0], 9)))


# ---------------------------------------------------------------------------
# Report and CSV writers
# ---------------------------------------------------------------------------

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


def _md_table(lines: list[str], headers: list[str], rows: list[list[Any]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")


def build_report(f3_summary: list[dict[str, Any]], main_summary: list[dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# F3 Robustness Evaluation Report",
        "",
        "> Generated by `summarize_robustness.py`.",
        "> Do not edit manually; re-run the summarizer after adding new results.",
        "",
        "## Protocol",
        "",
        "- **Task**: Zero-shot transfer to unlabelled F3 seismic field data; no field-data fine-tuning.",
        "- **Checkpoints**: `best.pt` from Image Impeccable training (selected on Image Impeccable val MS-SSIM).",
        "- **Main variants**: 2D, 3ch, 5ch (same as the paired experiment).",
        "- **Evaluation seeds**: seed42_run01, seed43_run02, seed44_run03 (training seeds only; one data-seed for F3 since it is unsplit field data).",
        "- **Metrics**: No-reference diagnostics only — MS-SSIM-R, residual energy fraction, residual-input correlation, amplitude ratio, low-frequency residual energy.",
        "- **Result dataset**: configurable via `--result-dataset`; default is `f3`.",
        "",
        "## Claim Boundaries",
        "",
        "- F3 results measure unlabelled field-domain transfer.",
        "- No accuracy metrics are reported for F3 because there is no clean ground truth.",
        "- Metrics are no-reference diagnostics only and supplement the paired Image Impeccable results.",
        "- `best.pt` checkpoints were selected only on Image Impeccable validation.",
        "- Main aggregate figures are restricted to the current main variants: 2D, 3ch, and 5ch.",
        "- Wider channel-window runs, if present, are listed only as per-run diagnostics unless the protocol is expanded.",
        "",
        "## Main F3 Aggregate",
        "",
        "Lower `MS-SSIM-R`, residual energy, residual-input correlation, and low-frequency residual energy are preferred. "
        "The amplitude ratio is a preservation check rather than a one-sided quality metric.",
        "",
    ]

    if main_summary:
        _md_table(
            lines,
            [
                "Dataset",
                "Variant",
                "Runs",
                "Seeds",
                "Samples/run",
                "MS-SSIM-R",
                "Residual energy",
                "Residual-input corr",
                "Amplitude ratio",
                "Low-freq residual",
            ],
            [
                [
                    r.get("dataset_key", "f3"),
                    r["variant_key"],
                    r["n_runs"],
                    r["seeds"],
                    r.get("samples_per_run", "-"),
                    _mean_std_text(r, "ms_ssim_r"),
                    _mean_std_text(r, "residual_energy_frac"),
                    _mean_std_text(r, "residual_input_corr"),
                    _mean_std_text(r, "denoised_input_amplitude_ratio"),
                    _mean_std_text(r, "low_freq_residual_energy_frac"),
                ]
                for r in main_summary
            ],
        )
    else:
        lines.append("No main F3 results found.")

    lines += ["", "## Per-Run F3 Diagnostics", ""]
    if f3_summary:
        _md_table(
            lines,
            [
                "Dataset",
                "Variant",
                "Group",
                "n",
                "Run",
                "Samples",
                "Orientations",
                "Sample request",
                "Common CR",
                "MS-SSIM-R",
                "Residual energy",
                "Residual-input corr",
                "Amplitude ratio",
                "Low-freq residual",
            ],
            [
                [
                    r.get("dataset_key", "f3"),
                    r["variant_key"],
                    r["run_group"],
                    r["n_vols"],
                    r["run_id"],
                    r["n_samples"],
                    r.get("orientations", "-"),
                    r.get("sample_count_request", "-"),
                    r.get("common_context_radius", "-"),
                    _fmt(r.get("ms_ssim_r_mean")),
                    _fmt(r.get("residual_energy_frac_mean")),
                    _fmt(r.get("residual_input_corr_mean")),
                    _fmt(r.get("denoised_input_amplitude_ratio_mean")),
                    _fmt(r.get("low_freq_residual_energy_frac_mean")),
                ]
                for r in f3_summary
            ],
        )
    else:
        lines.append("No F3 results found.")

    lines += [
        "",
        "## Generated Figures",
        "",
        "- `f3_main_metrics_bars.png`: compact poster-ready aggregate diagnostics for 2D, 3ch, and 5ch.",
        "- `f3_main_seed_metrics.png`: seed-to-seed stability for the three most interpretable diagnostics.",
        "- `f3_data_efficiency.png`: F3 no-reference diagnostics as Image Impeccable training volume count changes.",
        "- `f3_metric_tradeoff.png`: residual removal versus amplitude preservation.",
        "- `f3_comparison_grid.png`: shared unlabelled field panel for the main variants.",
        "- `f3_per_volume_scatter.png`: per-section scatter of MS-SSIM-R vs residual energy fraction, coloured by variant.",
        "- `f3_seed_consistency.png`: grouped bar chart of per-run MS-SSIM-R for all main seeds (no averaging), shows seed-to-seed variance.",
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
# Extended diagnostic plots
# ---------------------------------------------------------------------------

def plot_f3_per_volume_scatter(f3_records: list[dict[str, Any]], out_path: Path) -> None:
    """Scatter plot: per-file MS-SSIM-R vs residual energy fraction, coloured by variant.

    Reads raw rows from each run's f3_metrics.csv to show within-run variance across
    individual F3 volume sections.
    """
    plt = _load_pyplot()
    colors = {"2D": "#4c78a8", "3ch": "#f58518", "5ch": "#54a24b"}
    plotted_variants: set[str] = set()

    fig, ax = plt.subplots(figsize=(8, 6))
    for rec in f3_records:
        variant = str(rec.get("variant_key", ""))
        if variant not in MAIN_VARIANTS:
            continue
        try:
            rows = _read_csv(rec["csv_path"])
        except OSError:
            continue
        xs = [_float(r.get("residual_energy_frac")) for r in rows]
        ys = [_float(r.get("ms_ssim_r")) for r in rows]
        pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
        if not pairs:
            continue
        px, py = zip(*pairs)
        label = DISPLAY_VARIANT_LABELS.get(variant, variant) if variant not in plotted_variants else None
        ax.scatter(px, py, color=colors.get(variant, "#888888"), alpha=0.45, s=18, label=label)
        plotted_variants.add(variant)

    ax.set_xlabel("Residual energy fraction")
    ax.set_ylabel("MS-SSIM-R")
    ax.set_title("F3 per-volume-section scatter: MS-SSIM-R vs residual energy\n(each point = one F3 section; lower residual energy is better)")
    ax.grid(True, alpha=0.2)
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_f3_seed_consistency(f3_summary: list[dict[str, Any]], out_path: Path) -> None:
    """Grouped bar chart: MS-SSIM-R per run for main 2D/3ch/5ch variants.

    Shows all available main runs as separate bars (not averaged) so seed-to-seed
    variance is visible without relying on error bars.
    """
    rows = [
        r for r in f3_summary
        if r.get("run_group") == "main" and r.get("variant_key") in MAIN_VARIANTS
    ]
    if not rows:
        return

    plt = _load_pyplot()
    variants = sorted({str(r["variant_key"]) for r in rows}, key=lambda v: VARIANT_ORDER.get(v, 9))
    run_ids = sorted({str(r["run_id"]) for r in rows})
    run_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]
    x = np.arange(len(variants))
    width = 0.8 / max(len(run_ids), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 2.5), 5))
    for ri, run_id in enumerate(run_ids):
        vals = []
        xs = []
        for vi, variant in enumerate(variants):
            row = next((r for r in rows if r["variant_key"] == variant and r["run_id"] == run_id), None)
            v = _float((row or {}).get("ms_ssim_r_mean")) if row else None
            vals.append(v if v is not None else 0.0)
            xs.append(x[vi] + (ri - len(run_ids) / 2 + 0.5) * width)
        color = run_colors[ri % len(run_colors)]
        ax.bar(xs, vals, width, label=run_id.replace("_run0", " run"), color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("MS-SSIM-R (mean across F3 sections)")
    ax.set_title("F3 seed consistency: per-run MS-SSIM-R for main variants")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate F3 robustness evaluation results.")
    parser.add_argument("--robustness-root", required=True, help="Root of experiments/runs/robustness/.")
    parser.add_argument(
        "--result-dataset",
        default="f3",
        help="Result folder under robustness-root to summarize (default: f3).",
    )
    parser.add_argument(
        "--experiment-sets",
        default=None,
        help=(
            "Comma-separated experiment-set subdirectories to include "
            "(e.g. main_multidata,full_ft_multidata). "
            "Default: include everything under the result-dataset root."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Where to write summary files.")
    args = parser.parse_args()

    rob_root = Path(args.robustness_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_sets: list[str] | None = (
        [s.strip() for s in args.experiment_sets.split(",") if s.strip()]
        if args.experiment_sets
        else None
    )

    print(f"Robustness root: {rob_root}")
    print(f"Result dataset: {args.result_dataset}")
    print(f"Experiment sets: {exp_sets or 'all'}")
    print(f"Output dir: {out_dir}")

    f3_records = _discover_f3(rob_root, args.result_dataset, experiment_sets=exp_sets)
    print(f"Found {len(f3_records)} F3 run(s) after excluding non-main control variants.")

    if not f3_records:
        sys.exit(f"No F3 robustness results found under {rob_root / args.result_dataset}. Run evaluate_robustness.py first.")

    f3_summary = build_f3_summary(f3_records)
    main_summary = build_main_replicate_summary(f3_summary)
    _write_csv(f3_summary, out_dir / "f3_summary.csv")
    _write_csv(main_summary, out_dir / "f3_main_replicate_summary.csv")

    if _HAS_MPL:
        plot_f3_main_metrics(main_summary, out_dir / "f3_main_metrics_bars.png")
        plot_f3_main_seed_metrics(f3_summary, out_dir / "f3_main_seed_metrics.png")
        plot_f3_data_efficiency(f3_summary, out_dir / "f3_data_efficiency.png")
        plot_f3_metric_tradeoff(f3_summary, out_dir / "f3_metric_tradeoff.png")
        plot_f3_comparison_grid(f3_records, out_dir / "f3_comparison_grid.png")
        plot_f3_per_volume_scatter(f3_records, out_dir / "f3_per_volume_scatter.png")
        plot_f3_seed_consistency(f3_summary, out_dir / "f3_seed_consistency.png")
    else:
        print("matplotlib not installed; skipping plots.")

    build_report(f3_summary, main_summary, out_dir / "robustness_report.md")

    print("\nSummarization complete.")


if __name__ == "__main__":
    main()
