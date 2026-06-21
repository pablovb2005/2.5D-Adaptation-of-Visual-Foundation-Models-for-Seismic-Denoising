"""Generate the F3 field-data comparison grid for the thesis.

Layout: 3 image rows (Noisy input / Denoised / Residual) × 4 image columns
(2D-1ch / 2.5D-3ch / 2.5D-5ch / Median filter), wider than tall.

Pipeline
--------
1. **Cache phase** — run once per inline index.  Extracts 224×224 RGB crops for
   each variant and saves them as individual PNG files under
   ``experiments/summaries/f3_panel_crops/inline{IDX}/``.
   Re-run with ``--rebuild-cache`` to force regeneration.

2. **Figure phase** — load 12 crops from cache and assemble into the final PNG.
   Fast enough to iterate on layout without re-running any inference.

Cache structure::

    f3_panel_crops/inline0419/
        2D_row0_noisy.png
        2D_row1_denoised.png
        2D_row2_residual.png
        3ch_row{0,1,2}.png  ...
        5ch_row{0,1,2}.png  ...
        mf_row{0,1,2}.png   ...   ← rendered from f3_original.npy + f3_filtered_ref.npy

Usage::

    python evaluation/plot_f3_comparison_grid_with_median.py \\
        --project-root C:/UNI/Y3/RP \\
        [--inline-idx 419] \\
        [--rebuild-cache] \\
        [--out-dir experiments/summaries/f3_robustness]
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root

import numpy as np

try:
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_DIRS = {
    "2D":  ("2d",  "impeccable_repeated_stride5_lora_r16"),
    "3ch": ("3ch", "impeccable_neighbors3_stride5_lora_r16"),
    "5ch": ("5ch", "impeccable_neighbors5_stride5_patch_emb_lora_r16"),
}
SEED_RUN = "seed42_run01"

DISPLAY_LABELS = {
    "2D":  "2D-1ch",
    "3ch": "2.5D-3ch",
    "5ch": "2.5D-5ch",
    "MF":  "Median\nfilter",
}
ROW_LABELS = ["Noisy\ninput", "Denoised", "Residual"]
COL_KEYS   = ["2D", "3ch", "5ch", "MF"]
ORIENTATION_TITLES = {
    "timeslice": "Time orientation",
    "inline": "Inline/Crossline orientation",
}


# ---------------------------------------------------------------------------
# Panel-PNG crop extraction (from existing evaluate_robustness.py outputs)
# ---------------------------------------------------------------------------

def _color_runs(active: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_len:
                runs.append((start, idx))
            start = None
    if start is not None and len(active) - start >= min_len:
        runs.append((start, len(active)))
    return runs


def _extract_panel_crops(path: Path) -> list[np.ndarray]:
    """Extract [noisy, denoised, residual] RGB crops from a saved 3-panel PNG."""
    img = np.array(Image.open(path).convert("RGB"))
    spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    mask = (spread > 25) & (img.min(axis=2) < 245)

    row_active = mask.sum(axis=1) > max(8, img.shape[1] * 0.01)
    col_active = mask.sum(axis=0) > max(8, img.shape[0] * 0.04)
    row_runs = _color_runs(row_active, min_len=80)
    col_runs = _color_runs(col_active, min_len=120)

    if not row_runs or len(col_runs) < 3:
        h, w = img.shape[:2]
        body = img[int(round(h * 0.22)):, :, :]
        t = w // 3
        return [body[:, i * t:(i + 1) * t, :] for i in range(3)]

    y0 = min(s for s, _ in row_runs)
    y1 = max(e for _, e in row_runs)
    return [img[y0:y1, x0:x1, :] for x0, x1 in col_runs[:3]]


def _find_panel_png(rob_root: Path, variant: str, section_idx: int,
                    prefix: str = "inline", run_subdir: str = SEED_RUN) -> Path | None:
    family, vdir = VARIANT_DIRS[variant]
    run_dir = rob_root / family / vdir / run_subdir
    exact = run_dir / f"f3_panel_{prefix}_{section_idx:04d}.png"
    if exact.exists():
        return exact
    panels = sorted(run_dir.glob(f"f3_panel_{prefix}_*.png"))
    if not panels:
        return None

    def _i(p: Path) -> int:
        m = re.search(rf"{prefix}_(\d+)\.png$", p.name)
        return int(m.group(1)) if m else 9999

    panels.sort(key=lambda p: abs(_i(p) - section_idx))
    print(f"  [{variant}] f3_panel_{prefix}_{section_idx:04d}.png not found; "
          f"using {panels[0].name}")
    return panels[0]


# ---------------------------------------------------------------------------
# Median filter rendering
# ---------------------------------------------------------------------------

def _zscore(x: np.ndarray) -> np.ndarray:
    s = float(x.std())
    return (x - x.mean()) / s if s > 1e-8 else x - x.mean()


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    oh = max(0, (h - size) // 2)
    ow = max(0, (w - size) // 2)
    return arr[oh:oh + size, ow:ow + size]


def _float_to_rgb(arr: np.ndarray, vmin: float = -2.0, vmax: float = 2.0) -> np.ndarray:
    """Float 2D array → uint8 RGB using seismic colormap."""
    cmap = plt.get_cmap("seismic")
    rgba = cmap(Normalize(vmin=vmin, vmax=vmax)(arr))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _build_mf_crops(f3_npy: Path, filt_npy: Path, section_idx: int,
                    orientation: str = "inline",
                    crop_size: int = 224) -> list[np.ndarray]:
    """Return [noisy_rgb, filtered_rgb, residual_rgb] for the median filter column."""
    noisy_vol = np.load(str(f3_npy),   mmap_mode="r", allow_pickle=False)
    filt_vol  = np.load(str(filt_npy), mmap_mode="r", allow_pickle=False)

    if orientation == "timeslice":
        nc = _center_crop(noisy_vol[:, :, section_idx].astype(np.float32), crop_size)
        fc = _center_crop(filt_vol[:,  :, section_idx].astype(np.float32), crop_size)
    else:
        nc = _center_crop(noisy_vol[section_idx, :, :].astype(np.float32), crop_size)
        fc = _center_crop(filt_vol[section_idx,  :, :].astype(np.float32), crop_size)
    nc_z = _zscore(nc)
    fc_z = _zscore(fc)
    res  = nc_z - fc_z

    return [_float_to_rgb(nc_z), _float_to_rgb(fc_z), _float_to_rgb(res)]


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def _cache_dir(summaries_root: Path, section_idx: int, orientation: str = "inline") -> Path:
    return summaries_root / "f3_panel_crops" / f"{orientation}{section_idx:04d}"


def _crop_path(cache: Path, variant: str, row: int) -> Path:
    row_name = ["noisy", "denoised", "residual"][row]
    return cache / f"{variant}_row{row}_{row_name}.png"


def build_cache(
    rob_root: Path,
    f3_npy: Path,
    filt_npy: Path,
    summaries_root: Path,
    section_idx: int,
    orientation: str = "inline",
    prefix: str = "inline",
    run_subdir: str = SEED_RUN,
) -> Path:
    """Extract all 12 crops and save to PNG cache. Returns cache directory."""
    cache = _cache_dir(summaries_root, section_idx, orientation)
    cache.mkdir(parents=True, exist_ok=True)

    # Neural model variants
    for variant in ("2D", "3ch", "5ch"):
        panel_path = _find_panel_png(rob_root, variant, section_idx, prefix, run_subdir)
        if panel_path is None:
            print(f"  [{variant}] no panel PNG found — skipping cache entry.")
            continue
        crops = _extract_panel_crops(panel_path)
        if len(crops) < 3:
            print(f"  [{variant}] crop extraction returned fewer than 3 images; skipping.")
            continue
        for ri, crop in enumerate(crops[:3]):
            out = _crop_path(cache, variant, ri)
            Image.fromarray(crop).save(str(out))
        print(f"  [{variant}] cached 3 crops from {panel_path.name}")

    # Median filter
    mf_crops = _build_mf_crops(f3_npy, filt_npy, section_idx, orientation)
    for ri, crop in enumerate(mf_crops):
        out = _crop_path(cache, "mf", ri)
        Image.fromarray(crop).save(str(out))
    print(f"  [MF] cached 3 crops from numpy ({orientation} {section_idx})")

    return cache


def load_cache(summaries_root: Path, section_idx: int,
               orientation: str = "inline") -> dict[str, list[np.ndarray]] | None:
    """Load all crops from cache. Returns None if any crop is missing."""
    cache = _cache_dir(summaries_root, section_idx, orientation)
    result: dict[str, list[np.ndarray]] = {}
    for variant in ("2D", "3ch", "5ch", "MF"):
        key = variant.lower() if variant == "MF" else variant
        key = "mf" if variant == "MF" else variant
        crops = []
        for ri in range(3):
            p = _crop_path(cache, key if variant != "MF" else "mf", ri)
            if not p.exists():
                return None
            crops.append(np.array(Image.open(p).convert("RGB")))
        result[variant] = crops
    return result


# ---------------------------------------------------------------------------
# Figure assembly  (single-imshow approach — zero inter-panel gaps)
# ---------------------------------------------------------------------------

def _resize_crop(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a uint8 RGB array to (h, w) using PIL LANCZOS."""
    return np.array(Image.fromarray(arr).resize((w, h), Image.LANCZOS))


def assemble_figure(
    crops: dict[str, list[np.ndarray]],
    out_path: Path,
    title: str = "Denoising Results on the F3 Field Volume",
    xlabel: str = "Crossline position",
    ylabel: str = "Time sample",
) -> None:
    """Stitch all crops into one numpy canvas, then display with a single imshow.

    This guarantees zero gaps between panels — no matplotlib gridspec artifacts.
    The figure layout is computed in pixel space; matplotlib is used only for
    the title and axis-free text labels around the canvas.
    """
    col_keys = [k for k in COL_KEYS if k in crops]
    n_rows, n_cols = 3, len(col_keys)

    # Use the largest crop dimensions as the canonical cell size so that
    # neural-model crops (already high-res) are not up-scaled.
    all_h = [c.shape[0] for v in crops.values() for c in v]
    all_w = [c.shape[1] for v in crops.values() for c in v]
    cell_h, cell_w = max(all_h), max(all_w)

    # Pre-size all crops to cell_h × cell_w
    sized: dict[str, list[np.ndarray]] = {
        key: [_resize_crop(c, cell_h, cell_w) for c in crops[key]]
        for key in col_keys
    }

    # Assemble canvas with thin white separator strips between cells
    gap_col = 18  # pixels between columns
    gap_row_axis = 18  # pixels between first and second rows
    gap_row = 18  # pixels between lower rows
    sep_col = np.full((cell_h, gap_col, 3), 255, dtype=np.uint8)
    rows_img = []
    for ri in range(n_rows):
        strips = []
        for ci, key in enumerate(col_keys):
            if ci > 0:
                strips.append(sep_col)
            strips.append(sized[key][ri])
        rows_img.append(np.concatenate(strips, axis=1))
    row_w = rows_img[0].shape[1]
    interleaved = []
    row_offsets: list[int] = []
    cursor_y = 0
    for ri, row in enumerate(rows_img):
        if ri > 0:
            gap = gap_row_axis if ri == 1 else gap_row
            interleaved.append(np.full((gap, row_w, 3), 255, dtype=np.uint8))
            cursor_y += gap
        row_offsets.append(cursor_y)
        interleaved.append(row)
        cursor_y += row.shape[0]
    canvas = np.concatenate(interleaved, axis=0)

    # --- Matplotlib figure with one imshow ---
    # Reserve a label column on the left (label_frac of total image width)
    # and a header row on top (header_frac of total image height).
    label_frac  = 0.170   # width fraction for row labels plus the first-panel y-axis
    header_frac = 0.115   # height fraction for column headers
    title_frac  = 0.090   # height fraction for figure title
    cbar_frac   = 0.070   # width fraction reserved for colorbars on the right

    # Figure dimensions (width driven by canvas aspect ratio)
    fig_w = 18.0  # inches
    canvas_aspect = canvas.shape[0] / canvas.shape[1]
    # canvas occupies (1 - label_frac) wide and (1 - header_frac - title_frac) tall
    img_frac_w = 1.0 - label_frac
    img_frac_h = 1.0 - header_frac - title_frac
    fig_h = fig_w * (canvas_aspect * img_frac_w / img_frac_h)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Single axes for the image canvas
    # [left, bottom, width, height] in figure fraction
    ax_l = label_frac
    ax_b = 0.01
    ax_w = img_frac_w - cbar_frac
    ax_h = img_frac_h
    ax = fig.add_axes((ax_l, ax_b, ax_w, ax_h))
    ax.imshow(canvas, aspect="auto", interpolation="lanczos")
    ax.axis("off")

    # Draw coordinate axes only over one image tile at a time. The stitched
    # canvas spans multiple rows and columns, so the main imshow axes would make
    # the spines run far past tick 223.
    canvas_h, canvas_w = canvas.shape[:2]

    def _tile_bounds(row_idx: int, col_idx: int = 0) -> tuple[float, float, float, float]:
        tile_x = col_idx * (cell_w + gap_col)
        tile_y = row_offsets[row_idx]
        left = ax_l + ax_w * (tile_x / canvas_w)
        bottom = ax_b + ax_h * (1.0 - ((tile_y + cell_h) / canvas_h))
        width = ax_w * (cell_w / canvas_w)
        height = ax_h * (cell_h / canvas_h)
        return left, bottom, width, height

    y_ax = fig.add_axes(_tile_bounds(0), facecolor="none")
    y_ax.set_zorder(ax.get_zorder() + 1)
    y_ax.set_xlim(0, cell_w - 1)
    y_ax.set_ylim(cell_h - 1, 0)
    y_ax.set_xticks([])
    y_ax.set_yticks([0, cell_h // 2, cell_h - 1])
    y_ax.set_yticklabels(["0", "112", "223"], fontsize=14, fontweight="bold")
    for spine in ("top", "right", "bottom"):
        y_ax.spines[spine].set_visible(False)
    y_ax.spines["left"].set_linewidth(1.0)
    y_ax.tick_params(axis="y", length=3, width=1.0, pad=2)
    y_ax.tick_params(top=False, right=False, labeltop=False, labelright=False)

    x_ax = fig.add_axes(_tile_bounds(2), facecolor="none")
    x_ax.set_zorder(ax.get_zorder() + 1)
    x_ax.set_xlim(0, cell_w - 1)
    x_ax.set_ylim(cell_h - 1, 0)
    x_ax.set_xticks([0, cell_w // 2, cell_w - 1])
    x_ax.set_xticklabels(["0", "112", "223"], fontsize=14, fontweight="bold")
    x_ax.set_yticks([])
    for spine in ("top", "right", "left"):
        x_ax.spines[spine].set_visible(False)
    x_ax.spines["bottom"].set_linewidth(1.0)
    x_ax.tick_params(axis="x", length=3, width=1.0, pad=2)
    x_ax.tick_params(top=False, right=False, labeltop=False, labelright=False)
    x_ax.set_xlabel(xlabel, fontsize=16, fontweight="bold", labelpad=6)

    # --- Column labels ---
    # Horizontally: each column centre in figure coords
    col_centers_fig = [
        ax_l + ax_w * ((ci * (cell_w + gap_col) + cell_w / 2) / canvas_w)
        for ci in range(n_cols)
    ]
    label_y = ax_b + ax_h + header_frac * 0.35
    for ci, key in enumerate(col_keys):
        fig.text(
            col_centers_fig[ci], label_y,
            DISPLAY_LABELS[key],
            ha="center", va="center",
            fontsize=26, fontweight="bold",
        )

    # --- Row labels ---
    # Vertically: each row centre in figure coords (rows go top-to-bottom in image)
    row_centers_fig = [
        ax_b + ax_h * (1.0 - ((row_offsets[ri] + cell_h / 2) / canvas_h))
        for ri in range(n_rows)
    ]
    label_x = ax_l - 0.085
    for ri in range(n_rows):
        fig.text(
            label_x, row_centers_fig[ri],
            ROW_LABELS[ri],
            ha="right", va="center",
            fontsize=24, fontweight="bold",
        )

    # --- Title ---
    fig.text(
        0.5, ax_b + ax_h + header_frac + title_frac * 0.45,
        title,
        ha="center", va="center",
        fontsize=30, fontweight="bold",
    )

    # y-axis label placed left of row labels (which sit at label_x = ax_l-0.01 ≈ 0.12)
    fig.text(
        ax_l - 0.040, row_centers_fig[0],
        ylabel,
        ha="center", va="center",
        fontsize=16, fontweight="bold", rotation=90,
    )

    # Single shared colorbar — all panels use the same ±2 z-score scale
    cbar_x    = ax_l + ax_w + 0.012
    cbar_w_cb = 0.020

    shared_cax = fig.add_axes((cbar_x, ax_b, cbar_w_cb, ax_h))
    shared_sm = ScalarMappable(cmap="seismic", norm=Normalize(vmin=-2, vmax=2))
    shared_sm.set_array([])
    cb = fig.colorbar(shared_sm, cax=shared_cax)
    cb.set_label("Amplitude\n(z-score, ±2σ)", fontsize=13, fontweight="bold")
    cb.ax.tick_params(labelsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight", pad_inches=0.04,
                facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def assemble_orientation_comparison(
    timeslice_crops: dict[str, list[np.ndarray]],
    inline_crops: dict[str, list[np.ndarray]],
    out_path: Path,
    include_median: bool = True,
) -> None:
    """Assemble matched time-slice and rotated vertical-slice F3 panels.

    Both panels use the same cached image crops and the same seismic colour scale
    as the single-orientation figures. The layout is rebuilt as one canvas so
    row labels, axis labels, and the colourbar are shared rather than duplicated.
    """
    groups = [
        ("timeslice", timeslice_crops, "Inline position", "Crossline position"),
        ("inline", inline_crops, "Time sample", "Crossline position"),
    ]
    desired_cols = COL_KEYS if include_median else [k for k in COL_KEYS if k != "MF"]
    col_keys = [k for k in desired_cols if k in timeslice_crops and k in inline_crops]
    n_rows = 3
    n_cols_per_group = len(col_keys)

    all_crops = [
        c
        for _, crops, _, _ in groups
        for key in col_keys
        for c in crops[key]
    ]
    cell_h = max(c.shape[0] for c in all_crops)
    cell_w = max(c.shape[1] for c in all_crops)

    sized: dict[str, dict[str, list[np.ndarray]]] = {}
    for orient, crops, _, _ in groups:
        sized[orient] = {
            key: [_resize_crop(c, cell_h, cell_w) for c in crops[key]]
            for key in col_keys
        }

    gap_col = 12
    gap_group = 200
    gap_row = 12
    sep_col = np.full((cell_h, gap_col, 3), 255, dtype=np.uint8)
    sep_group = np.full((cell_h, gap_group, 3), 255, dtype=np.uint8)

    rows_img = []
    tile_positions: dict[tuple[str, int, int], tuple[int, int]] = {}
    for ri in range(n_rows):
        strips = []
        x_cursor = 0
        for gi, (orient, _, _, _) in enumerate(groups):
            if gi > 0:
                strips.append(sep_group)
                x_cursor += gap_group
            for ci, key in enumerate(col_keys):
                if ci > 0:
                    strips.append(sep_col)
                    x_cursor += gap_col
                tile_positions[(orient, ri, ci)] = (x_cursor, 0)
                strips.append(sized[orient][key][ri])
                x_cursor += cell_w
        rows_img.append(np.concatenate(strips, axis=1))

    row_w = rows_img[0].shape[1]
    interleaved = []
    row_offsets: list[int] = []
    y_cursor = 0
    for ri, row in enumerate(rows_img):
        if ri > 0:
            interleaved.append(np.full((gap_row, row_w, 3), 255, dtype=np.uint8))
            y_cursor += gap_row
        row_offsets.append(y_cursor)
        interleaved.append(row)
        y_cursor += row.shape[0]
    canvas = np.concatenate(interleaved, axis=0)
    for key, (x, _) in list(tile_positions.items()):
        orient, ri, ci = key
        tile_positions[key] = (x, row_offsets[ri])

    fig_w = 15.6
    canvas_aspect = canvas.shape[0] / canvas.shape[1]
    label_frac = 0.105
    header_frac = 0.140
    bottom_frac = 0.090
    cbar_frac = 0.045
    img_frac_w = 1.0 - label_frac - cbar_frac
    img_frac_h = 1.0 - header_frac - bottom_frac
    fig_h = fig_w * (canvas_aspect * img_frac_w / img_frac_h)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    ax_l = label_frac
    ax_b = bottom_frac
    ax_w = img_frac_w
    ax_h = img_frac_h
    ax = fig.add_axes((ax_l, ax_b, ax_w, ax_h))
    ax.imshow(canvas, aspect="auto", interpolation="lanczos")
    ax.axis("off")

    canvas_h, canvas_w = canvas.shape[:2]

    def _tile_bounds(orient: str, row_idx: int, col_idx: int = 0) -> tuple[float, float, float, float]:
        tile_x, tile_y = tile_positions[(orient, row_idx, col_idx)]
        left = ax_l + ax_w * (tile_x / canvas_w)
        bottom = ax_b + ax_h * (1.0 - ((tile_y + cell_h) / canvas_h))
        width = ax_w * (cell_w / canvas_w)
        height = ax_h * (cell_h / canvas_h)
        return left, bottom, width, height

    def _add_y_axis(orient: str, ylabel: str) -> None:
        y_ax = fig.add_axes(_tile_bounds(orient, 0), facecolor="none")
        y_ax.set_zorder(ax.get_zorder() + 1)
        y_ax.set_xlim(0, cell_w - 1)
        y_ax.set_ylim(cell_h - 1, 0)
        y_ax.set_xticks([])
        y_ax.set_yticks([0, cell_h // 2, cell_h - 1])
        y_ax.set_yticklabels(["0", "112", "223"], fontsize=8, fontweight="bold")
        for spine in ("top", "right", "bottom"):
            y_ax.spines[spine].set_visible(False)
        y_ax.spines["left"].set_linewidth(0.8)
        y_ax.tick_params(axis="y", length=2.5, width=0.8, pad=1.5)
        left, bottom, width, height = _tile_bounds(orient, 0)
        label_offset = 0.030 if orient == "inline" else 0.028
        fig.text(
            left - label_offset, bottom + height / 2,
            ylabel,
            ha="center", va="center",
            fontsize=9, fontweight="bold", rotation=90,
        )

    def _add_x_axis(orient: str) -> None:
        x_ax = fig.add_axes(_tile_bounds(orient, 2), facecolor="none")
        x_ax.set_zorder(ax.get_zorder() + 1)
        x_ax.set_xlim(0, cell_w - 1)
        x_ax.set_ylim(cell_h - 1, 0)
        x_ax.set_xticks([0, cell_w // 2, cell_w - 1])
        x_ax.set_xticklabels(["0", "112", "223"], fontsize=8, fontweight="bold")
        x_ax.set_yticks([])
        for spine in ("top", "right", "left"):
            x_ax.spines[spine].set_visible(False)
        x_ax.spines["bottom"].set_linewidth(0.8)
        x_ax.tick_params(axis="x", length=2.5, width=0.8, pad=1.5)

    for orient, _, ylabel, _ in groups:
        _add_y_axis(orient, ylabel)
        _add_x_axis(orient)

    # Shared row labels on the far left.
    row_centers_fig = [
        ax_b + ax_h * (1.0 - ((row_offsets[ri] + cell_h / 2) / canvas_h))
        for ri in range(n_rows)
    ]
    for ri, label in enumerate(ROW_LABELS):
        row_label_x = ax_l - (0.045 if ri == 0 else 0.010)
        fig.text(
            row_label_x, row_centers_fig[ri],
            label,
            ha="right", va="center",
            fontsize=11, fontweight="bold",
        )

    # Model column labels repeated inside each orientation panel.
    for orient, _, _, _ in groups:
        for ci, key in enumerate(col_keys):
            x, _ = tile_positions[(orient, 0, ci)]
            col_center = ax_l + ax_w * ((x + cell_w / 2) / canvas_w)
            fig.text(
                col_center, ax_b + ax_h + header_frac * 0.30,
                DISPLAY_LABELS[key].replace("\n", " "),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
            )

    # Orientation titles above each four-column group.
    for orient, _, _, _ in groups:
        first_x, _ = tile_positions[(orient, 0, 0)]
        last_x, _ = tile_positions[(orient, 0, n_cols_per_group - 1)]
        group_center = ax_l + ax_w * ((first_x + last_x + cell_w) / 2 / canvas_w)
        fig.text(
            group_center, ax_b + ax_h + header_frac * 0.76,
            ORIENTATION_TITLES[orient],
            ha="center", va="center",
            fontsize=12, fontweight="bold",
        )

    for orient, _, _, xlabel in groups:
        first_x, _ = tile_positions[(orient, 2, 0)]
        first_col_center = ax_l + ax_w * ((first_x + cell_w / 2) / canvas_w)
        fig.text(
            first_col_center, bottom_frac * 0.30,
            xlabel,
            ha="center", va="center",
            fontsize=10, fontweight="bold",
        )

    cbar_x = ax_l + ax_w + 0.012
    cbar_w_cb = 0.014
    shared_cax = fig.add_axes((cbar_x, ax_b, cbar_w_cb, ax_h))
    shared_sm = ScalarMappable(cmap="seismic", norm=Normalize(vmin=-2, vmax=2))
    shared_sm.set_array([])
    cb = fig.colorbar(shared_sm, cax=shared_cax)
    cb.set_label("Amplitude\n(z-score, +/-2 sigma)", fontsize=8, fontweight="bold")
    cb.ax.tick_params(labelsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=220, bbox_inches="tight", pad_inches=0.025,
                facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not _HAS_MPL:
        sys.exit("matplotlib is required: pip install matplotlib")
    if not _HAS_PIL:
        sys.exit("Pillow is required: pip install Pillow")

    parser = argparse.ArgumentParser(
        description="Generate F3 comparison grid with median filter column."
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=default_project_root(__file__),
    )
    parser.add_argument(
        "--orientation", choices=["inline", "timeslice"], default="inline",
        help="Slice orientation. 'timeslice' uses the training-matched horizontal "
             "inline x crossline panels from the f3_horizontal runs.",
    )
    parser.add_argument("--inline-idx", type=int, default=None,
                        help="Deprecated alias for --section-idx.")
    parser.add_argument("--section-idx", type=int, default=None,
                        help="Section index to render (default: 419 inline / 184 timeslice).")
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Force regeneration of crop cache even if it already exists.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--thesis-draft", type=str, default="draftv12",
                        help="Thesis draft folder to copy the figure into.")
    parser.add_argument(
        "--combined-orientation-comparison", action="store_true",
        help="Create the thesis-ready side-by-side matched time-slice and "
             "rotated vertical-slice F3 comparison figure.",
    )
    parser.add_argument(
        "--without-median-filter", action="store_true",
        help="When creating the combined orientation comparison, omit the "
             "Median filter columns from both orientation panels and save a "
             "separate no-median alternative figure.",
    )
    args = parser.parse_args()

    root        = args.project_root.resolve()
    summaries   = root / "experiments" / "summaries"
    f3_npy      = root / "Code" / "Dataset" / "F3" / "processed" / "f3_original.npy"
    filt_npy    = root / "Code" / "Dataset" / "F3" / "processed" / "f3_filtered_ref.npy"

    if args.combined_orientation_comparison:
        configs = {
            "timeslice": {
                "rob_root": root / "experiments" / "runs" / "robustness" / "f3_horizontal" / "main_multidata",
                "prefix": "timeslice",
                "run_subdir": "data_seed101/seed42_run01",
                "section_idx": args.section_idx if args.section_idx is not None else 184,
            },
            "inline": {
                "rob_root": root / "experiments" / "runs" / "robustness" / "f3",
                "prefix": "inline",
                "run_subdir": SEED_RUN,
                "section_idx": args.inline_idx if args.inline_idx is not None else 419,
            },
        }

        for p, label in [
            (f3_npy,   "F3 npy"),
            (filt_npy, "filtered ref npy"),
        ]:
            if not p.exists():
                sys.exit(f"{label} not found: {p}")
        for orient, cfg in configs.items():
            if not cfg["rob_root"].exists():
                sys.exit(f"{orient} robustness root not found: {cfg['rob_root']}")

        loaded: dict[str, dict[str, list[np.ndarray]]] = {}
        for orient, cfg in configs.items():
            section_idx = int(cfg["section_idx"])
            print(f"Orientation  : {orient}")
            print(f"Section index: {section_idx}")
            crops = None if args.rebuild_cache else load_cache(summaries, section_idx, orient)
            if crops is None:
                print("\nBuilding crop cache...")
                build_cache(
                    cfg["rob_root"], f3_npy, filt_npy, summaries, section_idx,
                    orientation=orient, prefix=cfg["prefix"],
                    run_subdir=cfg["run_subdir"],
                )
                crops = load_cache(summaries, section_idx, orient)
                if crops is None:
                    sys.exit(f"{orient} cache build failed - some crops are still missing.")
            else:
                cache_dir = _cache_dir(summaries, section_idx, orient)
                print(f"Loaded crops from cache: {cache_dir}")
            loaded[orient] = crops

        out_dir = (args.out_dir or summaries / "f3_robustness").resolve()
        out_name = (
            "f3_orientation_comparison_combined_no_median.png"
            if args.without_median_filter
            else "f3_orientation_comparison_combined.png"
        )
        out_path = out_dir / out_name
        print("\nAssembling combined orientation figure...")
        assemble_orientation_comparison(
            loaded["timeslice"], loaded["inline"], out_path,
            include_median=not args.without_median_filter,
        )

        thesis_fig = root / "Deliverables" / "Thesis" / args.thesis_draft / "figures"
        if thesis_fig.exists():
            dest = thesis_fig / out_path.name
            shutil.copy2(str(out_path), str(dest))
            print(f"  Copied to thesis: {dest}")
        return

    # Orientation-dependent configuration.
    if args.orientation == "timeslice":
        rob_root   = root / "experiments" / "runs" / "robustness" / "f3_horizontal" / "main_multidata"
        prefix     = "timeslice"
        run_subdir = "data_seed101/seed42_run01"
        out_name   = "f3_comparison_grid_timeslice.png"
        title      = "Denoising Results on the F3 Field Volume — Time Slice"
        xlabel     = "Crossline position"
        ylabel     = "Inline position"
        default_idx = 184
    else:
        rob_root   = root / "experiments" / "runs" / "robustness" / "f3"
        prefix     = "inline"
        run_subdir = SEED_RUN
        out_name   = "f3_comparison_grid.png"
        title      = "Denoising Results on the F3 Field Volume"
        xlabel     = "Crossline position"
        ylabel     = "Time sample"
        default_idx = 419

    section_idx = args.section_idx if args.section_idx is not None else (
        args.inline_idx if args.inline_idx is not None else default_idx
    )
    out_dir = (args.out_dir or summaries / "f3_robustness").resolve()

    for p, label in [
        (rob_root, "robustness root"),
        (f3_npy,   "F3 npy"),
        (filt_npy, "filtered ref npy"),
    ]:
        if not p.exists():
            sys.exit(f"{label} not found: {p}")

    print(f"Orientation  : {args.orientation}")
    print(f"Section index: {section_idx}")

    # --- Cache phase ---
    crops = None if args.rebuild_cache else load_cache(summaries, section_idx, args.orientation)
    if crops is None:
        print("\nBuilding crop cache...")
        build_cache(rob_root, f3_npy, filt_npy, summaries, section_idx,
                    orientation=args.orientation, prefix=prefix, run_subdir=run_subdir)
        crops = load_cache(summaries, section_idx, args.orientation)
        if crops is None:
            sys.exit("Cache build failed — some crops are still missing.")
    else:
        cache_dir = _cache_dir(summaries, section_idx, args.orientation)
        print(f"Loaded crops from cache: {cache_dir}")

    # --- Figure phase ---
    print("\nAssembling figure...")
    out_path = out_dir / out_name
    assemble_figure(crops, out_path, title=title, xlabel=xlabel, ylabel=ylabel)

    # Copy to thesis draft
    thesis_fig = root / "Deliverables" / "Thesis" / args.thesis_draft / "figures"
    if thesis_fig.exists():
        dest = thesis_fig / out_name
        shutil.copy2(str(out_path), str(dest))
        print(f"  Copied to thesis: {dest}")


if __name__ == "__main__":
    main()
