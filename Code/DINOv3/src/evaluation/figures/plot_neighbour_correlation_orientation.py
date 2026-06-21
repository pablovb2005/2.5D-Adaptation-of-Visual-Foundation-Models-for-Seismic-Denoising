"""Plot centre-to-neighbour Pearson correlations for Image Impeccable and F3.

This is a thesis figure helper for the F3 orientation discussion. It follows the
main data loaders' crop and z-score conventions, but computes correlations
directly from the prepared numpy volumes to avoid model/checkpoint dependencies.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OFFSETS = (-2, -1, 0, 1, 2)


@dataclass(frozen=True)
class CurveStats:
    label: str
    n_stacks: int
    values_by_offset: dict[int, np.ndarray]

    def rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for offset in OFFSETS:
            vals = self.values_by_offset[offset].astype(np.float64)
            n = int(vals.size)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            sem = float(std / np.sqrt(n)) if n > 1 else 0.0
            rows.append(
                {
                    "dataset": self.label,
                    "offset": offset,
                    "n_stacks": self.n_stacks,
                    "mean": mean,
                    "std": std,
                    "sem": sem,
                }
            )
        return rows


@dataclass(frozen=True)
class CurveSummary:
    label: str
    n_stacks: int
    summary_rows: list[dict[str, float | int | str]]

    def rows(self) -> list[dict[str, float | int | str]]:
        return self.summary_rows


def _zscore(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    std = float(x64.std())
    if std > 1e-8:
        return (x64 - float(x64.mean())) / std
    return x64 - float(x64.mean())


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation after flattening both 224 x 224 crops."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _crop_volume_slice(vol: np.ndarray, axis: int, t: int, oh: int, ow: int, crop_size: int) -> np.ndarray:
    if axis == 0:
        return vol[t, oh : oh + crop_size, ow : ow + crop_size]
    if axis == 1:
        return vol[oh : oh + crop_size, t, ow : ow + crop_size]
    if axis == 2:
        return vol[oh : oh + crop_size, ow : ow + crop_size, t]
    raise ValueError(f"Invalid slice axis: {axis}")


def _center_crop(arr: np.ndarray, crop_size: int) -> tuple[np.ndarray, int, int]:
    h, w = arr.shape
    oh = max(0, (h - crop_size) // 2)
    ow = max(0, (w - crop_size) // 2)
    return arr[oh : oh + crop_size, ow : ow + crop_size], oh, ow


def _append_correlations(
    values_by_offset: dict[int, list[float]],
    center: np.ndarray,
    neighbours: dict[int, np.ndarray],
) -> None:
    center_z = _zscore(center)
    for offset in OFFSETS:
        if offset == 0:
            values_by_offset[offset].append(1.0)
        else:
            values_by_offset[offset].append(_pearson_flat(center_z, _zscore(neighbours[offset])))


def compute_image_impeccable(
    root: Path,
    *,
    crop_size: int,
    slice_stride: int,
    context_radius: int,
) -> CurveStats:
    noisy_files = sorted(root.rglob("seismic_w_noise_vol_*.npy"))
    if not noisy_files:
        raise FileNotFoundError(f"No Image Impeccable noisy volumes found under {root}")

    values_by_offset: dict[int, list[float]] = {offset: [] for offset in OFFSETS}
    for noisy_path in noisy_files:
        vol = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
        axis = int(np.argmax(vol.shape))
        n_slices = int(vol.shape[axis])
        spatial_dims = [vol.shape[i] for i in range(3) if i != axis]
        if spatial_dims[0] < crop_size or spatial_dims[1] < crop_size:
            continue
        oh = (spatial_dims[0] - crop_size) // 2
        ow = (spatial_dims[1] - crop_size) // 2
        for t in range(context_radius, n_slices - context_radius, slice_stride):
            center = _crop_volume_slice(vol, axis, t, oh, ow, crop_size).astype(np.float32)
            neighbours = {
                offset: _crop_volume_slice(vol, axis, t + offset, oh, ow, crop_size).astype(np.float32)
                for offset in OFFSETS
                if offset != 0
            }
            _append_correlations(values_by_offset, center, neighbours)

    arrays = {offset: np.asarray(vals, dtype=np.float64) for offset, vals in values_by_offset.items()}
    n_stacks = int(arrays[0].size)
    return CurveStats("Image Impeccable", n_stacks, arrays)


def _f3_section_getter(vol: np.ndarray, orientation: str):
    if orientation == "inline":
        return lambda i: vol[i, :, :].astype(np.float32)
    if orientation == "crossline":
        return lambda i: vol[:, i, :].astype(np.float32)
    if orientation == "timeslice":
        return lambda i: vol[:, :, i].astype(np.float32)
    raise ValueError(f"Unsupported F3 orientation: {orientation}")


def compute_f3(
    npy_path: Path,
    *,
    label: str,
    orientations: tuple[str, ...],
    crop_size: int,
    context_radius: int,
    section_min: int | None = None,
    section_max: int | None = None,
) -> CurveStats:
    vol = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    n_by_orientation = {
        "inline": int(vol.shape[0]),
        "crossline": int(vol.shape[1]),
        "timeslice": int(vol.shape[2]),
    }
    values_by_offset: dict[int, list[float]] = {offset: [] for offset in OFFSETS}

    for orientation in orientations:
        get_section = _f3_section_getter(vol, orientation)
        lo = section_min if orientation == "timeslice" else None
        hi = section_max if orientation == "timeslice" else None
        start = max(context_radius, lo if lo is not None else 0)
        stop = min(n_by_orientation[orientation] - context_radius, hi if hi is not None else n_by_orientation[orientation])
        for sec_idx in range(start, stop):
            base = get_section(sec_idx)
            center, oh, ow = _center_crop(base, crop_size)
            neighbours: dict[int, np.ndarray] = {}
            for offset in OFFSETS:
                if offset == 0:
                    continue
                section = get_section(sec_idx + offset)
                neighbours[offset] = section[oh : oh + crop_size, ow : ow + crop_size]
            _append_correlations(values_by_offset, center, neighbours)

    arrays = {offset: np.asarray(vals, dtype=np.float64) for offset, vals in values_by_offset.items()}
    n_stacks = int(arrays[0].size)
    return CurveStats(label, n_stacks, arrays)


def write_csv(curves: list[CurveStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "offset", "n_stacks", "mean", "std", "sem"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for curve in curves:
            for row in curve.rows():
                writer.writerow(row)


def read_csv(in_path: Path) -> list[CurveSummary]:
    if not in_path.exists():
        raise FileNotFoundError(f"Correlation CSV not found: {in_path}")

    rows_by_label: dict[str, list[dict[str, float | int | str]]] = {}
    n_by_label: dict[str, int] = {}
    with in_path.open(newline="") as f:
        for row in csv.DictReader(f):
            label = str(row["dataset"])
            parsed = {
                "dataset": label,
                "offset": int(row["offset"]),
                "n_stacks": int(row["n_stacks"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "sem": float(row["sem"]),
            }
            rows_by_label.setdefault(label, []).append(parsed)
            n_by_label[label] = int(parsed["n_stacks"])

    curves: list[CurveSummary] = []
    for label in ("Image Impeccable", "F3 time slice", "F3 vertical slice"):
        rows = rows_by_label.get(label)
        if rows is None:
            raise ValueError(f"Missing curve {label!r} in {in_path}")
        rows = sorted(rows, key=lambda row: int(row["offset"]))
        offsets = [int(row["offset"]) for row in rows]
        if offsets != list(OFFSETS):
            raise ValueError(f"Unexpected offsets for {label!r}: {offsets}")
        curves.append(CurveSummary(label, n_by_label[label], rows))
    return curves


def plot_curves(curves: list[CurveStats] | list[CurveSummary], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.weight": "bold",
            "axes.labelsize": 8,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )
    styles = {
        "Image Impeccable": {"marker": "o", "linestyle": "-", "color": "#1f77b4"},
        "F3 time slice": {"marker": "s", "linestyle": "--", "color": "#d62728"},
        "F3 vertical slice": {"marker": "^", "linestyle": "-.", "color": "#2ca02c"},
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.85))
    x = np.asarray(OFFSETS, dtype=int)
    min_mean = 0.0
    for curve in curves:
        rows = curve.rows()
        means = np.asarray([float(row["mean"]) for row in rows])
        min_mean = min(min_mean, float(np.min(means)))
        ax.plot(
            x,
            means,
            linewidth=1.25,
            markersize=4.0,
            label=curve.label,
            **styles[curve.label],
        )

    ax.axhline(0.0, color="0.25", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Neighbour offset")
    ax.set_ylabel("Pearson correlation to centre slice")
    ax.set_xticks(x)
    ax.set_xticklabels(["-2", "-1", "0", "+1", "+2"])
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
    y_min = min(-0.5, min_mean - 0.05)
    ax.set_ylim(y_min, 1.05)
    ax.grid(True, axis="y", linewidth=0.4, color="0.86")
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02), frameon=True, framealpha=0.92)
    for text in legend.get_texts():
        text.set_fontweight("bold")
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.20, top=0.94)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--slice-stride", type=int, default=5)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--f3-section-min", type=int, default=50)
    parser.add_argument("--f3-section-max", type=int, default=None)
    parser.add_argument(
        "--recompute-data",
        action="store_true",
        help="Recompute the correlation CSV from the numpy volumes before plotting.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only regenerate the PNG from an existing correlation CSV.",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Only recompute the correlation CSV; do not regenerate the PNG.",
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("Deliverables/Thesis/draftv12/figures/f3_neighbour_correlation_orientation.png"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("experiments/summaries/f3_orientation_investigation/f3_neighbour_correlation_orientation.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only and args.compute_only:
        raise ValueError("--plot-only and --compute-only cannot be used together")
    root = args.project_root.resolve()
    image_root = root / "Code" / "Dataset" / "ThinkOnwards" / "training_data" / "extracted"
    f3_npy = root / "Code" / "Dataset" / "F3" / "processed" / "f3_original.npy"
    out_figure = args.output_figure if args.output_figure.is_absolute() else root / args.output_figure
    out_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv

    should_compute = args.recompute_data or args.compute_only or (not out_csv.exists() and not args.plot_only)
    if should_compute:
        curves = [
            compute_image_impeccable(
                image_root,
                crop_size=args.crop_size,
                slice_stride=args.slice_stride,
                context_radius=args.context_radius,
            ),
            compute_f3(
                f3_npy,
                label="F3 time slice",
                orientations=("timeslice",),
                crop_size=args.crop_size,
                context_radius=args.context_radius,
                section_min=args.f3_section_min,
                section_max=args.f3_section_max,
            ),
            compute_f3(
                f3_npy,
                label="F3 vertical slice",
                orientations=("inline", "crossline"),
                crop_size=args.crop_size,
                context_radius=args.context_radius,
            ),
        ]
        write_csv(curves, out_csv)
    else:
        curves = read_csv(out_csv)

    if not args.compute_only:
        if should_compute:
            curves = read_csv(out_csv)
        plot_curves(curves, out_figure)

    if not args.compute_only:
        print(f"Wrote figure: {out_figure}")
    if should_compute:
        print(f"Wrote CSV:    {out_csv}")
    else:
        print(f"Read CSV:     {out_csv}")
    for curve in curves:
        print(f"\n{curve.label} (n={curve.n_stacks})")
        for row in curve.rows():
            print(
                f"  offset {int(row['offset']):+d}: "
                f"mean={float(row['mean']):.6f}, "
                f"std={float(row['std']):.6f}, "
                f"sem={float(row['sem']):.6f}"
            )


if __name__ == "__main__":
    main()
