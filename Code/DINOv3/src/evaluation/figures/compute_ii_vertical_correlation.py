"""Centre-to-neighbour Pearson correlation for Image Impeccable VERTICAL slices.

Companion to ``plot_neighbour_correlation_orientation.py``, which computes the
horizontal (time-slice / argmax-axis) Image Impeccable curve and the two F3
orientation curves. This script computes the *vertical* (inline/crossline)
Image Impeccable curve, defined relative to the longest (time) axis exactly as
the training data loader does (see ``slice_orientation='vertical'`` in
``data/image_impeccable.py``).

It uses the same crop and z-score conventions as the existing figure script:
centre crop, per-crop z-score, Pearson over the flattened 224x224 crop. It reads
the noisy volumes (the model's input), matching the existing horizontal curve so
the two are directly comparable.

Run locally (data is under Code/Dataset/...):

    python Code/DINOv3/src/evaluation/figures/compute_ii_vertical_correlation.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

OFFSETS = (-2, -1, 0, 1, 2)


def _zscore(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    std = float(x64.std())
    if std > 1e-8:
        return (x64 - float(x64.mean())) / std
    return x64 - float(x64.mean())


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.sqrt(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _section(vol: np.ndarray, axis: int, t: int) -> np.ndarray:
    """Return the full 2D section at index ``t`` along ``axis``."""
    if axis == 0:
        return vol[t, :, :]
    if axis == 1:
        return vol[:, t, :]
    return vol[:, :, t]


def _accumulate_axis(
    vol: np.ndarray,
    axis: int,
    *,
    crop_size: int,
    slice_stride: int,
    context_radius: int,
    values_by_offset: dict[int, list[float]],
) -> None:
    n_slices = int(vol.shape[axis])
    spatial_dims = [int(vol.shape[i]) for i in range(3) if i != axis]
    if spatial_dims[0] < crop_size or spatial_dims[1] < crop_size:
        return
    oh = (spatial_dims[0] - crop_size) // 2
    ow = (spatial_dims[1] - crop_size) // 2
    for t in range(context_radius, n_slices - context_radius, slice_stride):
        center = _section(vol, axis, t)[oh : oh + crop_size, ow : ow + crop_size]
        center_z = _zscore(center.astype(np.float32))
        for offset in OFFSETS:
            if offset == 0:
                values_by_offset[offset].append(1.0)
                continue
            nb = _section(vol, axis, t + offset)[oh : oh + crop_size, ow : ow + crop_size]
            values_by_offset[offset].append(_pearson_flat(center_z, _zscore(nb.astype(np.float32))))


def _summary_rows(label: str, values_by_offset: dict[int, list[float]]) -> list[dict[str, object]]:
    n_stacks = len(values_by_offset[0])
    rows: list[dict[str, object]] = []
    for offset in OFFSETS:
        vals = np.asarray(values_by_offset[offset], dtype=np.float64)
        n = int(vals.size)
        mean = float(vals.mean()) if n else 0.0
        std = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = float(std / np.sqrt(n)) if n > 1 else 0.0
        rows.append(
            {"dataset": label, "offset": offset, "n_stacks": n_stacks,
             "mean": mean, "std": std, "sem": sem}
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Override Image Impeccable extracted/ dir (default: Code/Dataset/...).")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--slice-stride", type=int, default=5)
    parser.add_argument("--context-radius", type=int, default=2)
    parser.add_argument("--output-csv", type=Path,
                        default=Path("experiments/summaries/f3_orientation_investigation/"
                                     "ii_vertical_neighbour_correlation.csv"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    image_root = (args.image_root if args.image_root is not None
                  else root / "Code" / "Dataset" / "ThinkOnwards" / "training_data" / "extracted")
    out_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv

    noisy_files = sorted(Path(image_root).rglob("seismic_w_noise_vol_*.npy"))
    if not noisy_files:
        raise FileNotFoundError(f"No Image Impeccable noisy volumes found under {image_root}")

    # Accumulators: inline (first non-time axis), crossline (second), and pooled.
    inline_vals: dict[int, list[float]] = {o: [] for o in OFFSETS}
    crossline_vals: dict[int, list[float]] = {o: [] for o in OFFSETS}

    for path in noisy_files:
        vol = np.load(path, allow_pickle=False)  # full load: vertical slicing is strided
        time_axis = int(np.argmax(vol.shape))
        non_time = [i for i in range(3) if i != time_axis]
        inline_axis, crossline_axis = min(non_time), max(non_time)
        _accumulate_axis(vol, inline_axis, crop_size=args.crop_size,
                         slice_stride=args.slice_stride, context_radius=args.context_radius,
                         values_by_offset=inline_vals)
        _accumulate_axis(vol, crossline_axis, crop_size=args.crop_size,
                         slice_stride=args.slice_stride, context_radius=args.context_radius,
                         values_by_offset=crossline_vals)
        del vol

    pooled_vals = {o: inline_vals[o] + crossline_vals[o] for o in OFFSETS}

    all_rows: list[dict[str, object]] = []
    all_rows += _summary_rows("Image Impeccable vertical", pooled_vals)
    all_rows += _summary_rows("Image Impeccable inline", inline_vals)
    all_rows += _summary_rows("Image Impeccable crossline", crossline_vals)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "offset", "n_stacks", "mean", "std", "sem"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote CSV: {out_csv}")
    for label in ("Image Impeccable vertical", "Image Impeccable inline", "Image Impeccable crossline"):
        rows = [r for r in all_rows if r["dataset"] == label]
        print(f"\n{label} (n={rows[0]['n_stacks']})")
        for r in rows:
            print(f"  offset {int(r['offset']):+d}: mean={float(r['mean']):.6f}, "
                  f"std={float(r['std']):.6f}, sem={float(r['sem']):.6f}")


if __name__ == "__main__":
    main()
