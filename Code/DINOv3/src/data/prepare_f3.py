"""Prepare Netherlands F3 Demo seismic volume for robustness evaluation.

Converts a SEG-Y export (or an existing NumPy file) into a canonical
float32 .npy array plus a metadata JSON.

Usage:
    python data/prepare_f3.py \
        --input /path/to/f3.segy \
        --output Code/Dataset/F3/processed/

    python data/prepare_f3.py \
        --input /path/to/f3_volume.npy \
        --output Code/Dataset/F3/processed/

The output directory will contain:
    f3_original.npy   — float32 array, shape (n_inlines, n_xlines, n_samples)
    f3_meta.json      — shape, axis names, source file, preprocessing notes

Source: TerraNubis F3 Demo 2023 (https://terranubis.com/datainfo/F3-Demo-2023)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_segy(path: Path) -> tuple[np.ndarray, dict]:
    try:
        import segyio
    except ImportError:
        sys.exit(
            "segyio is not installed. Install it with:\n"
            "  pip install segyio\n"
            "Or export the F3 volume from OpendTect as a .npy file instead."
        )

    print(f"Reading SEG-Y: {path}")

    # Try strict geometry first; fall back to ignore_geometry for irregular surveys.
    try:
        vol, meta = _load_segy_strict(path, segyio)
    except ValueError as e:
        print(f"  Strict geometry failed ({e}); retrying with ignore_geometry=True ...")
        vol, meta = _load_segy_flat(path, segyio)

    return vol, meta


def _load_segy_strict(path: Path, segyio) -> tuple[np.ndarray, dict]:  # type: ignore[no-untyped-def]
    with segyio.open(str(path), ignore_geometry=False) as f:
        f.mmap()
        n_inlines = len(f.ilines)
        n_xlines = len(f.xlines)
        n_samples = f.samples.size
        sample_interval_us = int(f.bin[segyio.BinField.Interval])

        print(f"  Inlines: {n_inlines}, Crosslines: {n_xlines}, Samples: {n_samples}")
        print(f"  Sample interval: {sample_interval_us} µs")

        vol = np.zeros((n_inlines, n_xlines, n_samples), dtype=np.float32)
        for i, inline in enumerate(f.ilines):
            vol[i] = f.iline[inline].astype(np.float32)
            if (i + 1) % 50 == 0:
                print(f"  Reading inline {i + 1}/{n_inlines}", end="\r", flush=True)
        print()

    meta = {
        "source_file": str(path),
        "format": "segy",
        "shape": list(vol.shape),
        "axis_names": ["inline", "crossline", "samples"],
        "sample_interval_us": sample_interval_us,
        "sample_rate_hz": int(1_000_000 / sample_interval_us) if sample_interval_us > 0 else None,
    }
    return vol, meta


def _load_segy_flat(path: Path, segyio) -> tuple[np.ndarray, dict]:  # type: ignore[no-untyped-def]
    """Read trace-by-trace when the survey geometry is irregular."""
    with segyio.open(str(path), ignore_geometry=True) as f:
        f.mmap()
        n_traces = f.tracecount
        n_samples = len(f.samples)
        sample_interval_us = int(f.bin[segyio.BinField.Interval])

        print(f"  Total traces: {n_traces}, Samples per trace: {n_samples}")
        print(f"  Sample interval: {sample_interval_us} µs")
        print("  Reading inline/crossline headers ...")

        ilines_hdr = f.attributes(segyio.TraceField.INLINE_3D)[:]
        xlines_hdr = f.attributes(segyio.TraceField.CROSSLINE_3D)[:]

        unique_ilines = np.unique(ilines_hdr)
        unique_xlines = np.unique(xlines_hdr)
        n_il = len(unique_ilines)
        n_xl = len(unique_xlines)
        print(f"  Unique inlines: {n_il}, unique crosslines: {n_xl}")

        il_idx = {int(il): i for i, il in enumerate(unique_ilines)}
        xl_idx = {int(xl): j for j, xl in enumerate(unique_xlines)}

        vol = np.zeros((n_il, n_xl, n_samples), dtype=np.float32)
        print(f"  Reading {n_traces} traces (this may take a few minutes) ...")
        for k in range(n_traces):
            i = il_idx.get(int(ilines_hdr[k]), -1)
            j = xl_idx.get(int(xlines_hdr[k]), -1)
            if i >= 0 and j >= 0:
                vol[i, j] = f.trace[k].astype(np.float32)
            if (k + 1) % 50_000 == 0:
                print(f"  {k + 1}/{n_traces} traces read", end="\r", flush=True)
        print()

    meta = {
        "source_file": str(path),
        "format": "segy_flat",
        "shape": list(vol.shape),
        "axis_names": ["inline", "crossline", "samples"],
        "sample_interval_us": sample_interval_us,
        "sample_rate_hz": int(1_000_000 / sample_interval_us) if sample_interval_us > 0 else None,
        "note": "Loaded with ignore_geometry=True due to irregular trace count.",
    }
    return vol, meta


def _load_numpy(path: Path) -> tuple[np.ndarray, dict]:
    print(f"Loading NumPy volume: {path}")
    vol = np.load(str(path), allow_pickle=False).astype(np.float32)
    print(f"  Shape: {vol.shape}, dtype: {vol.dtype}")
    if vol.ndim != 3:
        sys.exit(f"Expected 3D volume, got shape {vol.shape}.")
    meta = {
        "source_file": str(path),
        "format": "numpy",
        "shape": list(vol.shape),
        "axis_names": ["inline", "crossline", "samples"],
        "sample_interval_us": None,
        "sample_rate_hz": None,
    }
    return vol, meta


def clip_outliers(vol: np.ndarray, n_sigma: float = 5.0) -> tuple[np.ndarray, dict]:
    """Clip amplitude outliers at ±n_sigma of the global distribution."""
    mu = float(vol.mean())
    sigma = float(vol.std())
    lo = mu - n_sigma * sigma
    hi = mu + n_sigma * sigma
    n_clipped = int(((vol < lo) | (vol > hi)).sum())
    vol = np.clip(vol, lo, hi).astype(np.float32)
    print(f"  Clipped {n_clipped} values at ±{n_sigma}σ  [{lo:.4f}, {hi:.4f}]")
    return vol, {"clip_lo": lo, "clip_hi": hi, "n_clipped": n_clipped, "clip_sigma": n_sigma}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare F3 volume for robustness evaluation.")
    parser.add_argument("--input", required=True, help="SEG-Y or .npy source file.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--no-clip", action="store_true", help="Skip outlier clipping.")
    parser.add_argument("--sigma", type=float, default=5.0, help="Clipping threshold in σ (default 5).")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        sys.exit(f"Input file not found: {src}")

    suffix = src.suffix.lower()
    if suffix in (".sgy", ".segy"):
        vol, meta = _load_segy(src)
    elif suffix == ".npy":
        vol, meta = _load_numpy(src)
    else:
        sys.exit(f"Unsupported file format: {suffix!r}. Provide a .segy or .npy file.")

    print(f"Volume shape: {vol.shape}, global range: [{vol.min():.4f}, {vol.max():.4f}]")

    clip_info: dict = {}
    if not args.no_clip:
        print("Clipping outliers...")
        vol, clip_info = clip_outliers(vol, args.sigma)

    out_npy = out_dir / "f3_original.npy"
    print(f"Saving {out_npy} ...")
    np.save(str(out_npy), vol)
    print(f"  Saved: {out_npy} ({out_npy.stat().st_size / 1e9:.2f} GB)")

    meta.update(clip_info)
    meta["preprocessing"] = (
        "Amplitude outliers clipped at ±{sigma}σ. "
        "No per-slice z-score applied here — normalization happens at load time "
        "in F3FieldDataset."
    ).format(sigma=args.sigma if not args.no_clip else "N/A (clipping skipped)")

    out_meta = out_dir / "f3_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {out_meta}")
    print("Done.")


if __name__ == "__main__":
    main()
