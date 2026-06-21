"""Prepare an algorithmically filtered F3 reference volume for filtered-reference evaluation.

Converts a SEG-Y export (or an existing NumPy file) of the dip-steered median-filter
volume into a canonical float32 .npy array plus a metadata JSON, aligned to the
same geometry as the existing f3_original.npy.

Usage:
    python data/prepare_filtered_ref.py \
        --input /path/to/f3_filtered_ref.segy \
        --reference-npy Code/Dataset/F3/processed/f3_original.npy \
        --output Code/Dataset/F3/processed/

    python data/prepare_filtered_ref.py \
        --input /path/to/f3_filtered_ref.npy \
        --reference-npy Code/Dataset/F3/processed/f3_original.npy \
        --output Code/Dataset/F3/processed/

The output directory will contain:
    f3_filtered_ref.npy   — float32 array, shape matching f3_original.npy
    f3_filtered_ref_meta.json

Source: F3 Demo 2023 OpendTect project (https://terranubis.com/datainfo/F3-Demo-2023)
Recommended volume: 4_Dip_steered_median_filter.cbvs exported as SEG-Y from OpendTect.

CBVS format note:
    CBVS is OpendTect's native binary format and cannot be read without OpendTect or
    its Python bindings (odpy). Export the volume as SEG-Y from OpendTect first:
      Survey > Export > Seismic > To SEG-Y
    then pass the exported .segy file to this script.
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
            "Or export the volume from OpendTect as a .npy file instead."
        )

    print(f"Reading SEG-Y: {path}")

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
        print(f"  Reading {n_traces} traces ...")
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
    mu = float(vol.mean())
    sigma = float(vol.std())
    lo = mu - n_sigma * sigma
    hi = mu + n_sigma * sigma
    n_clipped = int(((vol < lo) | (vol > hi)).sum())
    vol = np.clip(vol, lo, hi).astype(np.float32)
    print(f"  Clipped {n_clipped} values at +/-{n_sigma}sigma  [{lo:.4f}, {hi:.4f}]")
    return vol, {"clip_lo": lo, "clip_hi": hi, "n_clipped": n_clipped, "clip_sigma": n_sigma}


def _verify_geometry(vol: np.ndarray, ref_npy: Path) -> np.ndarray:
    """Assert the filtered volume has the same spatial shape as f3_original.npy.

    If the filtered volume has exactly one extra sample at the end of the time
    axis (common SEG-Y export off-by-one), the array is trimmed in-place and a
    warning is printed.  Any other mismatch is a fatal error.

    Returns the (possibly trimmed) volume.
    """
    if not ref_npy.exists():
        print(f"  Reference npy not found at {ref_npy}; skipping geometry check.")
        return vol
    ref_vol = np.load(str(ref_npy), mmap_mode="r", allow_pickle=False)
    if vol.shape == ref_vol.shape:
        print(f"  Geometry check passed: shape matches f3_original.npy {ref_vol.shape}")
        return vol

    # Allow a one-sample trailing difference on the time axis only.
    ni_ok = vol.shape[0] == ref_vol.shape[0]
    nx_ok = vol.shape[1] == ref_vol.shape[1]
    ns_diff = vol.shape[2] - ref_vol.shape[2]
    if ni_ok and nx_ok and ns_diff == 1:
        vol = vol[:, :, : ref_vol.shape[2]]
        print(
            f"  WARNING: filtered reference had {ref_vol.shape[2] + 1} samples; "
            f"trimmed to {ref_vol.shape[2]} to match f3_original.npy."
        )
        return vol

    sys.exit(
        f"Shape mismatch: filtered reference is {vol.shape} but "
        f"f3_original.npy is {ref_vol.shape}. Ensure both volumes cover the "
        "same inline/crossline/sample extent before preparing the filtered reference."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare F3 filtered-reference volume for filtered-reference evaluation."
    )
    parser.add_argument("--input", required=True, help="SEG-Y or .npy source file.")
    parser.add_argument(
        "--reference-npy",
        default=None,
        help="Path to existing f3_original.npy to verify shape alignment (recommended).",
    )
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
    elif suffix == ".cbvs":
        sys.exit(
            "CBVS format is not directly supported.\n"
            "Export the volume as SEG-Y from OpendTect:\n"
            "  Survey > Export > Seismic > To SEG-Y\n"
            "Then pass the exported .segy file to this script."
        )
    else:
        sys.exit(f"Unsupported file format: {suffix!r}. Provide a .segy or .npy file.")

    print(f"Volume shape: {vol.shape}, global range: [{vol.min():.4f}, {vol.max():.4f}]")

    if args.reference_npy is not None:
        print("Verifying geometry alignment ...")
        vol = _verify_geometry(vol, Path(args.reference_npy).resolve())

    # Keep metadata aligned with the saved array after any geometry repair.
    meta["shape"] = list(vol.shape)

    clip_info: dict = {}
    if not args.no_clip:
        print("Clipping outliers ...")
        vol, clip_info = clip_outliers(vol, args.sigma)

    out_npy = out_dir / "f3_filtered_ref.npy"
    print(f"Saving {out_npy} ...")
    np.save(str(out_npy), vol)
    print(f"  Saved: {out_npy} ({out_npy.stat().st_size / 1e9:.2f} GB)")

    meta.update(clip_info)
    meta["volume_type"] = "filtered_reference"
    meta["filter_description"] = (
        "Dip-steered median filter (F3 Demo 2023, OpendTect). "
        "This is a pseudo-clean reference, not clean ground truth."
    )
    meta["preprocessing"] = (
        "Amplitude outliers clipped at ±{sigma}σ. "
        "No per-slice z-score applied here — normalization happens at load time."
    ).format(sigma=args.sigma if not args.no_clip else "N/A (clipping skipped)")

    out_meta = out_dir / "f3_filtered_ref_meta.json"
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {out_meta}")
    print("Done.")


if __name__ == "__main__":
    main()
