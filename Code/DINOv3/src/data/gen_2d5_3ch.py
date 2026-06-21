"""Generate the 2.5D-3ch training dataset from a 3D SEGY volume.

Background
----------
The 2D baseline training data (seismic/ + label/) comes from Sheng et al.'s
public SFM dataset — 2D synthetic noisy/clean pairs with no 3D source volume.
True 3D neighbours cannot be recovered from those files.

The only 3D volume available is filt_mig.sgy (Teapot Dome, filtered/clean).
This script uses it as the clean reference and adds synthetic Gaussian noise
(varying SNR) to produce matched noisy inputs — the same noise model used by
the SFM dataset (Sheng et al., 2025, Eq. 1: X_obs = X_clean + eta).

For each inline slice t (boundaries skipped):
  - noisy input:  stack [t-1, t, t+1] crops from the noise-added volume
  - clean label:  crop from the original filtered volume at slice t

Output layout (mirrors the 2D dataset convention):
  <out_dir>/seismic_3ch/0.dat, 1.dat, ...   # 3-channel, flat [3*H*W] float32
  <out_dir>/label_3ch/0.dat,   1.dat, ...   # single-channel, flat [H*W] float32

Usage
-----
    python data/gen_2d5_3ch.py \
        --segy /path/to/filt_mig.sgy \
        --out_dir /path/to/Dataset/Denoise \
        [--crop_size 224] [--stride 7] \
        [--snr_min 1.0] [--snr_max 5.0] [--seed 42]

    # If you have a separate noisy SEGY (ask Jiahua):
    python data/gen_2d5_3ch.py \
        --segy /path/to/filt_mig.sgy \
        --noisy_segy /path/to/raw.sgy \
        --out_dir /path/to/Dataset/Denoise

Notes
-----
- First 150 time columns are skipped (zero-padded region in filt_mig.sgy),
  matching the pre-processing in Jiahua's data_gen.py.
- Boundary inlines (first and last) are skipped — no t-1 or t+1 neighbour.
- Crops that do not fit fully within the slice are skipped.
- Install segyio before running: pip install segyio
"""

import argparse
import numpy as np
from pathlib import Path

try:
    import segyio
except ImportError:
    raise ImportError("segyio is required: pip install segyio")

SKIP_COLS = 150  # zero-padded columns in filt_mig.sgy (from data_gen.py)


def load_volume(segy_path: Path) -> np.ndarray:
    """Load all inline slices → [n_inlines, n_crosslines, n_samples] float32."""
    print(f"  Loading {segy_path.name} ...")
    with segyio.open(str(segy_path), ignore_geometry=False) as f:
        f.mmap()
        vol = segyio.tools.cube(f)  # [n_inlines, n_crosslines, n_samples]
    print(f"  Shape: {vol.shape}, dtype: {vol.dtype}")
    return vol.astype(np.float32)


def add_gaussian_noise(clean: np.ndarray, snr: float) -> np.ndarray:
    """Add Gaussian noise at a given signal-to-noise ratio."""
    signal_power = np.mean(clean ** 2)
    noise_std = np.sqrt(signal_power / snr)
    noise = np.random.randn(*clean.shape).astype(np.float32) * noise_std
    return clean + noise


def generate(
    clean_vol: np.ndarray,
    noisy_vol: np.ndarray,
    out_dir: Path,
    crop_size: int,
    stride: int,
):
    n_inlines = clean_vol.shape[0]
    seismic_out = out_dir / "seismic_3ch"
    label_out = out_dir / "label_3ch"
    seismic_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    for t in range(1, n_inlines - 1):
        # Each inline slice: [n_crosslines, n_samples], skip first SKIP_COLS time columns
        prev_noisy = noisy_vol[t - 1, :, SKIP_COLS:]
        curr_noisy = noisy_vol[t,     :, SKIP_COLS:]
        next_noisy = noisy_vol[t + 1, :, SKIP_COLS:]
        curr_clean = clean_vol[t,     :, SKIP_COLS:]

        H, W = curr_noisy.shape
        r = 0
        while r + crop_size <= H:
            c = 0
            while c + crop_size <= W:
                triplet = np.stack([
                    prev_noisy[r:r + crop_size, c:c + crop_size],
                    curr_noisy[r:r + crop_size, c:c + crop_size],
                    next_noisy[r:r + crop_size, c:c + crop_size],
                ], axis=0).astype(np.float32)  # [3, crop_size, crop_size]

                label = curr_clean[r:r + crop_size, c:c + crop_size].astype(np.float32)

                triplet.tofile(seismic_out / f"{sample_idx}.dat")
                label.tofile(label_out / f"{sample_idx}.dat")

                sample_idx += 1
                c += stride
            r += stride

        if (t % 50 == 0) or (t == n_inlines - 2):
            print(f"  Inline {t}/{n_inlines - 2} — {sample_idx} samples so far")

    print(f"\nDone. {sample_idx} samples written.")
    print(f"  seismic_3ch/ → {seismic_out}")
    print(f"  label_3ch/   → {label_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2.5D-3ch seismic dataset from a 3D SEGY volume."
    )
    parser.add_argument("--segy", required=True, type=Path,
                        help="Path to the clean/filtered 3D SEGY (filt_mig.sgy).")
    parser.add_argument("--noisy_segy", type=Path, default=None,
                        help="Optional: separate noisy 3D SEGY. If omitted, synthetic "
                             "Gaussian noise is added to --segy to create the noisy input.")
    parser.add_argument("--out_dir", required=True, type=Path,
                        help="Output directory (seismic_3ch/ and label_3ch/ created here).")
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=7,
                        help="Sliding window stride (default 7).")
    parser.add_argument("--snr_min", type=float, default=1.0,
                        help="Min signal-to-noise ratio for synthetic noise (default 1.0).")
    parser.add_argument("--snr_max", type=float, default=5.0,
                        help="Max signal-to-noise ratio for synthetic noise (default 5.0).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("=== Loading volumes ===")
    clean_vol = load_volume(args.segy)

    if args.noisy_segy is not None:
        print("Using separate noisy SEGY.")
        noisy_vol = load_volume(args.noisy_segy)
        assert noisy_vol.shape == clean_vol.shape, "Volumes must have the same shape."
    else:
        snr = np.random.uniform(args.snr_min, args.snr_max)
        print(f"No noisy SEGY provided. Adding synthetic Gaussian noise (SNR={snr:.2f}).")
        noisy_vol = add_gaussian_noise(clean_vol, snr)

    n_inlines, n_crosslines, n_samples = clean_vol.shape
    effective_cols = n_samples - SKIP_COLS
    print(f"\nVolume: {n_inlines} inlines × {n_crosslines} crosslines × {n_samples} samples")
    print(f"Effective crop region: {n_crosslines} × {effective_cols} (after skipping {SKIP_COLS} cols)")

    if n_crosslines < args.crop_size or effective_cols < args.crop_size:
        print(f"WARNING: effective slice dimensions ({n_crosslines}×{effective_cols}) "
              f"< crop_size ({args.crop_size}). Reduce --crop_size or check volume dims.")

    print(f"\n=== Generating crops (crop={args.crop_size}, stride={args.stride}) ===")
    generate(clean_vol, noisy_vol, args.out_dir, args.crop_size, args.stride)
