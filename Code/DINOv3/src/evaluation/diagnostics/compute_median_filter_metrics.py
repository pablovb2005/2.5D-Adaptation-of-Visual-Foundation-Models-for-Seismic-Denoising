"""Compute no-reference F3 diagnostics for the dip-steered median filter.

Reports MS-SSIM-R and amplitude ratio on the same 64 F3 sections (32 inline +
32 crossline) sampled by the neural-model F3 evaluations, using
common_context_radius=2 to match the shared section IDs.

Usage:
    python evaluation/compute_median_filter_metrics.py \
        --f3-npy Code/Dataset/F3/processed/f3_original.npy \
        --filtered-npy Code/Dataset/F3/processed/f3_filtered_ref.npy \
        [--sample-count 32] [--crop-size 224] [--common-context-radius 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path

import numpy as np
import torch

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from evaluation.common.metrics import compute_ms_ssim_r


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    return (x - x.mean()) / std if std > 1e-8 else x - x.mean()


def _center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape
    oh = max(0, (h - size) // 2)
    ow = max(0, (w - size) // 2)
    return arr[oh:oh + size, ow:ow + size]


def _amplitude_ratio_raw(filt_raw: np.ndarray, noisy_raw: np.ndarray) -> float:
    """Std(filtered) / Std(noisy) in raw amplitude space.

    Matches the physical meaning of the neural-model amplitude ratio:
    the model output is in z-scored-input space (std ≈ output_std / 1.0),
    so for the median filter we compare filtered_std to noisy_std directly.
    """
    s_f = float(filt_raw.std())
    s_n = float(noisy_raw.std())
    return s_f / s_n if s_n > 1e-12 else 0.0


def _ms_ssim_r(filt_z: np.ndarray, noisy_z: np.ndarray) -> float:
    """MS-SSIM-R using independently z-scored crops.

    compute_ms_ssim_r applies _to_unit (min-max) internally, so the
    z-score scale does not affect the result.
    """
    den = torch.from_numpy(filt_z[None, None]).float()
    noi = torch.from_numpy(noisy_z[None, None]).float()
    return compute_ms_ssim_r(den, noi)


def _sample_indices(
    n_sections: int,
    sample_count: int,
    context_radius: int,
    lo: int | None = None,
    hi: int | None = None,
) -> list[int]:
    # ``lo``/``hi`` restrict the valid center window (used by the timeslice
    # orientation to skip the shallow no-data zone of the F3 survey), matching
    # F3FieldDataset.select_indices in data/robustness.py.
    start = max(context_radius, lo if lo is not None else 0)
    stop = min(n_sections - context_radius, hi if hi is not None else n_sections)
    n_valid = stop - start
    if n_valid <= 0:
        return []
    if sample_count is None or sample_count < 0 or sample_count >= n_valid:
        return list(range(start, stop))
    return np.linspace(start, stop - 1, sample_count, dtype=int).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute no-reference F3 diagnostics for the dip-steered median filter."
    )
    parser.add_argument("--f3-npy", required=True, help="Path to f3_original.npy")
    parser.add_argument("--filtered-npy", required=True, help="Path to f3_filtered_ref.npy")
    parser.add_argument("--sample-count", type=int, default=32, help="Sections per orientation")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--common-context-radius", type=int, default=2,
                        help="Shared context radius used by neural models (default 2)")
    parser.add_argument("--orientation", choices=["inline_crossline", "timeslice"],
                        default="inline_crossline",
                        help="Slice orientation. 'timeslice' takes horizontal inline x "
                             "crossline planes at fixed time samples, matching the Image "
                             "Impeccable training orientation (default: inline_crossline).")
    parser.add_argument("--section-min", type=int, default=None,
                        help="Lowest valid center index (inclusive) for timeslice; skips "
                             "the shallow no-data zone of F3, e.g. 50.")
    parser.add_argument("--section-max", type=int, default=None,
                        help="Highest valid center index (exclusive) for timeslice.")
    args = parser.parse_args()

    f3_npy = Path(args.f3_npy).resolve()
    filt_npy = Path(args.filtered_npy).resolve()

    print(f"Loading F3 volume: {f3_npy}")
    noisy_vol = np.load(str(f3_npy), mmap_mode="r", allow_pickle=False)
    print(f"  Shape: {noisy_vol.shape}")

    print(f"Loading filtered reference: {filt_npy}")
    filt_vol = np.load(str(filt_npy), mmap_mode="r", allow_pickle=False)
    print(f"  Shape: {filt_vol.shape}")

    if noisy_vol.shape != filt_vol.shape:
        sys.exit(f"Shape mismatch: {noisy_vol.shape} vs {filt_vol.shape}")

    n_inlines, n_xlines, n_samples = noisy_vol.shape
    cr = args.common_context_radius
    cs = args.crop_size
    sc = args.sample_count

    ms_ssim_r_vals: list[float] = []
    amp_ratio_vals: list[float] = []

    def _accumulate(noisy_sec: np.ndarray, filt_sec: np.ndarray) -> None:
        nc = _center_crop(noisy_sec.astype(np.float32), cs)
        fc = _center_crop(filt_sec.astype(np.float32),  cs)
        ms_ssim_r_vals.append(_ms_ssim_r(_zscore(fc), _zscore(nc)))
        amp_ratio_vals.append(_amplitude_ratio_raw(fc, nc))

    if args.orientation == "timeslice":
        ts_indices = _sample_indices(n_samples, sc, cr, lo=args.section_min, hi=args.section_max)
        print(f"\nSampling {len(ts_indices)} timeslices "
              f"(common_context_radius={cr}, section_min={args.section_min}, crop={cs}x{cs})")
        for k in ts_indices:
            _accumulate(noisy_vol[:, :, k], filt_vol[:, :, k])
    else:
        inline_indices = _sample_indices(n_inlines, sc, cr)
        xline_indices = _sample_indices(n_xlines, sc, cr)
        print(f"\nSampling {len(inline_indices)} inlines + {len(xline_indices)} crosslines "
              f"(common_context_radius={cr}, crop={cs}x{cs})")
        for idx in inline_indices:
            _accumulate(noisy_vol[idx, :, :], filt_vol[idx, :, :])
        for idx in xline_indices:
            _accumulate(noisy_vol[:, idx, :], filt_vol[:, idx, :])

    n = len(ms_ssim_r_vals)
    ms_r_mean = float(np.mean(ms_ssim_r_vals))
    ms_r_std  = float(np.std(ms_ssim_r_vals, ddof=1))
    ar_mean   = float(np.mean(amp_ratio_vals))
    ar_std    = float(np.std(amp_ratio_vals, ddof=1))

    print(f"\nResults over {n} sections:")
    print(f"  MS-SSIM-R:       {ms_r_mean:.4f} +/- {ms_r_std:.4f}")
    print(f"  Amplitude ratio: {ar_mean:.4f} +/- {ar_std:.4f}")
    print("\nFor LaTeX table row (Median filter, classical ref.):")
    print(f"  MS-SSIM-R:       ${ms_r_mean:.3f} \\pm {ms_r_std:.3f}$")
    print(f"  Amplitude ratio: ${ar_mean:.3f} \\pm {ar_std:.3f}$")


if __name__ == "__main__":
    main()
