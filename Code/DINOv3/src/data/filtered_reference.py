"""Filtered-reference field evaluation dataset.

FilteredReferenceDataset loads two aligned F3 volumes: the original noisy volume
and an algorithmically filtered pseudo-clean reference (e.g. dip-steered median
filter from F3 Demo 2023). It returns (noisy_tensor, ref_tensor, metadata) tuples
so downstream evaluation scripts can compute pseudo-paired agreement metrics.

The filtered reference is the center slice only (no multi-channel context),
cropped to the same (oh, ow) position as the noisy center channel, and
z-scored independently.

IMPORTANT: The filtered reference is a pseudo-clean teacher output, not clean
field ground truth. Report metrics as "filtered-reference agreement", never as
"real field ground-truth accuracy".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.input_modes import VALID_MODE_HELP, make_offsets as _make_offsets

_VALID_ORIENTATIONS = {"inline", "crossline", "both", "timeslice"}


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    return (x - x.mean()) / std if std > 1e-8 else x - x.mean()


def _center_crop(arr: np.ndarray, crop_size: int) -> tuple[np.ndarray, int, int]:
    h, w = arr.shape
    oh = max(0, (h - crop_size) // 2)
    ow = max(0, (w - crop_size) // 2)
    return arr[oh:oh + crop_size, ow:ow + crop_size], oh, ow


class FilteredReferenceDataset(Dataset):
    """Pseudo-paired field evaluation dataset using a filtered F3 reference.

    Yields (noisy_tensor, ref_tensor, metadata) tuples where:
      - noisy_tensor: [C, crop_size, crop_size] — multi-channel noisy input
      - ref_tensor:   [1, crop_size, crop_size] — center slice of filtered reference

    Both volumes must share the same (n_inlines, n_xlines, n_samples) shape.
    """

    def __init__(
        self,
        npy_path: str | Path,
        ref_npy_path: str | Path,
        mode: str = "2d",
        orientation: str = "both",
        sample_count: int | None = 32,
        crop_size: int = 224,
        neighbor_stride: int = 1,
        common_context_radius: int | None = None,
        meta_path: str | Path | None = None,
        ref_meta_path: str | Path | None = None,
        section_min: int | None = None,
        section_max: int | None = None,
    ) -> None:
        if orientation not in _VALID_ORIENTATIONS:
            raise ValueError(
                f"orientation must be one of {sorted(_VALID_ORIENTATIONS)}, got {orientation!r}"
            )
        if neighbor_stride < 1:
            raise ValueError(f"neighbor_stride must be >= 1, got {neighbor_stride}")
        if sample_count is not None and sample_count < 1:
            raise ValueError(f"sample_count must be >= 1 or None, got {sample_count}")
        try:
            offsets = _make_offsets(mode, neighbor_stride)
        except ValueError as exc:
            raise ValueError(f"{exc} (valid modes: {VALID_MODE_HELP})") from exc

        self.npy_path = Path(npy_path)
        self.ref_npy_path = Path(ref_npy_path)
        self.mode = mode
        self.orientation = orientation
        self.sample_count = sample_count
        self.crop_size = crop_size
        self.neighbor_stride = neighbor_stride
        if section_min is not None and section_max is not None and section_min > section_max:
            raise ValueError(f"section_min ({section_min}) must be <= section_max ({section_max})")
        self.section_min = section_min
        self.section_max = section_max
        self.offsets: list[int] = offsets
        self.context_radius: int = max(abs(offset) for offset in self.offsets) if self.offsets else 0
        if common_context_radius is not None and common_context_radius < 0:
            raise ValueError(f"common_context_radius must be >= 0, got {common_context_radius}")
        self.common_context_radius = common_context_radius
        self.effective_context_radius = max(
            self.context_radius,
            self.context_radius if common_context_radius is None else int(common_context_radius),
        )

        self.extra_meta: dict = {}
        if meta_path is not None and Path(meta_path).exists():
            with open(meta_path) as f:
                self.extra_meta = json.load(f)

        self.ref_extra_meta: dict = {}
        if ref_meta_path is not None and Path(ref_meta_path).exists():
            with open(ref_meta_path) as f:
                self.ref_extra_meta = json.load(f)

        self._vol: np.ndarray = np.load(str(self.npy_path), mmap_mode="r", allow_pickle=False)
        self._ref_vol: np.ndarray = np.load(str(self.ref_npy_path), mmap_mode="r", allow_pickle=False)

        self.vol_shape: tuple[int, ...] = self._vol.shape
        if len(self.vol_shape) != 3:
            raise ValueError(f"Expected 3D noisy volume, got shape {self.vol_shape}")
        if self._ref_vol.shape != self.vol_shape:
            raise ValueError(
                f"Filtered reference shape {self._ref_vol.shape} does not match "
                f"noisy volume shape {self.vol_shape}. Re-run prepare_filtered_ref.py "
                "with --reference-npy to verify alignment."
            )

        self._samples: list[tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        n_inlines, n_xlines, n_samples = self.vol_shape
        cr = self.effective_context_radius

        orientations_to_add: list[tuple[str, int]] = []

        def select_indices(n_sections: int, lo: int | None = None, hi: int | None = None) -> list[int]:
            # ``lo``/``hi`` restrict the valid center window (used by the timeslice
            # orientation to skip the shallow no-data zone of the F3 survey).
            start = max(cr, lo if lo is not None else 0)
            stop = min(n_sections - cr, hi if hi is not None else n_sections)
            n_valid = stop - start
            if n_valid <= 0:
                return []
            if self.sample_count is None or self.sample_count >= n_valid:
                return list(range(start, stop))
            return np.linspace(start, stop - 1, self.sample_count, dtype=int).tolist()

        if self.orientation in ("inline", "both"):
            indices = select_indices(n_inlines)
            orientations_to_add.extend(("inline", int(idx)) for idx in indices)
        if self.orientation in ("crossline", "both"):
            indices = select_indices(n_xlines)
            orientations_to_add.extend(("crossline", int(idx)) for idx in indices)
        if self.orientation == "timeslice":
            indices = select_indices(n_samples, lo=self.section_min, hi=self.section_max)
            orientations_to_add.extend(("timeslice", int(idx)) for idx in indices)

        self._samples.extend(orientations_to_add)

        sample_count_label = "all" if self.sample_count is None else str(self.sample_count)
        print(
            f"[FilteredReferenceDataset] mode={self.mode} | orientation={self.orientation} | "
            f"vol_shape={self.vol_shape} | neighbor_stride={self.neighbor_stride} | "
            f"context_radius={self.context_radius} | effective_context_radius={self.effective_context_radius} | "
            f"sample_count={sample_count_label} | samples={len(self._samples)}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        orient, sec_idx = self._samples[idx]
        crop_size = self.crop_size

        if orient == "inline":
            def get_section(vol: np.ndarray, i: int) -> np.ndarray:
                return vol[i, :, :].astype(np.float32)
        elif orient == "crossline":
            def get_section(vol: np.ndarray, i: int) -> np.ndarray:
                return vol[:, i, :].astype(np.float32)
        else:  # timeslice: horizontal (inline x crossline) plane at a fixed time sample
            def get_section(vol: np.ndarray, i: int) -> np.ndarray:
                return vol[:, :, i].astype(np.float32)

        # Determine crop position from center noisy slice.
        base = get_section(self._vol, sec_idx)
        if base.shape[0] < crop_size or base.shape[1] < crop_size:
            base = np.pad(
                base,
                [(max(0, crop_size - base.shape[0]), 0), (max(0, crop_size - base.shape[1]), 0)],
                mode="reflect",
            )
        _, oh, ow = _center_crop(base, crop_size)

        # Build multi-channel noisy tensor.
        channels: list[np.ndarray] = []
        for dt in self.offsets:
            section = get_section(self._vol, sec_idx + dt)
            if section.shape[0] < crop_size or section.shape[1] < crop_size:
                section = np.pad(
                    section,
                    [
                        (max(0, crop_size - section.shape[0]), 0),
                        (max(0, crop_size - section.shape[1]), 0),
                    ],
                    mode="reflect",
                )
            crop = section[oh:oh + crop_size, ow:ow + crop_size]
            channels.append(_zscore(crop))
        noisy = torch.from_numpy(np.stack(channels, axis=0))

        # Build single-channel filtered reference (center slice only).
        ref_section = get_section(self._ref_vol, sec_idx)
        if ref_section.shape[0] < crop_size or ref_section.shape[1] < crop_size:
            ref_section = np.pad(
                ref_section,
                [
                    (max(0, crop_size - ref_section.shape[0]), 0),
                    (max(0, crop_size - ref_section.shape[1]), 0),
                ],
                mode="reflect",
            )
        ref_crop = ref_section[oh:oh + crop_size, ow:ow + crop_size]
        ref_tensor = torch.from_numpy(_zscore(ref_crop)[np.newaxis])  # [1, H, W]

        meta = {
            "orientation": orient,
            "section_idx": sec_idx,
            "section_global_id": f"{orient}_{sec_idx:04d}",
            "sample_idx": idx,
            "oh": oh,
            "ow": ow,
            "mode": self.mode,
            "neighbor_stride": self.neighbor_stride,
            "context_radius": self.context_radius,
            "effective_context_radius": self.effective_context_radius,
            "common_context_radius": -1 if self.common_context_radius is None else self.common_context_radius,
            "sample_count_request": -1 if self.sample_count is None else self.sample_count,
        }
        return noisy, ref_tensor, meta
