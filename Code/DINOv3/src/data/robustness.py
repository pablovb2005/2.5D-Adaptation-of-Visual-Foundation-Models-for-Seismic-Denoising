"""Robustness evaluation datasets.

F3FieldDataset returns unlabelled real-field 3D sections from Netherlands F3
as (noisy_tensor, metadata_dict). There is no clean reference.

The dataset applies z-score normalization per channel, matching
image_impeccable.py.
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
    """Center-crop a 2D array to (crop_size, crop_size). Returns (crop, oh, ow)."""
    h, w = arr.shape
    oh = max(0, (h - crop_size) // 2)
    ow = max(0, (w - crop_size) // 2)
    return arr[oh:oh + crop_size, ow:ow + crop_size], oh, ow


class F3FieldDataset(Dataset):
    """Unlabelled field transfer dataset from Netherlands F3 Demo 2023.

    Yields (noisy_tensor, metadata) tuples where noisy_tensor is shaped
    [C, crop_size, crop_size] with C matching the chosen mode. There is no
    clean reference, so accuracy metrics must not be computed on this dataset.

    The 3D volume is expected at ``npy_path`` with shape
    (n_inlines, n_xlines, n_samples) float32.
    """

    def __init__(
        self,
        npy_path: str | Path,
        mode: str = "2d",
        orientation: str = "both",
        sample_count: int | None = 32,
        crop_size: int = 224,
        neighbor_stride: int = 1,
        common_context_radius: int | None = None,
        meta_path: str | Path | None = None,
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

        # Memory-map the volume once; individual __getitem__ calls read pages on demand.
        self._vol: np.ndarray = np.load(str(self.npy_path), mmap_mode="r", allow_pickle=False)
        self.vol_shape: tuple[int, ...] = self._vol.shape
        if len(self.vol_shape) != 3:
            raise ValueError(f"Expected 3D volume, got shape {self.vol_shape}")

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
            f"[F3FieldDataset] mode={self.mode} | orientation={self.orientation} | "
            f"vol_shape={self.vol_shape} | neighbor_stride={self.neighbor_stride} | "
            f"context_radius={self.context_radius} | effective_context_radius={self.effective_context_radius} | "
            f"sample_count={sample_count_label} | samples={len(self._samples)}"
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        orient, sec_idx = self._samples[idx]
        crop_size = self.crop_size

        if orient == "inline":

            def get_section(i: int) -> np.ndarray:
                return self._vol[i, :, :].astype(np.float32)

        elif orient == "crossline":

            def get_section(i: int) -> np.ndarray:
                return self._vol[:, i, :].astype(np.float32)

        else:  # timeslice: horizontal (inline x crossline) plane at a fixed time sample

            def get_section(i: int) -> np.ndarray:
                return self._vol[:, :, i].astype(np.float32)

        base = get_section(sec_idx)
        if base.shape[0] < crop_size or base.shape[1] < crop_size:
            base = np.pad(
                base,
                [(max(0, crop_size - base.shape[0]), 0), (max(0, crop_size - base.shape[1]), 0)],
                mode="reflect",
            )

        _, oh, ow = _center_crop(base, crop_size)

        channels: list[np.ndarray] = []
        for dt in self.offsets:
            section = get_section(sec_idx + dt)
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
        return noisy, meta
