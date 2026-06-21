"""ThinkOnward Image Impeccable dataset — on-the-fly .npy loader.

Volumes are memory-mapped so only the slices actually needed are read from disk.
Supports all input variants through the ``mode`` parameter.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.input_modes import VALID_MODE_HELP, make_offsets as _make_offsets


def _zscore(x: np.ndarray) -> np.ndarray:
    std = float(x.std())
    return (x - x.mean()) / std if std > 1e-8 else x - x.mean()


def _crop_slice(vol: np.ndarray, ax: int, t: int, oh: int, ow: int, cs: int) -> np.ndarray:
    """Return the requested crop without materialising the full 2D slice first."""
    if ax == 0:
        return vol[t, oh:oh + cs, ow:ow + cs]
    if ax == 1:
        return vol[oh:oh + cs, t, ow:ow + cs]
    if ax == 2:
        return vol[oh:oh + cs, ow:ow + cs, t]
    raise ValueError(f"Invalid slice axis: {ax}")


class ThinkOnwardDataset(Dataset):
    """On-the-fly loader for ThinkOnward Image Impeccable paired .npy volumes.

    Each item returns ``(noisy, clean)`` where:
      - ``noisy``: float32 tensor  [C, crop_size, crop_size]
      - ``clean``: float32 tensor  [1, crop_size, crop_size]

    Args:
        root_dir:        Path to ``extracted/`` directory.
        mode:            ``'2d'`` or ``'2.5d_{N}ch'`` for N in {3,5,7,9}.
        split:           ``'train'`` | ``'val'`` | ``'test'``
        n_train:         Volumes assigned to training (first n_train after shuffle).
        n_val:           Volumes assigned to validation.
        n_test:          Volumes assigned to testing.
        slice_stride:    Step between sampled central slice indices.
        crop_size:       Square spatial crop in pixels.
        seed:            RNG seed for reproducible volume-level split.
        train_subset_n:  If set, use only the first N volumes of the training split.
                         Val/test splits are unaffected. Use for data efficiency experiments.
        neighbor_stride: Spacing between 2.5D context channels. Default 1 gives direct
                         neighbors, e.g. [t-1,t,t+1] for 3ch or [t-3..t+3] for 7ch.
        crop_mode:       ``'center'`` (default) | ``'random'`` | ``'grid4'``.
                         ``center`` — single center crop per slice (current behavior).
                         ``random`` — one random crop per item, deterministic per sample index.
                         ``grid4`` — four corner crops per slice (~4× dataset length).
        crop_seed:       RNG seed for random crops. Defaults to ``seed`` if not set.
        slice_orientation: Which axis to slice volumes along.
                         ``'auto'`` (default) slices along the longest axis (the
                         time/depth axis for Image Impeccable), reproducing the
                         original horizontal time-slice behavior. ``'vertical'``
                         / ``'inline'`` slice along the first non-time axis and
                         ``'crossline'`` along the second, yielding vertical
                         sections. Orientation is defined relative to the longest
                         axis so it is robust to per-volume axis ordering.
    """

    def __init__(
        self,
        root_dir: str | Path,
        mode: str = "2d",
        split: str = "train",
        n_train: int = 20,
        n_val: int = 5,
        n_test: int = 5,
        slice_stride: int = 5,
        crop_size: int = 224,
        seed: int = 42,
        train_subset_n: int | None = None,
        neighbor_stride: int = 1,
        crop_mode: str = "center",
        crop_seed: int | None = None,
        shuffle_neighbors: bool = False,
        repeat_center: bool = False,
        cache_volumes: bool = False,
        slice_orientation: str = "auto",
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        if slice_orientation not in ("auto", "vertical", "inline", "crossline"):
            raise ValueError(
                "slice_orientation must be 'auto', 'vertical', 'inline', or "
                f"'crossline', got '{slice_orientation}'"
            )
        if neighbor_stride < 1:
            raise ValueError(f"neighbor_stride must be >= 1, got {neighbor_stride}")
        try:
            offsets = _make_offsets(mode, neighbor_stride)
        except ValueError as exc:
            raise ValueError(f"{exc} (valid modes: {VALID_MODE_HELP})") from exc
        if crop_mode not in ("center", "random", "grid4"):
            raise ValueError(f"crop_mode must be 'center', 'random', or 'grid4', got '{crop_mode}'")
        if repeat_center and mode != "2.5d_5ch":
            raise ValueError(f"repeat_center=True is only valid for mode='2.5d_5ch', got '{mode}'")

        self.root_dir = Path(root_dir)
        self.mode = mode
        self.split = split
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.slice_stride = slice_stride
        self.crop_size = crop_size
        self.seed = seed
        self.train_subset_n = train_subset_n
        self.neighbor_stride = neighbor_stride
        self.crop_mode = crop_mode
        self.crop_seed = seed if crop_seed is None else crop_seed
        self.shuffle_neighbors = shuffle_neighbors
        self.repeat_center = repeat_center
        self.cache_volumes = cache_volumes
        self.slice_orientation = slice_orientation
        self._current_epoch: int = 0
        self._volume_cache: dict[Path, np.ndarray] = {}
        self._donor_candidates: list[np.ndarray] = []
        self._donor_indices: np.ndarray | None = None

        self.offsets: list[int] = offsets
        if repeat_center:
            self.offsets = [0] * len(self.offsets)
        self.context_radius: int = max(abs(o) for o in self.offsets) if self.offsets else 0

        self.samples: list[tuple] = []
        self._build_index()
        if self.shuffle_neighbors and self.mode != "2d":
            self._build_donor_candidates()
            self.set_epoch(0)

    # ------------------------------------------------------------------
    def _build_index(self) -> None:
        noisy_files = sorted(self.root_dir.rglob("seismic_w_noise_vol_*.npy"))
        if not noisy_files:
            raise FileNotFoundError(
                f"No noisy .npy volumes found under {self.root_dir}. "
                "Check that the dataset is extracted correctly."
            )

        # Pair noisy with clean files by volume ID.
        pairs: list[tuple[int, Path, Path]] = []
        for noisy_path in noisy_files:
            vol_id = int(noisy_path.stem.split("_")[-1])
            clean_name = f"seismicCubes_RFC_fullstack_2024.{vol_id}.npy"
            clean_path = noisy_path.parent / clean_name
            if not clean_path.exists():
                warnings.warn(f"Clean file missing for volume {vol_id}; skipping.")
                continue
            pairs.append((vol_id, noisy_path, clean_path))

        if not pairs:
            raise FileNotFoundError("No complete noisy/clean pairs found.")

        # Reproducible volume-level shuffle, then split.
        pairs.sort(key=lambda x: x[0])
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(len(pairs))
        pairs = [pairs[i] for i in idx]

        split_slices = {
            "train": slice(0, self.n_train),
            "val":   slice(self.n_train, self.n_train + self.n_val),
            "test":  slice(self.n_train + self.n_val, self.n_train + self.n_val + self.n_test),
        }
        selected = pairs[split_slices[self.split]]
        if not selected:
            raise RuntimeError(
                f"No volumes for split='{self.split}' "
                f"(found {len(pairs)} total pairs, n_train={self.n_train}, "
                f"n_val={self.n_val}, n_test={self.n_test})"
            )

        if self.split == "train" and self.train_subset_n is not None:
            if not (1 <= self.train_subset_n <= len(selected)):
                raise ValueError(
                    f"train_subset_n={self.train_subset_n} out of range "
                    f"[1, {len(selected)}] for split='train'"
                )
            selected = selected[: self.train_subset_n]

        cr = self.context_radius
        cs = self.crop_size

        for vol_id, noisy_path, clean_path in selected:
            # Peek at shape without loading the full volume.
            try:
                vol = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
            except ValueError as exc:
                raise ValueError(
                    f"Volume file contains pickled data and cannot be memory-mapped: {noisy_path}\n"
                    "Re-save it as a plain float32 array first:\n"
                    "  python Code/DAIC/tools/fix_pickled_npy.py --fix <dataset_dir>"
                ) from exc
            vol_shape = vol.shape

            # The longest axis is the time/depth axis; 'auto' slices along it
            # (horizontal time slices). 'vertical'/'inline'/'crossline' slice
            # along a non-time axis, defined relative to the longest axis so the
            # choice is robust to per-volume axis ordering.
            time_axis = int(np.argmax(vol_shape))
            if self.slice_orientation == "auto":
                slice_axis = time_axis
            elif self.slice_orientation in ("vertical", "inline"):
                slice_axis = min(i for i in range(3) if i != time_axis)
            else:  # "crossline"
                slice_axis = max(i for i in range(3) if i != time_axis)
            n_slices = vol_shape[slice_axis]
            spatial_dims = [vol_shape[i] for i in range(3) if i != slice_axis]

            if spatial_dims[0] < cs or spatial_dims[1] < cs:
                warnings.warn(
                    f"Volume {vol_id}: spatial dims {spatial_dims} smaller than "
                    f"crop_size {cs}; skipping."
                )
                continue

            # Base center crop offsets (used for center and grid4).
            oh = (spatial_dims[0] - cs) // 2
            ow = (spatial_dims[1] - cs) // 2

            if self.crop_mode == "center":
                crop_positions: list[tuple[int | None, int | None]] = [(oh, ow)]
            elif self.crop_mode == "random":
                crop_positions = [(None, None)]
            else:  # grid4
                max_h = spatial_dims[0] - cs
                max_w = spatial_dims[1] - cs
                raw: list[tuple[int, int]] = [(0, 0), (0, max_w), (max_h, 0), (max_h, max_w)]
                seen: set[tuple[int, int]] = set()
                crop_positions = []
                for pos in raw:
                    if pos not in seen:
                        seen.add(pos)
                        crop_positions.append(pos)

            t_min = cr
            t_max = n_slices - 1 - cr

            for t in range(t_min, t_max + 1, self.slice_stride):
                for (ch, cw) in crop_positions:
                    self.samples.append((noisy_path, clean_path, slice_axis, t, ch, cw))

        if not self.samples:
            raise RuntimeError(f"No valid samples built for split='{self.split}'.")

        vol_ids = [str(p[0]) for p in selected]
        flags = ""
        if self.shuffle_neighbors:
            flags += " | shuffle_neighbors=True"
        if self.repeat_center:
            flags += " | repeat_center=True"
        print(
            f"[ThinkOnwardDataset] split={self.split} | mode={self.mode} | "
            f"orientation={self.slice_orientation} | "
            f"stride={self.slice_stride} | neighbor_stride={self.neighbor_stride} | "
            f"crop_mode={self.crop_mode} | volumes={len(selected)} | "
            f"samples={len(self.samples)} | vol_ids={vol_ids}{flags}"
        )

    # ------------------------------------------------------------------
    def _build_donor_candidates(self) -> None:
        """Precompute legal cross-volume donor pools for shuffled controls."""
        all_indices = np.arange(len(self.samples), dtype=np.int64)
        by_volume: dict[Path, list[int]] = {}
        for idx, (noisy_path, *_rest) in enumerate(self.samples):
            by_volume.setdefault(noisy_path, []).append(idx)

        candidates_by_volume: dict[Path, np.ndarray] = {}
        for noisy_path, same_volume_indices in by_volume.items():
            mask = np.ones(len(self.samples), dtype=bool)
            mask[np.asarray(same_volume_indices, dtype=np.int64)] = False
            candidates = all_indices[mask]
            if len(candidates) == 0:
                raise RuntimeError(
                    "shuffle_neighbors=True requires at least two volumes in the split; "
                    f"split='{self.split}' only has one usable volume."
                )
            candidates_by_volume[noisy_path] = candidates

        self._donor_candidates = [
            candidates_by_volume[noisy_path]
            for noisy_path, *_rest in self.samples
        ]

    def _load_volume(self, path: Path) -> np.ndarray:
        if not self.cache_volumes:
            try:
                return np.load(path, mmap_mode="r", allow_pickle=False)
            except ValueError as exc:
                raise ValueError(
                    f"Volume file contains pickled data and cannot be memory-mapped: {path}\n"
                    "Re-save it as a plain float32 array first:\n"
                    "  python Code/DAIC/tools/fix_pickled_npy.py --fix <dataset_dir>"
                ) from exc
        vol = self._volume_cache.get(path)
        if vol is None:
            try:
                vol = np.load(path, mmap_mode="r", allow_pickle=False)
            except ValueError as exc:
                raise ValueError(
                    f"Volume file contains pickled data and cannot be memory-mapped: {path}\n"
                    "Re-save it as a plain float32 array first:\n"
                    "  python Code/DAIC/tools/fix_pickled_npy.py --fix <dataset_dir>"
                ) from exc
            self._volume_cache[path] = vol
        return vol

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_volume_cache"] = {}
        return state

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each training epoch to vary shuffled-neighbor donors."""
        self._current_epoch = int(epoch)
        if not (self.shuffle_neighbors and self.mode != "2d"):
            return
        if not self._donor_candidates:
            self._build_donor_candidates()
        rng = np.random.RandomState(self.seed + self._current_epoch * 100003)
        donor_indices = np.empty(len(self.samples), dtype=np.int64)
        for idx, candidates in enumerate(self._donor_candidates):
            donor_indices[idx] = int(candidates[rng.randint(0, len(candidates))])
        self._donor_indices = donor_indices

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_path, clean_path, ax, t, oh, ow = self.samples[idx]
        cs = self.crop_size

        # Memory-mapped: only the pages containing the requested slices are read.
        noisy_vol = self._load_volume(noisy_path)
        clean_vol = self._load_volume(clean_path)

        if oh is None:  # random crop — resolve deterministically per sample index
            vol_shape = noisy_vol.shape
            spatial = [vol_shape[i] for i in range(3) if i != ax]
            rng = np.random.RandomState(self.crop_seed + idx)
            oh = int(rng.randint(0, max(1, spatial[0] - cs + 1)))
            ow = int(rng.randint(0, max(1, spatial[1] - cs + 1)))

        if all(dt == 0 for dt in self.offsets):
            center_slice = _crop_slice(noisy_vol, ax, t, oh, ow, cs).astype(np.float32)
            clean_slice = _crop_slice(clean_vol, ax, t, oh, ow, cs).astype(np.float32)

            center_slice = _zscore(center_slice)
            clean_slice = _zscore(clean_slice)

            noisy = torch.from_numpy(np.repeat(center_slice[None], len(self.offsets), axis=0))
            clean = torch.from_numpy(clean_slice[None])
            return noisy, clean

        if self.shuffle_neighbors and self.mode != "2d":
            # Replace neighbor channels with slices from a different volume.
            # Center channel (dt=0) always comes from the correct sample.
            # Donor varies per epoch to prevent the model memorising fixed pairings.
            if self._donor_indices is None:
                self.set_epoch(self._current_epoch)
            donor_idx = int(self._donor_indices[idx])
            donor_noisy_path, _, donor_ax, donor_t, donor_oh, donor_ow = self.samples[donor_idx]
            donor_vol = self._load_volume(donor_noisy_path)
            if donor_oh is None:
                d_shape = donor_vol.shape
                d_spatial = [d_shape[i] for i in range(3) if i != donor_ax]
                rng_dcrop = np.random.RandomState(self.crop_seed + donor_idx)
                donor_oh = int(rng_dcrop.randint(0, max(1, d_spatial[0] - cs + 1)))
                donor_ow = int(rng_dcrop.randint(0, max(1, d_spatial[1] - cs + 1)))

            noisy_slices = []
            for dt in self.offsets:
                if dt == 0:
                    noisy_slices.append(
                        _crop_slice(noisy_vol, ax, t, oh, ow, cs).astype(np.float32)
                    )
                else:
                    noisy_slices.append(
                        _crop_slice(
                            donor_vol, donor_ax, donor_t + dt, donor_oh, donor_ow, cs
                        ).astype(np.float32)
                    )
        else:
            noisy_slices = [
                _crop_slice(noisy_vol, ax, t + dt, oh, ow, cs).astype(np.float32)
                for dt in self.offsets
            ]

        clean_slice = _crop_slice(clean_vol, ax, t, oh, ow, cs).astype(np.float32)

        noisy_slices = [_zscore(s) for s in noisy_slices]
        clean_slice = _zscore(clean_slice)

        noisy = torch.from_numpy(np.stack(noisy_slices, axis=0))  # [C, H, W]
        clean = torch.from_numpy(clean_slice[None])                 # [1, H, W]
        return noisy, clean
