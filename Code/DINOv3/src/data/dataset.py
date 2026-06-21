from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CROP_SIZE = 224


def _load_dat(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(CROP_SIZE, CROP_SIZE)


def _load_dat_3ch(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(3, CROP_SIZE, CROP_SIZE)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    if std < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / std


class SeismicDataset(Dataset):
    """Paired noisy/clean dataset for seismic denoising.

    Pairs are matched by filename stem (e.g. seismic/2.dat <-> label/2.dat).

    mode="2d":
        Loads a single 224×224 .dat, replicates to 3 channels → [3, 224, 224].
        seismic_dir: directory of single-channel .dat files (224×224 float32).

    mode="2.5d_3ch":
        Loads a pre-generated triplet .dat containing [t-1, t, t+1] inline
        neighbours stacked at the same crop position → [3, 224, 224].
        seismic_dir: directory of 3-channel .dat files (3×224×224 float32).
        label_dir:   directory of single-channel .dat files (224×224 float32).
        Z-score normalisation applied independently per channel.
        Generate this dataset first with: src/data/gen_2d5_3ch.py
    """

    def __init__(
        self,
        seismic_dir: Path,
        label_dir: Path,
        val_split: float = 0.1,
        train: bool = True,
        seed: int = 42,
        mode: str = "2d",
    ):
        seismic_dir = Path(seismic_dir)
        label_dir = Path(label_dir)

        stems = sorted(
            p.stem for p in seismic_dir.glob("*.dat")
            if (label_dir / p.name).exists()
        )

        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(stems))
        n_val = max(1, int(len(stems) * val_split))

        val_idx = set(indices[:n_val].tolist())
        selected = [s for i, s in enumerate(stems) if (i in val_idx) != train]

        self.pairs = [(seismic_dir / f"{s}.dat", label_dir / f"{s}.dat") for s in selected]
        self.mode = mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        noisy_path, clean_path = self.pairs[idx]
        clean_t = torch.from_numpy(_zscore(_load_dat(clean_path))).unsqueeze(0)  # [1, 224, 224]

        if self.mode == "2d":
            noisy = _zscore(_load_dat(noisy_path))
            noisy_t = torch.from_numpy(noisy).unsqueeze(0).repeat(3, 1, 1)  # [3, 224, 224]

        elif self.mode == "2.5d_3ch":
            noisy_3ch = _load_dat_3ch(noisy_path)  # [3, 224, 224]
            noisy_norm = np.stack([_zscore(noisy_3ch[c]) for c in range(3)])
            noisy_t = torch.from_numpy(noisy_norm)  # [3, 224, 224]

        else:
            raise ValueError(f"Unknown mode: {self.mode!r}. Use '2d' or '2.5d_3ch'.")

        return noisy_t, clean_t


class FieldDataset(Dataset):
    """Noisy-only field test data for MS-SSIM-R evaluation."""

    def __init__(self, field_dir: Path):
        self.paths = sorted(Path(field_dir).glob("*.dat"))

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        noisy = _zscore(_load_dat(self.paths[idx]))
        noisy_t = torch.from_numpy(noisy).unsqueeze(0).repeat(3, 1, 1)
        return noisy_t, self.paths[idx].stem


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3] / "Dataset" / "Denoise"
    ds = SeismicDataset(root / "seismic", root / "label", train=True)
    noisy, clean = ds[0]
    print(f"Train size: {len(ds)}")
    print(f"Noisy: {tuple(noisy.shape)}, min={noisy.min():.3f}, max={noisy.max():.3f}")
    print(f"Clean: {tuple(clean.shape)}, min={clean.min():.3f}, max={clean.max():.3f}")

    val_ds = SeismicDataset(root / "seismic", root / "label", train=False)
    print(f"Val size: {len(val_ds)}")

    field_ds = FieldDataset(root / "field")
    noisy_f, stem = field_ds[0]
    print(f"Field size: {len(field_ds)}, sample: {stem}, shape: {tuple(noisy_f.shape)}")
