"""Smoke test for ThinkOnwardDataset.

Run locally (no GPU needed) to verify dataset indexing and sample loading.

Usage:
    python check_image_impeccable.py --root <path-to-extracted/>

Example:
    python check_image_impeccable.py \
        --root ../../../Dataset/ThinkOnwards/training_data/extracted
"""

import argparse
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from data.image_impeccable import ThinkOnwardDataset
from data.input_modes import mode_channel_count


def check_tensor(t: torch.Tensor, name: str) -> bool:
    ok = True
    if torch.isnan(t).any():
        print(f"  ERROR: {name} contains NaN")
        ok = False
    if torch.isinf(t).any():
        print(f"  ERROR: {name} contains Inf")
        ok = False
    print(
        f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}"
    )
    return ok


def check_split(root: Path, mode: str, split: str, n_train: int, n_val: int, n_test: int) -> bool:
    print(f"\n--- mode={mode!r} split={split!r} ---")
    try:
        ds = ThinkOnwardDataset(
            root_dir=root,
            mode=mode,
            split=split,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            slice_stride=5,
            crop_size=224,
            seed=42,
        )
    except Exception as e:
        print(f"  ERROR during construction: {e}")
        return False

    print(f"  len={len(ds)}")
    print(f"  offsets={ds.offsets}")
    if len(ds) == 0:
        print("  ERROR: empty dataset")
        return False

    ok = True
    for label, idx in [("first", 0), ("last", len(ds) - 1)]:
        print(f"  sample[{label}] (idx={idx}):")
        try:
            noisy, clean = ds[idx]
        except Exception as e:
            print(f"    ERROR loading sample: {e}")
            ok = False
            continue
        ok &= check_tensor(noisy, "noisy")
        ok &= check_tensor(clean, "clean")

        expected_c = mode_channel_count(mode)
        if tuple(noisy.shape) != (expected_c, 224, 224):
            print(f"    ERROR: expected noisy shape ({expected_c}, 224, 224), got {tuple(noisy.shape)}")
            ok = False
        if tuple(clean.shape) != (1, 224, 224):
            print(f"    ERROR: expected clean shape (1, 224, 224), got {tuple(clean.shape)}")
            ok = False

    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).parents[2] / "Dataset" / "ThinkOnwards" / "training_data" / "extracted"),
        help="Path to extracted/ directory containing volume sub-folders",
    )
    parser.add_argument("--n_train", type=int, default=20)
    parser.add_argument("--n_val", type=int, default=5)
    parser.add_argument("--n_test", type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Root directory: {root}")
    if not root.exists():
        print(f"ERROR: root directory does not exist: {root}")
        sys.exit(1)

    all_ok = True
    for mode in ("2d", "2.5d_3ch", "2.5d_5ch", "2.5d_7ch", "2.5d_9ch"):
        for split in ("train", "val", "test"):
            all_ok &= check_split(root, mode, split, args.n_train, args.n_val, args.n_test)

    print("\n" + ("=" * 50))
    if all_ok:
        print("All checks passed.")
    else:
        print("Some checks FAILED — review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
