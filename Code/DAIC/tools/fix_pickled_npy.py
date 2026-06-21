#!/usr/bin/env python3
"""Audit and repair .npy files that contain pickled (dtype=object) data.

Usage
-----
  # Dry-run: report bad files without changing anything
  python3 fix_pickled_npy.py <dataset_dir>

  # Apply fix: re-save bad files as plain float32 arrays
  python3 fix_pickled_npy.py <dataset_dir> --fix

Only files that fail np.load(..., allow_pickle=False) are examined further.
Files that load cleanly are skipped without any action.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def _load_raw(path):
    return np.load(str(path), allow_pickle=True)


def _check_file(path, fix):
    try:
        np.load(str(path), mmap_mode="r", allow_pickle=False)
        return "OK"
    except ValueError:
        pass

    raw = _load_raw(path)

    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
        raw = raw.item()

    py_type = type(raw).__name__

    if isinstance(raw, np.ndarray) and np.issubdtype(raw.dtype, np.number):
        status = "BAD  dtype={} shape={} type=ndarray".format(raw.dtype, raw.shape)
        if fix:
            np.save(str(path), raw.astype(np.float32))
            status += " -> FIXED (saved as float32)"
        else:
            status += " -> would fix (re-save as float32)"
        return status

    if isinstance(raw, dict):
        keys = list(raw.keys())
        status = "BAD  type=dict keys={}".format(keys)
        if fix:
            status += " -> SKIPPED (cannot auto-convert dict; inspect manually)"
        else:
            status += " -> would SKIP (dict — cannot auto-convert)"
        return status

    status = "BAD  type={}".format(py_type)
    if fix:
        status += " -> SKIPPED (unknown structure; inspect manually)"
    else:
        status += " -> would SKIP (unknown structure)"
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Root directory to scan (recursive)")
    parser.add_argument("--fix", action="store_true", help="Re-save bad files as float32")
    args = parser.parse_args()

    if not args.dataset_dir.is_dir():
        print("ERROR: {} is not a directory".format(args.dataset_dir), file=sys.stderr)
        sys.exit(1)

    npy_files = sorted(args.dataset_dir.rglob("*.npy"))
    if not npy_files:
        print("No .npy files found under {}".format(args.dataset_dir))
        return

    print("Scanning {} .npy files under {}".format(len(npy_files), args.dataset_dir))
    print("Mode: {}\n".format("FIX" if args.fix else "DRY-RUN (no changes)"))

    bad_count = 0
    for path in npy_files:
        status = _check_file(path, fix=args.fix)
        if status != "OK":
            bad_count += 1
            print("  {}".format(path))
            print("    {}".format(status))

    print()
    if bad_count == 0:
        print("All files are clean (no pickled data found).")
    else:
        noun = "file" if bad_count == 1 else "files"
        if args.fix:
            print("{} {} processed.".format(bad_count, noun))
            print("Re-run without --fix to confirm all are now clean.")
        else:
            print("{} {} need fixing.".format(bad_count, noun))
            print("Re-run with --fix to apply repairs.")


if __name__ == "__main__":
    main()
