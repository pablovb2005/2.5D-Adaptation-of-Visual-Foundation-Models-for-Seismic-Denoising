#!/usr/bin/env python3
"""Build and audit a canonical ThinkOnward Image Impeccable dataset.

The official archives contain a mix of pickle-backed arrays, float64 clean
targets, 2023/2024 filename suffixes, and clean-target orientations. This tool
normalizes each pair into the format expected by ThinkOnwardDataset:

  <output>/<volume_id>/seismic_w_noise_vol_<volume_id>.npy
  <output>/<volume_id>/seismicCubes_RFC_fullstack_2024.<volume_id>.npy

All outputs are plain float32 NPY arrays with matching shapes. Writes are
atomic, so interrupted DAIC jobs can be resubmitted safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


NOISY_RE = re.compile(r"(?:^|/)seismic_w_noise_vol_(\d+)\.npy$")
CLEAN_RE = re.compile(r"(?:^|/)seismicCubes_RFC_fullstack_(\d{4})\.(\d+)\.npy$")
MANIFEST_NAME = "dataset_manifest.json"
CANONICAL_SHAPE = (1259, 300, 300)
CANONICAL_DTYPE = np.dtype("float32")
ORIENTATION_POLICIES = ("auto", "identity", "swap-hw")


def _human_gb(n_bytes: int) -> str:
    return "{:.2f} GB".format(n_bytes / 1_000_000_000)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(".{}.tmp".format(path.name))
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


def _load_numeric(source: BinaryIO | Path, description: str) -> np.ndarray:
    try:
        if isinstance(source, Path):
            try:
                value = np.load(str(source), mmap_mode="r", allow_pickle=False)
            except ValueError:
                value = np.load(str(source), allow_pickle=True)
        else:
            value = np.load(source, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError("Could not load {}: {}".format(description, exc)) from exc
    if isinstance(value, np.ndarray) and value.dtype == object and value.ndim == 0:
        value = value.item()
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        raise TypeError(
            "{} must contain a numeric ndarray, got {}".format(description, type(value).__name__)
        )
    if value.ndim != 3:
        raise ValueError("{} must be 3D, got shape {}".format(description, value.shape))
    return value


def _zscore_slice(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x, dtype=np.float32)
    std = float(y.std())
    return (y - float(y.mean())) / std if std > 1e-8 else y - float(y.mean())


def _orientation_score(noisy: np.ndarray, clean: np.ndarray) -> float:
    """Score noisy/clean alignment on a small deterministic slice sample."""
    n_slices = int(noisy.shape[0])
    sample_count = min(9, n_slices)
    indices = np.linspace(0, n_slices - 1, sample_count, dtype=np.int64)
    scores: list[float] = []
    for idx in indices:
        a = _zscore_slice(noisy[int(idx)])
        b = _zscore_slice(clean[int(idx)])
        denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
        if denom <= 1e-12:
            continue
        scores.append(float(np.sum(a * b) / denom))
    if not scores:
        return float("-inf")
    return float(sum(scores) / len(scores))


def _clean_orientation_candidates(clean: np.ndarray, noisy_shape: tuple[int, ...]) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    if tuple(clean.shape) == tuple(noisy_shape):
        candidates.append(("identity", clean))
        swapped = clean.transpose(0, 2, 1)
        if tuple(swapped.shape) == tuple(noisy_shape):
            candidates.append(("swap_hw", swapped))
    transposed = clean.transpose(2, 0, 1)
    if tuple(transposed.shape) == tuple(noisy_shape):
        candidates.append(("transpose(2,0,1)", transposed))
    transposed_swapped = clean.transpose(2, 1, 0)
    if tuple(transposed_swapped.shape) == tuple(noisy_shape):
        candidates.append(("transpose(2,1,0)", transposed_swapped))
    return candidates


def _orient_clean(
    noisy: np.ndarray,
    clean: np.ndarray,
    description: str,
    policy: str,
) -> tuple[np.ndarray, str, dict[str, float]]:
    noisy_shape = tuple(noisy.shape)
    candidates = _clean_orientation_candidates(clean, noisy_shape)
    if not candidates:
        raise ValueError(
            "{} has shape {}, which cannot be aligned to noisy shape {}".format(
                description, clean.shape, noisy_shape
            )
        )

    if policy not in ORIENTATION_POLICIES:
        raise ValueError("Unknown clean orientation policy: {}".format(policy))

    scores = {name: _orientation_score(noisy, candidate) for name, candidate in candidates}
    by_name = {name: candidate for name, candidate in candidates}

    if policy == "identity":
        preferred = "identity" if "identity" in by_name else "transpose(2,0,1)"
    elif policy == "swap-hw":
        preferred = "swap_hw" if "swap_hw" in by_name else "transpose(2,1,0)"
    else:
        preferred = max(scores, key=lambda name: scores[name])

    if preferred not in by_name:
        raise ValueError(
            "{} cannot use clean orientation policy {} with shape {}".format(
                description, policy, clean.shape
            )
        )
    print(
        "{} clean orientation: {} | scores {}".format(
            description,
            preferred,
            ", ".join("{}={:.4f}".format(name, scores[name]) for name in sorted(scores)),
        )
    )
    return by_name[preferred], preferred, scores


def _validate_values(array: np.ndarray, description: str) -> None:
    for start in range(0, array.shape[0], 32):
        if not np.isfinite(array[start : start + 32]).all():
            raise ValueError("{} contains non-finite values".format(description))


def _canonical_paths(output_dir: Path, volume_id: str) -> tuple[Path, Path]:
    vol_dir = output_dir / volume_id
    return (
        vol_dir / "seismic_w_noise_vol_{}.npy".format(volume_id),
        vol_dir / "seismicCubes_RFC_fullstack_2024.{}.npy".format(volume_id),
    )


def _save_plain_float32(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(".{}.tmp".format(path.name))
    with tmp_path.open("wb") as f:
        np.save(f, np.asarray(array, dtype=CANONICAL_DTYPE), allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(path))


def _inspect_canonical(path: Path, expected_shape: tuple[int, ...] | None = None) -> dict[str, object]:
    try:
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise RuntimeError("{} is not a plain memmap-compatible NPY: {}".format(path, exc)) from exc
    if array.dtype != CANONICAL_DTYPE:
        raise ValueError("{} has dtype {}, expected float32".format(path, array.dtype))
    if array.ndim != 3:
        raise ValueError("{} has shape {}, expected a 3D array".format(path, array.shape))
    if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
        raise ValueError("{} has shape {}, expected {}".format(path, array.shape, expected_shape))
    _validate_values(array, str(path))
    return {
        "path": str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _pair_is_ready(output_dir: Path, volume_id: str) -> bool:
    noisy_path, clean_path = _canonical_paths(output_dir, volume_id)
    if not noisy_path.is_file() or not clean_path.is_file():
        return False
    try:
        noisy = np.load(str(noisy_path), mmap_mode="r", allow_pickle=False)
        clean = np.load(str(clean_path), mmap_mode="r", allow_pickle=False)
        return (
            noisy.dtype == CANONICAL_DTYPE
            and clean.dtype == CANONICAL_DTYPE
            and tuple(noisy.shape) == CANONICAL_SHAPE
            and tuple(clean.shape) == CANONICAL_SHAPE
        )
    except Exception:
        return False


def _write_pair(
    output_dir: Path,
    volume_id: str,
    noisy: np.ndarray,
    clean: np.ndarray,
    source: str,
    clean_orientation_policy: str,
) -> dict[str, object]:
    if tuple(noisy.shape) != CANONICAL_SHAPE:
        raise ValueError(
            "{} noisy shape {} does not match expected {}".format(volume_id, noisy.shape, CANONICAL_SHAPE)
        )
    clean, clean_orientation, clean_orientation_scores = _orient_clean(
        noisy, clean, "{} clean".format(volume_id), clean_orientation_policy
    )
    _validate_values(noisy, "{} noisy".format(volume_id))
    _validate_values(clean, "{} clean".format(volume_id))

    noisy_path, clean_path = _canonical_paths(output_dir, volume_id)
    _save_plain_float32(noisy_path, noisy)
    _save_plain_float32(clean_path, clean)

    noisy_info = _inspect_canonical(noisy_path, CANONICAL_SHAPE)
    clean_info = _inspect_canonical(clean_path, CANONICAL_SHAPE)
    return {
        "volume_id": volume_id,
        "source": source,
        "clean_orientation": clean_orientation,
        "clean_orientation_policy": clean_orientation_policy,
        "clean_orientation_scores": clean_orientation_scores,
        "noisy": noisy_info,
        "clean": clean_info,
    }


def _discover_directory_pairs(input_dir: Path) -> dict[str, tuple[Path, Path]]:
    noisy: dict[str, Path] = {}
    clean: dict[str, Path] = {}
    for path in sorted(input_dir.rglob("*.npy")):
        relative = path.relative_to(input_dir).as_posix()
        noisy_match = NOISY_RE.search(relative)
        clean_match = CLEAN_RE.search(relative)
        if noisy_match:
            noisy[noisy_match.group(1)] = path
        elif clean_match:
            clean[clean_match.group(2)] = path
    ids = sorted(set(noisy) | set(clean), key=int)
    missing = [volume_id for volume_id in ids if volume_id not in noisy or volume_id not in clean]
    if missing:
        raise RuntimeError("Directory contains incomplete pairs: {}".format(", ".join(missing)))
    return {volume_id: (noisy[volume_id], clean[volume_id]) for volume_id in ids}


def _discover_zip_pairs(archive: zipfile.ZipFile) -> dict[str, tuple[str, str]]:
    noisy: dict[str, str] = {}
    clean: dict[str, str] = {}
    for name in sorted(archive.namelist()):
        noisy_match = NOISY_RE.search(name)
        clean_match = CLEAN_RE.search(name)
        if noisy_match:
            noisy[noisy_match.group(1)] = name
        elif clean_match:
            clean[clean_match.group(2)] = name
    ids = sorted(set(noisy) | set(clean), key=int)
    missing = [volume_id for volume_id in ids if volume_id not in noisy or volume_id not in clean]
    if missing:
        raise RuntimeError("Archive contains incomplete pairs: {}".format(", ".join(missing)))
    return {volume_id: (noisy[volume_id], clean[volume_id]) for volume_id in ids}


def _write_part_report(
    output_dir: Path,
    part_name: str,
    rows: list[dict[str, object]],
    volume_ids: Iterable[str],
) -> None:
    recorded_ids = list(volume_ids)
    report = {
        "part": part_name,
        "created_at_epoch_s": int(time.time()),
        "pair_count": len(recorded_ids),
        "volume_ids": recorded_ids,
        "new_pair_count": len(rows),
        "pairs": rows,
    }
    _atomic_json(output_dir / "_manifests" / "{}.json".format(part_name), report)


def _check_free_space(path: Path, min_free_gb: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(path)).free
    required = int(min_free_gb * 1_000_000_000)
    print("Free space at {}: {}".format(path, _human_gb(free)))
    if free < required:
        raise RuntimeError(
            "Insufficient free space at {}: {} available, {:.2f} GB required".format(
                path, _human_gb(free), min_free_gb
            )
        )


def import_directory(args: argparse.Namespace) -> None:
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    _check_free_space(output_dir.parent, args.min_free_gb)
    pairs = _discover_directory_pairs(input_dir)
    if args.expected_pairs is not None and len(pairs) != args.expected_pairs:
        raise RuntimeError("Expected {} input pairs, found {}".format(args.expected_pairs, len(pairs)))

    rows = []
    for index, (volume_id, (noisy_path, clean_path)) in enumerate(pairs.items(), start=1):
        if _pair_is_ready(output_dir, volume_id):
            print("[{}/{}] {} already canonical; skipping".format(index, len(pairs), volume_id))
            continue
        print("[{}/{}] importing {}".format(index, len(pairs), volume_id))
        noisy = _load_numeric(noisy_path, str(noisy_path))
        clean = _load_numeric(clean_path, str(clean_path))
        rows.append(
            _write_pair(
                output_dir,
                volume_id,
                noisy,
                clean,
                "directory:{}".format(input_dir),
                args.clean_orientation_policy,
            )
        )
    _write_part_report(output_dir, args.part_name, rows, pairs)
    print("Imported {} new pairs from {}".format(len(rows), input_dir))


def build_archive(args: argparse.Namespace) -> None:
    archive_path = args.archive.resolve()
    output_dir = args.output.resolve()
    _check_free_space(output_dir.parent, args.min_free_gb)
    rows = []
    with zipfile.ZipFile(str(archive_path)) as archive:
        pairs = _discover_zip_pairs(archive)
        if args.expected_pairs is not None and len(pairs) != args.expected_pairs:
            raise RuntimeError("Expected {} archive pairs, found {}".format(args.expected_pairs, len(pairs)))
        for index, (volume_id, (noisy_name, clean_name)) in enumerate(pairs.items(), start=1):
            if _pair_is_ready(output_dir, volume_id):
                print("[{}/{}] {} already canonical; skipping".format(index, len(pairs), volume_id))
                continue
            print("[{}/{}] normalizing {} from {}".format(index, len(pairs), volume_id, archive_path.name))
            with archive.open(noisy_name) as noisy_file:
                noisy = _load_numeric(noisy_file, "{}:{}".format(archive_path, noisy_name))
            with archive.open(clean_name) as clean_file:
                clean = _load_numeric(clean_file, "{}:{}".format(archive_path, clean_name))
            rows.append(
                _write_pair(
                    output_dir,
                    volume_id,
                    noisy,
                    clean,
                    "archive:{}".format(archive_path.name),
                    args.clean_orientation_policy,
                )
            )
    _write_part_report(output_dir, args.part_name, rows, pairs)
    print("Normalized {} new pairs from {}".format(len(rows), archive_path))


def _canonical_pairs(output_dir: Path) -> dict[str, tuple[Path, Path]]:
    return _discover_directory_pairs(output_dir)


def _split_ids(volume_ids: list[str], seed: int, n_train: int, n_val: int, n_test: int) -> dict[str, list[str]]:
    if n_train + n_val + n_test > len(volume_ids):
        raise ValueError("Split sizes exceed available pairs")
    ordered = sorted(volume_ids, key=int)
    permutation = np.random.RandomState(seed).permutation(len(ordered))
    shuffled = [ordered[i] for i in permutation]
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def audit(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    pairs = _canonical_pairs(output_dir)
    if len(pairs) != args.expected_pairs:
        raise RuntimeError("Expected {} canonical pairs, found {}".format(args.expected_pairs, len(pairs)))

    rows = []
    for index, (volume_id, (noisy_path, clean_path)) in enumerate(pairs.items(), start=1):
        print("[{}/{}] auditing {}".format(index, len(pairs), volume_id))
        noisy_info = _inspect_canonical(noisy_path, CANONICAL_SHAPE)
        clean_info = _inspect_canonical(clean_path, CANONICAL_SHAPE)
        rows.append({"volume_id": volume_id, "noisy": noisy_info, "clean": clean_info})

    manifest = {
        "status": "ready",
        "created_at_epoch_s": int(time.time()),
        "root_dir": str(output_dir),
        "pair_count": len(rows),
        "canonical_shape": list(CANONICAL_SHAPE),
        "canonical_dtype": str(CANONICAL_DTYPE),
        "split": {
            "seed": args.seed,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "volume_ids": _split_ids(list(pairs), args.seed, args.n_train, args.n_val, args.n_test),
        },
        "pairs": rows,
    }
    _atomic_json(output_dir / MANIFEST_NAME, manifest)
    print("Dataset ready: {} pairs at {}".format(len(rows), output_dir))


def check_ready(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise RuntimeError("Missing ready manifest: {}".format(manifest_path))
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("status") != "ready":
        raise RuntimeError("{} does not have status=ready".format(manifest_path))
    if int(manifest.get("pair_count", -1)) != args.expected_pairs:
        raise RuntimeError(
            "{} has pair_count={}, expected {}".format(
                manifest_path, manifest.get("pair_count"), args.expected_pairs
            )
        )
    for row in manifest.get("pairs", []):
        for key in ("noisy", "clean"):
            path = Path(row[key]["path"])
            if not path.is_file():
                raise RuntimeError("Manifest references missing file: {}".format(path))
    print("Dataset ready manifest accepted: {} pairs at {}".format(args.expected_pairs, output_dir))


def check_part(args: argparse.Namespace) -> None:
    output_dir = args.output.resolve()
    report_path = output_dir / "_manifests" / "{}.json".format(args.part_name)
    if not report_path.is_file():
        raise RuntimeError("Missing part report: {}".format(report_path))
    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    volume_ids = [str(volume_id) for volume_id in report.get("volume_ids", [])]
    if not volume_ids:
        raise RuntimeError("{} does not record volume_ids".format(report_path))
    missing = [volume_id for volume_id in volume_ids if not _pair_is_ready(output_dir, volume_id)]
    if missing:
        raise RuntimeError(
            "{} references missing or invalid canonical pairs: {}".format(
                report_path, ", ".join(missing)
            )
        )
    print("Part report accepted: {} canonical pairs for {}".format(len(volume_ids), args.part_name))


def check_space(args: argparse.Namespace) -> None:
    path = args.path.resolve()
    marker = args.marker.resolve()
    if marker.is_file() and not args.force:
        print("Initial space gate already passed: {}".format(marker))
        return
    _check_free_space(path, args.min_free_gb)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "passed_at_epoch_s={}\npath={}\nmin_free_gb={}\n".format(
            int(time.time()), path, args.min_free_gb
        ),
        encoding="utf-8",
    )
    print("Recorded initial space gate: {}".format(marker))


def _add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True, help="Canonical extracted/ output directory")
    parser.add_argument("--part-name", required=True, help="Stable report name for this source part")
    parser.add_argument("--expected-pairs", type=int, help="Fail unless the input source has this many pairs")
    parser.add_argument("--min-free-gb", type=float, default=20.0, help="Required free space before processing")
    parser.add_argument(
        "--clean-orientation-policy",
        choices=ORIENTATION_POLICIES,
        default="auto",
        help="How to choose clean-target orientation when multiple shape-compatible candidates exist",
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import-dir", help="Import an existing extracted directory")
    import_parser.add_argument("--input", type=Path, required=True)
    _add_common_build_args(import_parser)
    import_parser.set_defaults(func=import_directory)

    archive_parser = subparsers.add_parser("build-archive", help="Normalize one official ZIP archive")
    archive_parser.add_argument("--archive", type=Path, required=True)
    _add_common_build_args(archive_parser)
    archive_parser.set_defaults(func=build_archive)

    audit_parser = subparsers.add_parser("audit", help="Fully validate the canonical dataset and write its manifest")
    audit_parser.add_argument("--output", type=Path, required=True)
    audit_parser.add_argument("--expected-pairs", type=int, default=250)
    audit_parser.add_argument("--seed", type=int, default=42)
    audit_parser.add_argument("--n-train", type=int, default=200)
    audit_parser.add_argument("--n-val", type=int, default=25)
    audit_parser.add_argument("--n-test", type=int, default=25)
    audit_parser.set_defaults(func=audit)

    ready_parser = subparsers.add_parser("check-ready", help="Quickly require a previously audited dataset")
    ready_parser.add_argument("--output", type=Path, required=True)
    ready_parser.add_argument("--expected-pairs", type=int, default=250)
    ready_parser.set_defaults(func=check_ready)

    part_parser = subparsers.add_parser("check-part", help="Quickly validate a completed source-part report")
    part_parser.add_argument("--output", type=Path, required=True)
    part_parser.add_argument("--part-name", required=True)
    part_parser.set_defaults(func=check_part)

    space_parser = subparsers.add_parser("check-space", help="Apply the one-time staff-bulk free-space gate")
    space_parser.add_argument("--path", type=Path, required=True)
    space_parser.add_argument("--marker", type=Path, required=True)
    space_parser.add_argument("--min-free-gb", type=float, default=350.0)
    space_parser.add_argument("--force", action="store_true")
    space_parser.set_defaults(func=check_space)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
