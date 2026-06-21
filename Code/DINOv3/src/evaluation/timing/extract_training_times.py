"""Extract training time statistics from run artifacts and SLURM log files.

Scans all run directories under experiments/runs/ and produces two CSVs:
  - experiments/summaries/timing/training_times_per_job.csv  (one row per SLURM job)
  - experiments/summaries/timing/training_times_per_run.csv  (one row per run directory)

Timing source priority:
  1. training_timing.csv train_step_time_s (exact train-step timing)
  2. history.csv epoch_time_s (approximate for train-step; includes validation)
  3. SLURM wallclock logs (approximate operational runtime)

Usage (on DAIC, use $PY310 not the system python):
    $PY310 evaluation/extract_training_times.py
    $PY310 evaluation/extract_training_times.py --runs-root /path/to/experiments/runs
    $PY310 evaluation/extract_training_times.py --family 2d 3ch 5ch
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

try:
    import yaml
except ImportError:  # pragma: no cover - DAIC env has PyYAML, local py_compile may not
    yaml = None

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

PROJECT_ROOT = default_project_root(__file__)  # Code/DINOv3/src -> Code/DINOv3 -> Code -> RP
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_OUT_DIR   = PROJECT_ROOT / "experiments" / "summaries" / "timing"


_JOB_STARTED_RE  = re.compile(
    r"Job started on \S+ at (\w+ +\w+ +\d+ +\d+:\d+:\d+) \w+ (\d{4})"
)
_JOB_FINISHED_RE = re.compile(
    r"Job finished at (\w+ +\w+ +\d+ +\d+:\d+:\d+) \w+ (\d{4})"
)
_CLEAN_STOP_RE   = re.compile(
    r"Stopping cleanly before walltime: elapsed=([\d.]+) min, avg_epoch=([\d.]+) min, next_epoch=(\d+)"
)
_RESUME_RE       = re.compile(r"Resuming from .+ at epoch (\d+)/(\d+)")
_STOPPED_RE      = re.compile(r"Training stopped at epoch (\d+)/(\d+)")
_TOTAL_PARAMS_RE = re.compile(r"Total params:\s+([\d,]+)")
_TRAIN_PARAMS_RE = re.compile(r"Trainable params:\s+([\d,]+)\s+\(([\d.]+)%\)")


def _parse_timestamp(dt_str: str, year: str) -> datetime | None:
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b  %d %H:%M:%S %Y"):
        try:
            return datetime.strptime(f"{dt_str} {year}", fmt)
        except ValueError:
            pass
    return None


def parse_log(path: Path) -> dict:
    result = {
        "job_id":              path.stem.replace("slurm_", ""),
        "start_ts":            None,
        "end_ts":              None,
        "wallclock_min":       None,
        "start_epoch":         1,
        "end_epoch":           None,
        "n_epochs_this_job":   None,
        "clean_stop_elapsed":  None,
        "clean_stop_avg_epoch": None,
        "clean_stop_next":     None,
        "total_params":        None,
        "trainable_params":    None,
        "trainable_pct":       None,
    }

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    for line in text.splitlines():
        if (m := _JOB_STARTED_RE.search(line)):
            result["start_ts"] = _parse_timestamp(m.group(1), m.group(2))
        elif (m := _JOB_FINISHED_RE.search(line)):
            result["end_ts"] = _parse_timestamp(m.group(1), m.group(2))
        elif (m := _CLEAN_STOP_RE.search(line)):
            result["clean_stop_elapsed"]   = float(m.group(1))
            result["clean_stop_avg_epoch"] = float(m.group(2))
            result["clean_stop_next"]      = int(m.group(3))
        elif (m := _RESUME_RE.search(line)):
            result["start_epoch"] = int(m.group(1))
        elif (m := _STOPPED_RE.search(line)):
            result["end_epoch"] = int(m.group(1))
        elif (m := _TOTAL_PARAMS_RE.search(line)):
            result["total_params"] = int(m.group(1).replace(",", ""))
        elif (m := _TRAIN_PARAMS_RE.search(line)):
            result["trainable_params"] = int(m.group(1).replace(",", ""))
            result["trainable_pct"]    = float(m.group(2))

    if result["start_ts"] and result["end_ts"]:
        delta = result["end_ts"] - result["start_ts"]
        result["wallclock_min"] = delta.total_seconds() / 60.0

    if result["end_epoch"] is not None:
        result["n_epochs_this_job"] = (
            result["end_epoch"] - result["start_epoch"] + 1
        )

    return result


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _int_or_none(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _round_or_blank(value: float | None, digits: int = 2) -> float | str:
    return round(value, digits) if value is not None else ""


def _read_gpu_count(run_dir: Path) -> int:
    if yaml is None:
        return 1
    meta_path = run_dir / "run_meta.yaml"
    if not meta_path.exists():
        return 1
    try:
        with meta_path.open() as f:
            meta = yaml.safe_load(f) or {}
    except OSError:
        return 1
    timing = meta.get("timing") or {}
    value = timing.get("gpu_count", 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _history_info(run_dir: Path) -> dict[str, object]:
    rows = _read_csv_rows(run_dir / "history.csv")
    epochs = [_int_or_none(r.get("epoch")) for r in rows]
    epochs = [e for e in epochs if e is not None]
    times = [_float_or_none(r.get("epoch_time_s")) for r in rows]
    times = [t for t in times if t is not None]
    return {
        "max_epoch": max(epochs) if epochs else None,
        "timed_epochs": len(times),
        "epoch_active_time_min": sum(times) / 60.0 if times else None,
    }


def _timing_artifact_info(run_dir: Path) -> dict[str, object] | None:
    rows = _read_csv_rows(run_dir / "training_timing.csv")
    timed_rows = [r for r in rows if _float_or_none(r.get("train_step_time_s")) is not None]
    if not timed_rows:
        return None

    train_step_s = sum(_float_or_none(r.get("train_step_time_s")) or 0.0 for r in timed_rows)
    validation_s = sum(_float_or_none(r.get("validation_time_s")) or 0.0 for r in timed_rows)
    checkpoint_s = sum(_float_or_none(r.get("checkpoint_time_s")) or 0.0 for r in timed_rows)
    epoch_total_s = sum(_float_or_none(r.get("epoch_total_time_s")) or 0.0 for r in timed_rows)
    epochs = [_int_or_none(r.get("epoch")) for r in timed_rows]
    epochs = [e for e in epochs if e is not None]
    schema = next((r.get("timing_schema_version", "") for r in timed_rows if r.get("timing_schema_version")), "")
    return {
        "timing_source": "training_timing.csv",
        "timing_is_exact": True,
        "timing_schema_version": schema,
        "timed_epochs": len(timed_rows),
        "max_epoch": max(epochs) if epochs else None,
        "train_step_time_min": train_step_s / 60.0,
        "validation_time_min": validation_s / 60.0,
        "checkpoint_time_min": checkpoint_s / 60.0,
        "epoch_total_time_min": epoch_total_s / 60.0,
        "epoch_active_time_min": (train_step_s + validation_s) / 60.0,
    }


def find_run_dirs(runs_root: Path) -> list[Path]:
    return sorted(p.parent for p in runs_root.rglob("run_meta.yaml"))


def process_run(run_dir: Path, runs_root: Path) -> tuple[list[dict], dict]:
    rel = run_dir.relative_to(runs_root)
    parts = rel.parts
    family  = parts[0]
    variant = parts[1] if len(parts) > 1 else ""
    run_id  = parts[2] if len(parts) > 2 else ""
    gpu_count = _read_gpu_count(run_dir)
    history = _history_info(run_dir)
    timing = _timing_artifact_info(run_dir)

    logs_dir = run_dir / "logs"
    log_files = sorted(logs_dir.glob("slurm_*.out")) if logs_dir.exists() else []

    per_job_rows = []
    for lf in log_files:
        parsed = parse_log(lf)
        row = {
            "run_dir":              str(rel).replace("\\", "/"),
            "family":               family,
            "variant":              variant,
            "run_id":               run_id,
            "job_id":               parsed["job_id"],
            "start_ts":             parsed["start_ts"].isoformat() if parsed["start_ts"] else "",
            "end_ts":               parsed["end_ts"].isoformat()   if parsed["end_ts"]   else "",
            "wallclock_min":        round(parsed["wallclock_min"], 2) if parsed["wallclock_min"] is not None else "",
            "start_epoch":          parsed["start_epoch"],
            "end_epoch":            parsed["end_epoch"] if parsed["end_epoch"] is not None else "",
            "n_epochs_this_job":    parsed["n_epochs_this_job"] if parsed["n_epochs_this_job"] is not None else "",
            "clean_stop_elapsed":   parsed["clean_stop_elapsed"]   if parsed["clean_stop_elapsed"]   is not None else "",
            "clean_stop_avg_epoch": parsed["clean_stop_avg_epoch"] if parsed["clean_stop_avg_epoch"] is not None else "",
            "total_params":         parsed["total_params"]    if parsed["total_params"]    is not None else "",
            "trainable_params":     parsed["trainable_params"] if parsed["trainable_params"] is not None else "",
            "trainable_pct":        parsed["trainable_pct"]   if parsed["trainable_pct"]   is not None else "",
        }
        per_job_rows.append(row)

    # Per-run summary --------------------------------------------------------
    total_wallclock = sum(
        float(r["wallclock_min"]) for r in per_job_rows if r["wallclock_min"] != ""
    ) if per_job_rows else None

    # Best avg_epoch estimate: minimum clean_stop value from jobs that ran >= 5 epochs
    # (excludes pathologically slow single-epoch outlier jobs)
    clean_avgs = [
        float(r["clean_stop_avg_epoch"])
        for r in per_job_rows
        if r["clean_stop_avg_epoch"] != "" and r["n_epochs_this_job"] != "" and int(r["n_epochs_this_job"]) >= 5
    ]
    best_avg_epoch = min(clean_avgs) if clean_avgs else None

    # Total epochs: highest end_epoch seen across all jobs
    end_epochs = [
        int(r["end_epoch"]) for r in per_job_rows if r["end_epoch"] != ""
    ]
    total_epochs_completed = max(end_epochs) if end_epochs else None

    # Computed avg from (total wallclock / total_epochs) — includes setup overhead
    computed_avg = None
    if total_wallclock and total_epochs_completed:
        computed_avg = round(total_wallclock / total_epochs_completed, 2)

    # Job-level GPU hours are operational wallclock, not exact train-step time.
    gpu_hours = round(total_wallclock / 60 * gpu_count, 2) if total_wallclock else None

    # Params from first log that has them
    total_params = trainable_params = trainable_pct = ""
    for r in per_job_rows:
        if r["total_params"] != "":
            total_params = r["total_params"]
            trainable_params = r["trainable_params"]
            trainable_pct    = r["trainable_pct"]
            break

    history_max_epoch = history.get("max_epoch")
    timing_max_epoch = timing.get("max_epoch") if timing else None
    epoch_candidates = [
        e for e in [total_epochs_completed, history_max_epoch, timing_max_epoch]
        if isinstance(e, int)
    ]
    total_epochs_completed = max(epoch_candidates) if epoch_candidates else None

    timing_source = ""
    timing_is_exact: bool | str = ""
    timing_schema_version = ""
    timed_epochs: int | str = ""
    train_step_time_min = None
    validation_time_min = None
    checkpoint_time_min = None
    epoch_total_time_min = None
    epoch_active_time_min = history.get("epoch_active_time_min")

    if timing:
        timing_source = str(timing["timing_source"])
        timing_is_exact = bool(timing["timing_is_exact"])
        timing_schema_version = str(timing.get("timing_schema_version", ""))
        timed_epochs = int(timing["timed_epochs"])
        train_step_time_min = float(timing["train_step_time_min"])
        validation_time_min = float(timing["validation_time_min"])
        checkpoint_time_min = float(timing["checkpoint_time_min"])
        epoch_total_time_min = float(timing["epoch_total_time_min"])
        epoch_active_time_min = float(timing["epoch_active_time_min"])
    elif history.get("epoch_active_time_min") is not None:
        timing_source = "history_epoch_time_s"
        timing_is_exact = False
        timed_epochs = int(history["timed_epochs"])
        train_step_time_min = float(history["epoch_active_time_min"])
        epoch_active_time_min = train_step_time_min
    elif total_wallclock is not None:
        timing_source = "slurm_wallclock"
        timing_is_exact = False
        timed_epochs = total_epochs_completed if total_epochs_completed is not None else ""
        train_step_time_min = total_wallclock

    train_step_gpu_hours = (
        train_step_time_min / 60.0 * gpu_count if train_step_time_min is not None else None
    )

    summary = {
        "run_dir":                  str(rel).replace("\\", "/"),
        "family":                   family,
        "variant":                  variant,
        "run_id":                   run_id,
        "n_jobs":                   len(per_job_rows),
        "total_epochs_completed":   total_epochs_completed if total_epochs_completed is not None else "",
        "timing_source":            timing_source,
        "timing_is_exact":          timing_is_exact,
        "timing_schema_version":    timing_schema_version,
        "timed_epochs":             timed_epochs,
        "gpu_count":                gpu_count,
        "train_step_time_min":      _round_or_blank(train_step_time_min),
        "train_step_gpu_hours":     _round_or_blank(train_step_gpu_hours),
        "epoch_active_time_min":    _round_or_blank(epoch_active_time_min),
        "validation_time_min":      _round_or_blank(validation_time_min),
        "checkpoint_time_min":      _round_or_blank(checkpoint_time_min),
        "epoch_total_time_min":     _round_or_blank(epoch_total_time_min),
        "job_wallclock_min":        _round_or_blank(total_wallclock),
        "job_gpu_hours":            gpu_hours if gpu_hours is not None else "",
        "total_wallclock_min":      round(total_wallclock, 2) if total_wallclock is not None else "",
        "gpu_hours":                gpu_hours if gpu_hours is not None else "",
        "best_avg_epoch_min":       round(best_avg_epoch, 2) if best_avg_epoch is not None else "",
        "computed_avg_epoch_min":   computed_avg if computed_avg is not None else "",
        "total_params":             total_params,
        "trainable_params":         trainable_params,
        "trainable_pct":            trainable_pct,
    }

    return per_job_rows, summary


PER_JOB_FIELDS = [
    "run_dir", "family", "variant", "run_id", "job_id",
    "start_ts", "end_ts", "wallclock_min",
    "start_epoch", "end_epoch", "n_epochs_this_job",
    "clean_stop_elapsed", "clean_stop_avg_epoch",
    "total_params", "trainable_params", "trainable_pct",
]

PER_RUN_FIELDS = [
    "run_dir", "family", "variant", "run_id",
    "n_jobs", "total_epochs_completed",
    "timing_source", "timing_is_exact", "timing_schema_version", "timed_epochs",
    "gpu_count",
    "train_step_time_min", "train_step_gpu_hours",
    "epoch_active_time_min", "validation_time_min", "checkpoint_time_min",
    "epoch_total_time_min",
    "job_wallclock_min", "job_gpu_hours",
    "total_wallclock_min", "gpu_hours",
    "best_avg_epoch_min", "computed_avg_epoch_min",
    "total_params", "trainable_params", "trainable_pct",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--family", nargs="*",
        help="Limit to these top-level family names (e.g. 2d 3ch 5ch)"
    )
    args = parser.parse_args()

    runs_root: Path = args.runs_root
    out_dir:   Path = args.out_dir

    if not runs_root.exists():
        print(f"Runs root not found: {runs_root}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = find_run_dirs(runs_root)
    if args.family:
        allowed = set(args.family)
        run_dirs = [d for d in run_dirs if d.relative_to(runs_root).parts[0] in allowed]

    print(f"Found {len(run_dirs)} run directories under {runs_root}")

    all_job_rows: list[dict] = []
    all_run_summaries: list[dict] = []

    for rd in run_dirs:
        job_rows, summary = process_run(rd, runs_root)
        all_job_rows.extend(job_rows)
        all_run_summaries.append(summary)
        n_jobs = summary["n_jobs"]
        epochs = summary["total_epochs_completed"]
        train_time = summary["train_step_time_min"]
        source = summary["timing_source"] or "none"
        exact = summary["timing_is_exact"]
        print(
            f"  {summary['run_dir']}: {n_jobs} job(s), {epochs} epochs, "
            f"{train_time} min training ({source}, exact={exact})"
        )

    job_csv = out_dir / "training_times_per_job.csv"
    with job_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_JOB_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_job_rows)
    print(f"\nPer-job CSV:  {job_csv}")

    run_csv = out_dir / "training_times_per_run.csv"
    with run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_run_summaries)
    print(f"Per-run CSV:  {run_csv}")


if __name__ == "__main__":
    main()
