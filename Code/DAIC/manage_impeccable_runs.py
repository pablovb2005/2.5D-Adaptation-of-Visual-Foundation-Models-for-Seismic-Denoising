#!/usr/bin/env python3
"""Manage Image Impeccable DAIC runs.

This helper is intentionally conservative:
  - ``status`` only reads history files.
  - ``resubmit`` defaults to dry-run and checks ``squeue`` before printing commands.
  - ``runtime`` reads logs and ``sacct`` to write per-chunk and aggregate runtime rows.

Run on DAIC, for example:

    python3 ~/RP/Code/DAIC/manage_impeccable_runs.py status --scope all
    python3 ~/RP/Code/DAIC/manage_impeccable_runs.py resubmit --scope all
    python3 ~/RP/Code/DAIC/manage_impeccable_runs.py resubmit --scope all --submit
    python3 ~/RP/Code/DAIC/manage_impeccable_runs.py runtime --scope all
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_STUDENT_DIR = Path(
    "/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal"
)


@dataclass(frozen=True)
class RunSpec:
    scope: str
    label: str
    rel_exp: str
    target_epochs: int
    submit_cmd: str
    array_index: int | None = None
    queue_marker: str = ""


@dataclass(frozen=True)
class ResolvedRunDir:
    path: Path
    layout: str
    canonical_path: Path
    legacy_path: Path | None = None


MAIN_SPECS = [
    RunSpec(
        "main",
        "MAIN 2D LoRA",
        "experiments/runs/2d/impeccable_repeated_stride5_lora_r16/seed42_run01",
        50,
        "~/RP/Code/DAIC/2d/impeccable_repeated_stride5_lora_r16/seed42_run01/submit.sh",
        queue_marker="2d_s5_r16_s42_r01",
    ),
    RunSpec(
        "main",
        "MAIN 2D LoRA seed43",
        "experiments/runs/2d/impeccable_repeated_stride5_lora_r16/seed43_run02",
        50,
        "~/RP/Code/DAIC/2d/impeccable_repeated_stride5_lora_r16/seed43_run02/submit.sh",
        queue_marker="2d_s5_r16_s43_r02",
    ),
    RunSpec(
        "main",
        "MAIN 2D LoRA seed44",
        "experiments/runs/2d/impeccable_repeated_stride5_lora_r16/seed44_run03",
        50,
        "~/RP/Code/DAIC/2d/impeccable_repeated_stride5_lora_r16/seed44_run03/submit.sh",
        queue_marker="2d_s5_r16_s44_r03",
    ),
    RunSpec(
        "main",
        "MAIN 3ch LoRA",
        "experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16/seed42_run01",
        50,
        "~/RP/Code/DAIC/3ch/impeccable_neighbors3_stride5_lora_r16/seed42_run01/submit.sh",
        queue_marker="3ch_s5_r16_s42_r01",
    ),
    RunSpec(
        "main",
        "MAIN 3ch LoRA seed43",
        "experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16/seed43_run02",
        50,
        "~/RP/Code/DAIC/3ch/impeccable_neighbors3_stride5_lora_r16/seed43_run02/submit.sh",
        queue_marker="3ch_s5_r16_s43_r02",
    ),
    RunSpec(
        "main",
        "MAIN 3ch LoRA seed44",
        "experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16/seed44_run03",
        50,
        "~/RP/Code/DAIC/3ch/impeccable_neighbors3_stride5_lora_r16/seed44_run03/submit.sh",
        queue_marker="3ch_s5_r16_s44_r03",
    ),
    RunSpec(
        "main",
        "MAIN 5ch control patch emb",
        "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_head/seed42_run01",
        50,
        "~/RP/Code/DAIC/5ch/impeccable_neighbors5_stride5_patch_emb_head/seed42_run01/submit.sh",
        queue_marker="5ch_ctrl_s5_s42_r01",
    ),
    RunSpec(
        "main",
        "MAIN 5ch patch emb + LoRA",
        "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed42_run01",
        50,
        "~/RP/Code/DAIC/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed42_run01/submit.sh",
        queue_marker="5ch_s5_r16_s42_r01",
    ),
    RunSpec(
        "main",
        "MAIN 5ch patch emb + LoRA seed43",
        "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed43_run02",
        50,
        "~/RP/Code/DAIC/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed43_run02/submit.sh",
        queue_marker="5ch_s5_r16_s43_r02",
    ),
    RunSpec(
        "main",
        "MAIN 5ch patch emb + LoRA seed44",
        "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed44_run03",
        50,
        "~/RP/Code/DAIC/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed44_run03/submit.sh",
        queue_marker="5ch_s5_r16_s44_r03",
    ),
]


DATA_EFFICIENCY_SPECS = [
    RunSpec("data_efficiency", "DE 0 | 2D n=5", "experiments/runs/2d/impeccable_repeated_stride5_lora_r16_n05vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 0, "de_"),
    RunSpec("data_efficiency", "DE 1 | 2D n=10", "experiments/runs/2d/impeccable_repeated_stride5_lora_r16_n10vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 1, "de_"),
    RunSpec("data_efficiency", "DE 2 | 2D n=15", "experiments/runs/2d/impeccable_repeated_stride5_lora_r16_n15vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 2, "de_"),
    RunSpec("data_efficiency", "DE 3 | 3ch n=5", "experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16_n05vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 3, "de_"),
    RunSpec("data_efficiency", "DE 4 | 3ch n=10", "experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16_n10vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 4, "de_"),
    RunSpec("data_efficiency", "DE 5 | 3ch n=15", "experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16_n15vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 5, "de_"),
    RunSpec("data_efficiency", "DE 6 | 5ch n=5", "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n05vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 6, "de_"),
    RunSpec("data_efficiency", "DE 7 | 5ch n=10", "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n10vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 7, "de_"),
    RunSpec("data_efficiency", "DE 8 | 5ch n=15", "experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n15vols/seed42_run01", 50, "~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh", 8, "de_"),
]


ABLATION_SPECS = [
    RunSpec("ablations", "ABL 0 | 2D stride=3 n=5", "experiments/runs/ablations/2d/stride3_n05vols/seed42_run01", 30, "~/RP/Code/DAIC/ablations/submit_ablations.sh", 0, "abl_"),
    RunSpec("ablations", "ABL 1 | 2D stride=1 n=5", "experiments/runs/ablations/2d/stride1_n05vols/seed42_run01", 10, "~/RP/Code/DAIC/ablations/submit_ablations.sh", 1, "abl_"),
    RunSpec("ablations", "ABL 2 | 3ch neighbor_stride=2 n=5", "experiments/runs/ablations/3ch/ns2_stride5_n05vols/seed42_run01", 50, "~/RP/Code/DAIC/ablations/submit_ablations.sh", 2, "abl_"),
    RunSpec("ablations", "ABL 3 | 3ch neighbor_stride=3 n=5", "experiments/runs/ablations/3ch/ns3_stride5_n05vols/seed42_run01", 50, "~/RP/Code/DAIC/ablations/submit_ablations.sh", 3, "abl_"),
    RunSpec("ablations", "ABL 4 | 3ch grid4 n=5", "experiments/runs/ablations/3ch/grid4_stride5_n05vols/seed42_run01", 13, "~/RP/Code/DAIC/ablations/submit_ablations.sh", 4, "abl_"),
]


ALL_SPECS = MAIN_SPECS + DATA_EFFICIENCY_SPECS + ABLATION_SPECS
LEGACY_MAIN_FAMILIES = {"2d", "3ch", "5ch"}
RUN_MARKERS = (
    "history.csv",
    "best.pt",
    "last.pt",
    "run_meta.yaml",
    "eval_results/results.csv",
)


def _specs(scope: str) -> list[RunSpec]:
    if scope == "all":
        return ALL_SPECS
    return [spec for spec in ALL_SPECS if spec.scope == scope]


def _exp_dir(student_dir: Path, spec: RunSpec) -> Path:
    return _resolve_exp_dir(student_dir, spec).path


def _canonical_exp_dir(student_dir: Path, spec: RunSpec) -> Path:
    return student_dir / spec.rel_exp


def _legacy_exp_dir(student_dir: Path, spec: RunSpec) -> Path | None:
    if spec.scope != "main":
        return None
    parts = Path(spec.rel_exp).parts
    if len(parts) < 4 or parts[0] != "experiments" or parts[1] != "runs":
        return None
    family = parts[2]
    if family not in LEGACY_MAIN_FAMILIES:
        return None
    return student_dir / "experiments" / Path(*parts[2:])


def _has_run_artifact(path: Path) -> bool:
    return any((path / marker).exists() for marker in RUN_MARKERS)


def _resolve_exp_dir(student_dir: Path, spec: RunSpec) -> ResolvedRunDir:
    canonical = _canonical_exp_dir(student_dir, spec)
    legacy = _legacy_exp_dir(student_dir, spec)

    if _has_run_artifact(canonical):
        return ResolvedRunDir(canonical, "canonical", canonical, legacy)
    if legacy is not None and _has_run_artifact(legacy):
        return ResolvedRunDir(legacy, "legacy", canonical, legacy)
    return ResolvedRunDir(canonical, "missing", canonical, legacy)


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_history(exp_dir: Path, target_epochs: int) -> dict[str, object]:
    path = exp_dir / "history.csv"
    if not path.exists():
        return {
            "exists": False,
            "last_epoch": 0,
            "total_epochs": target_epochs,
            "last_val_ms_ssim": None,
            "last_ms_ssim_r": None,
            "best_val_ms_ssim": None,
            "best_epoch": None,
            "complete": False,
        }

    rows: list[dict[str, str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = row.get("epoch")
            if epoch and epoch.isdigit():
                rows.append(row)

    if not rows:
        return {
            "exists": True,
            "last_epoch": 0,
            "total_epochs": target_epochs,
            "last_val_ms_ssim": None,
            "last_ms_ssim_r": None,
            "best_val_ms_ssim": None,
            "best_epoch": None,
            "complete": False,
        }

    last = rows[-1]
    validated_rows = [r for r in rows if _float_or_none(r.get("val_ms_ssim")) is not None]
    best = max(validated_rows, key=lambda r: _float_or_none(r.get("val_ms_ssim")) or float("-inf")) if validated_rows else None
    last_validated = validated_rows[-1] if validated_rows else None
    last_epoch = int(last["epoch"])
    total_epochs = int(last.get("total_epochs") or target_epochs)
    return {
        "exists": True,
        "last_epoch": last_epoch,
        "total_epochs": total_epochs,
        "last_val_ms_ssim": _float_or_none(last_validated.get("val_ms_ssim")) if last_validated else None,
        "last_ms_ssim_r": _float_or_none(last_validated.get("val_ms_ssim_r")) if last_validated else None,
        "best_val_ms_ssim": _float_or_none(best.get("val_ms_ssim")) if best else None,
        "best_epoch": int(best["epoch"]) if best else None,
        "complete": last_epoch >= total_epochs,
    }


def _fmt_float(value: object) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _status_lines(student_dir: Path, specs: Iterable[RunSpec]) -> list[str]:
    lines: list[str] = []
    for spec in specs:
        resolved = _resolve_exp_dir(student_dir, spec)
        exp_dir = resolved.path
        hist = _parse_history(exp_dir, spec.target_epochs)
        layout_note = "" if resolved.layout == "canonical" else f" | layout={resolved.layout}"
        if not hist["exists"]:
            if resolved.legacy_path is not None and resolved.layout == "missing":
                lines.append(
                    f"{spec.label}: MISSING history.csv | {exp_dir} "
                    f"| checked legacy: {resolved.legacy_path}"
                )
            else:
                lines.append(f"{spec.label}: MISSING history.csv | {exp_dir}{layout_note}")
            continue
        if hist["complete"]:
            status = "DONE"
        elif resolved.layout == "legacy":
            status = "LEGACY_REVIEW"
        else:
            status = "NEEDS_RESUBMIT"
        lines.append(
            f"{spec.label}: {status} | "
            f"epoch {hist['last_epoch']}/{hist['total_epochs']} | "
            f"last={_fmt_float(hist['last_val_ms_ssim'])} | "
            f"best={_fmt_float(hist['best_val_ms_ssim'])} @ {hist['best_epoch']}"
            f"{layout_note}"
        )
    return lines


def _run(args: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return None


def _queued_jobs() -> list[tuple[str, str, str]]:
    user = os.environ.get("USER", "")
    cmd = ["squeue", "-h", "-u", user, "-o", "%i|%j|%T"] if user else ["squeue", "-h", "-o", "%i|%j|%T"]
    result = _run(cmd)
    if result is None or result.returncode != 0:
        return []
    jobs = []
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 3:
            jobs.append((parts[0], parts[1], parts[2]))
    return jobs


def _queue_has_marker(marker: str, queued: list[tuple[str, str, str]]) -> bool:
    if not marker:
        return False
    for _job_id, name, _state in queued:
        if name == marker or name.startswith(marker):
            return True
    return False


def _submit_args(script: str, indices: list[int] | None) -> list[str]:
    script_path = os.path.expanduser(script)
    if indices:
        return ["sbatch", f"--array={','.join(str(i) for i in indices)}", script_path]
    return ["sbatch", script_path]


def _print_shell_command(args: list[str]) -> str:
    return " ".join(args).replace(os.path.expanduser("~"), "~")


def command_status(args: argparse.Namespace) -> int:
    for line in _status_lines(args.student_dir, _specs(args.scope)):
        print(line)
    return 0


def _print_legacy_review(items: list[tuple[RunSpec, ResolvedRunDir, dict[str, object]]]) -> None:
    print("Manual review required for legacy-layout incomplete runs.")
    print("These are not auto-submitted into experiments/runs to avoid overwriting old evidence.")
    for spec, resolved, hist in items:
        print(
            f"  {spec.label}: epoch {hist['last_epoch']}/{hist['total_epochs']} "
            f"| legacy={resolved.path} | canonical={resolved.canonical_path}"
        )


def command_resubmit(args: argparse.Namespace) -> int:
    specs = _specs(args.scope)
    unfinished: list[RunSpec] = []
    legacy_review: list[tuple[RunSpec, ResolvedRunDir, dict[str, object]]] = []
    for spec in specs:
        resolved = _resolve_exp_dir(args.student_dir, spec)
        hist = _parse_history(resolved.path, spec.target_epochs)
        if hist["complete"]:
            continue
        if resolved.layout == "legacy":
            legacy_review.append((spec, resolved, hist))
            continue
        unfinished.append(spec)

    if not unfinished:
        if legacy_review:
            _print_legacy_review(legacy_review)
            print("\nNo automatic resubmit commands generated for legacy-only incomplete runs.")
        else:
            print("All selected runs are complete.")
        return 0

    if legacy_review:
        _print_legacy_review(legacy_review)
        print("")

    queued = _queued_jobs()
    grouped: dict[str, list[RunSpec]] = {}
    for spec in unfinished:
        grouped.setdefault(spec.submit_cmd, []).append(spec)

    submitted = 0
    for submit_cmd, group in grouped.items():
        marker = group[0].queue_marker
        if _queue_has_marker(marker, queued):
            labels = ", ".join(spec.label for spec in group)
            print(f"SKIP queued/running marker '{marker}': {labels}")
            continue

        indices = [spec.array_index for spec in group if spec.array_index is not None]
        submit_args = _submit_args(submit_cmd, indices if indices else None)
        print(_print_shell_command(submit_args))
        if args.submit:
            result = _run(submit_args)
            if result is None:
                print("ERROR: sbatch not found", file=sys.stderr)
                return 2
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            if result.returncode != 0:
                return result.returncode
            submitted += 1

    if not args.submit:
        print("\nDry-run only. Add --submit to run these sbatch commands.")
    else:
        print(f"\nSubmitted {submitted} command(s).")
    return 0


def _parse_elapsed_seconds(value: str | None) -> int | None:
    if not value:
        return None
    days = 0
    rest = value
    if "-" in value:
        day_str, rest = value.split("-", 1)
        try:
            days = int(day_str)
        except ValueError:
            return None
    parts = rest.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(p) for p in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(p) for p in parts)
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_mem_mb(value: str | None) -> float | None:
    if not value:
        return None
    match = re.match(r"^([0-9.]+)([KMGTP]?)", value.strip())
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2)
    factors = {"": 1.0 / (1024 * 1024), "K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 * 1024.0}
    return amount * factors.get(unit, 1.0)


def _parse_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="ignore")
    job_id_match = re.search(r"SLURM_JOB_ID:\s*([^\s]+)", text)
    task_match = re.search(r"ARRAY_TASK_ID:\s*([^\s]+)", text)
    epochs = [int(x) for x in re.findall(r"Epoch\s+([0-9]+)\s*/", text)]
    job_id = job_id_match.group(1) if job_id_match else path.stem.replace("slurm_", "")
    array_task_id = task_match.group(1) if task_match else ""
    sacct_id = f"{job_id}_{array_task_id}" if array_task_id else job_id
    return {
        "path": str(path),
        "job_id": job_id,
        "array_task_id": array_task_id,
        "sacct_id": sacct_id,
        "start_epoch": min(epochs) if epochs else "",
        "end_epoch": max(epochs) if epochs else "",
        "epochs_completed": len(set(epochs)) if epochs else "",
    }


def _sacct_records(job_ids: Iterable[str]) -> dict[str, dict[str, object]]:
    ids = sorted(set(job_id for job_id in job_ids if job_id))
    if not ids:
        return {}
    result = _run(
        [
            "sacct",
            "-j",
            ",".join(ids),
            "--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS",
            "--parsable2",
            "--noheader",
        ]
    )
    if result is None or result.returncode != 0:
        return {}

    records: dict[str, dict[str, object]] = {}
    max_mem: dict[str, tuple[str, float]] = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        job_id, job_name, state, exit_code, elapsed, max_rss = parts[:6]
        base = job_id.split(".", 1)[0]
        mem_mb = _parse_mem_mb(max_rss)
        if mem_mb is not None and (base not in max_mem or mem_mb > max_mem[base][1]):
            max_mem[base] = (max_rss, mem_mb)
        if job_id == base and base in ids:
            records[base] = {
                "job_name": job_name,
                "state": state,
                "exit_code": exit_code,
                "elapsed": elapsed,
                "elapsed_seconds": _parse_elapsed_seconds(elapsed),
            }

    for base, (max_rss, mem_mb) in max_mem.items():
        records.setdefault(base, {})
        records[base]["max_rss"] = max_rss
        records[base]["max_rss_mb"] = mem_mb
    return records


def command_runtime(args: argparse.Namespace) -> int:
    rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    for spec in _specs(args.scope):
        exp_dir = _exp_dir(args.student_dir, spec)
        hist = _parse_history(exp_dir, spec.target_epochs)
        logs = sorted((exp_dir / "logs").glob("slurm_*.out"))
        log_infos = [_parse_log(path) for path in logs]
        sacct = _sacct_records(info["sacct_id"] for info in log_infos)

        total_seconds = 0
        max_rss_mb = 0.0
        max_rss = ""
        for info in log_infos:
            rec = sacct.get(str(info["sacct_id"]), {})
            elapsed_seconds = rec.get("elapsed_seconds")
            if isinstance(elapsed_seconds, int):
                total_seconds += elapsed_seconds
            mem_mb = rec.get("max_rss_mb")
            if isinstance(mem_mb, float) and mem_mb > max_rss_mb:
                max_rss_mb = mem_mb
                max_rss = str(rec.get("max_rss", ""))
            row = _runtime_row("chunk", spec, exp_dir, hist, info, rec)
            chunk_rows.append(row)
            rows.append(row)

        aggregate = _runtime_row("aggregate", spec, exp_dir, hist, {}, {})
        aggregate["job_id"] = "TOTAL"
        aggregate["elapsed_minutes"] = f"{total_seconds / 60:.2f}" if total_seconds else ""
        aggregate["max_rss"] = max_rss
        aggregate["max_rss_mb"] = f"{max_rss_mb:.1f}" if max_rss_mb else ""
        rows.append(aggregate)

    output = args.output or args.student_dir / "experiments" / "summaries" / "index" / "runtime_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "row_type",
        "scope",
        "label",
        "run_path",
        "submit_cmd",
        "job_id",
        "array_task_id",
        "state",
        "exit_code",
        "elapsed",
        "elapsed_minutes",
        "max_rss",
        "max_rss_mb",
        "start_epoch",
        "end_epoch",
        "epochs_completed",
        "last_epoch",
        "total_epochs",
        "complete",
        "last_val_ms_ssim",
        "last_ms_ssim_r",
        "best_val_ms_ssim",
        "best_epoch",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")
    print(f"Rows: {len(rows)} ({len(chunk_rows)} chunks + {len(rows) - len(chunk_rows)} aggregates)")
    return 0


def _runtime_row(
    row_type: str,
    spec: RunSpec,
    exp_dir: Path,
    hist: dict[str, object],
    info: dict[str, object],
    rec: dict[str, object],
) -> dict[str, object]:
    elapsed_seconds = rec.get("elapsed_seconds")
    return {
        "row_type": row_type,
        "scope": spec.scope,
        "label": spec.label,
        "run_path": str(exp_dir),
        "submit_cmd": spec.submit_cmd,
        "job_id": info.get("sacct_id", ""),
        "array_task_id": info.get("array_task_id", ""),
        "state": rec.get("state", ""),
        "exit_code": rec.get("exit_code", ""),
        "elapsed": rec.get("elapsed", ""),
        "elapsed_minutes": f"{elapsed_seconds / 60:.2f}" if isinstance(elapsed_seconds, int) else "",
        "max_rss": rec.get("max_rss", ""),
        "max_rss_mb": f"{rec['max_rss_mb']:.1f}" if isinstance(rec.get("max_rss_mb"), float) else "",
        "start_epoch": info.get("start_epoch", ""),
        "end_epoch": info.get("end_epoch", ""),
        "epochs_completed": info.get("epochs_completed", ""),
        "last_epoch": hist.get("last_epoch", ""),
        "total_epochs": hist.get("total_epochs", ""),
        "complete": hist.get("complete", ""),
        "last_val_ms_ssim": hist.get("last_val_ms_ssim", ""),
        "last_ms_ssim_r": hist.get("last_ms_ssim_r", ""),
        "best_val_ms_ssim": hist.get("best_val_ms_ssim", ""),
        "best_epoch": hist.get("best_epoch", ""),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student-dir",
        type=Path,
        default=DEFAULT_STUDENT_DIR,
        help=f"Student staff-bulk root (default: {DEFAULT_STUDENT_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Print completion status from history.csv.")
    _add_scope_arg(status)
    status.set_defaults(func=command_status)

    resubmit = sub.add_parser("resubmit", help="Print or run sbatch commands for unfinished runs.")
    _add_scope_arg(resubmit)
    resubmit.add_argument("--submit", action="store_true", help="Actually run sbatch. Default is dry-run.")
    resubmit.set_defaults(func=command_resubmit)

    runtime = sub.add_parser("runtime", help="Write runtime_summary.csv from logs and sacct.")
    _add_scope_arg(runtime)
    runtime.add_argument("--output", type=Path, default=None, help="Output CSV path.")
    runtime.set_defaults(func=command_runtime)
    return parser


def _add_scope_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=["all", "main", "data_efficiency", "ablations"],
        default="all",
        help="Run group to manage.",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
