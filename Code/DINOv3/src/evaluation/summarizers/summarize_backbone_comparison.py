"""Summarize SFM/SwinV2 backbone-comparison runs.

This intentionally does not feed the existing main DINOv3 summarizer. The
backbone comparison is a separate extension and should not silently alter the
main thesis tables.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root
from typing import Any

import yaml

F3_METRICS_DISPLAY = [
    ("ms_ssim_r", "MS-SSIM-R"),
    ("residual_energy_frac", "Residual energy"),
    ("residual_input_corr", "Residual-input corr"),
    ("denoised_input_amplitude_ratio", "Amplitude ratio"),
    ("low_freq_residual_energy_frac", "Low-freq residual"),
]

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

PROJECT_ROOT = default_project_root(__file__)
MATRIX_PATH = PROJECT_ROOT / "Code" / "DAIC" / "backbone_comparison" / "matrix.csv"
RUNS_ROOT = PROJECT_ROOT / "experiments" / "runs"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "summaries" / "backbone_comparison"

RUN_FIELDS = [
    "task_id",
    "backbone",
    "variant",
    "data_seed",
    "training_seed",
    "config",
    "checkpoint_dir",
    "local_run_dir",
    "epoch_status",
    "is_complete",
    "has_eval",
    "best_epoch",
    "best_val_ms_ssim",
    "total_params",
    "trainable_params",
    "trainable_pct",
    "test_ms_ssim",
    "test_ms_ssim_r",
    "test_mse",
    "test_psnr",
]

AGG_FIELDS = [
    "backbone",
    "variant",
    "data_seed",
    "n_runs",
    "complete_runs",
    "evaluated_runs",
    "training_seeds",
    "test_ms_ssim_mean",
    "test_ms_ssim_std",
    "test_ms_ssim_r_mean",
    "test_ms_ssim_r_std",
    "test_mse_mean",
    "test_mse_std",
    "test_psnr_mean",
    "test_psnr_std",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _local_run_dir(checkpoint_dir: str) -> Path:
    marker = "/experiments/runs/"
    if marker in checkpoint_dir:
        rel = checkpoint_dir.split(marker, 1)[1]
        return RUNS_ROOT / Path(rel)
    return Path(checkpoint_dir)


def _history_status(run_dir: Path) -> tuple[str, bool, int | None]:
    history = run_dir / "history.csv"
    if not history.exists():
        return "missing", False, None
    rows = _read_csv(history)
    if not rows:
        return "empty", False, None
    last = rows[-1]
    epoch = _safe_int(last.get("epoch"))
    total = _safe_int(last.get("total_epochs"))
    if epoch is None or total is None:
        return "unknown", False, epoch
    complete = epoch >= total
    return f"{epoch}/{total} {'complete' if complete else 'partial'}", complete, epoch


def _best_from_history(run_dir: Path) -> tuple[int | None, float | None]:
    history = run_dir / "history.csv"
    if not history.exists():
        return None, None
    best_epoch = None
    best_score = None
    for row in _read_csv(history):
        score = _safe_float(row.get("val_ms_ssim"))
        epoch = _safe_int(row.get("epoch"))
        if score is None or epoch is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
    return best_epoch, best_score


def _eval_metrics(run_dir: Path) -> dict[str, float | None]:
    path = run_dir / "eval_results" / "results.csv"
    empty = {
        "test_ms_ssim": None,
        "test_ms_ssim_r": None,
        "test_mse": None,
        "test_psnr": None,
    }
    if not path.exists():
        return empty
    rows = [row for row in _read_csv(path) if row.get("split") == "test"]
    if not rows:
        return empty

    def mean_col(name: str) -> float | None:
        values = [_safe_float(row.get(name)) for row in rows]
        values = [value for value in values if value is not None]
        return statistics.fmean(values) if values else None

    return {
        "test_ms_ssim": mean_col("ms_ssim"),
        "test_ms_ssim_r": mean_col("ms_ssim_r"),
        "test_mse": mean_col("mse"),
        "test_psnr": mean_col("psnr"),
    }


def _row_from_matrix(row: dict[str, str]) -> dict[str, Any]:
    run_dir = _local_run_dir(row["checkpoint_dir"])
    meta = _read_yaml(run_dir / "run_meta.yaml")
    params = meta.get("params") or {}
    epoch_status, complete, last_epoch = _history_status(run_dir)
    best_epoch, best_val = _best_from_history(run_dir)
    metrics = _eval_metrics(run_dir)
    if best_epoch is None:
        best_epoch = last_epoch

    return {
        "task_id": _safe_int(row.get("task_id")),
        "backbone": row.get("backbone"),
        "variant": row.get("variant"),
        "data_seed": _safe_int(row.get("data_seed")),
        "training_seed": _safe_int(row.get("training_seed")),
        "config": row.get("config"),
        "checkpoint_dir": row.get("checkpoint_dir"),
        "local_run_dir": str(run_dir),
        "epoch_status": epoch_status,
        "is_complete": complete,
        "has_eval": metrics["test_ms_ssim"] is not None,
        "best_epoch": best_epoch,
        "best_val_ms_ssim": best_val,
        "total_params": _safe_int(params.get("total")),
        "trainable_params": _safe_int(params.get("trainable")),
        "trainable_pct": _safe_float(params.get("trainable_pct")),
        **metrics,
    }


def _stat(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _aggregate(rows: list[dict[str, Any]], by_data_seed: bool) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["backbone"],
            row["variant"],
            row["data_seed"] if by_data_seed else "all",
        )
        groups.setdefault(key, []).append(row)

    out = []
    for (backbone, variant, data_seed), group_rows in sorted(groups.items()):
        eval_rows = [row for row in group_rows if row.get("has_eval")]
        metrics = {}
        for key in ("test_ms_ssim", "test_ms_ssim_r", "test_mse", "test_psnr"):
            values = [row[key] for row in eval_rows if row.get(key) is not None]
            mean, std = _stat(values)
            metrics[f"{key}_mean"] = mean
            metrics[f"{key}_std"] = std
        out.append(
            {
                "backbone": backbone,
                "variant": variant,
                "data_seed": data_seed,
                "n_runs": len(group_rows),
                "complete_runs": sum(1 for row in group_rows if row.get("is_complete")),
                "evaluated_runs": len(eval_rows),
                "training_seeds": ",".join(
                    str(row["training_seed"])
                    for row in sorted(group_rows, key=lambda r: r["training_seed"])
                    if row.get("training_seed") is not None
                ),
                **metrics,
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_f3_summary(f3_path: Path) -> list[dict[str, Any]]:
    if not f3_path.exists():
        return []
    with f3_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_report(
    path: Path,
    rows: list[dict[str, Any]],
    aggregate: list[dict[str, Any]],
    f3_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Backbone Comparison Summary",
        "",
        "This extension is separate from the DINOv3 main thesis tables.",
        "",
        "## Status",
        "",
        f"- Matrix rows: {len(rows)}",
        f"- Complete runs: {sum(1 for row in rows if row.get('is_complete'))}",
        f"- Evaluated runs: {sum(1 for row in rows if row.get('has_eval'))}",
        "",
        "## Image Impeccable Paired Results",
        "",
        "| Backbone | Variant | Runs | Evaluated | Test MS-SSIM | Test MSE | Test PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| "
            f"{row['backbone']} | {row['variant']} | {row['n_runs']} | "
            f"{row['evaluated_runs']} | {_fmt(row['test_ms_ssim_mean'])} | "
            f"{_fmt(row['test_mse_mean'])} | {_fmt(row['test_psnr_mean'])} |"
        )

    lines.extend(["", "## F3 Field-Transfer Robustness", ""])
    if f3_rows:
        header_cols = " | ".join(label for _, label in F3_METRICS_DISPLAY)
        sep_cols = " | ".join("---:" for _ in F3_METRICS_DISPLAY)
        lines.append(f"| Backbone | Variant | Runs | {header_cols} |")
        lines.append(f"|---|---|---:| {sep_cols} |")
        for row in f3_rows:
            metric_cells = []
            for key, _ in F3_METRICS_DISPLAY:
                mean = _safe_float(row.get(f"{key}_mean"))
                std = _safe_float(row.get(f"{key}_std"))
                if mean is None:
                    metric_cells.append("pending")
                elif std is None:
                    metric_cells.append(f"{mean:.4f}")
                else:
                    metric_cells.append(f"`{mean:.4f} +/- {std:.4f}`")
            n = row.get("n", "?")
            lines.append(
                f"| {row['backbone']} | {row['variant']} | {n} | "
                + " | ".join(metric_cells)
                + " |"
            )
        lines.extend([
            "",
            "No accuracy claim is allowed; F3 has no clean target.",
            "MS-SSIM-R near 1.0 means the output resembles the input (minimal denoising).",
        ])
    else:
        lines.append("Pending — run `summarize_backbone_f3_robustness.py` first.")

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Use these rows only as an additional-backbone extension after all compared cells are complete.",
            "- Do not merge this summary into the DINOv3 main result snapshot.",
            "- SFM `2D` is native one-channel center-slice input; DINOv3/SwinV2 `2D` is repeated three-channel input.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


DEFAULT_F3_SUMMARY = PROJECT_ROOT / "experiments" / "summaries" / "f3_backbone_robustness" / "backbone_f3_replicate_summary.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--f3-summary",
        type=Path,
        default=DEFAULT_F3_SUMMARY,
        help="Path to backbone_f3_replicate_summary.csv (from summarize_backbone_f3_robustness.py).",
    )
    args = parser.parse_args()

    matrix_rows = _read_csv(args.matrix)
    rows = [_row_from_matrix(row) for row in matrix_rows]
    aggregate_all = _aggregate(rows, by_data_seed=False)
    aggregate_by_split = _aggregate(rows, by_data_seed=True)
    f3_rows = _load_f3_summary(args.f3_summary)
    if not f3_rows:
        print(f"[WARN] No F3 backbone summary found at {args.f3_summary}; F3 section will show as pending.")

    out_dir = args.out_dir
    _write_csv(out_dir / "backbone_run_summary.csv", rows, RUN_FIELDS)
    _write_csv(out_dir / "backbone_replicate_summary.csv", aggregate_all, AGG_FIELDS)
    _write_csv(out_dir / "backbone_by_data_seed_summary.csv", aggregate_by_split, AGG_FIELDS)
    _write_report(out_dir / "backbone_comparison_report.md", rows, aggregate_all, f3_rows)

    print(f"Wrote backbone comparison summaries to {out_dir}")


if __name__ == "__main__":
    main()
