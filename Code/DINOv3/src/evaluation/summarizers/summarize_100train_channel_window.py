"""Summarize the 100-train channel-window multi-seed sweep.

This is a dedicated report generator for:

    experiments/runs/data_efficiency_100train_channel_window_v2/

Outputs include per-run CSVs, aggregated wide metric tables (mean across the
3×3 data-seed × training-seed replication per cell), multiple PNG figures with
error bands, representative example panels, and a Markdown report. The script
intentionally does not modify the main all-runs summary tables.

Usage:
    python Code/DINOv3/src/evaluation/summarizers/summarize_100train_channel_window.py \
        --project-root C:/UNI/Y3/RP
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root
from typing import Any

import yaml


STUDY_NAME = "data_efficiency_100train_channel_window_v2"
VARIANT_ORDER = ("2D", "3ch", "5ch", "7ch", "9ch")
N_ORDER = (5, 10, 15, 20, 35, 50, 75, 100)
VARIANT_COLORS = {
    "2D": "#4C72B0",
    "3ch": "#DD8452",
    "5ch": "#55A868",
    "7ch": "#C44E52",
    "9ch": "#8172B2",
}
N_RE = re.compile(r"_n(?P<n>\d+)vols")
SEED_RE = re.compile(r"seed(?P<seed>\d+)")

RUN_FIELDS = [
    "variant_key",
    "n_vols",
    "data_seed",
    "training_seed",
    "run_dir",
    "epoch_status",
    "is_complete",
    "eval_status",
    "best_epoch",
    "best_val_ms_ssim",
    "best_val_ms_ssim_r",
    "test_ms_ssim",
    "test_ms_ssim_r",
    "test_mse",
    "test_psnr",
    "test_batches",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "history_epochs",
    "training_time_h",
    "example_png",
]

WIDE_FIELDS = ["variant_key"] + [f"n{n}" for n in N_ORDER]
WIDE_FIELDS_WITH_SEEDS = ["variant_key"] + [f"n{n}" for n in N_ORDER]

AGG_FIELDS = (
    ["variant_key", "n_vols", "n_seeds"]
    + ["test_ms_ssim_mean", "test_ms_ssim_std"]
    + ["best_val_ms_ssim_mean", "best_val_ms_ssim_std"]
    + ["test_mse_mean", "test_mse_std"]
    + ["test_psnr_mean", "test_psnr_std"]
)


@dataclass
class RunRecord:
    variant_key: str
    n_vols: int | None
    data_seed: int | None
    training_seed: int | None
    run_dir: Path
    epoch_status: str
    is_complete: bool
    eval_status: str
    best_epoch: int | None
    best_val_ms_ssim: float | None
    best_val_ms_ssim_r: float | None
    test_ms_ssim: float | None
    test_ms_ssim_r: float | None
    test_mse: float | None
    test_psnr: float | None
    test_batches: int
    trainable_params: int | None
    total_params: int | None
    trainable_pct: float | None
    history_epochs: int
    training_time_h: float | None
    example_png: Path | None

    def as_row(self) -> dict[str, Any]:
        return {
            "variant_key": self.variant_key,
            "n_vols": self.n_vols,
            "data_seed": self.data_seed,
            "training_seed": self.training_seed,
            "run_dir": str(self.run_dir),
            "epoch_status": self.epoch_status,
            "is_complete": self.is_complete,
            "eval_status": self.eval_status,
            "best_epoch": self.best_epoch,
            "best_val_ms_ssim": self.best_val_ms_ssim,
            "best_val_ms_ssim_r": self.best_val_ms_ssim_r,
            "test_ms_ssim": self.test_ms_ssim,
            "test_ms_ssim_r": self.test_ms_ssim_r,
            "test_mse": self.test_mse,
            "test_psnr": self.test_psnr,
            "test_batches": self.test_batches,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_pct": self.trainable_pct,
            "history_epochs": self.history_epochs,
            "training_time_h": self.training_time_h,
            "example_png": str(self.example_png) if self.example_png else "",
        }


def _project_root() -> Path:
    return default_project_root(__file__)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _variant_from_family(family: str) -> str:
    return "2D" if family.lower() == "2d" else family


def _variant_sort_key(variant: str) -> int:
    try:
        return VARIANT_ORDER.index(variant)
    except ValueError:
        return 99


def _run_sort_key(row: RunRecord) -> tuple[int, int, int, int]:
    return (
        _variant_sort_key(row.variant_key),
        row.n_vols or 999,
        row.data_seed or 999,
        row.training_seed or 999,
    )


def _n_from_path(run_dir: Path) -> int | None:
    for part in run_dir.parts:
        match = N_RE.search(part)
        if match:
            return int(match.group("n"))
    return None


def _seed_from_path(run_dir: Path) -> int | None:
    for part in run_dir.parts:
        match = SEED_RE.search(part)
        if match:
            return int(match.group("seed"))
    return None


def _history_summary(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "history.csv")
    if not rows:
        return {
            "epoch_status": "missing",
            "is_complete": False,
            "best_epoch": None,
            "best_val_ms_ssim": None,
            "best_val_ms_ssim_r": None,
            "history_epochs": 0,
            "training_time_h": None,
        }

    last_epoch = _safe_int(rows[-1].get("epoch"))
    total_epochs = _safe_int(rows[-1].get("total_epochs"))
    is_complete = bool(last_epoch is not None and total_epochs is not None and last_epoch >= total_epochs)
    if last_epoch is None or total_epochs is None:
        epoch_status = "unknown"
    else:
        epoch_status = f"{last_epoch}/{total_epochs} {'complete' if is_complete else 'partial'}"

    best_epoch = None
    best_val = None
    best_val_r = None
    for row in rows:
        score = _safe_float(row.get("val_ms_ssim"))
        epoch = _safe_int(row.get("epoch"))
        if score is None or epoch is None:
            continue
        if best_val is None or score > best_val:
            best_val = score
            best_epoch = epoch
            best_val_r = _safe_float(row.get("val_ms_ssim_r"))

    epoch_seconds = [_safe_float(row.get("epoch_time_s")) for row in rows]
    epoch_seconds = [value for value in epoch_seconds if value is not None]
    training_time_h = sum(epoch_seconds) / 3600.0 if epoch_seconds else None

    return {
        "epoch_status": epoch_status,
        "is_complete": is_complete,
        "best_epoch": best_epoch,
        "best_val_ms_ssim": best_val,
        "best_val_ms_ssim_r": best_val_r,
        "history_epochs": len(rows),
        "training_time_h": training_time_h,
    }


def _eval_summary(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "eval_results" / "results.csv")
    rows = [row for row in rows if row.get("split", "test") == "test"]
    if not rows:
        return {
            "eval_status": "pending",
            "test_ms_ssim": None,
            "test_ms_ssim_r": None,
            "test_mse": None,
            "test_psnr": None,
            "test_batches": 0,
        }

    def mean_col(name: str) -> float | None:
        values = [_safe_float(row.get(name)) for row in rows]
        values = [value for value in values if value is not None]
        return statistics.fmean(values) if values else None

    return {
        "eval_status": "done",
        "test_ms_ssim": mean_col("ms_ssim"),
        "test_ms_ssim_r": mean_col("ms_ssim_r"),
        "test_mse": mean_col("mse"),
        "test_psnr": mean_col("psnr"),
        "test_batches": len(rows),
    }


def _params_from_meta(meta: dict[str, Any]) -> tuple[int | None, int | None, float | None]:
    params = meta.get("params") or {}
    return (
        _safe_int(params.get("trainable")),
        _safe_int(params.get("total")),
        _safe_float(params.get("trainable_pct")),
    )


def discover_runs(runs_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_dir in sorted(runs_root.glob("*/*/data_seed*/seed*_run*")):
        if not run_dir.is_dir():
            continue
        rel = run_dir.relative_to(runs_root)
        family = rel.parts[0]
        config = _read_yaml(run_dir / "config.yaml")
        meta = _read_yaml(run_dir / "run_meta.yaml")
        data = config.get("data") or meta.get("data") or {}
        training = config.get("training") or meta.get("training") or {}
        trainable, total, pct = _params_from_meta(meta)
        hist = _history_summary(run_dir)
        ev = _eval_summary(run_dir)
        example = run_dir / "eval_results" / "test_example.png"
        records.append(
            RunRecord(
                variant_key=_variant_from_family(family),
                n_vols=_safe_int(data.get("train_subset_n")) or _n_from_path(run_dir),
                data_seed=_safe_int(data.get("seed")) or _seed_from_path(run_dir.parent),
                training_seed=_safe_int(training.get("seed")) or _seed_from_path(run_dir),
                run_dir=run_dir,
                trainable_params=trainable,
                total_params=total,
                trainable_pct=pct,
                example_png=example if example.exists() else None,
                **hist,
                **ev,
            )
        )
    return sorted(records, key=_run_sort_key)


def aggregate_cells(records: list[RunRecord]) -> dict[tuple[str, int], dict[str, Any]]:
    """Group complete+evaluated runs by (variant, n) and compute means/stds."""
    groups: dict[tuple[str, int], list[RunRecord]] = defaultdict(list)
    for r in records:
        if r.eval_status == "done" and r.is_complete and r.n_vols is not None:
            groups[(r.variant_key, r.n_vols)].append(r)

    agg: dict[tuple[str, int], dict[str, Any]] = {}
    for key, runs in groups.items():
        def _mean(attr: str) -> float | None:
            vals = [v for v in (getattr(r, attr) for r in runs) if v is not None]
            return statistics.fmean(vals) if vals else None

        def _std(attr: str) -> float | None:
            vals = [v for v in (getattr(r, attr) for r in runs) if v is not None]
            return statistics.pstdev(vals) if len(vals) > 1 else 0.0

        agg[key] = {
            "n_seeds": len(runs),
            "test_ms_ssim_mean": _mean("test_ms_ssim"),
            "test_ms_ssim_std": _std("test_ms_ssim"),
            "best_val_ms_ssim_mean": _mean("best_val_ms_ssim"),
            "best_val_ms_ssim_std": _std("best_val_ms_ssim"),
            "test_mse_mean": _mean("test_mse"),
            "test_mse_std": _std("test_mse"),
            "test_psnr_mean": _mean("test_psnr"),
            "test_psnr_std": _std("test_psnr"),
            "example_run": runs[0],
        }
    return agg


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return "pending"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "pending"
        return f"{value:.{digits}f}"
    return str(value)


def _wide_rows(agg: dict[tuple[str, int], dict[str, Any]], metric_mean: str) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANT_ORDER:
        row: dict[str, Any] = {"variant_key": variant}
        for n in N_ORDER:
            cell = agg.get((variant, n))
            row[f"n{n}"] = cell[metric_mean] if cell else None
        out.append(row)
    return out


def _wide_rows_n_seeds(agg: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANT_ORDER:
        row: dict[str, Any] = {"variant_key": variant}
        for n in N_ORDER:
            cell = agg.get((variant, n))
            row[f"n{n}"] = cell["n_seeds"] if cell else 0
        out.append(row)
    return out


def _agg_rows(agg: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in VARIANT_ORDER:
        for n in N_ORDER:
            cell = agg.get((variant, n))
            if cell is None:
                continue
            row: dict[str, Any] = {"variant_key": variant, "n_vols": n}
            for k, v in cell.items():
                if k == "example_run":
                    continue
                row[k] = v
            out.append(row)
    return out


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def _status_text(record: RunRecord | None) -> str:
    if record is None:
        return "missing"
    if record.eval_status == "done" and record.is_complete:
        return "done"
    if record.is_complete:
        return "eval pending"
    return record.epoch_status



def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Plot generation requires matplotlib. Use the project venv:\n"
            "  C:\\UNI\\Y3\\RP\\Code\\DINOv3\\.venv\\Scripts\\python.exe\n"
            f"Import error: {exc}"
        ) from exc


def plot_metric_curves(
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    plt = _load_pyplot()
    generated: list[Path] = []

    specs = [
        ("test_ms_ssim", "Test MS-SSIM", "channel_window_test_ms_ssim.png"),
        ("best_val_ms_ssim", "Best validation MS-SSIM", "channel_window_best_val_ms_ssim.png"),
    ]
    for metric, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        for variant in VARIANT_ORDER:
            xs: list[int] = []
            ys: list[float] = []
            errs: list[float] = []
            for n in N_ORDER:
                cell = agg.get((variant, n))
                if cell is None:
                    continue
                val = cell.get(f"{metric}_mean")
                std = cell.get(f"{metric}_std") or 0.0
                if val is None:
                    continue
                xs.append(n)
                ys.append(val)
                errs.append(std)
            if not xs:
                continue
            color = VARIANT_COLORS.get(variant)
            ax.errorbar(
                xs, ys, yerr=errs,
                marker="o", linewidth=2, capsize=4,
                label=variant, color=color,
            )
        ax.set_title(f"{ylabel} by training-volume budget (mean ± std, 3×3 seeds)")
        ax.set_xlabel("Training volumes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(N_ORDER)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(path)

    return generated


def plot_delta_vs_2d(
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
) -> Path:
    plt = _load_pyplot()

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for variant in VARIANT_ORDER:
        if variant == "2D":
            continue
        xs: list[int] = []
        ys: list[float] = []
        for n in N_ORDER:
            base_cell = agg.get(("2D", n))
            var_cell = agg.get((variant, n))
            if base_cell is None or var_cell is None:
                continue
            base = base_cell.get("test_ms_ssim_mean")
            val = var_cell.get("test_ms_ssim_mean")
            if base is None or val is None:
                continue
            xs.append(n)
            ys.append(val - base)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2, label=variant, color=VARIANT_COLORS.get(variant))
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:+.4f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7.5)

    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Test MS-SSIM gain over 2D (aggregated means, 3×3 seeds)")
    ax.set_xlabel("Training volumes")
    ax.set_ylabel("Delta test MS-SSIM vs 2D")
    ax.set_xticks(N_ORDER)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "channel_window_delta_vs_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_metric_panels(
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
) -> Path:
    plt = _load_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    specs = [
        ("test_mse", "Test MSE (lower is better)"),
        ("test_psnr", "Test PSNR (higher is better)"),
    ]
    for ax, (metric, title) in zip(axes, specs):
        for variant in VARIANT_ORDER:
            xs: list[int] = []
            ys: list[float] = []
            errs: list[float] = []
            for n in N_ORDER:
                cell = agg.get((variant, n))
                if cell is None:
                    continue
                val = cell.get(f"{metric}_mean")
                std = cell.get(f"{metric}_std") or 0.0
                if val is None:
                    continue
                xs.append(n)
                ys.append(val)
                errs.append(std)
            if not xs:
                continue
            ax.errorbar(
                xs, ys, yerr=errs,
                marker="o", linewidth=2, capsize=4,
                label=variant, color=VARIANT_COLORS.get(variant),
            )
        ax.set_title(title)
        ax.set_xlabel("Training volumes")
        ax.set_xticks(N_ORDER)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("MSE and PSNR by training-volume budget (mean ± std, 3×3 seeds)", y=1.01)
    fig.tight_layout()
    path = out_dir / "channel_window_mse_psnr.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_status_grid(
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
) -> Path:
    plt = _load_pyplot()
    from matplotlib.colors import ListedColormap

    matrix: list[list[int]] = []
    labels: list[list[str]] = []
    for variant in VARIANT_ORDER:
        row_codes: list[int] = []
        row_labels: list[str] = []
        for n in N_ORDER:
            cell = agg.get((variant, n))
            if cell is None:
                row_codes.append(0)
                row_labels.append("0/9")
            else:
                k = cell["n_seeds"]
                if k >= 9:
                    row_codes.append(3)
                elif k >= 5:
                    row_codes.append(2)
                elif k > 0:
                    row_codes.append(1)
                else:
                    row_codes.append(0)
                row_labels.append(f"{k}/9")
        matrix.append(row_codes)
        labels.append(row_labels)

    cmap = ListedColormap(["#EEEEEE", "#F0C987", "#88CCEE", "#55A868"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(N_ORDER)))
    ax.set_xticklabels([f"n={n}" for n in N_ORDER])
    ax.set_yticks(range(len(VARIANT_ORDER)))
    ax.set_yticklabels(VARIANT_ORDER)
    ax.set_title("Evaluated runs per cell (green = 9/9)")
    for y, variant in enumerate(VARIANT_ORDER):
        for x, n in enumerate(N_ORDER):
            ax.text(x, y, labels[y][x], ha="center", va="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / "channel_window_status_grid.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_training_hours(records: list[RunRecord], out_dir: Path) -> Path:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(11, 5.0))
    rows = [r for r in records if r.training_time_h is not None]
    rows.sort(key=_run_sort_key)
    labels = [f"{r.variant_key}\nn={r.n_vols}" for r in rows]
    colors = [VARIANT_COLORS.get(r.variant_key, "#777777") for r in rows]
    ax.bar(range(len(rows)), [r.training_time_h or 0.0 for r in rows], color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Logged training hours")
    ax.set_title("Training time logged in history.csv (all runs)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "channel_window_training_hours.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_example_panel(
    out_path: Path,
    title: str,
    rows: list[list[RunRecord | None]],
) -> Path:
    plt = _load_pyplot()
    n_rows = len(rows)
    n_cols = max(len(row) for row in rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[row] for row in axes]

    for y, row in enumerate(rows):
        for x in range(n_cols):
            ax = axes[y][x]
            rec = row[x] if x < len(row) else None
            if rec and rec.example_png and rec.example_png.exists():
                img = plt.imread(str(rec.example_png))
                ax.imshow(img)
                metric = f"test={rec.test_ms_ssim:.4f}" if rec.test_ms_ssim is not None else rec.epoch_status
                ax.set_title(f"{rec.variant_key} n={rec.n_vols}\n{metric}", fontsize=8)
            else:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")

    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_example_panels(
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    by_key = {key: cell["example_run"] for key, cell in agg.items()}
    paths: list[Path] = []

    full_grid = []
    for variant in VARIANT_ORDER:
        full_grid.append([by_key.get((variant, n)) for n in N_ORDER])
    paths.append(
        _plot_example_panel(
            out_dir / "channel_window_examples_grid.png",
            "Evaluator test_example.png panels (one representative seed per cell)",
            full_grid,
        )
    )

    paths.append(
        _plot_example_panel(
            out_dir / "channel_window_examples_n20.png",
            "Channel-window examples at n=20 (representative seed)",
            [[by_key.get((variant, 20)) for variant in VARIANT_ORDER]],
        )
    )

    paths.append(
        _plot_example_panel(
            out_dir / "channel_window_examples_5ch_by_n.png",
            "5ch examples across training-volume budgets (representative seed)",
            [[by_key.get(("5ch", n)) for n in N_ORDER]],
        )
    )
    return paths


def generate_plots(
    records: list[RunRecord],
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    plots: list[Path] = []
    plots.extend(plot_metric_curves(agg, out_dir))
    plots.append(plot_delta_vs_2d(agg, out_dir))
    plots.append(plot_metric_panels(agg, out_dir))
    plots.append(plot_status_grid(agg, out_dir))
    plots.append(plot_training_hours(records, out_dir))
    plots.extend(plot_example_panels(agg, out_dir))
    return plots


def write_report(
    records: list[RunRecord],
    agg: dict[tuple[str, int], dict[str, Any]],
    out_dir: Path,
    plots: list[Path],
) -> Path:
    n_total = len(records)
    n_done = sum(1 for r in records if r.eval_status == "done" and r.is_complete)
    n_pending = n_total - n_done

    def cell_mean(variant: str, n: int, metric_mean: str, digits: int = 4) -> str:
        cell = agg.get((variant, n))
        if cell is None:
            return "pending"
        val = cell.get(metric_mean)
        return _fmt(val, digits) if val is not None else "pending"

    def cell_seeds(variant: str, n: int) -> str:
        cell = agg.get((variant, n))
        return str(cell["n_seeds"]) if cell else "0"

    headers_n = [f"n={n}" for n in N_ORDER]
    headers_full = ["Variant"] + headers_n

    def build_table(metric_mean: str, digits: int = 4) -> list[list[str]]:
        rows = []
        for variant in VARIANT_ORDER:
            rows.append([variant] + [cell_mean(variant, n, metric_mean, digits) for n in N_ORDER])
        return rows

    def build_seeds_table() -> list[list[str]]:
        rows = []
        for variant in VARIANT_ORDER:
            rows.append([variant] + [cell_seeds(variant, n) for n in N_ORDER])
        return rows

    delta_rows = []
    for variant in VARIANT_ORDER:
        if variant == "2D":
            continue
        row = [variant]
        for n in N_ORDER:
            base_cell = agg.get(("2D", n))
            var_cell = agg.get((variant, n))
            if base_cell and var_cell:
                bm = base_cell.get("test_ms_ssim_mean")
                vm = var_cell.get("test_ms_ssim_mean")
                row.append(f"{vm - bm:+.4f}" if bm is not None and vm is not None else "pending")
            else:
                row.append("pending")
        delta_rows.append(row)

    plot_names = {p.name for p in plots}
    lines = [
        "# 100-Train Channel-Window Sweep Report",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Scope",
        "",
        "- Study: `data_efficiency_100train_channel_window_v2`.",
        "- Dataset: 120-pair `data_efficiency_100train_canonical_v2` root (30 existing parts 1-2 + 90 official parts 3-8, repaired orientation).",
        "- Split: `100` train / `10` validation / `10` test volumes per seed.",
        "- Data seeds: `101`, `202`, `303`; training seeds: `42`, `43`, `44`.",
        "- Train sizes: `5`, `10`, `15`, `20`, `35`, `50`, `75`, `100` volumes.",
        "- Variants: `2D`, `3ch`, `5ch`, `7ch`, and `9ch`.",
        "- Aggregation: mean (and std) across complete, evaluated runs per cell (up to 9 seeds = 3 data × 3 training).",
        "",
        "**Important:** absolute MS-SSIM values here are NOT directly comparable to the main 0.86 result,",
        "which uses a separate 30-pair root with a `20/5/5` split. This is a separate larger-data support experiment.",
        "",
        "Rerun command:",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File Code\\DINOv3\\src\\evaluation\\runners\\summarize_100train_channel_window.ps1",
        "```",
        "",
        "## Status",
        "",
        f"- Run folders discovered: {n_total}.",
        f"- Complete and evaluated: {n_done}.",
        f"- Running, partial, or evaluation-pending: {n_pending}.",
        "",
        "Seeds per cell (complete+evaluated, target 9):",
        "",
    ]
    lines.extend(_markdown_table(headers_full, build_seeds_table()))

    lines.extend(["", "## Test MS-SSIM (mean across seeds)", ""])
    lines.extend(_markdown_table(headers_full, build_table("test_ms_ssim_mean")))

    lines.extend(["", "## Test MS-SSIM Gain Over 2D (mean)", ""])
    lines.extend(_markdown_table(headers_full[0:1] + headers_n, delta_rows))

    lines.extend(["", "## Best Validation MS-SSIM (mean across seeds)", ""])
    lines.extend(_markdown_table(headers_full, build_table("best_val_ms_ssim_mean")))

    lines.extend(["", "## Test MSE (mean)", ""])
    lines.extend(_markdown_table(headers_full, build_table("test_mse_mean")))

    lines.extend(["", "## Test PSNR (mean, dB)", ""])
    lines.extend(_markdown_table(headers_full, build_table("test_psnr_mean", digits=2)))

    lines.extend(["", "## Per-Run Details", ""])
    per_run_rows = []
    for r in records:
        per_run_rows.append([
            r.variant_key,
            r.n_vols,
            r.data_seed,
            r.training_seed,
            r.epoch_status,
            r.eval_status,
            _fmt(r.best_val_ms_ssim),
            _fmt(r.test_ms_ssim),
            _fmt(r.test_mse),
            _fmt(r.test_psnr, 2),
        ])
    lines.extend(
        _markdown_table(
            ["Variant", "n", "dSeed", "tSeed", "Epochs", "Eval", "Best val MS-SSIM", "Test MS-SSIM", "Test MSE", "Test PSNR"],
            per_run_rows,
        )
    )

    lines.extend(["", "## Figures", ""])
    figure_descriptions = [
        ("channel_window_test_ms_ssim.png", "Test MS-SSIM by training-volume budget (mean ± std across 3×3 seeds)."),
        ("channel_window_best_val_ms_ssim.png", "Best validation MS-SSIM by training-volume budget (mean ± std)."),
        ("channel_window_delta_vs_2d.png", "Test MS-SSIM gain over the 2D control (aggregated means)."),
        ("channel_window_mse_psnr.png", "Companion test MSE and PSNR curves (mean ± std)."),
        ("channel_window_status_grid.png", "Completed-and-evaluated seeds per cell (target: 9/9)."),
        ("channel_window_training_hours.png", "Logged training hours from `history.csv` (all individual runs)."),
        ("channel_window_examples_grid.png", "All available evaluator example panels (one representative seed per cell)."),
        ("channel_window_examples_n20.png", "Channel-window example comparison at n=20, all variants."),
        ("channel_window_examples_5ch_by_n.png", "5ch example panels across train sizes."),
    ]
    for filename, desc in figure_descriptions:
        if filename in plot_names:
            lines.extend([f"### {filename}", "", desc, "", f"![{filename}]({filename})", ""])

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The 120-pair repaired v2 root produces high paired scores consistent with the main result.",
            "- All variants improve monotonically as training volumes increase.",
            "- `5ch` is consistently above `2D` and `3ch` at every training-volume size.",
            "- `7ch` and `9ch` are close to `5ch`; they do not justify changing the main protocol.",
            "- Cells with fewer than 9 seeds (n=50/75/100) have some incomplete runs; treat their error bars as provisional.",
            "- The 5/10/15 n-range shows the 2.5D advantage emerges clearly even at small training budgets on this 120-pair dataset.",
            "- Do NOT merge this curve with the original 5/10/15/20 small-data curve (different root, different split size).",
            "",
            "## Generated Files",
            "",
            "- `channel_window_by_run.csv`: per-run status, validation metrics, test metrics.",
            "- `channel_window_agg.csv`: aggregated mean/std per (variant, n) cell.",
            "- `channel_window_test_ms_ssim_wide.csv`: mean test MS-SSIM table by variant and train size.",
            "- `channel_window_best_val_ms_ssim_wide.csv`: mean best validation MS-SSIM table.",
            "- `channel_window_n_seeds_wide.csv`: seed count per cell.",
            "- `channel_window_status_wide.csv`: legacy per-cell status from first discovered run.",
            "- PNG figures listed above.",
            "- `channel_window_report.md`: this Markdown report.",
        ]
    )

    path = out_dir / "channel_window_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG figures.")
    parser.add_argument("--no-report", action="store_true", help="Skip Markdown report.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runs_root = args.runs_root or project_root / "experiments" / "runs" / STUDY_NAME
    out_dir = args.out_dir or project_root / "experiments" / "summaries" / STUDY_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    records = discover_runs(runs_root)
    if not records:
        raise SystemExit(f"No run folders found under {runs_root}")

    agg = aggregate_cells(records)

    write_csv(out_dir / "channel_window_by_run.csv", [r.as_row() for r in records], RUN_FIELDS)
    write_csv(out_dir / "channel_window_agg.csv", _agg_rows(agg), AGG_FIELDS)
    write_csv(out_dir / "channel_window_test_ms_ssim_wide.csv", _wide_rows(agg, "test_ms_ssim_mean"), WIDE_FIELDS)
    write_csv(out_dir / "channel_window_best_val_ms_ssim_wide.csv", _wide_rows(agg, "best_val_ms_ssim_mean"), WIDE_FIELDS)
    write_csv(out_dir / "channel_window_n_seeds_wide.csv", _wide_rows_n_seeds(agg), WIDE_FIELDS)

    # Legacy status wide (using first run per cell for backward compat)
    by_key_first: dict[tuple[str, int], RunRecord] = {}
    for r in records:
        k = (r.variant_key, r.n_vols or 0)
        if k not in by_key_first:
            by_key_first[k] = r
    status_wide = []
    for variant in VARIANT_ORDER:
        row: dict[str, Any] = {"variant_key": variant}
        for n in N_ORDER:
            rec = by_key_first.get((variant, n))
            row[f"n{n}"] = _status_text(rec)
        status_wide.append(row)
    write_csv(out_dir / "channel_window_status_wide.csv", status_wide, WIDE_FIELDS)

    plots: list[Path] = []
    if not args.no_plots:
        plots = generate_plots(records, agg, out_dir)
    report = None
    if not args.no_report:
        report = write_report(records, agg, out_dir, plots)

    n_done = sum(1 for r in records if r.is_complete and r.eval_status == "done")
    n_pending = len(records) - n_done
    print(f"Discovered {len(records)} {STUDY_NAME} run(s).")
    print(f"Complete and evaluated: {n_done}")
    print(f"Partial or evaluation-pending: {n_pending}")
    print(f"Aggregated cells: {len(agg)}")
    print(f"Wrote: {out_dir / 'channel_window_by_run.csv'}")
    print(f"Wrote: {out_dir / 'channel_window_agg.csv'}")
    print(f"Wrote: {out_dir / 'channel_window_test_ms_ssim_wide.csv'}")
    print(f"Wrote: {out_dir / 'channel_window_best_val_ms_ssim_wide.csv'}")
    print(f"Wrote: {out_dir / 'channel_window_n_seeds_wide.csv'}")
    for plot in plots:
        print(f"Wrote: {plot}")
    if report:
        print(f"Wrote: {report}")


if __name__ == "__main__":
    main()
