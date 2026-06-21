"""Dedicated ablation study summariser.

Reads completed runs from experiments/runs/ablations/ and the main n=20
replicates (used as baselines) from experiments/runs/3ch/ and
experiments/runs/5ch/, then produces:

  ablation_runs_per_seed.csv  -- one row per individual seed run
  ablation_aggregated.csv     -- mean +/- std across seeds per condition
  study_{id}_bars.png         -- per-study bar + training-curve figure (6 files)
  ablation_overview.png       -- 2x3 delta-MS-SSIM panel across all studies
  ablation_report.md          -- self-contained Markdown report

Usage:
    python Code/DINOv3/src/evaluation/summarize_ablations.py --project-root .
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root
from typing import Any

# ---------------------------------------------------------------------------
# Study taxonomy
# ---------------------------------------------------------------------------

# Path fragment (forward-slash, relative to experiments/runs/) ->
# (study_id, study_title, variant, condition_label, condition_order)
_STUDY_MAP: dict[str, tuple[str, str, str, str, int]] = {
    "ablations/3ch/stride3_n20vols":       ("A",     "3ch Slice Stride",     "3ch", "stride=3",     1),
    "ablations/3ch/stride1_n20vols":       ("A",     "3ch Slice Stride",     "3ch", "stride=1",     2),
    "ablations/5ch/ns2_stride5_n20vols":   ("B",     "5ch Neighbour Stride", "5ch", "ns=2",         1),
    "ablations/5ch/ns3_stride5_n20vols":   ("B",     "5ch Neighbour Stride", "5ch", "ns=3",         2),
    "ablations/5ch/grid4_stride5_n20vols": ("C",     "5ch Grid4 Crop",       "5ch", "grid4",        1),
    "ablations/3ch/lora_r4_n20vols":       ("D",     "LoRA Rank",            "3ch", "r=4",          1),
    "ablations/3ch/lora_r8_n20vols":       ("D",     "LoRA Rank",            "3ch", "r=8",          2),
    "ablations/3ch/lora_r32_n20vols":      ("D",     "LoRA Rank",            "3ch", "r=32",         4),
    "ablations/3ch/lambda0_n20vols":       ("E",     "Loss Weight",          "3ch", "λ=0.0",   1),
    "ablations/3ch/lambda1_n20vols":       ("E",     "Loss Weight",          "3ch", "λ=1.0",   3),
    "ablations/5ch/rand_outer_n20vols":    ("F",     "Patch Emb Init",       "5ch", "random_outer", 1),
    "ablations/5ch/all_mean_n20vols":      ("F",     "Patch Emb Init",       "5ch", "all_mean",     2),
}

# Path fragment (within experiments/runs/, no trailing slash) ->
# tuple of study IDs this run serves as baseline for
_BASELINE_STUDIES: dict[str, tuple[str, ...]] = {
    "3ch/impeccable_neighbors3_stride5_lora_r16":            ("A", "D", "E"),
    "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16":  ("B", "C", "F"),
}

# Per-study, what to call the baseline condition in charts and tables
_BASELINE_LABEL: dict[str, str] = {
    "A":     "stride=5 (base)",
    "B":     "ns=1 (base)",
    "C":     "center (base)",
    "D":     "r=16 (base)",
    "E":     "λ=0.5 (base)",
    "F":     "mixed (base)",
}

_STUDY_ORDER = ["A", "B", "C", "D", "E", "F"]

_STUDY_QUESTION: dict[str, str] = {
    "A":     "Does slice stride matter for 3ch?",
    "B":     "Does neighbour spacing matter for 5ch?",
    "C":     "Does grid4 crop help 5ch?",
    "D":     "Is LoRA rank r=16 the right choice?",
    "E":     "Is loss weight λ=0.5 robust?",
    "F":     "Does patch-emb init strategy matter?",
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    study_id: str
    study_title: str
    variant: str
    condition: str
    condition_order: int
    is_baseline: bool
    seed: int
    run_dir: Path
    test_ms_ssim: float | None = None
    test_ms_ssim_r: float | None = None
    test_mse: float | None = None
    test_psnr: float | None = None
    best_val_ms_ssim: float | None = None
    n_epochs_completed: int = 0
    total_epochs: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AggCondition:
    study_id: str
    study_title: str
    variant: str
    condition: str
    condition_order: int
    is_baseline: bool
    n_seeds: int
    seeds: list[int]
    test_ms_ssim_mean: float | None = None
    test_ms_ssim_std: float | None = None
    test_ms_ssim_r_mean: float | None = None
    test_ms_ssim_r_std: float | None = None
    test_mse_mean: float | None = None
    test_mse_std: float | None = None
    test_psnr_mean: float | None = None
    test_psnr_std: float | None = None
    best_val_ms_ssim_mean: float | None = None
    best_val_ms_ssim_std: float | None = None
    history_mean: list[dict[str, Any]] = field(default_factory=list)
    run_dirs: list[Path] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _mean_std(vals: list[float | None]) -> tuple[float | None, float | None]:
    good = [v for v in vals if v is not None and math.isfinite(v)]
    if not good:
        return None, None
    m = sum(good) / len(good)
    if len(good) == 1:
        return m, None
    var = sum((v - m) ** 2 for v in good) / (len(good) - 1)
    return m, math.sqrt(var)


def _project_root_default(script_path: Path) -> Path:
    return default_project_root(script_path)


def _extract_seed(run_dir: Path) -> int:
    m = re.search(r"seed(\d+)", run_dir.name)
    return int(m.group(1)) if m else 0


def _is_run_dir(d: Path) -> bool:
    return (
        (d / "eval_results" / "results.csv").exists()
        or (d / "history.csv").exists()
        or (d / "best.pt").exists()
    )

# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _read_eval(run_dir: Path) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "test_ms_ssim": None, "test_ms_ssim_r": None,
        "test_mse": None, "test_psnr": None,
    }
    results_path = run_dir / "eval_results" / "results.csv"
    if not results_path.exists():
        return out
    rows: list[dict[str, str]] = []
    with results_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split", "").strip().lower() == "test":
                rows.append(row)
    if not rows:
        return out
    for csv_col, key in [
        ("ms_ssim", "test_ms_ssim"),
        ("ms_ssim_r", "test_ms_ssim_r"),
        ("mse", "test_mse"),
        ("psnr", "test_psnr"),
    ]:
        good = [v for r in rows if (v := _safe_float(r.get(csv_col))) is not None]
        out[key] = sum(good) / len(good) if good else None
    return out


def _read_history(run_dir: Path) -> list[dict[str, Any]]:
    hist_path = run_dir / "history.csv"
    if not hist_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with hist_path.open(newline="") as f:
        for row in csv.DictReader(f):
            epoch = _safe_float(row.get("epoch"))
            if epoch is None:
                continue
            rows.append({
                "epoch": int(epoch),
                "total_epochs": int(_safe_float(row.get("total_epochs")) or 0),
                "train_loss": _safe_float(row.get("train_loss")),
                "val_loss": _safe_float(row.get("val_loss")),
                "val_ms_ssim": _safe_float(row.get("val_ms_ssim")),
                "val_ms_ssim_r": _safe_float(row.get("val_ms_ssim_r")),
            })
    return rows


def _best_val_ms_ssim(history: list[dict[str, Any]]) -> float | None:
    vals = [r["val_ms_ssim"] for r in history if r.get("val_ms_ssim") is not None]
    return max(vals) if vals else None

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _title_for_study(sid: str) -> str:
    for info in _STUDY_MAP.values():
        if info[0] == sid:
            return info[1]
    return sid


def _variant_for_baseline_frag(frag: str) -> str:
    return "3ch" if frag.startswith("3ch/") else "5ch"


def _discover_runs(project_root: Path) -> list[RunRecord]:
    runs_root = project_root / "experiments" / "runs"
    records: list[RunRecord] = []

    # Ablation runs
    abl_root = runs_root / "ablations"
    if abl_root.exists():
        for d in sorted(abl_root.rglob("*")):
            if not d.is_dir() or not _is_run_dir(d):
                continue
            rel_fwd = str(d.relative_to(runs_root)).replace("\\", "/")
            match = next(
                (info for frag, info in _STUDY_MAP.items() if frag in rel_fwd),
                None,
            )
            if match is None:
                continue
            study_id, study_title, variant, condition, corder = match
            history = _read_history(d)
            eval_m = _read_eval(d)
            records.append(RunRecord(
                study_id=study_id,
                study_title=study_title,
                variant=variant,
                condition=condition,
                condition_order=corder,
                is_baseline=False,
                seed=_extract_seed(d),
                run_dir=d,
                test_ms_ssim=eval_m["test_ms_ssim"],
                test_ms_ssim_r=eval_m["test_ms_ssim_r"],
                test_mse=eval_m["test_mse"],
                test_psnr=eval_m["test_psnr"],
                best_val_ms_ssim=_best_val_ms_ssim(history),
                n_epochs_completed=history[-1]["epoch"] if history else 0,
                total_epochs=history[-1]["total_epochs"] if history else 0,
                history=history,
            ))

    # Baseline runs (main n=20 replicates)
    for frag, study_ids in _BASELINE_STUDIES.items():
        base_dir = runs_root / Path(frag)
        if not base_dir.exists():
            continue
        for d in sorted(base_dir.iterdir()):
            if not d.is_dir() or not _is_run_dir(d):
                continue
            history = _read_history(d)
            eval_m = _read_eval(d)
            seed = _extract_seed(d)
            for sid in study_ids:
                records.append(RunRecord(
                    study_id=sid,
                    study_title=_title_for_study(sid),
                    variant=_variant_for_baseline_frag(frag),
                    condition=_BASELINE_LABEL[sid],
                    condition_order=0,
                    is_baseline=True,
                    seed=seed,
                    run_dir=d,
                    test_ms_ssim=eval_m["test_ms_ssim"],
                    test_ms_ssim_r=eval_m["test_ms_ssim_r"],
                    test_mse=eval_m["test_mse"],
                    test_psnr=eval_m["test_psnr"],
                    best_val_ms_ssim=_best_val_ms_ssim(history),
                    n_epochs_completed=history[-1]["epoch"] if history else 0,
                    total_epochs=history[-1]["total_epochs"] if history else 0,
                    history=history,
                ))

    return records

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_history(histories: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not histories:
        return []
    epoch_vals: dict[int, list[float | None]] = {}
    for h in histories:
        for row in h:
            v = row.get("val_ms_ssim")
            if v is not None:
                epoch_vals.setdefault(row["epoch"], []).append(v)
    result = []
    for epoch in sorted(epoch_vals):
        m, s = _mean_std(epoch_vals[epoch])
        result.append({
            "epoch": epoch,
            "val_ms_ssim_mean": m,
            "val_ms_ssim_std": s if s is not None else 0.0,
        })
    return result


def _aggregate(runs: list[RunRecord]) -> dict[str, list[AggCondition]]:
    from collections import defaultdict
    groups: dict[tuple[str, str], list[RunRecord]] = defaultdict(list)
    for r in runs:
        groups[(r.study_id, r.condition)].append(r)

    result: dict[str, list[AggCondition]] = {}
    for (study_id, condition), group in groups.items():
        agg = AggCondition(
            study_id=study_id,
            study_title=group[0].study_title,
            variant=group[0].variant,
            condition=condition,
            condition_order=group[0].condition_order,
            is_baseline=group[0].is_baseline,
            n_seeds=len(group),
            seeds=sorted({r.seed for r in group}),
            run_dirs=[r.run_dir for r in group],
            history_mean=_aggregate_history([r.history for r in group]),
        )
        agg.test_ms_ssim_mean, agg.test_ms_ssim_std = _mean_std([r.test_ms_ssim for r in group])
        agg.test_ms_ssim_r_mean, agg.test_ms_ssim_r_std = _mean_std([r.test_ms_ssim_r for r in group])
        agg.test_mse_mean, agg.test_mse_std = _mean_std([r.test_mse for r in group])
        agg.test_psnr_mean, agg.test_psnr_std = _mean_std([r.test_psnr for r in group])
        agg.best_val_ms_ssim_mean, agg.best_val_ms_ssim_std = _mean_std([r.best_val_ms_ssim for r in group])
        result.setdefault(study_id, []).append(agg)

    for sid in result:
        result[sid].sort(key=lambda a: a.condition_order)

    return result

# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

_PER_SEED_FIELDS = [
    "study_id", "study_title", "variant", "condition", "is_baseline", "seed",
    "test_ms_ssim", "test_ms_ssim_r", "test_mse", "test_psnr",
    "best_val_ms_ssim", "n_epochs_completed", "total_epochs", "run_dir",
]

_AGG_FIELDS = [
    "study_id", "study_title", "variant", "condition", "is_baseline",
    "n_seeds", "seeds",
    "test_ms_ssim_mean", "test_ms_ssim_std",
    "test_ms_ssim_r_mean", "test_ms_ssim_r_std",
    "test_mse_mean", "test_mse_std",
    "test_psnr_mean", "test_psnr_std",
    "best_val_ms_ssim_mean", "best_val_ms_ssim_std",
]


def _fmt(v: Any, decimals: int = 6) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _write_per_seed_csv(path: Path, runs: list[RunRecord]) -> None:
    def sort_key(r: RunRecord) -> tuple[int, int, int]:
        order = _STUDY_ORDER.index(r.study_id) if r.study_id in _STUDY_ORDER else 99
        return (order, r.condition_order, r.seed)

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_PER_SEED_FIELDS)
        w.writeheader()
        for r in sorted(runs, key=sort_key):
            w.writerow({
                "study_id": r.study_id,
                "study_title": r.study_title,
                "variant": r.variant,
                "condition": r.condition,
                "is_baseline": r.is_baseline,
                "seed": r.seed,
                "test_ms_ssim": _fmt(r.test_ms_ssim),
                "test_ms_ssim_r": _fmt(r.test_ms_ssim_r),
                "test_mse": _fmt(r.test_mse),
                "test_psnr": _fmt(r.test_psnr),
                "best_val_ms_ssim": _fmt(r.best_val_ms_ssim),
                "n_epochs_completed": r.n_epochs_completed,
                "total_epochs": r.total_epochs,
                "run_dir": str(r.run_dir),
            })


def _write_agg_csv(path: Path, all_studies: dict[str, list[AggCondition]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_AGG_FIELDS)
        w.writeheader()
        for sid in _STUDY_ORDER:
            if sid not in all_studies:
                continue
            for a in all_studies[sid]:
                w.writerow({
                    "study_id": a.study_id,
                    "study_title": a.study_title,
                    "variant": a.variant,
                    "condition": a.condition,
                    "is_baseline": a.is_baseline,
                    "n_seeds": a.n_seeds,
                    "seeds": ";".join(str(s) for s in a.seeds),
                    "test_ms_ssim_mean": _fmt(a.test_ms_ssim_mean),
                    "test_ms_ssim_std": _fmt(a.test_ms_ssim_std),
                    "test_ms_ssim_r_mean": _fmt(a.test_ms_ssim_r_mean),
                    "test_ms_ssim_r_std": _fmt(a.test_ms_ssim_r_std),
                    "test_mse_mean": _fmt(a.test_mse_mean),
                    "test_mse_std": _fmt(a.test_mse_std),
                    "test_psnr_mean": _fmt(a.test_psnr_mean),
                    "test_psnr_std": _fmt(a.test_psnr_std),
                    "best_val_ms_ssim_mean": _fmt(a.best_val_ms_ssim_mean),
                    "best_val_ms_ssim_std": _fmt(a.best_val_ms_ssim_std),
                })

# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

_C_BASELINE = "#95A5A6"     # neutral grey for baseline bars/lines
_C_ABLATION = ["#2E86C1", "#E67E22", "#8E44AD", "#16A085"]
_C_POS      = "#27AE60"     # green  — positive delta
_C_NEG      = "#E74C3C"     # red    — negative delta

# ---------------------------------------------------------------------------
# Per-study bar figure
# ---------------------------------------------------------------------------

def _plot_study(
    study_id: str,
    conds: list[AggCondition],
    out_dir: Path,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    is_lora_rank = study_id == "D"
    is_loss_weight = study_id == "E"
    ncols = 3 if is_loss_weight else 2
    fig_w = 16.0 if is_loss_weight else 12.0

    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, 4.8))
    fig.patch.set_facecolor("white")
    title = conds[0].study_title if conds else study_id
    fig.suptitle(f"Study {study_id}: {title}", fontsize=12, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Left panel: MS-SSIM bars (standard) or rank line (LoRA rank study) #
    # ------------------------------------------------------------------ #
    ax0 = axes[0]
    ax0.set_ylabel("Test MS-SSIM", fontsize=10)
    ax0.grid(axis="y", alpha=0.35, linewidth=0.8, zorder=0)
    ax0.spines[["top", "right"]].set_visible(False)

    if is_lora_rank:
        # Connected line plot over rank (4, 8, 16, 32)
        rank_order = [4, 8, 16, 32]
        cond_by_rank: dict[int, AggCondition] = {}
        for c in conds:
            for r in rank_order:
                label = f"r={r}"
                if label in c.condition or (c.is_baseline and r == 16):
                    cond_by_rank[r] = c
        xs = list(range(len(rank_order)))
        ms_vals = [
            (cond_by_rank[r].test_ms_ssim_mean if r in cond_by_rank else None)
            for r in rank_order
        ]
        ms_errs = [
            (cond_by_rank[r].test_ms_ssim_std or 0.0 if r in cond_by_rank else 0.0)
            for r in rank_order
        ]
        good = [(xi, v, e) for xi, v, e in zip(xs, ms_vals, ms_errs) if v is not None]
        if good:
            gx, gv, ge = zip(*good)
            ax0.errorbar(
                gx, gv, yerr=ge,
                fmt="o-", color=_C_ABLATION[0], linewidth=2.0,
                markersize=7, capsize=4, label="Test MS-SSIM",
            )
            # Baseline at rank=16 (index 2)
            if 16 in cond_by_rank and cond_by_rank[16].test_ms_ssim_mean is not None:
                bv = cond_by_rank[16].test_ms_ssim_mean
                ax0.axhline(bv, color=_C_BASELINE, linestyle="--", linewidth=1.2, alpha=0.8)
                ax0.text(3.05, bv, "r=16 (base)", fontsize=8, va="center", color="grey")
        tick_labels = [f"r={r}" + (" (base)" if r == 16 else "") for r in rank_order]
        ax0.set_xticks(xs)
        ax0.set_xticklabels(tick_labels, fontsize=9)
        ax0.set_xlabel("LoRA rank", fontsize=10)
        ax0.set_title("Test MS-SSIM vs LoRA rank", fontsize=10)
        # y zoom
        valid = [v for v in ms_vals if v is not None]
        if valid:
            ax0.set_ylim(min(valid) - 0.012, max(valid) + 0.012)
    else:
        # Standard grouped bar chart
        x = list(range(len(conds)))
        baseline_val: float | None = None
        abl_idx = 0
        for xi, c in enumerate(conds):
            v = c.test_ms_ssim_mean
            e = c.test_ms_ssim_std or 0.0
            if c.is_baseline:
                col, hatch = _C_BASELINE, "//"
                baseline_val = v
            else:
                col, hatch = _C_ABLATION[abl_idx % len(_C_ABLATION)], ""
                abl_idx += 1
            if v is not None:
                ax0.bar(
                    xi, v, width=0.6, color=col, hatch=hatch,
                    edgecolor="black", linewidth=0.7,
                    yerr=e, capsize=4, error_kw={"linewidth": 1.2},
                    zorder=3,
                )
        if baseline_val is not None:
            ax0.axhline(
                baseline_val, color=_C_BASELINE, linestyle="--",
                linewidth=1.2, alpha=0.8, zorder=2,
            )
        # Zoom y to data range
        vals = [c.test_ms_ssim_mean for c in conds if c.test_ms_ssim_mean is not None]
        if vals:
            ax0.set_ylim(max(0.0, min(vals) - 0.015), min(1.0, max(vals) + 0.015))
        ax0.set_xticks(x)
        ax0.set_xticklabels([c.condition for c in conds], fontsize=9, rotation=15, ha="right")
        ax0.set_title("Test MS-SSIM (mean ± std)", fontsize=10)

    # ------------------------------------------------------------------ #
    # Middle panel: validation MS-SSIM training curves                   #
    # ------------------------------------------------------------------ #
    ax1 = axes[1]
    abl_idx = 0
    for c in conds:
        h = c.history_mean
        if not h:
            continue
        epochs = [r["epoch"] for r in h]
        means  = [r["val_ms_ssim_mean"] for r in h]
        stds   = [r.get("val_ms_ssim_std") or 0.0 for r in h]
        if c.is_baseline:
            col, ls, lw, zorder = _C_BASELINE, "--", 1.6, 2
        else:
            col, ls, lw, zorder = _C_ABLATION[abl_idx % len(_C_ABLATION)], "-", 1.8, 3
            abl_idx += 1
        ax1.plot(epochs, means, color=col, linestyle=ls, linewidth=lw,
                 label=c.condition, zorder=zorder)
        if any(s > 1e-6 for s in stds):
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax1.fill_between(epochs, lo, hi, color=col, alpha=0.15, zorder=zorder - 1)
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Val MS-SSIM", fontsize=10)
    ax1.set_title("Validation MS-SSIM (mean ± 1σ)", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.35, linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)

    # ------------------------------------------------------------------ #
    # Right panel (loss-weight study only): Test MSE bars                #
    # ------------------------------------------------------------------ #
    if is_loss_weight:
        ax2 = axes[2]
        abl_idx = 0
        for xi, c in enumerate(conds):
            v = c.test_mse_mean
            e = c.test_mse_std or 0.0
            if c.is_baseline:
                col, hatch = _C_BASELINE, "//"
            else:
                col, hatch = _C_ABLATION[abl_idx % len(_C_ABLATION)], ""
                abl_idx += 1
            if v is not None:
                ax2.bar(
                    xi, v, width=0.6, color=col, hatch=hatch,
                    edgecolor="black", linewidth=0.7,
                    yerr=e, capsize=4, error_kw={"linewidth": 1.2},
                    zorder=3,
                )
        vals = [c.test_mse_mean for c in conds if c.test_mse_mean is not None]
        if vals:
            ax2.set_ylim(0.0, max(vals) * 1.15)
        ax2.set_xticks(list(range(len(conds))))
        ax2.set_xticklabels([c.condition for c in conds], fontsize=9, rotation=15, ha="right")
        ax2.set_ylabel("Test MSE", fontsize=10)
        ax2.set_title("Test MSE (lower = better)", fontsize=10)
        ax2.grid(axis="y", alpha=0.35, linewidth=0.8, zorder=0)
        ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = out_dir / f"study_{study_id}_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------------------------------
# Overview figure (2x3 delta panel)
# ---------------------------------------------------------------------------

def _plot_overview(all_studies: dict[str, list[AggCondition]], out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    present = [sid for sid in _STUDY_ORDER if sid in all_studies]
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 9))
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Ablation Overview — Δ Test MS-SSIM vs Baseline",
        fontsize=13, fontweight="bold",
    )

    for cell_i, ax in enumerate(axes.flatten()):
        if cell_i >= len(present):
            ax.set_visible(False)
            continue
        sid = present[cell_i]
        conds = all_studies[sid]

        base = next((c for c in conds if c.is_baseline), None)
        base_ms = base.test_ms_ssim_mean if base else None
        base_std = base.test_ms_ssim_std if base else None

        ablations = [c for c in conds if not c.is_baseline]
        y_labels, deltas, delta_errs, bar_colors = [], [], [], []
        for c in ablations:
            y_labels.append(c.condition)
            v = c.test_ms_ssim_mean
            if v is not None and base_ms is not None:
                d = v - base_ms
                s_abl = c.test_ms_ssim_std or 0.0
                s_base = base_std or 0.0
                deltas.append(d)
                delta_errs.append(math.sqrt(s_abl ** 2 + s_base ** 2))
                bar_colors.append(_C_POS if d >= 0 else _C_NEG)
            else:
                deltas.append(None)
                delta_errs.append(0.0)
                bar_colors.append("lightgrey")

        y_pos = list(range(len(ablations)))
        for yi, (d, de, col) in enumerate(zip(deltas, delta_errs, bar_colors)):
            if d is not None:
                ax.barh(
                    yi, d, xerr=de, color=col, alpha=0.80,
                    edgecolor="black", linewidth=0.6, height=0.55,
                    capsize=3, error_kw={"linewidth": 1.0},
                )
            else:
                ax.barh(yi, 0, color="lightgrey", height=0.55)

        ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_xlabel("Δ MS-SSIM vs baseline", fontsize=8)
        question = _STUDY_QUESTION.get(sid, sid)
        ax.set_title(f"Study {sid}: {question}", fontsize=9, fontweight="bold", pad=18)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="x", alpha=0.3, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

        if base_ms is not None:
            ax.annotate(
                f"base = {base_ms:.4f}",
                xy=(1.0, 1.0),
                xycoords="axes fraction",
                xytext=(0, 4),
                textcoords="offset points",
                fontsize=8.5,
                ha="right",
                va="bottom",
                color="dimgray",
                annotation_clip=False,
            )

    fig.tight_layout()
    out_path = out_dir / "ablation_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ---------------------------------------------------------------------------
# Markdown report helpers
# ---------------------------------------------------------------------------

def _fmt_ms(mean: float | None, std: float | None = None) -> str:
    if mean is None:
        return "—"
    s = f"{mean:.4f}"
    if std is not None and math.isfinite(std):
        return f"{s} ± {std:.4f}"
    return s


def _delta_str(v: float | None, base: float | None) -> str:
    if v is None or base is None:
        return "—"
    d = v - base
    return f"{'+' if d >= 0 else ''}{d:.4f}"


def _auto_conclusion(ablations: list[AggCondition], base_ms: float | None) -> str:
    if not ablations or base_ms is None:
        return "pending"
    deltas = [
        (c.test_ms_ssim_mean or 0.0) - base_ms
        for c in ablations
        if c.test_ms_ssim_mean is not None
    ]
    if not deltas:
        return "pending"
    max_abs = max(abs(d) for d in deltas)
    if max_abs < 0.003:
        return "Robust — all alternatives within noise"
    if max(deltas) > 0.003:
        return "Alternative may improve — check variance overlap"
    return "Alternatives underperform baseline"


def _auto_interpretation(
    conds: list[AggCondition], base_ms: float | None
) -> str:
    ablations = [c for c in conds if not c.is_baseline]
    if not ablations or base_ms is None:
        return "Results pending."
    deltas = {
        c.condition: (c.test_ms_ssim_mean or 0.0) - base_ms
        for c in ablations
        if c.test_ms_ssim_mean is not None
    }
    if not deltas:
        return "Results pending."
    max_abs = max(abs(d) for d in deltas.values())
    best_cond = max(deltas, key=lambda k: deltas[k])
    worst_cond = min(deltas, key=lambda k: deltas[k])
    if max_abs < 0.003:
        return (
            f"All alternatives differ from baseline by at most {max_abs:.4f} MS-SSIM, "
            f"well within three-seed noise. The protocol choice is robust."
        )
    return (
        f"The largest deviation from baseline is {max_abs:.4f} MS-SSIM "
        f"({best_cond}: {deltas[best_cond]:+.4f}, "
        f"{worst_cond}: {deltas[worst_cond]:+.4f}). "
        f"The baseline protocol {'is not clearly suboptimal' if max(deltas.values()) <= 0.003 else 'may not be globally optimal'}."
    )

# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _write_report(
    all_studies: dict[str, list[AggCondition]],
    out_dir: Path,
) -> Path:
    today = date.today().isoformat()
    lines: list[str] = []

    lines += [
        "# Ablation Study Report",
        f"_Generated: {today}_",
        "",
        "All improved ablation studies included here were run at "
        "n=20 training volumes with three seeds (42, 43, 44). "
        "Baselines are the main experimental replicates at the same scale.",
        "",
        "## Protocol",
        "",
        "- **Dataset**: ThinkOnward Image Impeccable (parts 1–2, 30 volumes); volume-level split.",
        "- **Training volumes**: n=20 for all ablation conditions and baselines.",
        "- **Seeds**: training seeds 42, 43, 44 (3 replicates per condition).",
        "- **Baselines**: matched main replicates for each study at the same training scale.",
        "- **PEFT**: LoRA on `qkv` + `proj`, no TTA.",
        "- **Primary comparison metric**: Test MS-SSIM; secondary: MS-SSIM-R, MSE, PSNR.",
        "",
        "## Figures",
        "",
        "| Filename | Description |",
        "|---|---|",
        "| `study_{id}_bars.png` (6 files) | Per-study bar chart comparing all conditions + baseline with ±std error bars |",
        "| `ablation_overview.png` | 2×3 panel of Δ MS-SSIM per condition for all studies |",
        "| `ablation_delta_summary.png` | Ranked horizontal bar chart of Δ MS-SSIM vs baseline across all studies |",
        "",
    ]

    # Summary table
    lines += ["## Summary", ""]
    lines += ["| Study | Question | Baseline MS-SSIM | Best alternative | Δ | Conclusion |"]
    lines += ["|---|---|---|---|---|---|"]
    for sid in _STUDY_ORDER:
        if sid not in all_studies:
            continue
        conds = all_studies[sid]
        base = next((c for c in conds if c.is_baseline), None)
        base_ms = base.test_ms_ssim_mean if base else None
        base_std = base.test_ms_ssim_std if base else None
        ablations = [c for c in conds if not c.is_baseline]
        if ablations and base_ms is not None:
            best = max(ablations, key=lambda c: c.test_ms_ssim_mean or -999.0)
            d = (best.test_ms_ssim_mean or 0.0) - base_ms
            delta_str = f"{'+' if d >= 0 else ''}{d:.4f}"
            best_str = f"{best.condition}: {_fmt_ms(best.test_ms_ssim_mean)}"
        else:
            delta_str, best_str = "—", "—"
        conclusion = _auto_conclusion(ablations, base_ms)
        q = _STUDY_QUESTION.get(sid, "")
        lines.append(
            f"| {sid} | {q} | {_fmt_ms(base_ms, base_std)} "
            f"| {best_str} | {delta_str} | {conclusion} |"
        )
    lines += [""]

    # Per-study sections
    for sid in _STUDY_ORDER:
        if sid not in all_studies:
            continue
        conds = all_studies[sid]
        title = conds[0].study_title
        base = next((c for c in conds if c.is_baseline), None)
        base_ms = base.test_ms_ssim_mean if base else None
        lines += [
            "---", "",
            f"## Study {sid}: {title}", "",
            f"**Question:** {_STUDY_QUESTION.get(sid, '')}", "",
        ]

        # Results table
        lines += [
            "| Condition | MS-SSIM | ±std | MS-SSIM-R | ±std | "
            "MSE | PSNR | Δ vs baseline |"
        ]
        lines += ["|---|---|---|---|---|---|---|---|"]
        for c in conds:
            delta = "— (baseline)" if c.is_baseline else _delta_str(c.test_ms_ssim_mean, base_ms)
            lines.append(
                f"| {c.condition} "
                f"| {_fmt_ms(c.test_ms_ssim_mean)} "
                f"| {_fmt_ms(c.test_ms_ssim_std) if c.test_ms_ssim_std is not None else '—'} "
                f"| {_fmt_ms(c.test_ms_ssim_r_mean)} "
                f"| {_fmt_ms(c.test_ms_ssim_r_std) if c.test_ms_ssim_r_std is not None else '—'} "
                f"| {_fmt_ms(c.test_mse_mean)} "
                f"| {_fmt_ms(c.test_psnr_mean)} "
                f"| {delta} |"
            )
        lines += [""]

        seed_str = ", ".join(str(s) for s in (conds[0].seeds if conds else []))
        lines += [f"_n=20 training volumes, seeds: {seed_str}_", ""]
        lines += [f"![Study {sid}](study_{sid}_bars.png)", ""]
        lines += [
            f"**Interpretation:** {_auto_interpretation(conds, base_ms)}", ""
        ]

    # Overview figure
    lines += [
        "---", "",
        "## Overview Panel", "",
        "Each cell shows Δ MS-SSIM relative to the baseline. "
        "Green bars indicate improvement, red bars indicate degradation. "
        "Error bars show propagated ± 1σ uncertainty.",
        "",
        "![Overview](ablation_overview.png)", "",
    ]

    # Protocol conclusions
    lines += [
        "---", "",
        "## Protocol Conclusions", "",
        "All studies were run at n=20/3-seeds to match the main experiment scale. "
        "The following decisions are justified by these results:", "",
    ]
    for sid in _STUDY_ORDER:
        if sid not in all_studies:
            continue
        conds = all_studies[sid]
        title = conds[0].study_title
        base = next((c for c in conds if c.is_baseline), None)
        base_ms = base.test_ms_ssim_mean if base else None
        ablations = [c for c in conds if not c.is_baseline]
        conclusion = _auto_conclusion(ablations, base_ms)
        lines.append(f"- **Study {sid} — {title}:** {conclusion}")
    lines += [
        "",
        "## Claim Boundaries",
        "",
        "- All comparisons are relative to the matched study baseline at n=20, 3 seeds; do not generalise to other training scales.",
        "- `neighbor_stride > 1` conditions are ablation-only; the main 3ch setup always uses stride 1.",
        "- The patch-embedding-only 5ch control (without LoRA) is excluded from main summary tables.",
        "- These ablations do not cover alternative backbones, different datasets, or TTA.",
    ]

    report_path = out_dir / "ablation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

# ---------------------------------------------------------------------------
# Delta summary plot (all studies in one ranked chart)
# ---------------------------------------------------------------------------

def _plot_delta_summary(all_studies: dict[str, list["AggCondition"]], out_dir: Path) -> Path:
    """Horizontal bar chart: Δ MS-SSIM vs baseline for every ablation condition.

    All studies are pooled into a single ranked chart so the relative importance
    of each design choice is immediately visible.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  [skip] ablation_delta_summary.png — matplotlib not available")
        return out_dir / "ablation_delta_summary.png"

    entries: list[tuple[str, float, str]] = []  # (label, delta, study_id)
    for sid in _STUDY_ORDER:
        conds = all_studies.get(sid, [])
        base = next((c for c in conds if c.is_baseline), None)
        if base is None or base.test_ms_ssim_mean is None:
            continue
        for c in conds:
            if c.is_baseline:
                continue
            if c.test_ms_ssim_mean is None:
                continue
            delta = c.test_ms_ssim_mean - base.test_ms_ssim_mean
            short_cond = c.condition if len(c.condition) <= 30 else c.condition[:28] + "…"
            label = f"{sid}: {short_cond}"
            entries.append((label, delta, sid))

    if not entries:
        print("  [skip] ablation_delta_summary.png — no evaluated ablation conditions")
        return out_dir / "ablation_delta_summary.png"

    entries.sort(key=lambda t: t[1])  # ascending = worst on top, best on bottom
    labels, deltas, study_ids = zip(*entries)

    study_palette = {
        "A": "#4c78a8", "B": "#f58518", "C": "#54a24b",
        "D": "#e45756", "E": "#72b7b2", "F": "#b279a2",
    }
    bar_colors = [study_palette.get(sid, "#888888") for sid in study_ids]

    fig_h = max(4, 0.45 * len(entries) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(range(len(labels)), deltas, color=bar_colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Δ Test MS-SSIM vs study baseline")
    ax.set_title("Ablation Δ MS-SSIM — all studies ranked\n(positive = better than baseline; dashed = baseline)")
    ax.grid(True, axis="x", alpha=0.25)

    # Add legend for study colour coding
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(color=study_palette.get(sid, "#888888"), label=sid)
        for sid in _STUDY_ORDER if sid in {e[2] for e in entries}
    ]
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8, title="Study")

    fig.tight_layout()
    out_path = out_dir / "ablation_delta_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise DINOv3 seismic denoising ablation studies."
    )
    parser.add_argument(
        "--project-root", type=Path, default=None,
        help="Project root. Default: auto-detect from script location.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory. Default: <project-root>/experiments/summaries/ablations/",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figures.")
    parser.add_argument("--no-report", action="store_true", help="Skip Markdown report.")
    args = parser.parse_args()

    project_root = (
        args.project_root or _project_root_default(Path(__file__))
    ).resolve()
    out_dir = (
        args.out_dir or project_root / "experiments" / "summaries" / "ablations"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root : {project_root}")
    print(f"Output dir   : {out_dir}")

    print("\nDiscovering runs...")
    runs = _discover_runs(project_root)
    n_abl = sum(1 for r in runs if not r.is_baseline)
    n_base = sum(1 for r in runs if r.is_baseline)
    print(f"  {n_abl} ablation run(s), {n_base} baseline run(s) found")

    print("\nAggregating across seeds...")
    all_studies = _aggregate(runs)
    for sid in _STUDY_ORDER:
        if sid not in all_studies:
            print(f"  [missing] Study {sid}")
            continue
        conds = all_studies[sid]
        abl_conds = [c for c in conds if not c.is_baseline]
        base_cond = next((c for c in conds if c.is_baseline), None)
        print(
            f"  Study {sid}: {len(abl_conds)} ablation condition(s), "
            f"baseline n_seeds={base_cond.n_seeds if base_cond else 0}"
        )

    print("\nWriting CSVs...")
    per_seed_path = out_dir / "ablation_runs_per_seed.csv"
    agg_path = out_dir / "ablation_aggregated.csv"
    _write_per_seed_csv(per_seed_path, runs)
    _write_agg_csv(agg_path, all_studies)
    print(f"  {per_seed_path}")
    print(f"  {agg_path}")

    if not args.no_plots:
        print("\nGenerating figures...")
        try:
            for sid in _STUDY_ORDER:
                if sid not in all_studies:
                    print(f"  [skip] Study {sid} — no data")
                    continue
                p = _plot_study(sid, all_studies[sid], out_dir)
                print(f"  {p.name}")
            p = _plot_overview(all_studies, out_dir)
            print(f"  {p.name}")
            p = _plot_delta_summary(all_studies, out_dir)
            print(f"  {p.name}")
        except Exception as exc:
            import traceback
            print(f"  WARNING: figure generation failed — {exc}")
            traceback.print_exc()
    else:
        print("\nPlots skipped (--no-plots).")

    if not args.no_report:
        print("\nWriting report...")
        p = _write_report(all_studies, out_dir)
        print(f"  {p.name}")
    else:
        print("\nReport skipped (--no-report).")

    print("\nDone.")


if __name__ == "__main__":
    main()
