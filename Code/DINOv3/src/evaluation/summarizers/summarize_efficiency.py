"""Summarize full fine-tuning vs PEFT efficiency comparison.

Reads per-seed eval_results/results.csv files, exact train-step timing from
training_times_per_run.csv or training_timing.csv when available, and optionally
inference_benchmark.csv. Legacy history/SLURM timings are retained only as
approximate fallbacks and are flagged in the output.

Run catalogue: multi-data-seed protocol only (data seeds 101/202/303 x training
seeds 42/43/44 = 9 runs per variant). PEFT runs live under
experiments/runs/main_multidata/; full-FT runs under full_ft_multidata/.

Outputs:
  experiments/summaries/efficiency/efficiency_comparison.csv
  experiments/summaries/efficiency/performance_comparison.png
  experiments/summaries/efficiency/efficiency_overview.png

Usage:
    python evaluation/summarize_efficiency.py
    python evaluation/summarize_efficiency.py --no-plots
    python evaluation/summarize_efficiency.py --out-dir /path/to/dir
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise RuntimeError("matplotlib is required") from exc

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

PROJECT_ROOT = default_project_root(__file__)  # src -> DINOv3 -> Code -> RP
RUNS_ROOT    = PROJECT_ROOT / "experiments" / "runs"
TIMING_CSV   = PROJECT_ROOT / "experiments" / "summaries" / "timing" / "training_times_per_run.csv"
INFERENCE_CSV = PROJECT_ROOT / "experiments" / "summaries" / "timing" / "inference_benchmark.csv"
DEFAULT_OUT  = PROJECT_ROOT / "experiments" / "summaries" / "efficiency"

# ---------------------------------------------------------------------------
# Run catalogue — multi-data-seed protocol only
# Each entry: (display_name, variant_label, run_dir_relative_to_RUNS_ROOT, runs)
# runs: list of (data_seed, train_seed, run_id) strings
# ---------------------------------------------------------------------------
_MULTIDATA_RUNS = [
    ("101", "42", "01"), ("101", "43", "02"), ("101", "44", "03"),
    ("202", "42", "01"), ("202", "43", "02"), ("202", "44", "03"),
    ("303", "42", "01"), ("303", "43", "02"), ("303", "44", "03"),
]

CATALOGUE = [
    (
        "2D PEFT", "2D",
        "main_multidata/2d/impeccable_repeated_stride5_lora_r16",
        _MULTIDATA_RUNS,
    ),
    (
        "3ch PEFT", "3ch",
        "main_multidata/3ch/impeccable_neighbors3_stride5_lora_r16",
        _MULTIDATA_RUNS,
    ),
    (
        "5ch PEFT", "5ch",
        "main_multidata/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16",
        _MULTIDATA_RUNS,
    ),
    (
        "2D full-FT", "full_ft_2d",
        "full_ft_multidata/2d/impeccable_repeated_stride5_full_ft",
        _MULTIDATA_RUNS,
    ),
    (
        "3ch full-FT", "full_ft_3ch",
        "full_ft_multidata/3ch/impeccable_neighbors3_stride5_full_ft",
        _MULTIDATA_RUNS,
    ),
    (
        "5ch full-FT", "full_ft_5ch",
        "full_ft_multidata/5ch/impeccable_neighbors5_stride5_patch_emb_full_ft",
        _MULTIDATA_RUNS,
    ),
]

# Known parameter counts (deterministic from architecture).
PARAMS = {
    "2D":          {"total": 23_337_457, "trainable": 1_736_305,  "pct": 7.44},
    "3ch":         {"total": 23_337_457, "trainable": 1_736_305,  "pct": 7.44},
    "5ch":         {"total": 23_515_633, "trainable": 2_209_777,  "pct": 9.40},
    "full_ft_2d":  {"total": 23_337_457, "trainable": 23_337_457, "pct": 100.0},
    "full_ft_3ch": {"total": 23_337_457, "trainable": 23_337_457, "pct": 100.0},
    "full_ft_5ch": {"total": 23_515_633, "trainable": 23_515_633, "pct": 100.0},
}

COLORS = {
    "2D":          "#62727A",
    "3ch":         "#008C95",
    "5ch":         "#D48210",
    "full_ft_2d":  "#C0392B",
    "full_ft_3ch": "#922B21",
    "full_ft_5ch": "#641E16",
}

DISPLAY = {
    "2D":          "2D PEFT",
    "3ch":         "3ch PEFT",
    "5ch":         "5ch PEFT",
    "full_ft_2d":  "2D full-FT",
    "full_ft_3ch": "3ch full-FT",
    "full_ft_5ch": "5ch full-FT",
}

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_results_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _seed_test_metrics(run_dir: Path) -> dict | None:
    p = run_dir / "eval_results" / "results.csv"
    if not p.exists():
        return None
    rows = _load_results_csv(p)
    if not rows:
        return None
    return {
        "ms_ssim":   float(np.mean([float(r["ms_ssim"])   for r in rows])),
        "ms_ssim_r": float(np.mean([float(r["ms_ssim_r"]) for r in rows])),
        "mse":       float(np.mean([float(r["mse"])        for r in rows])),
        "psnr":      float(np.mean([float(r["psnr"])       for r in rows])),
    }


def _float_or_none(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool_from_csv(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _history_epoch_time_min(run_dir: Path) -> float | None:
    """Sum epoch_time_s from legacy history.csv as an approximate fallback."""
    p = run_dir / "history.csv"
    if not p.exists():
        return None
    rows = _load_results_csv(p)
    times = [_float_or_none(r.get("epoch_time_s")) for r in rows]
    times = [t for t in times if t is not None]
    if not times:
        return None
    return sum(times) / 60.0


def _training_timing_time_min(run_dir: Path) -> float | None:
    p = run_dir / "training_timing.csv"
    if not p.exists():
        return None
    rows = _load_results_csv(p)
    times = [_float_or_none(r.get("train_step_time_s")) for r in rows]
    times = [t for t in times if t is not None]
    if not times:
        return None
    return sum(times) / 60.0


def _load_training_times_csv() -> dict[str, dict]:
    """Returns dict keyed by run_dir (forward slash), value is the CSV row."""
    if not TIMING_CSV.exists():
        return {}
    with TIMING_CSV.open(newline="") as f:
        return {r["run_dir"]: r for r in csv.DictReader(f)}


def _run_timing_record(run_dir: Path, rel_key: str, timing_lookup: dict[str, dict]) -> dict:
    row = timing_lookup.get(rel_key, {})
    wallclock = _float_or_none(row.get("job_wallclock_min") or row.get("total_wallclock_min"))

    train_step = _float_or_none(row.get("train_step_time_min"))
    if train_step is not None:
        return {
            "minutes": train_step,
            "source": row.get("timing_source") or "training_times_per_run.csv",
            "exact": _bool_from_csv(row.get("timing_is_exact")),
            "wallclock_min": wallclock,
        }

    direct = _training_timing_time_min(run_dir)
    if direct is not None:
        return {
            "minutes": direct,
            "source": "training_timing.csv",
            "exact": True,
            "wallclock_min": wallclock,
        }

    history = _history_epoch_time_min(run_dir)
    if history is not None:
        return {
            "minutes": history,
            "source": "history_epoch_time_s",
            "exact": False,
            "wallclock_min": wallclock,
        }

    if wallclock is not None:
        return {
            "minutes": wallclock,
            "source": "slurm_wallclock",
            "exact": False,
            "wallclock_min": wallclock,
        }

    return {
        "minutes": None,
        "source": "",
        "exact": False,
        "wallclock_min": wallclock,
    }


def _load_inference_csv() -> list[dict]:
    if not INFERENCE_CSV.exists():
        return []
    with INFERENCE_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_group_stats(
    variant_label: str,
    variant_dir: str,
    seeds_runids: list[tuple[str, str, str]],
    timing_lookup: dict[str, dict],
) -> dict:
    """Compute per-group aggregate statistics across seeds.

    seeds_runids: list of (data_seed, train_seed, run_id) triples.
    Run directory: RUNS_ROOT / variant_dir / data_seed{ds} / seed{ts}_run{ri}.
    """
    ms_ssim_vals, ms_ssim_r_vals, mse_vals, psnr_vals = [], [], [], []
    train_time_vals: list[float] = []
    wallclock_vals: list[float] = []
    timing_sources: list[str] = []
    timing_exact_flags: list[bool] = []
    n_evaluated = 0
    available_seeds: list[str] = []

    for data_seed, seed, run_id in seeds_runids:
        run_dir = RUNS_ROOT / variant_dir / f"data_seed{data_seed}" / f"seed{seed}_run{run_id}"
        metrics = _seed_test_metrics(run_dir)
        if metrics:
            ms_ssim_vals.append(metrics["ms_ssim"])
            ms_ssim_r_vals.append(metrics["ms_ssim_r"])
            mse_vals.append(metrics["mse"])
            psnr_vals.append(metrics["psnr"])
            n_evaluated += 1
            available_seeds.append(f"d{data_seed}:s{seed}")

        # Primary timing is exact train-step time when available. Legacy
        # history/SLURM values remain approximate fallbacks and are flagged.
        rel_key = f"{variant_dir}/data_seed{data_seed}/seed{seed}_run{run_id}".replace("\\", "/")
        timing = _run_timing_record(run_dir, rel_key, timing_lookup)
        if timing["minutes"] is not None:
            train_time_vals.append(float(timing["minutes"]))
            timing_sources.append(str(timing["source"]))
            timing_exact_flags.append(bool(timing["exact"]))
        if timing["wallclock_min"] is not None:
            wallclock_vals.append(float(timing["wallclock_min"]))

    def _stat(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    p = PARAMS[variant_label]

    ms_mean, ms_std        = _stat(ms_ssim_vals)
    ms_r_mean, ms_r_std    = _stat(ms_ssim_r_vals)
    mse_mean, mse_std      = _stat(mse_vals)
    psnr_mean, psnr_std    = _stat(psnr_vals)
    tt_mean, tt_std        = _stat(train_time_vals)
    wc_mean, wc_std        = _stat(wallclock_vals)
    exact_count = sum(1 for flag in timing_exact_flags if flag)
    unique_sources = sorted(set(s for s in timing_sources if s))

    return {
        "variant_label":        variant_label,
        "display_name":         DISPLAY[variant_label],
        "n_seeds_total":        len(seeds_runids),
        "n_seeds_evaluated":    n_evaluated,
        "available_seeds":      ",".join(available_seeds),
        "total_params":         p["total"],
        "trainable_params":     p["trainable"],
        "trainable_pct":        p["pct"],
        "test_ms_ssim_mean":    ms_mean,
        "test_ms_ssim_std":     ms_std,
        "test_ms_ssim_r_mean":  ms_r_mean,
        "test_ms_ssim_r_std":   ms_r_std,
        "test_mse_mean":        mse_mean,
        "test_mse_std":         mse_std,
        "test_psnr_mean":       psnr_mean,
        "test_psnr_std":        psnr_std,
        "train_step_time_mean_min":  tt_mean,
        "train_step_time_std_min":   tt_std,
        "train_step_gpu_hours_mean": round(tt_mean / 60, 2) if tt_mean is not None else None,
        "train_step_gpu_hours_std":  round(tt_std / 60, 2) if tt_std is not None else None,
        "timing_source":        "+".join(unique_sources),
        "timing_is_exact":      bool(timing_exact_flags) and all(timing_exact_flags),
        "n_exact_timed_runs":   exact_count,
        "job_wallclock_mean_min": wc_mean,
        "job_wallclock_std_min":  wc_std,
        "job_wallclock_gpu_hours_mean": round(wc_mean / 60, 2) if wc_mean is not None else None,
        "job_wallclock_gpu_hours_std":  round(wc_std / 60, 2) if wc_std is not None else None,
        # Backwards-compatible aliases now point to the primary train-step
        # timing value, not to SLURM wallclock unless no better source exists.
        "train_time_mean_min":  tt_mean,
        "train_time_std_min":   tt_std,
        "train_gpu_hours_mean": round(tt_mean / 60, 2) if tt_mean is not None else None,
        "train_gpu_hours_std":  round(tt_std / 60, 2) if tt_std is not None else None,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "display_name", "variant_label",
    "n_seeds_total", "n_seeds_evaluated", "available_seeds",
    "total_params", "trainable_params", "trainable_pct",
    "test_ms_ssim_mean", "test_ms_ssim_std",
    "test_ms_ssim_r_mean", "test_ms_ssim_r_std",
    "test_mse_mean", "test_mse_std",
    "test_psnr_mean", "test_psnr_std",
    "train_step_time_mean_min", "train_step_time_std_min",
    "train_step_gpu_hours_mean", "train_step_gpu_hours_std",
    "timing_source", "timing_is_exact", "n_exact_timed_runs",
    "job_wallclock_mean_min", "job_wallclock_std_min",
    "job_wallclock_gpu_hours_mean", "job_wallclock_gpu_hours_std",
    "train_time_mean_min", "train_time_std_min",
    "train_gpu_hours_mean", "train_gpu_hours_std",
    "infer_latency_bs1_mean_ms", "infer_latency_bs16_mean_ms",
    "infer_throughput_bs16_slices_per_s",
]


def _merge_inference(groups: list[dict], infer_rows: list[dict]) -> None:
    """Attach per-variant inference stats (batch_size 1 and 16) to group dicts."""
    if not infer_rows:
        return
    for g in groups:
        g["infer_latency_bs1_mean_ms"] = None
        g["infer_latency_bs16_mean_ms"] = None
        g["infer_throughput_bs16_slices_per_s"] = None
        vl = g["variant_label"]
        # inference benchmark uses 'variant' column matching family (2d/3ch/5ch)
        bench_family = {
            "2D": "2d", "3ch": "3ch", "5ch": "5ch",
        }.get(vl)
        if bench_family is None:
            continue
        bs1  = [r for r in infer_rows if r.get("variant") == bench_family and r.get("batch_size") == "1"]
        bs16 = [r for r in infer_rows if r.get("variant") == bench_family and r.get("batch_size") == "16"]
        g["infer_latency_bs1_mean_ms"]           = float(np.mean([float(r["latency_mean_ms"]) for r in bs1]))  if bs1  else None
        g["infer_latency_bs16_mean_ms"]          = float(np.mean([float(r["latency_mean_ms"]) for r in bs16])) if bs16 else None
        g["infer_throughput_bs16_slices_per_s"]  = float(np.mean([float(r["throughput_slices_per_s"]) for r in bs16])) if bs16 else None


def write_csv(groups: list[dict], out_dir: Path) -> None:
    path = out_dir / "efficiency_comparison.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for g in groups:
            row = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in g.items()}
            writer.writerow(row)
    print(f"CSV saved: {path}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.22)
    ax.tick_params(axis="both", labelsize=11)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_horizontalalignment("right")


def _bar_group(ax, groups: list[dict], metric: str, err_metric: str | None,
               ylabel: str, title: str, fmt: str = ".4f") -> None:
    names  = [g["display_name"] for g in groups]
    vals   = [g[metric] for g in groups]
    errs   = [g[err_metric] if err_metric and g.get(err_metric) else 0 for g in groups]
    colors = [COLORS[g["variant_label"]] for g in groups]
    # Grey out bars with no data
    bars = ax.bar(names, [v if v is not None else 0 for v in vals],
                  color=colors, edgecolor="white", linewidth=0.8,
                  yerr=[e if e else 0 for e in errs],
                  capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "0.3"})
    for bar, v in zip(bars, vals):
        if v is None:
            bar.set_alpha(0.25)
            ax.text(bar.get_x() + bar.get_width() / 2, 0.01,
                    "pending", ha="center", va="bottom", fontsize=9, color="0.5")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (errs[bars.index(bar)] or 0) + 0.002,
                    f"{v:{fmt}}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    _style(ax)


def plot_performance(groups: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("PEFT vs Full Fine-Tuning: Denoising Performance", fontsize=14, fontweight="bold")

    _bar_group(axes[0], groups,
               "test_ms_ssim_mean", "test_ms_ssim_std",
               "MS-SSIM (↑)", "Test MS-SSIM", fmt=".4f")
    axes[0].set_ylim(0.82, 0.93)

    _bar_group(axes[1], groups,
               "test_psnr_mean", "test_psnr_std",
               "PSNR dB (↑)", "Test PSNR", fmt=".2f")

    fig.tight_layout()
    path = out_dir / "performance_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def plot_efficiency(groups: list[dict], out_dir: Path) -> None:
    has_inference = any(g.get("infer_latency_bs1_mean_ms") is not None for g in groups)
    ncols = 3 if has_inference else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols + 1, 5))
    fig.suptitle("PEFT vs Full Fine-Tuning: Efficiency", fontsize=14, fontweight="bold")

    # Trainable params (M)
    ax = axes[0]
    names  = [g["display_name"] for g in groups]
    vals   = [g["trainable_params"] / 1e6 for g in groups]
    colors = [COLORS[g["variant_label"]] for g in groups]
    bars = ax.bar(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, v, g in zip(bars, vals, groups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{v:.1f}M\n({g['trainable_pct']:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Trainable parameters (M)", fontsize=12)
    ax.set_title("Trainable Parameters", fontsize=13, fontweight="bold")
    _style(ax)

    # Primary training GPU-hours
    _bar_group(axes[1], groups,
               "train_step_gpu_hours_mean", "train_step_gpu_hours_std",
               "GPU-hours (1 GPU)", "Train-Step Time", fmt=".1f")

    if has_inference:
        _bar_group(axes[2], groups,
                   "infer_latency_bs1_mean_ms", None,
                   "Latency ms/slice (batch=1)", "Inference Latency", fmt=".1f")

    fig.tight_layout()
    path = out_dir / "efficiency_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {path}")


def print_table(groups: list[dict]) -> None:
    header = f"{'Variant':<14} {'Trainable':>12} {'%':>6} {'Train GPU-h':>12} {'Exact':>7} {'Test MS-SSIM':>14} {'Seeds':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for g in groups:
        ms = g["test_ms_ssim_mean"]
        ms_s = g["test_ms_ssim_std"]
        tt = g["train_step_gpu_hours_mean"]
        ms_str = f"{ms:.4f}+/-{ms_s:.4f}" if ms and ms_s else ("pending" if ms is None else f"{ms:.4f}")
        tt_str = f"{tt:.1f}" if tt else "-"
        exact = f"{g['n_exact_timed_runs']}/{g['n_seeds_total']}"
        seeds  = f"{g['n_seeds_evaluated']}/{g['n_seeds_total']}"
        print(f"{g['display_name']:<14} {g['trainable_params']:>12,} {g['trainable_pct']:>5.1f}% {tt_str:>12} {exact:>7} {ms_str:>14} {seeds:>7}")
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",   type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-plots",  action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    timing_lookup = _load_training_times_csv()
    infer_rows    = _load_inference_csv()

    if not timing_lookup:
        print("Warning: training_times_per_run.csv not found; timing will come from per-run artifacts when available")
    if not infer_rows:
        print("Note: inference_benchmark.csv not found — inference columns will be empty (run benchmark_inference.py first)")

    groups = []
    for display_name, variant_label, variant_dir, seeds_runids in CATALOGUE:
        g = compute_group_stats(variant_label, variant_dir, seeds_runids, timing_lookup)
        groups.append(g)
        status = f"{g['n_seeds_evaluated']}/{g['n_seeds_total']} seeds evaluated"
        print(f"  {display_name}: {status}")

    _merge_inference(groups, infer_rows)

    print_table(groups)
    write_csv(groups, out_dir)

    if not args.no_plots:
        plot_performance(groups, out_dir)
        plot_efficiency(groups, out_dir)

    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
