"""Training time feasibility study across model families and data scales.

Reads exact training_timing.csv artifacts (data_efficiency_100train, backbone_comparison)
and approximate history.csv timing (full_ft) to produce a markdown report and
matplotlib figures comparing training costs across:

  - DINOv3 PEFT (2D / 3ch / 5ch) across n_vols 20-100 (data_efficiency_100train)
  - SFM PEFT     (2D / 3ch / 5ch) at n=20  (backbone_comparison)
  - SwinV2 PEFT  (2D / 3ch / 5ch) at n=20  (backbone_comparison)
  - DINOv3 Full-FT (2D / 3ch / 5ch) at n=20 (full_ft, approximate)

Usage:
    python Code/DINOv3/src/evaluation/training_time_study.py
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC = ensure_src_on_path(__file__)
PROJECT_ROOT = default_project_root(__file__)
RUNS = PROJECT_ROOT / "experiments" / "runs"
OUT_DIR = PROJECT_ROOT / "experiments" / "summaries" / "training_time_study"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class RunTiming(NamedTuple):
    label: str           # human-readable group label
    variant: str         # "2D", "3ch", "5ch"
    n_vols: int          # training volumes
    total_train_min: float   # sum of train_step_time_s across all epochs, in minutes
    total_epochs: int
    timing_exact: bool
    seed_tag: str        # e.g. "seed42"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_training_timing(path: Path) -> tuple[float, int]:
    """Return (total_train_step_seconds, n_epochs) from training_timing.csv."""
    total_s = 0.0
    n = 0
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                val = row.get("train_step_time_s", "")
                if val:
                    total_s += float(val)
                    n += 1
    except OSError:
        pass
    return total_s, n


def _read_history_timing(path: Path) -> tuple[float, int]:
    """Return (total_epoch_time_seconds, n_epochs) from history.csv.
    epoch_time_s includes validation, so this is an overestimate of train-step time.
    """
    total_s = 0.0
    n = 0
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                val = row.get("epoch_time_s", "")
                if val:
                    total_s += float(val)
                    n += 1
    except OSError:
        pass
    return total_s, n


def _variant_from_name(name: str) -> str:
    """Map directory component to canonical variant name."""
    n = name.lower()
    # Match longest/most-specific patterns first to avoid 'n35vols' → 3ch confusion
    if "neighbors5" in n or "neighbors_5" in n or "5ch" in n:
        return "5ch"
    if "neighbors3" in n or "neighbors_3" in n or "3ch" in n:
        return "3ch"
    if n in ("2d", "2d_native_stride5_lora_r16", "2d_repeated_stride5_lora_r16"):
        return "2D"
    # Fallback: use the raw dir name to avoid substring false-positives
    if n.startswith("2d"):
        return "2D"
    return "2D"


def _n_vols_from_name(name: str) -> int:
    """Extract n_vols from a run directory name like *_n20vols_* -> 20."""
    import re
    m = re.search(r"_n(\d+)vols", name)
    return int(m.group(1)) if m else 20


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------
def collect_data_efficiency_100train(data_seed: str = "data_seed101") -> list[RunTiming]:
    """Collect DINOv3 PEFT timing from data_efficiency_100train runs."""
    base = RUNS / "data_efficiency_100train"
    if not base.exists():
        return []
    # Top-level variant dirs are "2d", "3ch", "5ch" — use these directly
    variant_name_map = {"2d": "2D", "3ch": "3ch", "5ch": "5ch"}
    rows: list[RunTiming] = []
    for variant_dir in sorted(base.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = variant_name_map.get(variant_dir.name.lower())
        if variant is None:
            continue
        for cfg_dir in sorted(variant_dir.iterdir()):
            if not cfg_dir.is_dir():
                continue
            n_vols = _n_vols_from_name(cfg_dir.name)
            seed_root = cfg_dir / data_seed
            if not seed_root.exists():
                continue
            for seed_dir in sorted(seed_root.iterdir()):
                if not seed_dir.is_dir():
                    continue
                timing_path = seed_dir / "training_timing.csv"
                if not timing_path.exists():
                    continue
                total_s, epochs = _read_training_timing(timing_path)
                if epochs == 0:
                    continue
                rows.append(RunTiming(
                    label="DINOv3 PEFT",
                    variant=variant,
                    n_vols=n_vols,
                    total_train_min=total_s / 60.0,
                    total_epochs=epochs,
                    timing_exact=True,
                    seed_tag=seed_dir.name,
                ))
    return rows


def collect_backbone_comparison(backbone: str) -> list[RunTiming]:
    """Collect timing from backbone_comparison/<backbone> runs (n=20 vols)."""
    label_map = {
        "sfm_vit_base_patch16": "SFM PEFT",
        "swin_v2_t": "SwinV2 PEFT",
    }
    label = label_map.get(backbone, backbone)
    base = RUNS / "backbone_comparison" / backbone
    if not base.exists():
        return []
    rows: list[RunTiming] = []
    for variant_dir in sorted(base.iterdir()):
        if not variant_dir.is_dir():
            continue
        variant = _variant_from_name(variant_dir.name)
        for data_seed_dir in sorted(variant_dir.iterdir()):
            if not data_seed_dir.is_dir():
                continue
            for seed_dir in sorted(data_seed_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                timing_path = seed_dir / "training_timing.csv"
                if not timing_path.exists():
                    continue
                total_s, epochs = _read_training_timing(timing_path)
                if epochs == 0:
                    continue
                rows.append(RunTiming(
                    label=label,
                    variant=variant,
                    n_vols=20,
                    total_train_min=total_s / 60.0,
                    total_epochs=epochs,
                    timing_exact=True,
                    seed_tag=f"{data_seed_dir.name}/{seed_dir.name}",
                ))
    return rows


def collect_full_ft() -> list[RunTiming]:
    """Collect approximate timing from full_ft runs (history.csv only)."""
    variant_map = {
        "impeccable_repeated_stride5_full_ft": "2D",
        "impeccable_neighbors3_stride5_full_ft": "3ch",
        "impeccable_neighbors5_stride5_patch_emb_full_ft": "5ch",
    }
    base = RUNS / "full_ft"
    if not base.exists():
        return []
    rows: list[RunTiming] = []
    for cfg_dir in sorted(base.iterdir()):
        if not cfg_dir.is_dir():
            continue
        variant = variant_map.get(cfg_dir.name, _variant_from_name(cfg_dir.name))
        for seed_dir in sorted(cfg_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            hist = seed_dir / "history.csv"
            if not hist.exists():
                continue
            total_s, epochs = _read_history_timing(hist)
            if epochs == 0:
                continue
            rows.append(RunTiming(
                label="DINOv3 Full-FT",
                variant=variant,
                n_vols=20,
                total_train_min=total_s / 60.0,
                total_epochs=epochs,
                timing_exact=False,
                seed_tag=seed_dir.name,
            ))
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(rows: list[RunTiming], label: str, variant: str, n_vols: int) -> dict | None:
    """Return mean/median/min/std stats for a given (label, variant, n_vols) group."""
    subset = [r for r in rows
              if r.label == label and r.variant == variant and r.n_vols == n_vols]
    if not subset:
        return None
    vals = [r.total_train_min for r in subset]
    exact = subset[0].timing_exact
    return {
        "label": label, "variant": variant, "n_vols": n_vols,
        "n_seeds": len(vals),
        "mean_min": statistics.mean(vals),
        "median_min": statistics.median(vals),
        "min_min": min(vals),
        "max_min": max(vals),
        "std_min": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "mean_h": statistics.mean(vals) / 60.0,
        "median_h": statistics.median(vals) / 60.0,
        "min_h": min(vals) / 60.0,
        "exact": exact,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------
def _fmt(val: float, digits: int = 1) -> str:
    return f"{val:.{digits}f}"


def _fmt_cell(agg: dict | None) -> str:
    """median (min–max) min, with † for approximate."""
    if agg is None:
        return "—"
    med = agg["median_min"]
    lo = agg["min_min"]
    hi = agg["max_min"]
    exact_flag = "" if agg["exact"] else "†"
    return f"{med:.0f} ({lo:.0f}–{hi:.0f}){exact_flag}"


def _fmt_hours_cell(agg: dict | None) -> str:
    if agg is None:
        return "—"
    h = agg["median_h"]
    exact_flag = "" if agg["exact"] else "†"
    return f"{h:.2f}{exact_flag}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Collect all runs ---
    dinov3_rows = collect_data_efficiency_100train("data_seed101")
    sfm_rows = collect_backbone_comparison("sfm_vit_base_patch16")
    swinv2_rows = collect_backbone_comparison("swin_v2_t")
    fullft_rows = collect_full_ft()

    all_rows = dinov3_rows + sfm_rows + swinv2_rows + fullft_rows

    if not all_rows:
        print("No timing data found — check RUNS path:", RUNS)
        return

    # --- Dump raw CSV ---
    raw_csv = OUT_DIR / "raw_run_timings.csv"
    with raw_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "variant", "n_vols", "total_train_min", "total_epochs",
                    "timing_exact", "seed_tag"])
        for r in all_rows:
            w.writerow([r.label, r.variant, r.n_vols, f"{r.total_train_min:.2f}",
                        r.total_epochs, r.timing_exact, r.seed_tag])
    print(f"Raw CSV: {raw_csv}")

    # --- Per-n_vols DINOv3 PEFT table ---
    peft_n_vols = [20, 35, 50, 75, 100]
    variants = ["2D", "3ch", "5ch"]
    labels_at20 = ["DINOv3 PEFT", "SFM PEFT", "SwinV2 PEFT", "DINOv3 Full-FT"]

    # Build aggregated lookup
    agg_cache: dict[tuple, dict | None] = {}
    for label in labels_at20:
        for v in variants:
            agg_cache[(label, v, 20)] = aggregate(all_rows, label, v, 20)

    for n in peft_n_vols:
        for v in variants:
            agg_cache[("DINOv3 PEFT", v, n)] = aggregate(dinov3_rows, "DINOv3 PEFT", v, n)

    # --- Markdown report ---
    report_lines: list[str] = []
    report_lines.append("# Training Time Feasibility Study\n")
    report_lines.append(
        "Exact timing: `train_step_time_s` from `training_timing.csv` "
        "(excludes validation, checkpointing, queue time).  \n"
        "† = approximate: `epoch_time_s` from `history.csv` (includes validation — upper bound).  \n"
        "All times are for 50 training epochs on one GPU.  \n\n"
        "**Table format: median (min–max) min across seeds.**  \n"
        "High variance is caused by DAIC NFS I/O wait, which is included in `train_step_time_s`. "
        "The minimum is the best-node estimate (pure GPU compute); "
        "the median is the representative operational time.  \n"
        "GPU-hours column uses the median.  \n"
    )

    # Table 1: At n=20 across all model families
    report_lines.append("\n## Table 1: Training time at n=20 training volumes (median (min–max) min)\n")
    header = "| Model | " + " | ".join(variants) + " | n_seeds |"
    sep = "|---|" + "---|" * len(variants) + "---|"
    report_lines.append(header)
    report_lines.append(sep)
    for label in labels_at20:
        cells = [_fmt_cell(agg_cache.get((label, v, 20))) for v in variants]
        first_agg = agg_cache.get((label, variants[0], 20))
        n_seeds_str = str(first_agg["n_seeds"]) if first_agg else "?"
        report_lines.append(f"| {label} | " + " | ".join(cells) + f" | {n_seeds_str} |")

    report_lines.append("\n### GPU-hours at n=20 (median)\n")
    header_h = "| Model | " + " | ".join(variants) + " |"
    report_lines.append(header_h)
    report_lines.append("|---|" + "---|" * len(variants))
    for label in labels_at20:
        cells = [_fmt_hours_cell(agg_cache.get((label, v, 20))) for v in variants]
        report_lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Table 2: DINOv3 PEFT scaling with n_vols
    report_lines.append(
        "\n## Table 2: DINOv3 PEFT training time vs n_vols — data_seed101 "
        "(median (min–max) min, 3 training seeds)\n"
    )
    header2 = "| n_vols | " + " | ".join(variants) + " |"
    sep2 = "|---|" + "---|" * len(variants)
    report_lines.append(header2)
    report_lines.append(sep2)
    for n in peft_n_vols:
        cells = [_fmt_cell(agg_cache.get(("DINOv3 PEFT", v, n))) for v in variants]
        report_lines.append(f"| {n} | " + " | ".join(cells) + " |")

    # Per-seed detail for n=20
    report_lines.append("\n## Per-seed detail at n=20 training volumes\n")
    for label in labels_at20:
        for v in variants:
            seed_rows = [r for r in all_rows
                         if r.label == label and r.variant == v and r.n_vols == 20]
            if not seed_rows:
                continue
            report_lines.append(f"\n**{label} / {v}** ({len(seed_rows)} run(s)):")
            for sr in sorted(seed_rows, key=lambda r: r.seed_tag):
                exact_flag = "" if sr.timing_exact else " (approx)"
                report_lines.append(
                    f"  - `{sr.seed_tag}`: {sr.total_train_min:.1f} min "
                    f"({sr.total_train_min/60:.2f} h, {sr.total_epochs} epochs){exact_flag}"
                )

    # Scaling note
    report_lines.append("\n## Scaling note (medians, DINOv3 PEFT, data_seed101)\n")
    for v in variants:
        agg20 = agg_cache.get(("DINOv3 PEFT", v, 20))
        agg100 = agg_cache.get(("DINOv3 PEFT", v, 100))
        if agg20 and agg100 and agg20["median_min"] > 0:
            ratio = agg100["median_min"] / agg20["median_min"]
            report_lines.append(
                f"- **{v}**: n=20 → {agg20['median_min']:.0f} min, "
                f"n=100 → {agg100['median_min']:.0f} min "
                f"(×{ratio:.1f} scale-up)"
            )

    report_md = OUT_DIR / "training_time_study.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Report:  {report_md}")

    # --- Figures ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        COLORS = {"2D": "#4C72B0", "3ch": "#55A868", "5ch": "#C44E52"}
        HATCHES = {
            "DINOv3 PEFT": "",
            "SFM PEFT": "//",
            "SwinV2 PEFT": "\\\\",
            "DINOv3 Full-FT": "xx",
        }
        VARIANT_MARKERS = {"2D": "o", "3ch": "s", "5ch": "^"}

        # ----------------------------------------------------------------
        # Figure 1: Bar chart — training time at n=20, all model families
        # ----------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        group_labels = labels_at20
        n_groups = len(group_labels)
        n_variants = len(variants)
        bar_width = 0.22
        x = np.arange(n_groups)

        for i, v in enumerate(variants):
            medians = []
            lo_errs = []
            hi_errs = []
            for label in group_labels:
                agg = agg_cache.get((label, v, 20))
                if agg:
                    med = agg["median_min"]
                    medians.append(med)
                    lo_errs.append(med - agg["min_min"])
                    hi_errs.append(agg["max_min"] - med)
                else:
                    medians.append(0)
                    lo_errs.append(0)
                    hi_errs.append(0)
            offset = (i - 1) * bar_width
            bars = ax1.bar(
                x + offset, medians, bar_width,
                label=v, color=COLORS[v],
                yerr=[lo_errs, hi_errs], capsize=4, error_kw={"elinewidth": 1.2},
                edgecolor="black", linewidth=0.6,
            )
            for j, bar in enumerate(bars):
                hatch = list(HATCHES.values())[j]
                bar.set_hatch(hatch)

        ax1.set_xticks(x)
        ax1.set_xticklabels(group_labels, fontsize=11)
        ax1.set_ylabel("Total train-step time (min, 50 epochs)", fontsize=11)
        ax1.set_title("Training time at n=20 vols — all model families\n"
                      "(bars = median; error bars = min–max across seeds)", fontsize=11)
        ax1.legend(title="Variant", fontsize=10)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax1.text(
            0.98, 0.97, "† Full-FT: approx (includes validation)",
            transform=ax1.transAxes, ha="right", va="top",
            fontsize=8, color="gray", style="italic",
        )

        fig1.tight_layout()
        fig1_path = OUT_DIR / "fig1_training_time_at_n20_bar.png"
        fig1.savefig(fig1_path, dpi=150)
        plt.close(fig1)
        print(f"Figure 1: {fig1_path}")

        # ----------------------------------------------------------------
        # Figure 2: Line chart — DINOv3 PEFT scaling with n_vols
        # ----------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        for v in variants:
            xs, ys, lo_errs, hi_errs = [], [], [], []
            for n in peft_n_vols:
                agg = agg_cache.get(("DINOv3 PEFT", v, n))
                if agg and agg["median_min"] > 0:
                    xs.append(n)
                    med = agg["median_min"]
                    ys.append(med)
                    lo_errs.append(med - agg["min_min"])
                    hi_errs.append(agg["max_min"] - med)
            if xs:
                ax2.errorbar(
                    xs, ys, yerr=[lo_errs, hi_errs],
                    label=v, color=COLORS[v],
                    marker=VARIANT_MARKERS[v], markersize=7,
                    linewidth=2, capsize=4,
                )

        ax2.set_xlabel("Training volumes (n_vols)", fontsize=11)
        ax2.set_ylabel("Total train-step time (min, 50 epochs)", fontsize=11)
        ax2.set_title("DINOv3 PEFT training time vs. training volumes (data_seed101)\n"
                      "(median ± min–max across 3 training seeds)", fontsize=11)
        ax2.set_xticks(peft_n_vols)
        ax2.legend(title="Variant", fontsize=10)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig2.tight_layout()
        fig2_path = OUT_DIR / "fig2_dinov3_peft_scaling.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)
        print(f"Figure 2: {fig2_path}")

        # ----------------------------------------------------------------
        # Figure 3: Combined — PEFT variants at n=20 + scaling context
        # ----------------------------------------------------------------
        fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax3a, ax3b = axes

        # Left: model comparison bars (2D only for clarity, then separate grouped)
        # Use grouped bar for all variants
        group_labels_short = ["DINOv3\nPEFT", "SFM\nPEFT", "SwinV2\nPEFT", "Full-FT†"]
        x3 = np.arange(len(group_labels_short))
        for i, v in enumerate(variants):
            medians = []
            lo_e = []
            hi_e = []
            for label in labels_at20:
                agg = agg_cache.get((label, v, 20))
                if agg:
                    med = agg["median_min"]
                    medians.append(med)
                    lo_e.append(med - agg["min_min"])
                    hi_e.append(agg["max_min"] - med)
                else:
                    medians.append(0)
                    lo_e.append(0)
                    hi_e.append(0)
            offset = (i - 1) * bar_width
            ax3a.bar(
                x3 + offset, medians, bar_width,
                label=v, color=COLORS[v],
                yerr=[lo_e, hi_e], capsize=3, error_kw={"elinewidth": 1},
                edgecolor="black", linewidth=0.5,
            )

        ax3a.set_xticks(x3)
        ax3a.set_xticklabels(group_labels_short, fontsize=10)
        ax3a.set_ylabel("Training time (min)", fontsize=10)
        ax3a.set_title("Model comparison at n=20 vols", fontsize=11)
        ax3a.legend(title="Variant", fontsize=9)
        ax3a.spines["top"].set_visible(False)
        ax3a.spines["right"].set_visible(False)

        # Right: DINOv3 PEFT scaling
        for v in variants:
            xs, ys, ys_err = [], [], []
            for n in peft_n_vols:
                agg = agg_cache.get(("DINOv3 PEFT", v, n))
                if agg and agg["mean_min"] > 0:
                    xs.append(n)
                    ys.append(agg["mean_min"])
                    ys_err.append(agg["std_min"])
            if xs:
                ax3b.errorbar(
                    xs, ys, yerr=ys_err,
                    label=v, color=COLORS[v],
                    marker=VARIANT_MARKERS[v], markersize=7,
                    linewidth=2, capsize=3,
                )

        ax3b.set_xlabel("Training volumes", fontsize=10)
        ax3b.set_ylabel("Training time (min)", fontsize=10)
        ax3b.set_title("DINOv3 PEFT: time vs. training volumes", fontsize=11)
        ax3b.set_xticks(peft_n_vols)
        ax3b.legend(title="Variant", fontsize=9)
        ax3b.spines["top"].set_visible(False)
        ax3b.spines["right"].set_visible(False)

        fig3.tight_layout()
        fig3_path = OUT_DIR / "fig3_combined_training_time.png"
        fig3.savefig(fig3_path, dpi=150)
        plt.close(fig3)
        print(f"Figure 3: {fig3_path}")

        # ----------------------------------------------------------------
        # Figure 4: GPU-hours bar at n=20 (absolute cost view)
        # ----------------------------------------------------------------
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        for i, v in enumerate(variants):
            medians_h = []
            lo_e_h = []
            hi_e_h = []
            for label in labels_at20:
                agg = agg_cache.get((label, v, 20))
                if agg:
                    med_h = agg["median_h"]
                    medians_h.append(med_h)
                    lo_e_h.append(med_h - agg["min_h"])
                    hi_e_h.append(agg["max_min"] / 60.0 - med_h)
                else:
                    medians_h.append(0)
                    lo_e_h.append(0)
                    hi_e_h.append(0)
            offset = (i - 1) * bar_width
            ax4.bar(
                x + offset, medians_h, bar_width,
                label=v, color=COLORS[v],
                yerr=[lo_e_h, hi_e_h], capsize=4, error_kw={"elinewidth": 1.2},
                edgecolor="black", linewidth=0.6,
            )

        ax4.set_xticks(x)
        ax4.set_xticklabels(group_labels, fontsize=11)
        ax4.set_ylabel("GPU-hours (50 epochs, 1 GPU)", fontsize=11)
        ax4.set_title("GPU-hours at n=20 training volumes\n"
                      "(bars = median; error bars = min–max)", fontsize=11)
        ax4.legend(title="Variant", fontsize=10)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

        fig4.tight_layout()
        fig4_path = OUT_DIR / "fig4_gpu_hours_bar.png"
        fig4.savefig(fig4_path, dpi=150)
        plt.close(fig4)
        print(f"Figure 4: {fig4_path}")

    except ImportError as e:
        print(f"matplotlib/numpy not available, skipping figures: {e}")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
