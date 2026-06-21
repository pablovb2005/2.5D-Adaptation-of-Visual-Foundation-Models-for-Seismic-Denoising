"""Summarize trained mechanism controls for the 3x3 multidata protocol.

Reads ``eval_results/results.csv`` from each controls_multidata run and from the
matched main_multidata aligned baselines. The script writes aggregate tables,
per-run tables, plots, a representative example grid assembled from existing
``test_example.png`` files, and a Markdown report.

Protocol:
  data seeds:     101, 202, 303
  training seeds: 42, 43, 44
  variants:       2D, 3ch, 5ch, 3ch_shuffled,
                  5ch_repeated_center, 5ch_shuffled

Usage:
    python -m evaluation.summarizers.summarize_mechanism_controls_multidata --project-root .
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

SRC = ensure_src_on_path(__file__)
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


DATA_SEEDS = [101, 202, 303]
TRAINING_SEEDS = [42, 43, 44]
SEED_TO_RUN = {42: "seed42_run01", 43: "seed43_run02", 44: "seed44_run03"}
METRICS = ["ms_ssim", "mse", "psnr"]

VARIANTS: dict[str, dict[str, str]] = {
    "2D": {
        "label": "2D [t,t,t], aligned training",
        "role": "baseline",
        "subdir": "main_multidata/2d",
        "variant_dir": "impeccable_repeated_stride5_lora_r16",
    },
    "3ch": {
        "label": "3ch aligned neighbors",
        "role": "baseline",
        "subdir": "main_multidata/3ch",
        "variant_dir": "impeccable_neighbors3_stride5_lora_r16",
    },
    "3ch_shuffled": {
        "label": "3ch shuffled-neighbor training",
        "role": "control",
        "subdir": "controls_multidata/3ch_shuffled",
        "variant_dir": "impeccable_shuffled3_stride5_lora_r16",
    },
    "5ch": {
        "label": "5ch aligned neighbors",
        "role": "baseline",
        "subdir": "main_multidata/5ch",
        "variant_dir": "impeccable_neighbors5_stride5_patch_emb_lora_r16",
    },
    "5ch_repeated_center": {
        "label": "5ch repeated-center capacity control",
        "role": "control",
        "subdir": "controls_multidata/5ch_repeated_center",
        "variant_dir": "impeccable_repeated_center_stride5_patch_emb_lora_r16",
    },
    "5ch_shuffled": {
        "label": "5ch shuffled-neighbor training",
        "role": "control",
        "subdir": "controls_multidata/5ch_shuffled",
        "variant_dir": "impeccable_shuffled5_stride5_patch_emb_lora_r16",
    },
}

VARIANT_ORDER = [
    "2D",
    "3ch",
    "3ch_shuffled",
    "5ch",
    "5ch_repeated_center",
    "5ch_shuffled",
]

PLOT_LABELS = {
    "2D": "2D\naligned",
    "3ch": "3ch\naligned",
    "3ch_shuffled": "3ch\nshuffled train",
    "5ch": "5ch\naligned",
    "5ch_repeated_center": "5ch\nrepeated center",
    "5ch_shuffled": "5ch\nshuffled train",
}

ROLE_COLORS = {
    "baseline": "#2F6FB0",
    "control": "#D9782D",
}

DATA_SEED_COLORS = {
    101: "#2F6FB0",
    202: "#2E8B57",
    303: "#B23A48",
}

TRAINING_SEED_MARKERS = {
    42: "o",
    43: "s",
    44: "^",
}


def _safe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None, None
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
    return float(np.mean(finite)), std


def _read_mean_from_csv(csv_path: Path) -> dict[str, float] | None:
    if not csv_path.exists():
        return None
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") == "test":
                rows.append(row)
    if not rows:
        return None

    result: dict[str, float] = {}
    for metric in METRICS:
        vals = [_safe_float(row.get(metric)) for row in rows]
        vals = [v for v in vals if v is not None]
        result[metric] = float(np.mean(vals)) if vals else float("nan")
    return result


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _collect_run_rows(project_root: Path) -> list[dict[str, Any]]:
    runs_root = project_root / "experiments" / "runs"
    rows: list[dict[str, Any]] = []

    for variant_key in VARIANT_ORDER:
        cfg = VARIANTS[variant_key]
        group_root = runs_root / cfg["subdir"] / cfg["variant_dir"]
        for data_seed in DATA_SEEDS:
            for training_seed in TRAINING_SEEDS:
                run_id = SEED_TO_RUN[training_seed]
                run_dir = group_root / f"data_seed{data_seed}" / run_id
                results_csv = run_dir / "eval_results" / "results.csv"
                example_path = run_dir / "eval_results" / "test_example.png"
                metrics = _read_mean_from_csv(results_csv)

                row: dict[str, Any] = {
                    "variant_key": variant_key,
                    "label": cfg["label"],
                    "role": cfg["role"],
                    "data_seed": data_seed,
                    "training_seed": training_seed,
                    "run_key": f"ds{data_seed}_ts{training_seed}",
                    "run_dir": _rel(run_dir, project_root),
                    "results_csv": _rel(results_csv, project_root) if results_csv.exists() else "",
                    "example_path": _rel(example_path, project_root) if example_path.exists() else "",
                    "status": "complete" if metrics is not None else "missing_eval",
                }
                for metric in METRICS:
                    row[metric] = metrics.get(metric, float("nan")) if metrics else float("nan")
                rows.append(row)

    return rows


def _aggregate_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)

    out: list[dict[str, Any]] = []
    for group_key, group_rows in groups.items():
        template = group_rows[0]
        complete = [row for row in group_rows if row["status"] == "complete"]
        agg: dict[str, Any] = {
            key: group_key[i] for i, key in enumerate(keys)
        }
        agg.update(
            {
                "label": template["label"],
                "role": template["role"],
                "n_complete": len(complete),
                "n_expected": len(group_rows),
                "missing": ";".join(row["run_key"] for row in group_rows if row["status"] != "complete"),
            }
        )
        for metric in METRICS:
            vals = [_safe_float(row.get(metric)) for row in complete]
            vals = [v for v in vals if v is not None]
            mean, std = _mean_std(vals)
            agg[f"{metric}_mean"] = mean if mean is not None else float("nan")
            agg[f"{metric}_std"] = std if std is not None else float("nan")
        out.append(agg)

    return sorted(out, key=_sort_agg_row)


def _sort_agg_row(row: dict[str, Any]) -> tuple[int, int, int]:
    variant = str(row.get("variant_key"))
    data_seed = int(row.get("data_seed", 0) or 0)
    return (
        VARIANT_ORDER.index(variant) if variant in VARIANT_ORDER else 999,
        DATA_SEEDS.index(data_seed) if data_seed in DATA_SEEDS else 999,
        data_seed,
    )


def _sort_run_row(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        VARIANT_ORDER.index(str(row["variant_key"])),
        DATA_SEEDS.index(int(row["data_seed"])),
        TRAINING_SEEDS.index(int(row["training_seed"])),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path} ({len(rows)} rows)")


def _load_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_aggregate(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _load_pyplot()
    by_variant = {row["variant_key"]: row for row in summary_rows}
    means = [_safe_float(by_variant.get(key, {}).get("ms_ssim_mean")) for key in VARIANT_ORDER]
    stds = [_safe_float(by_variant.get(key, {}).get("ms_ssim_std")) or 0.0 for key in VARIANT_ORDER]
    colors = [ROLE_COLORS[VARIANTS[key]["role"]] for key in VARIANT_ORDER]

    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    y = np.arange(len(VARIANT_ORDER))
    bars = ax.barh(
        y,
        [mean if mean is not None else 0.0 for mean in means],
        xerr=stds,
        capsize=6,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        error_kw={"elinewidth": 2.0, "ecolor": "#333333", "capthick": 2.0},
    )

    two_d = _safe_float(by_variant.get("2D", {}).get("ms_ssim_mean"))
    if two_d is not None:
        ax.axvline(two_d, color="#444444", linestyle="--", linewidth=1.8, label=f"2D baseline ({two_d:.4f})")

    for bar, mean, std in zip(bars, means, stds):
        if mean is None:
            continue
        ax.text(
            mean + std + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.4f}",
            ha="left",
            va="center",
            fontsize=15,
            fontweight="bold",
        )

    y_values = [m for m in means if m is not None]
    x_min = min(y_values) - 0.035 if y_values else 0.7
    x_max = max((mean or 0.0) + std for mean, std in zip(means, stds)) + 0.05 if y_values else 0.95
    ax.set_xlim(max(0.0, x_min), min(1.0, x_max))
    ax.set_yticks(y)
    ax.set_yticklabels([PLOT_LABELS[key].replace("\n", " ") for key in VARIANT_ORDER], fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", labelsize=15)
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    ax.set_xlabel("Test MS-SSIM (mean +/- std over 9 runs)", fontsize=17, fontweight="bold")
    ax.set_title("Trained Mechanism Controls, Multidata Protocol", fontsize=21, fontweight="bold", pad=14)
    ax.grid(True, axis="x", alpha=0.25, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    from matplotlib import patches as mpatches

    handles = [
        mpatches.Patch(color=ROLE_COLORS["baseline"], label="Aligned-training baseline"),
        mpatches.Patch(color=ROLE_COLORS["control"], label="Trained control"),
    ]
    if two_d is not None:
        from matplotlib import lines as mlines

        handles.append(mlines.Line2D([0], [0], color="#444444", linestyle="--", label="2D baseline"))
    ax.legend(
        handles=handles,
        prop={"weight": "bold", "size": 13},
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=1,
        frameon=True,
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_seed_consistency(run_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _load_pyplot()
    fig, ax = plt.subplots(figsize=(9.5, 5))

    ds_offsets = {101: -0.17, 202: 0.0, 303: 0.17}
    ts_offsets = {42: -0.035, 43: 0.0, 44: 0.035}

    for vi, variant in enumerate(VARIANT_ORDER):
        for row in run_rows:
            if row["variant_key"] != variant or row["status"] != "complete":
                continue
            data_seed = int(row["data_seed"])
            training_seed = int(row["training_seed"])
            value = _safe_float(row["ms_ssim"])
            if value is None:
                continue
            ax.scatter(
                vi + ds_offsets[data_seed] + ts_offsets[training_seed],
                value,
                color=DATA_SEED_COLORS[data_seed],
                marker=TRAINING_SEED_MARKERS[training_seed],
                s=60,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    ax.set_xticks(range(len(VARIANT_ORDER)))
    ax.set_xticklabels([PLOT_LABELS[key] for key in VARIANT_ORDER], fontsize=9)
    ax.set_ylabel("Test MS-SSIM per run")
    ax.set_title("Run Consistency Across Data Seeds and Training Seeds")
    ax.grid(True, axis="y", alpha=0.25)

    from matplotlib import lines as mlines

    seed_handles = [
        mlines.Line2D(
            [0],
            [0],
            color=DATA_SEED_COLORS[seed],
            marker="o",
            linestyle="",
            markersize=7,
            label=f"data_seed={seed}",
        )
        for seed in DATA_SEEDS
    ]
    training_handles = [
        mlines.Line2D(
            [0],
            [0],
            color="#555555",
            marker=TRAINING_SEED_MARKERS[seed],
            linestyle="",
            markersize=7,
            label=f"training_seed={seed}",
        )
        for seed in TRAINING_SEEDS
    ]
    legend1 = ax.legend(handles=seed_handles, fontsize=8.5, loc="lower right")
    ax.add_artist(legend1)
    ax.legend(handles=training_handles, fontsize=8.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_by_data_seed(data_seed_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _load_pyplot()
    by_pair = {(row["variant_key"], int(row["data_seed"])): row for row in data_seed_rows}

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(DATA_SEEDS))
    width = 0.13
    offsets = np.linspace(-0.33, 0.33, len(VARIANT_ORDER))

    for offset, variant in zip(offsets, VARIANT_ORDER):
        means = []
        stds = []
        for data_seed in DATA_SEEDS:
            row = by_pair.get((variant, data_seed), {})
            means.append(_safe_float(row.get("ms_ssim_mean")) or 0.0)
            stds.append(_safe_float(row.get("ms_ssim_std")) or 0.0)
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=variant,
            color=ROLE_COLORS[VARIANTS[variant]["role"]],
            alpha=0.55 if VARIANTS[variant]["role"] == "control" else 0.88,
            edgecolor="white",
            linewidth=0.6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"data_seed={seed}" for seed in DATA_SEEDS])
    ax.set_ylabel("Test MS-SSIM (mean +/- std across training seeds)")
    ax.set_title("Trained Controls by Data Split")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _plot_mse_psnr(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    plt = _load_pyplot()
    by_variant = {row["variant_key"]: row for row in summary_rows}
    colors = [ROLE_COLORS[VARIANTS[key]["role"]] for key in VARIANT_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in [
        (axes[0], "mse", "Test MSE (lower is better)"),
        (axes[1], "psnr", "Test PSNR"),
    ]:
        means = [_safe_float(by_variant.get(key, {}).get(f"{metric}_mean")) for key in VARIANT_ORDER]
        stds = [_safe_float(by_variant.get(key, {}).get(f"{metric}_std")) or 0.0 for key in VARIANT_ORDER]
        x = np.arange(len(VARIANT_ORDER))
        ax.bar(
            x,
            [mean if mean is not None else 0.0 for mean in means],
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([PLOT_LABELS[key] for key in VARIANT_ORDER], fontsize=8)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _make_example_grid(project_root: Path, run_rows: list[dict[str, Any]], out_path: Path) -> bool:
    plt = _load_pyplot()
    chosen: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        rows = [
            row
            for row in run_rows
            if row["variant_key"] == variant and row["example_path"] and row["status"] == "complete"
        ]
        rows = sorted(rows, key=lambda row: (int(row["data_seed"]) != 101, int(row["training_seed"]) != 42))
        if rows:
            chosen.append(rows[0])

    if not chosen:
        return False

    ncols = 2
    nrows = math.ceil(len(chosen) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows))
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, row in zip(axes_arr, chosen):
        image_path = project_root / row["example_path"]
        try:
            image = plt.imread(image_path)
        except OSError:
            ax.text(0.5, 0.5, "example unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.imshow(image)
        ax.set_title(
            f"{row['variant_key']} | data_seed={row['data_seed']}, training_seed={row['training_seed']}",
            fontsize=10,
        )
        ax.set_axis_off()

    for ax in axes_arr[len(chosen):]:
        ax.set_axis_off()

    fig.suptitle("Representative Test Examples from Generated Evaluation Artifacts", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
    return True


def _fmt_mean_std(row: dict[str, Any], metric: str, precision: int = 4) -> str:
    mean = _safe_float(row.get(f"{metric}_mean"))
    std = _safe_float(row.get(f"{metric}_std"))
    if mean is None:
        return "pending"
    if std is None:
        std = 0.0
    if metric == "psnr":
        return f"{mean:.2f} +/- {std:.2f}"
    return f"{mean:.{precision}f} +/- {std:.{precision}f}"


def _fmt_metric(value: object, metric: str) -> str:
    val = _safe_float(value)
    if val is None:
        return "pending"
    return f"{val:.2f}" if metric == "psnr" else f"{val:.4f}"


def _delta(summary_by_variant: dict[str, dict[str, Any]], left: str, right: str, metric: str = "ms_ssim") -> float | None:
    left_value = _safe_float(summary_by_variant.get(left, {}).get(f"{metric}_mean"))
    right_value = _safe_float(summary_by_variant.get(right, {}).get(f"{metric}_mean"))
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _write_report(
    out_path: Path,
    summary_rows: list[dict[str, Any]],
    data_seed_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    generated_files: list[Path],
) -> None:
    summary_by_variant = {row["variant_key"]: row for row in summary_rows}
    two_d = summary_by_variant.get("2D", {})

    lines = [
        "# Trained Mechanism Controls - Multidata Summary",
        "",
        f"Generated: {date.today().isoformat()}",
        "",
        "## Protocol and Scope",
        "",
        "This report summarizes trained mechanism controls under the 3 data seed x 3 training seed protocol.",
        "It compares aligned-trained main baselines against controls trained with shuffled neighbors or repeated-center 5-channel input.",
        "",
        "- Data seeds: `101`, `202`, `303`.",
        "- Training seeds: `42`, `43`, `44`.",
        "- Metrics are means of `eval_results/results.csv` rows with `split == test` for each run.",
        "- This is an Image Impeccable paired evaluation; F3 robustness is not included here.",
        "",
        "## Aggregate Results",
        "",
        "| Variant | Role | n | MS-SSIM mean +/- std | MSE mean +/- std | PSNR mean +/- std | Delta vs 2D | Missing |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    two_d_ms = _safe_float(two_d.get("ms_ssim_mean"))
    for row in summary_rows:
        ms = _safe_float(row.get("ms_ssim_mean"))
        delta_vs_2d = "pending" if ms is None or two_d_ms is None else f"{ms - two_d_ms:+.4f}"
        missing = row.get("missing") or "-"
        lines.append(
            "| {variant} | {role} | {n}/{expected} | {ms} | {mse} | {psnr} | {delta} | {missing} |".format(
                variant=row["variant_key"],
                role=row["role"],
                n=row["n_complete"],
                expected=row["n_expected"],
                ms=_fmt_mean_std(row, "ms_ssim"),
                mse=_fmt_mean_std(row, "mse"),
                psnr=_fmt_mean_std(row, "psnr"),
                delta=delta_vs_2d,
                missing=missing,
            )
        )

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Aggregate MS-SSIM bars](trained_controls_multidata_bar.png)",
            "",
            "Aggregate test MS-SSIM with +/- 1 std across the nine runs for each variant.",
            "",
            "![Seed consistency](trained_controls_multidata_seed_consistency.png)",
            "",
            "Per-run test MS-SSIM. Color encodes data seed and marker encodes training seed.",
            "",
            "![By data seed](trained_controls_multidata_by_data_seed.png)",
            "",
            "Per-data-seed mean +/- std across the three training seeds.",
            "",
            "![MSE and PSNR](trained_controls_multidata_mse_psnr.png)",
            "",
            "Secondary paired metrics for the same runs.",
            "",
            "![Representative examples](trained_controls_multidata_example_grid.png)",
            "",
            "Representative generated `test_example.png` panels assembled from existing evaluation outputs.",
            "",
            "## Per-Data-Seed Summary",
            "",
            "| Variant | Data seed | n | MS-SSIM mean +/- std | MSE mean +/- std | PSNR mean +/- std | Missing |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )

    for row in data_seed_rows:
        missing = row.get("missing") or "-"
        lines.append(
            "| {variant} | {data_seed} | {n}/{expected} | {ms} | {mse} | {psnr} | {missing} |".format(
                variant=row["variant_key"],
                data_seed=row["data_seed"],
                n=row["n_complete"],
                expected=row["n_expected"],
                ms=_fmt_mean_std(row, "ms_ssim"),
                mse=_fmt_mean_std(row, "mse"),
                psnr=_fmt_mean_std(row, "psnr"),
                missing=missing,
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Table",
            "",
            "| Variant | Data seed | Training seed | Status | MS-SSIM | MSE | PSNR | Results CSV |",
            "|---|---:|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in sorted(run_rows, key=_sort_run_row):
        results = row["results_csv"] or "-"
        lines.append(
            "| {variant} | {data_seed} | {training_seed} | {status} | {ms} | {mse} | {psnr} | `{results}` |".format(
                variant=row["variant_key"],
                data_seed=row["data_seed"],
                training_seed=row["training_seed"],
                status=row["status"],
                ms=_fmt_metric(row.get("ms_ssim"), "ms_ssim"),
                mse=_fmt_metric(row.get("mse"), "mse"),
                psnr=_fmt_metric(row.get("psnr"), "psnr"),
                results=results,
            )
        )

    d_3ch = _delta(summary_by_variant, "3ch", "2D")
    d_5ch = _delta(summary_by_variant, "5ch", "2D")
    d_3ch_shuf = _delta(summary_by_variant, "3ch_shuffled", "2D")
    d_5ch_rep = _delta(summary_by_variant, "5ch_repeated_center", "2D")
    d_5ch_shuf = _delta(summary_by_variant, "5ch_shuffled", "2D")
    d_content = _delta(summary_by_variant, "5ch", "5ch_repeated_center")

    def fmt_delta(value: float | None) -> str:
        return "pending" if value is None else f"{value:+.4f}"

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- 3ch aligned vs 2D: `{fmt_delta(d_3ch)}` MS-SSIM.",
            f"- 5ch aligned vs 2D: `{fmt_delta(d_5ch)}` MS-SSIM.",
            f"- 3ch shuffled-neighbor training vs 2D: `{fmt_delta(d_3ch_shuf)}` MS-SSIM.",
            f"- 5ch repeated-center capacity control vs 2D: `{fmt_delta(d_5ch_rep)}` MS-SSIM.",
            f"- 5ch shuffled-neighbor training vs 2D: `{fmt_delta(d_5ch_shuf)}` MS-SSIM.",
            f"- 5ch aligned neighbor-content contribution over repeated-center 5ch: `{fmt_delta(d_content)}` MS-SSIM.",
            "",
            "Reading rule: shuffled-neighbor trained controls test whether correctly aligned neighboring slices are needed during training, while the 5ch repeated-center control separates extra patch-embedding capacity from neighboring-slice content.",
            "",
            "## Rerun Command",
            "",
            "```powershell",
            "cd C:\\UNI\\Y3\\RP\\Code\\DINOv3\\src",
            "python -m evaluation.summarizers.summarize_mechanism_controls_multidata --project-root C:\\UNI\\Y3\\RP",
            "```",
            "",
            "## Generated Files",
            "",
        ]
    )
    for path in generated_files:
        lines.append(f"- `{path.name}`")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true", help="Write CSVs/report without PNG plots.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (
        project_root / "experiments" / "summaries" / "mechanism_analysis" / "trained_controls_multidata"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = sorted(_collect_run_rows(project_root), key=_sort_run_row)
    summary_rows = _aggregate_rows(run_rows, ["variant_key"])
    data_seed_rows = _aggregate_rows(run_rows, ["variant_key", "data_seed"])

    by_run_path = out_dir / "trained_controls_multidata_by_run.csv"
    by_data_seed_path = out_dir / "trained_controls_multidata_by_data_seed.csv"
    summary_path = out_dir / "trained_controls_multidata_summary.csv"
    report_path = out_dir / "trained_controls_multidata_report.md"

    _write_csv(
        by_run_path,
        run_rows,
        [
            "variant_key",
            "label",
            "role",
            "data_seed",
            "training_seed",
            "run_key",
            "status",
            "ms_ssim",
            "mse",
            "psnr",
            "run_dir",
            "results_csv",
            "example_path",
        ],
    )
    _write_csv(
        by_data_seed_path,
        data_seed_rows,
        [
            "variant_key",
            "data_seed",
            "label",
            "role",
            "n_complete",
            "n_expected",
            "ms_ssim_mean",
            "ms_ssim_std",
            "mse_mean",
            "mse_std",
            "psnr_mean",
            "psnr_std",
            "missing",
        ],
    )
    _write_csv(
        summary_path,
        summary_rows,
        [
            "variant_key",
            "label",
            "role",
            "n_complete",
            "n_expected",
            "ms_ssim_mean",
            "ms_ssim_std",
            "mse_mean",
            "mse_std",
            "psnr_mean",
            "psnr_std",
            "missing",
        ],
    )

    generated_files = [by_run_path, by_data_seed_path, summary_path]

    if not args.no_plots:
        plot_paths = [
            out_dir / "trained_controls_multidata_bar.png",
            out_dir / "trained_controls_multidata_seed_consistency.png",
            out_dir / "trained_controls_multidata_by_data_seed.png",
            out_dir / "trained_controls_multidata_mse_psnr.png",
            out_dir / "trained_controls_multidata_example_grid.png",
        ]
        _plot_aggregate(summary_rows, plot_paths[0])
        _plot_seed_consistency(run_rows, plot_paths[1])
        _plot_by_data_seed(data_seed_rows, plot_paths[2])
        _plot_mse_psnr(summary_rows, plot_paths[3])
        if not _make_example_grid(project_root, run_rows, plot_paths[4]):
            print("No example images found; skipped trained_controls_multidata_example_grid.png")
            plot_paths = plot_paths[:-1]
        generated_files.extend(plot_paths)

    _write_report(report_path, summary_rows, data_seed_rows, run_rows, generated_files + [report_path])

    incomplete = [row for row in run_rows if row["status"] != "complete"]
    print(f"\nSummarized {len(run_rows) - len(incomplete)}/{len(run_rows)} expected runs.")
    if incomplete:
        print("Missing evaluations:")
        for row in incomplete:
            print(f"  - {row['variant_key']} {row['run_key']}")
    print(f"Summary saved to: {out_dir}")


if __name__ == "__main__":
    main()
