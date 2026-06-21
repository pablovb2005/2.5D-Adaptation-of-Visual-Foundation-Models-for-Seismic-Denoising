"""Generate data_efficiency_main.png — compact main-results data-efficiency curve.

Shows only the three main input variants (2D-1ch, 2.5D-3ch, 2.5D-5ch) for
training-volume budgets up to 20, with mean +/- std error bars over the full
3 data seed by 3 training seed protocol. Wider 7ch/9ch windows are deliberately
excluded; those belong in the appendix context-window figure.

Usage:
    python Code/DINOv3/src/evaluation/figures/plot_data_efficiency_main.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

STUDY = "data_efficiency_100train_channel_window_v2"
VARIANT_ORDER = ("2D", "3ch", "5ch")
VARIANT_LABELS = {
    "2D":  "2D-1ch",
    "3ch": "2.5D-3ch",
    "5ch": "2.5D-5ch",
}
N_ORDER = (5, 10, 15, 20)
VARIANT_COLORS = {
    "2D":  "#4C72B0",
    "3ch": "#DD8452",
    "5ch": "#55A868",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _safe_float(v: str) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    root = _project_root()
    agg_csv = root / "experiments" / "summaries" / STUDY / "channel_window_agg.csv"
    out_dir = root / "experiments" / "summaries" / STUDY

    mean: dict[tuple[str, int], float] = {}
    std: dict[tuple[str, int], float] = {}
    with agg_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant = row["variant_key"]
            n = int(row["n_vols"])
            m = _safe_float(row.get("test_ms_ssim_mean", ""))
            s = _safe_float(row.get("test_ms_ssim_std", ""))
            if m is not None:
                mean[(variant, n)] = m
            if s is not None:
                std[(variant, n)] = s

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for variant in VARIANT_ORDER:
        xs: list[int] = []
        ys: list[float] = []
        es: list[float] = []
        for n in N_ORDER:
            m = mean.get((variant, n))
            if m is not None:
                xs.append(n)
                ys.append(m)
                es.append(std.get((variant, n), 0.0))
        if not xs:
            continue
        ax.errorbar(
            xs, ys, yerr=es, marker="o", linewidth=2, capsize=3,
            label=VARIANT_LABELS[variant], color=VARIANT_COLORS.get(variant),
        )

    ax.set_xlabel("Training volumes")
    ax.set_ylabel("Test MS-SSIM")
    ax.set_xticks(list(N_ORDER))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = out_dir / "data_efficiency_main.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
