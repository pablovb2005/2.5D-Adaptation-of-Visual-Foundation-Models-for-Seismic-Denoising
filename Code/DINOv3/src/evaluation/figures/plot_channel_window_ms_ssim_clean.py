"""Generate channel_window_test_ms_ssim_clean.png — same as the main curve plot
but without error bars, for cleaner thesis figures.

Usage:
    python Code/DINOv3/src/evaluation/figures/plot_channel_window_ms_ssim_clean.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

STUDY = "data_efficiency_100train_channel_window_v2"
VARIANT_ORDER = ("2D", "3ch", "5ch", "7ch", "9ch")
N_ORDER = (5, 10, 15, 20, 35, 50, 75, 100)
VARIANT_COLORS = {
    "2D":  "#4C72B0",
    "3ch": "#DD8452",
    "5ch": "#55A868",
    "7ch": "#C44E52",
    "9ch": "#8172B2",
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

    # Load agg table
    agg: dict[tuple[str, int], float] = {}
    with agg_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant = row["variant_key"]
            n = int(row["n_vols"])
            val = _safe_float(row.get("test_ms_ssim_mean", ""))
            if val is not None:
                agg[(variant, n)] = val

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for variant in VARIANT_ORDER:
        xs: list[int] = []
        ys: list[float] = []
        for n in N_ORDER:
            val = agg.get((variant, n))
            if val is not None:
                xs.append(n)
                ys.append(val)
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", linewidth=2, label=variant,
                color=VARIANT_COLORS.get(variant))

    ax.set_title("Test MS-SSIM by training-volume budget (mean across 3×3 seeds)")
    ax.set_xlabel("Training volumes")
    ax.set_ylabel("Test MS-SSIM")
    ax.set_xticks(list(N_ORDER))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = out_dir / "channel_window_test_ms_ssim_clean.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
