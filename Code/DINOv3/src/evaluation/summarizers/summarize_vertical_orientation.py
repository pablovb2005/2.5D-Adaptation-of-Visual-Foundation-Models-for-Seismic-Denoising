"""Aggregate the vertical-slice training-orientation control runs.

Each run under ``experiments/runs/vertical_orientation/<family>/<run_base>/
data_seed<NNN>/<run_id>/eval_results/results.csv`` holds per-test-split metric
rows written by ``evaluation/evaluators/evaluate.py`` (columns ``split``,
``ms_ssim``, ``ms_ssim_r``, ``mse``, ``psnr``). This script mirrors the main
summariser's metric extraction (average over test rows per run) and then reports
mean +/- std across the replicate runs for each variant (2D / 3ch / 5ch),
matching the horizontal main-results protocol so the two tables are comparable.

It can run locally against a pulled copy or directly on DAIC against the
staff-bulk experiments root via ``--runs-root``.

    python Code/DINOv3/src/evaluation/summarizers/summarize_vertical_orientation.py \
        --runs-root experiments/runs/vertical_orientation
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

METRICS = ("ms_ssim", "ms_ssim_r", "mse", "psnr")
VARIANT_ORDER = {"2D": 0, "3ch": 1, "5ch": 2}


def _variant_from_path(rel: str) -> str | None:
    low = rel.lower().replace("\\", "/")
    if "/2d/" in f"/{low}/" or low.startswith("2d/"):
        return "2D"
    for ch in ("3ch", "5ch"):
        if f"/{ch}/" in f"/{low}/" or low.startswith(f"{ch}/"):
            return ch
    return None


def _seeds_from_path(rel: str) -> tuple[int | None, int | None]:
    ds = re.search(r"data_seed(\d+)", rel)
    ts = re.search(r"seed(\d+)", rel.split("data_seed")[-1]) if "data_seed" in rel else re.search(r"seed(\d+)", rel)
    return (int(ds.group(1)) if ds else None, int(ts.group(1)) if ts else None)


def _run_metrics(results_csv: Path) -> dict[str, float] | None:
    """Average each metric over the test-split rows of one run."""
    acc: dict[str, list[float]] = {m: [] for m in METRICS}
    with results_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            for m in METRICS:
                raw = row.get(m)
                if raw in (None, ""):
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(v):
                    acc[m].append(v)
    if not any(acc[m] for m in METRICS):
        return None
    return {m: (sum(acc[m]) / len(acc[m]) if acc[m] else float("nan")) for m in METRICS}


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(var)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=Path("experiments/runs/vertical_orientation"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("experiments/summaries/vertical_orientation"))
    args = ap.parse_args()

    runs_root = args.runs_root
    if not runs_root.exists():
        raise FileNotFoundError(f"runs-root not found: {runs_root.resolve()}")

    per_run: list[dict[str, object]] = []
    for results_csv in sorted(runs_root.rglob("eval_results/results.csv")):
        run_dir = results_csv.parent.parent
        rel = run_dir.relative_to(runs_root).as_posix()
        variant = _variant_from_path(rel)
        if variant is None:
            continue
        metrics = _run_metrics(results_csv)
        if metrics is None:
            continue
        data_seed, train_seed = _seeds_from_path(rel)
        per_run.append({"variant": variant, "data_seed": data_seed,
                        "training_seed": train_seed, "run": rel, **metrics})

    if not per_run:
        print(f"No evaluated runs found under {runs_root.resolve()} yet.")
        return

    # Per-variant aggregation across replicate runs.
    agg: list[dict[str, object]] = []
    by_variant: dict[str, list[dict[str, object]]] = {}
    for r in per_run:
        by_variant.setdefault(str(r["variant"]), []).append(r)
    for variant in sorted(by_variant, key=lambda v: VARIANT_ORDER.get(v, 9)):
        rows = by_variant[variant]
        out: dict[str, object] = {"variant": variant, "n_runs": len(rows)}
        for m in METRICS:
            mean, std = _mean_std([float(x[m]) for x in rows])  # type: ignore[arg-type]
            out[f"{m}_mean"] = mean
            out[f"{m}_std"] = std
        agg.append(out)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "vertical_per_run.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "data_seed", "training_seed", "run", *METRICS])
        w.writeheader()
        w.writerows(per_run)

    with (out_dir / "vertical_replicate_summary.csv").open("w", newline="") as f:
        fields = ["variant", "n_runs"] + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(agg)

    def fmt(x: object, d: int = 4) -> str:
        return f"{float(x):.{d}f}" if isinstance(x, (int, float)) and x is not None else "NA"

    # Console + LaTeX-ready table (MS-SSIM and MS-SSIM-R mean +/- std).
    print(f"\nVertical-orientation Image Impeccable results  (runs found: {len(per_run)})\n")
    print(f"{'Variant':8} {'n':>3}  {'MS-SSIM':>16}  {'MS-SSIM-R':>16}  {'PSNR':>14}")
    latex = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Variant & $n$ & MS-SSIM & MS-SSIM-R & PSNR \\", r"\midrule"]
    for o in agg:
        v = str(o["variant"])
        ms = f"{fmt(o['ms_ssim_mean'])} +/- {fmt(o['ms_ssim_std'])}"
        msr = f"{fmt(o['ms_ssim_r_mean'])} +/- {fmt(o['ms_ssim_r_std'])}"
        ps = f"{fmt(o['psnr_mean'], 2)} +/- {fmt(o['psnr_std'], 2)}"
        print(f"{v:8} {int(o['n_runs']):>3}  {ms:>16}  {msr:>16}  {ps:>14}")
        latex.append(
            f"{v} & {int(o['n_runs'])} & "
            f"${fmt(o['ms_ssim_mean'])}\\pm{fmt(o['ms_ssim_std'])}$ & "
            f"${fmt(o['ms_ssim_r_mean'])}\\pm{fmt(o['ms_ssim_r_std'])}$ & "
            f"${fmt(o['psnr_mean'], 2)}\\pm{fmt(o['psnr_std'], 2)}$ \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}"]
    (out_dir / "vertical_table.tex").write_text("\n".join(latex) + "\n")

    print(f"\nWrote: {out_dir / 'vertical_per_run.csv'}")
    print(f"Wrote: {out_dir / 'vertical_replicate_summary.csv'}")
    print(f"Wrote: {out_dir / 'vertical_table.tex'}")


if __name__ == "__main__":
    main()
