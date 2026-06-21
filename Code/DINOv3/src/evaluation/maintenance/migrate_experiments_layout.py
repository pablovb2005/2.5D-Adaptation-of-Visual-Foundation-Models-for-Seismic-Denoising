"""Migrate experiments/ into runs/ and summaries/.

Dry-run by default. Use --apply to perform moves.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root


RUN_DIR_MOVES = [
    ("2d", "runs/2d"),
    ("3ch", "runs/3ch"),
    ("5ch", "runs/5ch"),
    ("ablations", "runs/ablations"),
    ("robustness/f3", "runs/robustness/f3"),
    ("summary/logs", "runs/system/setup_py310/logs"),
]

SUMMARY_DIR_MOVES = [
    ("robustness/summary", "summaries/f3_robustness"),
    ("comparison_panels", "summaries/comparison_panels"),
    ("poster_figures", "summaries/poster_figures"),
]

SUMMARY_FILE_ROUTES = {
    "index": {
        "impeccable_report.md",
        "impeccable_run_summary.csv",
        "impeccable_training_history.csv",
        "legacy_summary.csv",
        "test_metrics_bars.png",
    },
    "main_experiment": {
        "main_comparison.csv",
        "main_replicate_summary.csv",
        "main_stats_tests.csv",
        "main_loss_curves.png",
        "main_test_metrics_bars.png",
        "main_val_ms_ssim_curves.png",
        "main_val_ms_ssim_r_curves.png",
    },
    "data_efficiency": {
        "data_efficiency.png",
        "data_efficiency_loss_curves.png",
        "data_efficiency_summary.csv",
        "data_efficiency_val_ms_ssim_curves.png",
        "data_efficiency_val_ms_ssim_r_curves.png",
    },
    "ablations": {
        "ablation_study_a_slice_stride.png",
        "ablation_study_b_neighbor_stride.png",
        "ablation_study_c_crop_coverage.png",
        "ablation_summary.csv",
        "ablation_test_metrics_bars.png",
        "ablations_loss_curves.png",
        "ablations_val_ms_ssim_curves.png",
        "ablations_val_ms_ssim_r_curves.png",
    },
    "mechanism_analysis/stratification/by_2d_ms_ssim": {
        "difficulty_correlation_summary.csv",
        "difficulty_delta_scatter.png",
        "difficulty_gain_by_quartile.png",
        "difficulty_stratification_summary.csv",
        "difficulty_top_examples.csv",
        "difficulty_volume_delta.png",
        "difficulty_volume_summary.csv",
        "mechanism_analysis_report.md",
    },
}


def _project_root() -> Path:
    return default_project_root(__file__)


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _move(src: Path, dst: Path, root: Path, apply: bool) -> None:
    if not src.exists():
        return
    if not _inside(src, root) or not _inside(dst, root):
        raise RuntimeError(f"Refusing move outside experiments/: {src} -> {dst}")
    if dst.exists():
        raise FileExistsError(f"Target already exists, refusing to overwrite: {dst}")
    print(f"MOVE {src.relative_to(root)} -> {dst.relative_to(root)}")
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


def _route_summary_files(exp_root: Path, apply: bool) -> None:
    summary = exp_root / "summary"
    if not summary.exists():
        return
    for target_name, filenames in SUMMARY_FILE_ROUTES.items():
        for filename in sorted(filenames):
            _move(summary / filename, exp_root / "summaries" / target_name / filename, exp_root, apply)


def _cleanup_empty_dirs(exp_root: Path, apply: bool) -> None:
    for rel in ["summary", "robustness"]:
        path = exp_root / rel
        if not path.exists() or any(path.iterdir()):
            continue
        print(f"RMDIR {path.relative_to(exp_root)}")
        if apply:
            path.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--apply", action="store_true", help="Perform moves. Default is dry-run.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    exp_root = project_root / "experiments"
    if not exp_root.exists():
        raise FileNotFoundError(f"Missing experiments directory: {exp_root}")

    print(f"Experiments root: {exp_root}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    for src_rel, dst_rel in RUN_DIR_MOVES:
        _move(exp_root / src_rel, exp_root / dst_rel, exp_root, args.apply)
    for src_rel, dst_rel in SUMMARY_DIR_MOVES:
        _move(exp_root / src_rel, exp_root / dst_rel, exp_root, args.apply)
    _route_summary_files(exp_root, args.apply)
    _cleanup_empty_dirs(exp_root, args.apply)

    print("Migration scan complete.")


if __name__ == "__main__":
    main()
