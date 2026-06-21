"""Summarize experiment folders and draw result plots.

The expected run layout is:

    experiments/runs/<family>/<version>/
        best.pt
        last.pt
        config.yaml              # copied by training/train.py for future runs
        run_meta.yaml            # copied by training/train.py for future runs
        history.csv              # per-epoch history for future runs
        logs/
            slurm_<JOBID>.out
            slurm_<JOBID>.err
        eval_results/
            results.csv
            test_example.png

For older runs, the script also looks in the legacy matching DAIC script folder:

    Code/DAIC/<family>/<version>/*.out

It does not require job IDs. If old SLURM logs are gone, it can still summarize
``best.pt`` / ``last.pt`` and test metrics, but it cannot recreate loss curves.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from evaluation.common.paths import project_root as default_project_root
from typing import Any

import yaml

try:
    from scipy import stats as _scipy_stats
except ImportError:
    _scipy_stats = None  # type: ignore[assignment]
_SCIPY_AVAILABLE = _scipy_stats is not None


EPOCH_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*/\s*(?P<total_epochs>\d+)\s*\|\s*"
    r"train_loss=(?P<train_loss>[-+0-9.eE]+)\s*\|\s*"
    r"val_loss=(?P<val_loss>[-+0-9.eE]+)\s*\|\s*"
    r"val_MS-SSIM=(?P<val_ms_ssim>[-+0-9.eE]+)\s*\|\s*"
    r"val_MS-SSIM-R=(?P<val_ms_ssim_r>[-+0-9.eE]+)\s*\|\s*"
    r"lr=(?P<lr>[-+0-9.eE]+)"
)
TOTAL_PARAMS_RE = re.compile(r"Total params:\s+([0-9,]+)")
TRAINABLE_PARAMS_RE = re.compile(r"Trainable params:\s+([0-9,]+)\s+\(([0-9.]+)%\)")
MODE_RE = re.compile(r"^2\.5d_(\d+)ch$")


HISTORY_FIELDS = [
    "run_id",
    "label",
    "epoch",
    "total_epochs",
    "train_loss",
    "val_loss",
    "val_ms_ssim",
    "val_ms_ssim_r",
    "lr",
    "source",
]

SUMMARY_FIELDS = [
    "run_id",
    "label",
    "variant_key",
    "n_vols",
    "training_seed",
    "checkpoint_dir",
    "best_pt",
    "last_pt",
    "best_epoch",
    "last_epoch",
    "total_epochs",
    "timed_out_or_partial",
    "best_val_ms_ssim",
    "history_epochs",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "test_ms_ssim",
    "test_ms_ssim_r",
    "test_mse",
    "test_psnr",
    "checkpoint_load_error",
    "study_group",
    "display_name",
    "condition",
    "is_complete",
    "has_eval",
    "is_legacy",
    "epoch_status",
    "eval_status",
]

FOCUSED_FIELDS = [
    "display_name",
    "study_group",
    "variant_key",
    "condition",
    "n_vols",
    "training_seed",
    "epoch_status",
    "eval_status",
    "best_epoch",
    "best_val_ms_ssim",
    "test_ms_ssim",
    "test_ms_ssim_r",
    "test_mse",
    "test_psnr",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "checkpoint_dir",
]

REPLICATE_FIELDS = [
    "variant_key",
    "n_runs",
    "complete_runs",
    "evaluated_runs",
    "seeds",
    "test_ms_ssim_mean",
    "test_ms_ssim_std",
    "test_ms_ssim_r_mean",
    "test_ms_ssim_r_std",
    "test_mse_mean",
    "test_mse_std",
    "test_psnr_mean",
    "test_psnr_std",
]

PLATEAU_THRESHOLD = 0.005
PLATEAU_REQUIRED_VARIANTS = ("2D", "3ch", "5ch")
PLATEAU_CONFIRMATION_SEEDS = (42, 43, 44)
PLATEAU_DECISION_FIELDS = [
    "variant_key",
    "n_vols",
    "training_seeds",
    "n_runs",
    "complete_runs",
    "evaluated_runs",
    "best_val_ms_ssim",
    "test_ms_ssim",
    "test_ms_ssim_r",
    "test_mse",
    "test_psnr",
    "delta_val_to_n100",
    "within_0p005_val_of_n100",
    "confirmation_status",
    "candidate_protocol_n",
    "selected_protocol_n",
    "decision_note",
]

VARIANT_ORDER = {"2D": 0, "3ch": 1, "5ch": 2, "7ch": 3, "9ch": 4, "5ch-control": 99}
EXCLUDED_VARIANTS = {"5ch-control"}
MAIN_FAMILIES = {"2d", "3ch", "5ch"}
MECHANISM_CONTROL_FAMILIES = {"3ch_shuffled", "5ch_repeated_center", "5ch_shuffled"}
PLOT_FILENAMES = [
    "main_val_ms_ssim_curves.png",
    "main_val_ms_ssim_r_curves.png",
    "main_loss_curves.png",
    "main_test_metrics_bars.png",
    "data_efficiency_val_ms_ssim_curves.png",
    "data_efficiency_val_ms_ssim_r_curves.png",
    "data_efficiency_loss_curves.png",
    "data_efficiency.png",
    "data_efficiency_100train_val_ms_ssim_curves.png",
    "data_efficiency_100train_val_ms_ssim_r_curves.png",
    "data_efficiency_100train_loss_curves.png",
    "data_efficiency_100train.png",
    "ablations_val_ms_ssim_curves.png",
    "ablations_val_ms_ssim_r_curves.png",
    "ablations_loss_curves.png",
    "ablation_test_metrics_bars.png",
    "ablation_study_a_slice_stride.png",
    "ablation_study_b_neighbor_stride.png",
    "ablation_study_c_crop_coverage.png",
    "test_metrics_bars.png",
]

PLOT_ROUTES = {
    "main_experiment": {
        "main_val_ms_ssim_curves.png",
        "main_val_ms_ssim_r_curves.png",
        "main_loss_curves.png",
        "main_test_metrics_bars.png",
    },
    "data_efficiency": {
        "data_efficiency_val_ms_ssim_curves.png",
        "data_efficiency_val_ms_ssim_r_curves.png",
        "data_efficiency_loss_curves.png",
        "data_efficiency.png",
    },
    "data_efficiency_100train": {
        "data_efficiency_100train_val_ms_ssim_curves.png",
        "data_efficiency_100train_val_ms_ssim_r_curves.png",
        "data_efficiency_100train_loss_curves.png",
        "data_efficiency_100train.png",
    },
    "ablations": {
        "ablations_val_ms_ssim_curves.png",
        "ablations_val_ms_ssim_r_curves.png",
        "ablations_loss_curves.png",
        "ablation_test_metrics_bars.png",
        "ablation_study_a_slice_stride.png",
        "ablation_study_b_neighbor_stride.png",
        "ablation_study_c_crop_coverage.png",
    },
    "index": {
        "test_metrics_bars.png",
    },
}


@dataclass(frozen=True)
class Run:
    run_id: str
    label: str
    rel_dir: Path
    exp_dir: Path


def _project_root() -> Path:
    return default_project_root(__file__)


def _experiments_root(project_root: Path) -> Path:
    return project_root / "experiments"


def _runs_root(project_root: Path) -> Path:
    experiments = _experiments_root(project_root)
    runs = experiments / "runs"
    return runs if runs.exists() else experiments


def _summary_dirs(summary_root: Path) -> dict[str, Path]:
    return {
        "root": summary_root,
        "index": summary_root / "index",
        "main_experiment": summary_root / "main_experiment",
        "data_efficiency": summary_root / "data_efficiency",
        "data_efficiency_100train": summary_root / "data_efficiency_100train",
        "data_efficiency_100train_v2": summary_root / "data_efficiency_100train_v2",
        "data_efficiency_100train_channel_window_v2": summary_root / "data_efficiency_100train_channel_window_v2",
        "ablations": summary_root / "ablations",
    }


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)
    except (TypeError, ValueError):
        return None


def _mode_channels(mode: object) -> int | None:
    if not isinstance(mode, str):
        return None
    if mode == "2d":
        return 3
    match = MODE_RE.fullmatch(mode)
    return int(match.group(1)) if match else None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except OSError:
        return {}


def _format_label(rel_dir: Path, meta: dict[str, Any], config: dict[str, Any]) -> str:
    data = meta.get("data") or config.get("data") or {}
    model = meta.get("model") or config.get("model") or {}
    mode = data.get("mode")
    in_chans = model.get("in_chans", 3)
    lora_rank = model.get("lora_rank")

    if mode == "2d":
        return f"{rel_dir.as_posix()} | 2D [t,t,t]"
    channels = _mode_channels(mode)
    if channels is not None and mode != "2d":
        if channels == 3:
            return f"{rel_dir.as_posix()} | 2.5D-3ch"
        adapter = "PatchEmb+LoRA" if _safe_int(lora_rank) and _safe_int(lora_rank) > 0 else "PatchEmb+decoder"
        return f"{rel_dir.as_posix()} | 2.5D-{channels}ch {adapter}"
    if _safe_int(in_chans) and _safe_int(in_chans) != 3:
        return f"{rel_dir.as_posix()} | {_safe_int(in_chans)}ch"
    return rel_dir.as_posix()


def _n_vols_from_config(exp_dir: Path, config: dict[str, Any]) -> int:
    v = config.get("data", {}).get("train_subset_n")
    if v is not None:
        return int(v)
    m = re.search(r"_n(\d+)vols", str(exp_dir))
    if m:
        return int(m.group(1))
    return int(config.get("data", {}).get("n_train", 20))


def _variant_key_from_config(config: dict[str, Any], rel_dir: Path | None = None) -> str:
    data = config.get("data") or {}
    model = config.get("model") or {}
    mode = data.get("mode")
    lora_rank = _safe_int(model.get("lora_rank", 0))
    if mode == "2d":
        return "2D"
    channels = _mode_channels(mode)
    if channels is not None and mode != "2d":
        if channels == 5 and not (lora_rank and lora_rank > 0):
            return "5ch-control"
        return f"{channels}ch"
    if rel_dir is not None:
        text = rel_dir.as_posix().lower()
        for channels in (3, 5, 7, 9):
            label = f"{channels}ch"
            if text.startswith(f"{label}/") or f"/{label}/" in text:
                if channels == 5 and ("patch_emb_head" in text or "5ch_a" in text or "_a_" in text):
                    return "5ch-control"
                return label
        if text.startswith("2d/") or "/2d/" in text:
            return "2D"
    return mode or "unknown"


def _is_legacy_run(run: Run, config: dict[str, Any]) -> bool:
    rel = run.rel_dir.as_posix().lower()
    if config:
        return False
    return "impeccable_v1" in rel or "_v1" in rel


def _study_group(run: Run, config: dict[str, Any]) -> str:
    if _is_legacy_run(run, config):
        return "legacy"
    family = run.rel_dir.parts[0].lower() if run.rel_dir.parts else ""
    if family == "ablations":
        return "ablations"
    if family == "data_efficiency_100train":
        return "data_efficiency_100train"
    if family == "data_efficiency_100train_v2":
        return "data_efficiency_100train_v2"
    if family == "data_efficiency_100train_channel_window_v2":
        return "data_efficiency_100train_channel_window_v2"
    if re.search(r"_n\d+vols", run.rel_dir.as_posix()):
        return "data_efficiency"
    if family == "full_ft":
        return "full_ft"
    if family in MECHANISM_CONTROL_FAMILIES:
        return "mechanism_controls"
    if family in {"robustness", "system"}:
        return family
    if family in MAIN_FAMILIES or family == "main_multidata":
        return "main"
    return "excluded"


def _condition_from_config(run: Run, config: dict[str, Any], n_vols: int | None) -> str:
    data = config.get("data") or {}
    parts = []
    if n_vols is not None:
        parts.append(f"n={n_vols}")
    family = run.rel_dir.parts[0].lower() if run.rel_dir.parts else ""
    if family == "main_multidata":
        dseed = _data_seed(config)
        if dseed is not None:
            parts.append(f"data_seed={dseed}")
    base_stride = data.get("slice_stride")
    train_stride = data.get("train_slice_stride")
    val_stride = data.get("val_slice_stride") or data.get("eval_slice_stride")
    test_stride = data.get("test_slice_stride") or data.get("eval_slice_stride")
    if train_stride is not None or val_stride is not None or test_stride is not None:
        if train_stride is not None:
            parts.append(f"train_stride={train_stride}")
        elif base_stride is not None:
            parts.append(f"train_stride={base_stride}")
        if val_stride is not None:
            parts.append(f"val_stride={val_stride}")
        if test_stride is not None and test_stride != val_stride:
            parts.append(f"test_stride={test_stride}")
    elif base_stride is not None:
        parts.append(f"stride={base_stride}")
    mode_channels = _mode_channels(data.get("mode"))
    if (mode_channels is not None and data.get("mode") != "2d") or re.search(
        r"(^|[/_])(3|5|7|9)ch([/_]|$)", run.rel_dir.as_posix()
    ):
        parts.append(f"ns={data.get('neighbor_stride', 1)}")
    crop = data.get("train_crop_mode") or data.get("crop_mode")
    if crop and crop != "center":
        parts.append(f"crop={crop}")
    return ", ".join(parts)


def _training_seed(config: dict[str, Any]) -> int | None:
    training = config.get("training") or {}
    data = config.get("data") or {}
    return _safe_int(training.get("seed")) or _safe_int(data.get("seed"))


def _data_seed(config: dict[str, Any]) -> int | None:
    return _safe_int((config.get("data") or {}).get("seed"))


def _display_name(run: Run, config: dict[str, Any], variant: str, group: str, n_vols: int | None) -> str:
    rel = run.rel_dir.as_posix().lower()
    if group == "main":
        seed = _training_seed(config)
        family = run.rel_dir.parts[0].lower() if run.rel_dir.parts else ""
        if family == "main_multidata":
            dseed = _data_seed(config)
            if seed is not None and dseed is not None:
                return f"Main {variant} d{dseed} t{seed}"
            return f"Main {variant} seed{seed}" if seed is not None else f"Main {variant}"
        return f"Main {variant} seed{seed}" if seed is not None else f"Main {variant}"
    if group == "full_ft":
        seed = _training_seed(config)
        return f"Full-FT {variant} seed{seed}" if seed is not None else f"Full-FT {variant}"
    if group == "mechanism_controls":
        seed = _training_seed(config)
        family = run.rel_dir.parts[0] if run.rel_dir.parts else variant
        return f"Control {family} seed{seed}" if seed is not None else f"Control {family}"
    if group == "data_efficiency":
        return f"DE {variant} n={n_vols}"
    if group == "data_efficiency_100train":
        return f"DE100 {variant} n={n_vols}"
    if group == "data_efficiency_100train_v2":
        return f"DE100v2 {variant} n={n_vols}"
    if group == "data_efficiency_100train_channel_window_v2":
        return f"DE100pilot {variant} n={n_vols}"
    if group == "legacy":
        return f"Legacy {variant}"
    if "stride1" in rel:
        return "ABL A2 2D stride=1"
    if "stride3" in rel:
        return "ABL A1 2D stride=3"
    if "ns2" in rel:
        return "ABL B1 3ch ns=2"
    if "ns3" in rel:
        return "ABL B2 3ch ns=3"
    if "grid4" in rel:
        return "ABL C1 3ch grid4"
    return f"ABL {variant}"


def _epoch_status(last_epoch: int | None, total_epochs: int | None) -> tuple[bool, str]:
    if last_epoch is None or total_epochs is None:
        return False, "unknown"
    if last_epoch >= total_epochs:
        return True, f"{last_epoch}/{total_epochs} complete"
    return False, f"{last_epoch}/{total_epochs} partial"


def _sort_key(row: dict[str, Any]) -> tuple:
    group_order = {
        "main": 0,
        "data_efficiency": 1,
        "data_efficiency_100train": 2,
        "data_efficiency_100train_v2": 3,
        "data_efficiency_100train_channel_window_v2": 4,
        "ablations": 5,
        "mechanism_controls": 6,
        "full_ft": 7,
        "robustness": 8,
        "system": 9,
        "legacy": 10,
        "excluded": 11,
    }
    return (
        group_order.get(str(row.get("study_group")), 9),
        VARIANT_ORDER.get(str(row.get("variant_key")), 9),
        row.get("n_vols") or 999,
        row.get("training_seed") or 999,
        str(row.get("display_name") or row.get("label") or ""),
    )


def _included_summary_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in summary if row.get("variant_key") not in EXCLUDED_VARIANTS]


def _discover_runs(project_root: Path) -> list[Run]:
    experiments = _runs_root(project_root)
    if not experiments.exists():
        return []

    runs = []
    for path in sorted(p for p in experiments.rglob("*") if p.is_dir()):
        rel = path.relative_to(experiments)
        if not rel.parts or rel.parts[0] in {"summary", "summaries"}:
            continue
        markers = [
            path / "best.pt",
            path / "last.pt",
            path / "history.csv",
            path / "run_meta.yaml",
            path / "eval_results" / "results.csv",
        ]
        if not any(marker.exists() for marker in markers):
            continue

        meta = _read_yaml(path / "run_meta.yaml")
        config = _read_yaml(path / "config.yaml")
        label = _format_label(rel, meta, config)
        run_id = "__".join(rel.parts)
        runs.append(Run(run_id=run_id, label=label, rel_dir=rel, exp_dir=path))
    return runs


def _candidate_logs(project_root: Path, run: Run) -> list[Path]:
    candidates: list[Path] = []
    for base in [
        run.exp_dir,
        run.exp_dir / "logs",
        project_root / "Code" / "DAIC" / run.rel_dir,
    ]:
        if not base.exists():
            continue
        for pattern in ("*.out", "*.err", "*.log"):
            candidates.extend(base.rglob(pattern))
    return sorted(set(candidates), key=lambda p: (p.stat().st_mtime, str(p)))


def _parse_history_csv(run: Run) -> list[dict[str, Any]]:
    path = run.exp_dir / "history.csv"
    if not path.exists():
        return []

    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "epoch": _safe_int(row.get("epoch")),
                    "total_epochs": _safe_int(row.get("total_epochs")),
                    "train_loss": _safe_float(row.get("train_loss")),
                    "val_loss": _safe_float(row.get("val_loss")),
                    "val_ms_ssim": _safe_float(row.get("val_ms_ssim")),
                    "val_ms_ssim_r": _safe_float(row.get("val_ms_ssim_r")),
                    "lr": _safe_float(row.get("lr")),
                    "source": str(path),
                    "_mtime": path.stat().st_mtime,
                }
            )
    return [row for row in rows if row["epoch"] is not None]


def _parse_log_history(project_root: Path, run: Run) -> list[dict[str, Any]]:
    rows = []
    for path in _candidate_logs(project_root, run):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for match in EPOCH_RE.finditer(text):
            gd = match.groupdict()
            rows.append(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "epoch": int(gd["epoch"]),
                    "total_epochs": int(gd["total_epochs"]),
                    "train_loss": float(gd["train_loss"]),
                    "val_loss": float(gd["val_loss"]),
                    "val_ms_ssim": float(gd["val_ms_ssim"]),
                    "val_ms_ssim_r": float(gd["val_ms_ssim_r"]),
                    "lr": float(gd["lr"]),
                    "source": str(path),
                    "_mtime": path.stat().st_mtime,
                }
            )
    return rows


def _history_rows(project_root: Path, runs: list[Run]) -> list[dict[str, Any]]:
    dedup: dict[tuple[str, int], dict[str, Any]] = {}
    for run in runs:
        rows = _parse_history_csv(run) or _parse_log_history(project_root, run)
        for row in rows:
            key = (row["run_id"], row["epoch"])
            old = dedup.get(key)
            if old is None or row["_mtime"] >= old["_mtime"]:
                dedup[key] = row
    clean = []
    for row in dedup.values():
        row = dict(row)
        row.pop("_mtime", None)
        clean.append(row)
    return sorted(clean, key=lambda r: (r["run_id"], r["epoch"]))


def _checkpoint_meta(path: Path) -> dict[str, Any]:
    meta = {"exists": path.exists(), "epoch": None, "val_ms_ssim": None, "load_error": ""}
    if not path.exists():
        return meta
    try:
        import torch
    except Exception as exc:
        meta["load_error"] = f"torch import failed: {exc}"
        return meta
    try:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        meta["epoch"] = _safe_int(ckpt.get("epoch"))
        meta["val_ms_ssim"] = _safe_float(ckpt.get("val_ms_ssim"))
    except Exception as exc:
        meta["load_error"] = str(exc)
    return meta


def _params_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
    return {
        "trainable_params": _safe_int(params.get("trainable")),
        "total_params": _safe_int(params.get("total")),
        "trainable_pct": _safe_float(params.get("trainable_pct")),
    }


def _params_from_logs(project_root: Path, run: Run) -> dict[str, Any]:
    result = {"trainable_params": None, "total_params": None, "trainable_pct": None}
    for path in _candidate_logs(project_root, run):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        total = TOTAL_PARAMS_RE.search(text)
        trainable = TRAINABLE_PARAMS_RE.search(text)
        if total:
            result["total_params"] = int(total.group(1).replace(",", ""))
        if trainable:
            result["trainable_params"] = int(trainable.group(1).replace(",", ""))
            result["trainable_pct"] = float(trainable.group(2))
        if result["total_params"] and result["trainable_params"]:
            break
    return result


def _eval_metrics(path: Path) -> dict[str, float | None]:
    metrics = {
        "test_ms_ssim": None,
        "test_ms_ssim_r": None,
        "test_mse": None,
        "test_psnr": None,
    }
    if not path.exists():
        return metrics

    values = {key: [] for key in metrics}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "test":
                continue
            mapping = {
                "test_ms_ssim": row.get("ms_ssim"),
                "test_ms_ssim_r": row.get("ms_ssim_r"),
                "test_mse": row.get("mse"),
                "test_psnr": row.get("psnr"),
            }
            for key, raw in mapping.items():
                val = _safe_float(raw)
                if val is not None and math.isfinite(val):
                    values[key].append(val)

    for key, vals in values.items():
        if vals:
            metrics[key] = sum(vals) / len(vals)
    return metrics


def _summary_rows(project_root: Path, runs: list[Run], history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in history:
        by_run.setdefault(row["run_id"], []).append(row)

    rows = []
    for run in runs:
        hist = sorted(by_run.get(run.run_id, []), key=lambda r: r["epoch"])
        valid_hist = [r for r in hist if r.get("val_ms_ssim") is not None]
        best_hist = max(valid_hist, key=lambda r: r["val_ms_ssim"]) if valid_hist else None
        last_hist = hist[-1] if hist else None
        best_ckpt = _checkpoint_meta(run.exp_dir / "best.pt")
        last_ckpt = _checkpoint_meta(run.exp_dir / "last.pt")
        meta = _read_yaml(run.exp_dir / "run_meta.yaml")
        config = _read_yaml(run.exp_dir / "config.yaml")
        n_vols = _n_vols_from_config(run.exp_dir, config)
        variant_key = _variant_key_from_config(config, run.rel_dir)
        group = _study_group(run, config)
        training_seed = _training_seed(config)
        is_legacy = group == "legacy"
        condition = _condition_from_config(run, config, n_vols)
        display_name = _display_name(run, config, variant_key, group, n_vols)
        params = _params_from_meta(meta)
        if not params["trainable_params"] or not params["total_params"]:
            params = _params_from_logs(project_root, run)

        best_epoch = best_ckpt["epoch"] or (best_hist["epoch"] if best_hist else None)
        best_val = best_ckpt["val_ms_ssim"] or (best_hist["val_ms_ssim"] if best_hist else None)
        last_epoch = last_ckpt["epoch"] or (last_hist["epoch"] if last_hist else None)
        total_epochs = (
            last_hist["total_epochs"]
            if last_hist
            else _safe_int((meta.get("training") or {}).get("epochs")) or 50
        )
        is_complete, epoch_status = _epoch_status(last_epoch, total_epochs)
        eval_metrics = _eval_metrics(run.exp_dir / "eval_results" / "results.csv")
        has_eval = any(eval_metrics[k] is not None for k in ("test_ms_ssim", "test_ms_ssim_r", "test_mse", "test_psnr"))

        rows.append(
            {
                "run_id": run.run_id,
                "label": run.label,
                "variant_key": variant_key,
                "n_vols": n_vols,
                "training_seed": training_seed,
                "checkpoint_dir": str(run.exp_dir),
                "best_pt": best_ckpt["exists"],
                "last_pt": last_ckpt["exists"],
                "best_epoch": best_epoch,
                "last_epoch": last_epoch,
                "total_epochs": total_epochs,
                "timed_out_or_partial": bool(last_epoch and total_epochs and last_epoch < total_epochs),
                "best_val_ms_ssim": best_val,
                "history_epochs": len(hist),
                **params,
                **eval_metrics,
                "checkpoint_load_error": best_ckpt["load_error"] or last_ckpt["load_error"],
                "study_group": group,
                "display_name": display_name,
                "condition": condition,
                "is_complete": is_complete,
                "has_eval": has_eval,
                "is_legacy": is_legacy,
                "epoch_status": epoch_status,
                "eval_status": "done" if has_eval else "pending",
            }
        )
    return sorted(rows, key=_sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_focused_csvs(out_dir: Path, summary: list[dict[str, Any]]) -> None:
    groups = {
        "main_comparison.csv": _main_rows(summary),
        "data_efficiency_summary.csv": [r for r in summary if r.get("study_group") == "data_efficiency"],
        "data_efficiency_100train_summary.csv": [
            r for r in summary if r.get("study_group") == "data_efficiency_100train"
        ],
        "data_efficiency_100train_v2_summary.csv": [
            r for r in summary if r.get("study_group") == "data_efficiency_100train_v2"
        ],
        "data_efficiency_100train_channel_window_v2_summary.csv": [
            r for r in summary if r.get("study_group") == "data_efficiency_100train_channel_window_v2"
        ],
        "ablation_summary.csv": [r for r in summary if r.get("study_group") == "ablations"],
        "legacy_summary.csv": _legacy_rows(summary),
    }
    for name, rows in groups.items():
        _write_csv(out_dir / name, sorted(rows, key=_sort_key), FOCUSED_FIELDS)
    _write_csv(out_dir / "main_replicate_summary.csv", _main_replicate_summary(summary), REPLICATE_FIELDS)


def _write_split_csvs(dirs: dict[str, Path], summary: list[dict[str, Any]]) -> None:
    _write_csv(dirs["main_experiment"] / "main_comparison.csv", sorted(_main_rows(summary), key=_sort_key), FOCUSED_FIELDS)
    _write_csv(dirs["main_experiment"] / "main_replicate_summary.csv", _main_replicate_summary(summary), REPLICATE_FIELDS)
    _write_csv(
        dirs["data_efficiency"] / "data_efficiency_summary.csv",
        sorted([r for r in summary if r.get("study_group") == "data_efficiency"], key=_sort_key),
        FOCUSED_FIELDS,
    )
    _write_csv(
        dirs["data_efficiency_100train"] / "data_efficiency_100train_summary.csv",
        sorted([r for r in summary if r.get("study_group") == "data_efficiency_100train"], key=_sort_key),
        FOCUSED_FIELDS,
    )
    _write_csv(
        dirs["data_efficiency_100train"] / "plateau_decision.csv",
        _plateau_decision_rows(summary),
        PLATEAU_DECISION_FIELDS,
    )
    _write_csv(
        dirs["data_efficiency_100train_v2"] / "data_efficiency_100train_v2_summary.csv",
        sorted([r for r in summary if r.get("study_group") == "data_efficiency_100train_v2"], key=_sort_key),
        FOCUSED_FIELDS,
    )
    _write_csv(
        dirs["data_efficiency_100train_v2"] / "plateau_decision.csv",
        _plateau_decision_rows(summary, group="data_efficiency_100train_v2"),
        PLATEAU_DECISION_FIELDS,
    )
    _write_csv(
        dirs["data_efficiency_100train_channel_window_v2"] / "data_efficiency_100train_channel_window_v2_summary.csv",
        sorted([r for r in summary if r.get("study_group") == "data_efficiency_100train_channel_window_v2"], key=_sort_key),
        FOCUSED_FIELDS,
    )
    _write_csv(
        dirs["ablations"] / "ablation_summary.csv",
        sorted([r for r in summary if r.get("study_group") == "ablations"], key=_sort_key),
        FOCUSED_FIELDS,
    )
    _write_csv(dirs["index"] / "legacy_summary.csv", sorted(_legacy_rows(summary), key=_sort_key), FOCUSED_FIELDS)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pending(value: Any, digits: int = 4) -> str:
    return _fmt(value, digits) if value is not None and value != "" else "pending"


def _md(value: Any, digits: int | None = None) -> str:
    if digits is None:
        text = _fmt(value)
    else:
        text = _fmt(value, digits)
    return text.replace("|", "\\|")


def _param_text(row: dict[str, Any]) -> str:
    if row.get("trainable_params") and row.get("total_params"):
        return (
            f"{int(row['trainable_params']):,} / {int(row['total_params']):,} "
            f"({_fmt(row.get('trainable_pct'), 2)}%)"
        )
    return ""


def _write_md_table(lines: list[str], headers: list[str], rows: list[list[Any]]) -> None:
    lines.append("| " + " | ".join(_md(h) for h in headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_md(cell) for cell in row) + " |")


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib, but it could not be imported.\n"
            f"Python executable: {sys.executable}\n"
            f"Import error: {exc}\n\n"
            "Use the project venv on Windows:\n"
            "  C:\\UNI\\Y3\\RP\\Code\\DINOv3\\.venv\\Scripts\\python.exe "
            "Code\\DINOv3\\src\\evaluation\\summarize_impeccable_runs.py "
            "--project-root C:\\UNI\\Y3\\RP --out-dir C:\\UNI\\Y3\\RP\\experiments\\summaries\n\n"
            "Or install the dependency in the active environment:\n"
            "  python -m pip install matplotlib\n\n"
            "Use --no-plots only when you intentionally want tables without PNG figures."
        ) from exc


def _metric(row: dict[str, Any] | None, key: str, digits: int = 4) -> str:
    if not row:
        return "pending"
    value = row.get(key)
    if value is None or value == "":
        return "pending"
    return _fmt(value, digits)


def _mean_std_text(row: dict[str, Any], metric: str, digits: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if mean is None or mean == "":
        return "pending"
    if std is None or std == "":
        return _fmt(mean, digits)
    return f"{_fmt(mean, digits)} +/- {_fmt(std, digits)}"


def _compact_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    compact = []
    for row in sorted(rows, key=_sort_key):
        compact.append(
            [
                row["display_name"],
                row.get("condition") or "",
                row.get("epoch_status") or "",
                _fmt(row.get("best_epoch"), 0),
                _fmt(row.get("best_val_ms_ssim")),
                _metric(row, "test_ms_ssim"),
                _metric(row, "test_ms_ssim_r"),
                _metric(row, "test_mse", 6),
                _metric(row, "test_psnr", 2),
                _param_text(row),
            ]
        )
    return compact


def _main_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in summary if r.get("study_group") == "main"]


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return None, None
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, None
    variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return mean, math.sqrt(variance)


def _main_replicate_summary(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _main_rows(summary):
        grouped.setdefault(str(row.get("variant_key")), []).append(row)

    rows: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(grouped.items(), key=lambda item: VARIANT_ORDER.get(item[0], 9)):
        out: dict[str, Any] = {
            "variant_key": variant,
            "n_runs": len(variant_rows),
            "complete_runs": sum(1 for r in variant_rows if r.get("is_complete")),
            "evaluated_runs": sum(1 for r in variant_rows if r.get("has_eval")),
            "seeds": ",".join(str(r.get("training_seed")) for r in sorted(variant_rows, key=_sort_key) if r.get("training_seed") is not None),
        }
        for metric in ("test_ms_ssim", "test_ms_ssim_r", "test_mse", "test_psnr"):
            vals = [_safe_float(r.get(metric)) for r in variant_rows]
            mean, std = _mean_std([v for v in vals if v is not None])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        rows.append(out)
    return rows


def _plateau_decision_rows(
    summary: list[dict[str, Any]],
    group: str = "data_efficiency_100train",
) -> list[dict[str, Any]]:
    rows = [
        r for r in summary
        if r.get("study_group") == group
        and r.get("variant_key") in PLATEAU_REQUIRED_VARIANTS
        and r.get("n_vols") is not None
    ]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["variant_key"]), int(row["n_vols"])), []).append(row)

    aggregates: dict[tuple[str, int], dict[str, Any]] = {}
    for key, group_rows in grouped.items():
        seeds = sorted(
            {
                seed
                for seed in (_safe_int(r.get("training_seed")) for r in group_rows)
                if seed is not None
            }
        )
        aggregate: dict[str, Any] = {
            "variant_key": key[0],
            "n_vols": key[1],
            "training_seeds": ",".join(str(seed) for seed in seeds),
            "n_runs": len(group_rows),
            "complete_runs": sum(1 for r in group_rows if r.get("is_complete")),
            "evaluated_runs": sum(1 for r in group_rows if r.get("has_eval")),
        }
        for metric in (
            "best_val_ms_ssim",
            "test_ms_ssim",
            "test_ms_ssim_r",
            "test_mse",
            "test_psnr",
        ):
            vals = [_safe_float(r.get(metric)) for r in group_rows]
            mean, _std = _mean_std([v for v in vals if v is not None])
            aggregate[metric] = mean
        aggregates[key] = aggregate

    for variant in PLATEAU_REQUIRED_VARIANTS:
        baseline = aggregates.get((variant, 100))
        baseline_val = _safe_float(baseline.get("best_val_ms_ssim")) if baseline else None
        for (row_variant, _n_vols), row in aggregates.items():
            if row_variant != variant:
                continue
            current_val = _safe_float(row.get("best_val_ms_ssim"))
            if baseline_val is None or current_val is None:
                row["delta_val_to_n100"] = None
                row["within_0p005_val_of_n100"] = "pending"
                continue
            delta = current_val - baseline_val
            row["delta_val_to_n100"] = delta
            row["within_0p005_val_of_n100"] = (
                "yes" if (baseline_val - current_val) <= (PLATEAU_THRESHOLD + 1e-12) else "no"
            )

    candidate_n: int | None = None
    candidate_note = "pending: missing n=100 validation baseline for at least one variant"
    if all(
        _safe_float(aggregates.get((variant, 100), {}).get("best_val_ms_ssim")) is not None
        for variant in PLATEAU_REQUIRED_VARIANTS
    ):
        available_ns = sorted({int(key[1]) for key in aggregates})
        for n_vols in available_ns:
            if all(
                aggregates.get((variant, n_vols), {}).get("within_0p005_val_of_n100") == "yes"
                for variant in PLATEAU_REQUIRED_VARIANTS
            ):
                candidate_n = n_vols
                candidate_note = (
                    f"candidate: n={n_vols} is within {PLATEAU_THRESHOLD:.3f} "
                    "validation MS-SSIM of n=100 for all variants"
                )
                break
        if candidate_n is None:
            candidate_note = "pending: no common training size is within threshold for all variants"

    confirmation_status = "pending_sweep"
    selected_protocol_n: int | None = None
    if candidate_n is not None:
        required_seed_set = set(PLATEAU_CONFIRMATION_SEEDS)
        confirmation_keys = [(variant, candidate_n) for variant in PLATEAU_REQUIRED_VARIANTS]
        if candidate_n != 100:
            confirmation_keys.extend((variant, 100) for variant in PLATEAU_REQUIRED_VARIANTS)

        missing_confirmation = []
        for key in confirmation_keys:
            seeds = {
                _safe_int(seed)
                for seed in str(aggregates.get(key, {}).get("training_seeds", "")).split(",")
                if seed != ""
            }
            missing = sorted(seed for seed in required_seed_set if seed not in seeds)
            if missing:
                missing_confirmation.append(f"{key[0]} n={key[1]} missing {missing}")

        if missing_confirmation:
            confirmation_status = "pending_confirmation"
            candidate_note = f"{candidate_note}; " + "; ".join(missing_confirmation)
        else:
            confirmation_status = "confirmed"
            selected_protocol_n = candidate_n
            candidate_note = f"{candidate_note}; confirmation seeds 42/43/44 complete"

    decision_rows = []
    for key, row in sorted(
        aggregates.items(),
        key=lambda item: (VARIANT_ORDER.get(item[0][0], 9), item[0][1]),
    ):
        out = dict(row)
        out["confirmation_status"] = confirmation_status
        out["candidate_protocol_n"] = candidate_n
        out["selected_protocol_n"] = selected_protocol_n
        out["decision_note"] = candidate_note
        decision_rows.append(out)

    decision_rows.append(
        {
            "variant_key": "ALL",
            "n_vols": candidate_n,
            "training_seeds": "",
            "n_runs": sum(int(r.get("n_runs") or 0) for r in aggregates.values()),
            "complete_runs": sum(int(r.get("complete_runs") or 0) for r in aggregates.values()),
            "evaluated_runs": sum(int(r.get("evaluated_runs") or 0) for r in aggregates.values()),
            "best_val_ms_ssim": None,
            "test_ms_ssim": None,
            "test_ms_ssim_r": None,
            "test_mse": None,
            "test_psnr": None,
            "delta_val_to_n100": None,
            "within_0p005_val_of_n100": "",
            "confirmation_status": confirmation_status,
            "candidate_protocol_n": candidate_n,
            "selected_protocol_n": selected_protocol_n,
            "decision_note": candidate_note,
        }
    )
    return decision_rows


_STATS_NOTE = "n=3; p-values are indicative only — Cohen's d is the primary effect-size estimate"


def _cohens_d(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((v - mean_a) ** 2 for v in a) / (len(a) - 1)
    var_b = sum((v - mean_b) ** 2 for v in b) / (len(b) - 1)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / pooled_std if pooled_std > 1e-12 else None


def _compute_stats_tests(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired statistical comparisons between 2D and the 2.5D variants across seeds."""
    main = _main_rows(summary)
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in main:
        by_variant.setdefault(str(row.get("variant_key")), []).append(row)

    metrics = ["test_ms_ssim", "test_ms_ssim_r", "test_mse", "test_psnr"]
    comparisons = [("3ch", "3ch vs 2D"), ("5ch", "5ch vs 2D")]
    rows: list[dict[str, Any]] = []

    baseline_rows = by_variant.get("2D", [])
    if not baseline_rows:
        return rows

    for variant_key, label in comparisons:
        variant_rows = by_variant.get(variant_key, [])
        if not variant_rows:
            continue

        # Align by training_seed so pairs are genuine matched observations.
        baseline_by_seed = {r.get("training_seed"): r for r in baseline_rows}
        variant_by_seed = {r.get("training_seed"): r for r in variant_rows}
        shared_seeds = sorted(
            s for s in set(baseline_by_seed) & set(variant_by_seed) if s is not None
        )
        if len(shared_seeds) < 2:
            continue

        for metric in metrics:
            a_vals = [_safe_float(baseline_by_seed[s].get(metric)) for s in shared_seeds]
            b_vals = [_safe_float(variant_by_seed[s].get(metric)) for s in shared_seeds]
            paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
            if not paired:
                continue
            pa, pb = zip(*paired)
            pa, pb = list(pa), list(pb)

            mean_a = sum(pa) / len(pa)
            mean_b = sum(pb) / len(pb)
            delta = mean_b - mean_a
            cohens_d = _cohens_d(pb, pa)

            wilcoxon_stat = wilcoxon_p = None
            ttest_stat = ttest_p = None
            if _SCIPY_AVAILABLE and _scipy_stats is not None and len(pa) >= 2:
                _st = _scipy_stats
                try:
                    diffs = [b - a for a, b in zip(pa, pb)]
                    if any(d != 0 for d in diffs):
                        res = _st.wilcoxon(pb, pa)
                        wilcoxon_stat = float(res.statistic)  # type: ignore[union-attr]
                        wilcoxon_p = float(res.pvalue)        # type: ignore[union-attr]
                except Exception:
                    pass
                try:
                    res = _st.ttest_rel(pb, pa)
                    ttest_stat = float(res.statistic)  # type: ignore[union-attr]
                    ttest_p = float(res.pvalue)        # type: ignore[union-attr]
                except Exception:
                    pass

            rows.append({
                "comparison": label,
                "metric": metric,
                "mean_2d": mean_a,
                "mean_variant": mean_b,
                "delta": delta,
                "cohens_d": cohens_d,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "ttest_stat": ttest_stat,
                "ttest_p": ttest_p,
                "n_seeds": len(pa),
                "note": _STATS_NOTE,
            })
    return rows


def _data_efficiency_rows(
    summary: list[dict[str, Any]],
    include_main_reference: bool = True,
    group: str = "data_efficiency",
) -> list[dict[str, Any]]:
    rows = [r for r in summary if r.get("study_group") == group]
    if include_main_reference and group == "data_efficiency":
        rows.extend(
            r
            for r in summary
            if r.get("study_group") == "main" and r.get("training_seed") == 42
        )
    return sorted(rows, key=_sort_key)


def _legacy_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in summary if r.get("study_group") == "legacy"]


def _find_row(rows: list[dict[str, Any]], variant: str, **contains: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("variant_key") != variant:
            continue
        rel = str(row.get("checkpoint_dir", "")).lower()
        if all(value in rel for value in contains.values()):
            return row
    return None


def _ablation_studies(summary: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    de = [r for r in summary if r.get("study_group") == "data_efficiency"]
    abl = [r for r in summary if r.get("study_group") == "ablations"]

    study_a = [
        _find_row(de, "2D", path="n05vols"),
        _find_row(abl, "2D", path="stride3"),
        _find_row(abl, "2D", path="stride1"),
    ]
    study_b = [
        _find_row(de, "3ch", path="n05vols"),
        _find_row(abl, "3ch", path="ns2"),
        _find_row(abl, "3ch", path="ns3"),
    ]
    study_c = [
        _find_row(de, "3ch", path="n05vols"),
        _find_row(abl, "3ch", path="grid4"),
    ]
    return {
        "Study A - 2D slice stride": [r for r in study_a if r],
        "Study B - 3ch neighbor stride": [r for r in study_b if r],
        "Study C - 3ch crop coverage": [r for r in study_c if r],
    }


def _write_report(path: Path, summary: list[dict[str, Any]]) -> None:
    metric_headers = [
        "Run",
        "Condition",
        "Status",
        "Best epoch",
        "Best val MS-SSIM",
        "Test MS-SSIM",
        "Test MS-SSIM-R",
        "MSE",
        "PSNR",
        "Trainable params",
    ]
    lines = [
        "# Image Impeccable Run Summary",
        "",
        "This report separates current protocol runs from data-efficiency, ablation, and legacy runs.",
        "Missing test metrics mean `evaluation/evaluate.py` has not finished for that checkpoint yet.",
        "",
        "## Protocol",
        "",
        "- **Dataset**: ThinkOnward Image Impeccable, parts 1–2 (30 paired 3D volumes).",
        "- **Split policy**: Volume-level split (never slice- or crop-level). Test set is held out and evaluated once per checkpoint via `evaluate.py`.",
        "- **Data seeds**: 101, 202, 303 (control which volumes go to train/val/test).",
        "- **Training seeds**: 42, 43, 44 (control weight initialisation and data augmentation).",
        "- **Full protocol**: 3 data seeds × 3 training seeds = 9 replicate runs per variant.",
        "- **Variants**: `2D` (repeated-channel control), `3ch` (aligned nearest neighbour), `5ch` (aligned ±2 with patch-emb init).",
        "- **PEFT method**: LoRA on `qkv` + `proj` targets, rank 16. No TTA.",
        "- **Primary metrics**: MS-SSIM ↑, PSNR ↑, MSE ↓, MS-SSIM-R ↓ (residual diagnostic).",
        "- **n_vols default**: 20 training volumes unless otherwise stated.",
        "",
        "## Main 2D vs 2.5D Comparison",
        "",
    ]
    main = _main_rows(summary)
    if main:
        _write_md_table(lines, metric_headers, _compact_rows(main))
    else:
        lines.append("No main runs discovered.")

    replicate_rows = _main_replicate_summary(summary)
    if replicate_rows:
        lines.extend(["", "### Main replicate aggregate", ""])
        _write_md_table(
            lines,
            ["Variant", "Runs", "Complete", "Evaluated", "Seeds", "MS-SSIM mean +/- std", "MS-SSIM-R mean +/- std", "MSE mean +/- std", "PSNR mean +/- std"],
            [
                [
                    r["variant_key"],
                    r["n_runs"],
                    r["complete_runs"],
                    r["evaluated_runs"],
                    r["seeds"],
                    _mean_std_text(r, "test_ms_ssim"),
                    _mean_std_text(r, "test_ms_ssim_r"),
                    _mean_std_text(r, "test_mse", 6),
                    _mean_std_text(r, "test_psnr", 2),
                ]
                for r in replicate_rows
            ],
        )

    lines.extend(
        [
            "",
            "## Data Efficiency",
            "",
            "Training-volume sweep. The n=20 entries are active main-run references when available.",
            "",
        ]
    )

    de_rows = _data_efficiency_rows(summary)
    de_variants: list[str] = sorted({str(r["variant_key"]) for r in de_rows}, key=lambda v: VARIANT_ORDER.get(v, 9))
    de_n_vols: list[int] = sorted({int(r["n_vols"]) for r in de_rows if r.get("n_vols") is not None})
    if de_variants and de_n_vols:
        by_variant_n: dict[tuple[str, int], dict[str, Any]] = {
            (r["variant_key"], r["n_vols"]): r
            for r in de_rows
            if r.get("variant_key") and r.get("n_vols") is not None
        }
        de_headers = ["Variant"] + [f"n={n}" for n in de_n_vols]
        lines.append("### Test MS-SSIM")
        lines.append("")
        de_metric_rows = []
        for variant in de_variants:
            cells = [variant]
            for n in de_n_vols:
                cells.append(_metric(by_variant_n.get((variant, n)), "test_ms_ssim"))
            de_metric_rows.append(cells)
        _write_md_table(lines, de_headers, de_metric_rows)

        lines.extend(["", "### Test MS-SSIM-R", ""])
        de_metric_rows = []
        for variant in de_variants:
            cells = [variant]
            for n in de_n_vols:
                cells.append(_metric(by_variant_n.get((variant, n)), "test_ms_ssim_r"))
            de_metric_rows.append(cells)
        _write_md_table(lines, de_headers, de_metric_rows)
    else:
        lines.append("No data-efficiency runs discovered.")

    lines.extend(
        [
            "",
            "## Large Data Efficiency: 100-Train Split",
            "",
            "Separate larger-data sweep on the 120-pair canonical dataset with a fixed 100/10/10 split.",
            "",
        ]
    )
    de100_rows = _data_efficiency_rows(summary, include_main_reference=False, group="data_efficiency_100train")
    de100_variants: list[str] = sorted(
        {str(r["variant_key"]) for r in de100_rows},
        key=lambda v: VARIANT_ORDER.get(v, 9),
    )
    de100_n_vols: list[int] = sorted(
        {int(r["n_vols"]) for r in de100_rows if r.get("n_vols") is not None}
    )
    if de100_variants and de100_n_vols:
        by_variant_n = {
            (r["variant_key"], r["n_vols"]): r
            for r in de100_rows
            if r.get("variant_key") and r.get("n_vols") is not None
        }
        de100_headers = ["Variant"] + [f"n={n}" for n in de100_n_vols]
        lines.append("### Test MS-SSIM")
        lines.append("")
        de100_metric_rows = []
        for variant in de100_variants:
            cells = [variant]
            for n in de100_n_vols:
                cells.append(_metric(by_variant_n.get((variant, n)), "test_ms_ssim"))
            de100_metric_rows.append(cells)
        _write_md_table(lines, de100_headers, de100_metric_rows)

        lines.extend(["", "### Test MS-SSIM-R", ""])
        de100_metric_rows = []
        for variant in de100_variants:
            cells = [variant]
            for n in de100_n_vols:
                cells.append(_metric(by_variant_n.get((variant, n)), "test_ms_ssim_r"))
            de100_metric_rows.append(cells)
        _write_md_table(lines, de100_headers, de100_metric_rows)
    else:
        lines.append("No large data-efficiency runs discovered.")

    lines.extend(["", "## Ablation Studies", ""])
    for title, rows in _ablation_studies(summary).items():
        lines.extend([f"### {title}", ""])
        if rows:
            _write_md_table(lines, metric_headers, _compact_rows(rows))
        else:
            lines.append("No runs discovered for this study.")
        lines.append("")

    legacy = _legacy_rows(summary)
    if legacy:
        lines.extend(["## Legacy Partial Runs", ""])
        lines.append("Old partial checkpoints are context only and should not be mixed with the current protocol.")
        lines.append("")
        _write_md_table(lines, metric_headers, _compact_rows(legacy))

    pending = [r for r in summary if not r.get("has_eval") and not r.get("is_legacy")]
    if pending:
        lines.extend(["", "## Missing Evaluations / Partial Runs", ""])
        _write_md_table(
            lines,
            ["Run", "Status", "Eval"],
            [[r["display_name"], r["epoch_status"], r["eval_status"]] for r in sorted(pending, key=_sort_key)],
        )

    stats_rows = _compute_stats_tests(summary)
    if stats_rows:
        lines.extend(["", "## Statistical Comparisons", ""])
        lines.append(
            "> **Note:** n=3 seeds per condition. "
            "The minimum achievable Wilcoxon p-value at n=3 is 0.125, so formal significance (p<0.05) "
            "cannot be reached. **Cohen's d is the primary effect-size estimate.** "
            "p-values are shown for completeness only."
        )
        lines.append("")
        stats_headers = [
            "Comparison", "Metric", "Mean 2D", "Mean variant", "Δ (variant−2D)",
            "Cohen's d", "Wilcoxon p", "t-test p", "n",
        ]
        stats_table_rows = [
            [
                r["comparison"],
                r["metric"],
                _fmt(r["mean_2d"]),
                _fmt(r["mean_variant"]),
                _fmt(r["delta"]),
                _fmt(r["cohens_d"]),
                _fmt(r["wilcoxon_p"]) if r["wilcoxon_p"] is not None else "n/a",
                _fmt(r["ttest_p"]) if r["ttest_p"] is not None else "n/a",
                str(r["n_seeds"]),
            ]
            for r in stats_rows
        ]
        _write_md_table(lines, stats_headers, stats_table_rows)
        if not _SCIPY_AVAILABLE:
            lines.append("")
            lines.append("> scipy not installed — Wilcoxon and t-test columns are n/a. Run `pip install scipy` to enable.")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `best.pt` is selected by validation MS-SSIM; held-out test metrics appear only after evaluation finishes.",
            "- Higher MS-SSIM and PSNR are better; lower MS-SSIM-R and MSE are better.",
            "- Partial rows are usable for posters only if explicitly labeled partial/checkpoint results.",
            "- Legacy rows are not part of the current 2D vs 2.5D protocol.",
            "",
            "## Figures",
            "",
            "| Filename | Description |",
            "|---|---|",
            "| `main_val_ms_ssim_curves.png` | Validation MS-SSIM learning curves for main runs |",
            "| `main_val_ms_ssim_r_curves.png` | Validation MS-SSIM-R learning curves for main runs |",
            "| `main_loss_curves.png` | Train/val loss curves for main runs |",
            "| `main_test_metrics_bars.png` | Bar chart of all four test metrics per main run |",
            "| `main_per_run_distribution.png` | Box plot of per-batch MS-SSIM distribution per run, grouped by variant |",
            "| `main_seed_consistency.png` | Strip plot of per-run test MS-SSIM, coloured by data seed |",
            "| `main_per_data_seed_bars.png` | Grouped bars: mean test MS-SSIM per variant, one group per data seed |",
            "| `data_efficiency.png` | MS-SSIM vs n_vols curve for data-efficiency sweep |",
            "| `data_efficiency_100train.png` | MS-SSIM vs n_vols for 100-train sweep |",
            "| `ablation_test_metrics_bars.png` | Bar chart of test metrics for ablation runs |",
            "| `test_metrics_bars.png` | All non-legacy runs: test metrics bar chart |",
            "",
            "## Claim Boundaries",
            "",
            "- Results are specific to Image Impeccable parts 1–2, DINOv3 backbone, LoRA rank 16, n=20 training volumes, PEFT-only (no TTA).",
            "- Do **not** compare these numbers directly to Jiahua's PEFT+TTA numbers.",
            "- The `[t,t,t]` repeated-channel 2D control is not 2.5D; it is the 2D baseline.",
            "- F3 field-transfer numbers are supplementary and have no clean ground truth — do not call them accuracy.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(history: list[dict[str, Any]], summary: list[dict[str, Any]], out_dir: Path) -> None:
    plt = _load_pyplot()

    by_run: dict[str, list[dict[str, Any]]] = {}
    labels = {row["run_id"]: row["display_name"] for row in summary}
    for row in history:
        by_run.setdefault(row["run_id"], []).append(row)

    def history_line_plot(rows_for_plot: list[dict[str, Any]], metric: str, title: str, ylabel: str, filename: str) -> None:
        run_ids = {r["run_id"] for r in rows_for_plot}
        if not run_ids:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for run_id, rows in sorted(by_run.items()):
            if run_id not in run_ids:
                continue
            rows = sorted(rows, key=lambda r: r["epoch"])
            vals = [r.get(metric) for r in rows]
            if not any(v is not None for v in vals):
                continue
            ax.plot(
                [r["epoch"] for r in rows],
                vals,
                marker="o",
                linewidth=1.7,
                markersize=3,
                label=labels.get(run_id, run_id),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def loss_plot(rows_for_plot: list[dict[str, Any]], title: str, filename: str) -> None:
        run_ids = {r["run_id"] for r in rows_for_plot}
        if not run_ids:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plotted = False
        for run_id, rows in sorted(by_run.items()):
            if run_id not in run_ids:
                continue
            rows = sorted(rows, key=lambda r: r["epoch"])
            axes[0].plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label=labels.get(run_id, run_id))
            axes[1].plot([r["epoch"] for r in rows], [r["val_loss"] for r in rows], label=labels.get(run_id, run_id))
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        axes[0].set_title("Training loss")
        axes[1].set_title("Validation loss")
        fig.suptitle(title)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Loss")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    def metric_grid(rows_for_plot: list[dict[str, Any]], title: str, filename: str) -> None:
        rows_for_plot = [r for r in rows_for_plot if r.get("has_eval")]
        if not rows_for_plot:
            return
        metrics = [
            ("test_ms_ssim", "Test MS-SSIM"),
            ("test_ms_ssim_r", "Test MS-SSIM-R"),
            ("test_mse", "Test MSE"),
            ("test_psnr", "Test PSNR"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        run_labels = [r["display_name"] for r in rows_for_plot]
        for ax, (key, metric_title) in zip(axes.flat, metrics):
            metric_rows = [r for r in rows_for_plot if r.get(key) is not None]
            ax.set_title(metric_title)
            if not metric_rows:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                continue
            labels_for_metric = [r["display_name"] for r in metric_rows]
            ax.bar(labels_for_metric, [r[key] for r in metric_rows])
            ax.tick_params(axis="x", labelrotation=25)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=180)
        plt.close(fig)

    groups = [
        ("main", _main_rows(summary), "Main runs"),
        ("data_efficiency", [r for r in summary if r.get("study_group") == "data_efficiency"], "Data efficiency runs"),
        (
            "data_efficiency_100train",
            [r for r in summary if r.get("study_group") == "data_efficiency_100train"],
            "100-train data efficiency runs",
        ),
        ("ablations", [r for r in summary if r.get("study_group") == "ablations"], "Ablation runs"),
    ]
    for prefix, rows_for_group, title in groups:
        if not rows_for_group:
            continue
        history_line_plot(rows_for_group, "val_ms_ssim", f"{title}: validation MS-SSIM", "MS-SSIM", f"{prefix}_val_ms_ssim_curves.png")
        history_line_plot(rows_for_group, "val_ms_ssim_r", f"{title}: validation MS-SSIM-R", "MS-SSIM-R", f"{prefix}_val_ms_ssim_r_curves.png")
        loss_plot(rows_for_group, f"{title}: loss curves", f"{prefix}_loss_curves.png")

    metric_grid(_main_rows(summary), "Main comparison test metrics", "main_test_metrics_bars.png")
    metric_grid([r for r in summary if r.get("study_group") == "ablations"], "Ablation test metrics", "ablation_test_metrics_bars.png")
    metric_grid([r for r in summary if not r.get("is_legacy")], "Available test metrics", "test_metrics_bars.png")
    _plot_data_efficiency(summary, out_dir)
    _plot_data_efficiency(
        summary,
        out_dir,
        group="data_efficiency_100train",
        include_main_reference=False,
        filename="data_efficiency_100train.png",
        title_prefix="100-train split ",
    )
    _plot_ablation_studies(summary, out_dir)
    _plot_main_extended(summary, out_dir)


def _extract_data_seed_from_condition(condition: str | None) -> int | None:
    m = re.search(r"data_seed=(\d+)", condition or "")
    return int(m.group(1)) if m else None


def _read_per_batch_test_ms_ssim(results_csv: Path) -> list[float]:
    """Read per-batch test MS-SSIM values from a run's eval_results/results.csv."""
    vals = []
    if not results_csv.exists():
        return vals
    try:
        with results_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("split") != "test":
                    continue
                v = _safe_float(row.get("ms_ssim"))
                if v is not None and math.isfinite(v):
                    vals.append(v)
    except OSError:
        pass
    return vals


def _plot_main_extended(summary: list[dict[str, Any]], out_dir: Path) -> None:
    """Three supplementary main-result plots: batch distribution, seed consistency, per-data-seed."""
    plt = _load_pyplot()
    main = [r for r in _main_rows(summary) if r.get("has_eval") and r.get("variant_key") in {"2D", "3ch", "5ch"}]
    if not main:
        return

    variant_order = ["2D", "3ch", "5ch"]
    variant_colors = {"2D": "#4C72B0", "3ch": "#DD8452", "5ch": "#55A868"}
    data_seed_markers = {101: "o", 202: "s", 303: "^"}
    data_seed_colors  = {101: "#4C72B0", 202: "#DD8452", 303: "#55A868", None: "#888888"}

    # -----------------------------------------------------------------------
    # Plot 1: per-run batch distribution box plots (one box per run, grouped by variant)
    # Shows within-run variance and between-run (seed) spread for main runs.
    # -----------------------------------------------------------------------
    grouped_batches: dict[str, list[list[float]]] = {v: [] for v in variant_order}
    grouped_labels:  dict[str, list[str]] = {v: [] for v in variant_order}
    for row in sorted(main, key=_sort_key):
        variant = str(row.get("variant_key", ""))
        if variant not in grouped_batches:
            continue
        csv_path = Path(str(row.get("checkpoint_dir", ""))) / "eval_results" / "results.csv"
        vals = _read_per_batch_test_ms_ssim(csv_path)
        if vals:
            grouped_batches[variant].append(vals)
            grouped_labels[variant].append(str(row.get("display_name", row.get("run_id", ""))))

    any_batches = any(grouped_batches[v] for v in variant_order)
    if any_batches:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        fig.suptitle("Per-run batch MS-SSIM distribution (main runs)", fontsize=12)
        for ax, variant in zip(axes, variant_order):
            batch_lists = grouped_batches[variant]
            run_labels  = grouped_labels[variant]
            if not batch_lists:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(variant)
                continue
            ax.boxplot(batch_lists, labels=run_labels, patch_artist=True,
                       boxprops=dict(facecolor=variant_colors.get(variant, "#888888"), alpha=0.6),
                       medianprops=dict(color="black", linewidth=1.5),
                       whiskerprops=dict(linestyle="--"))
            ax.set_title(variant)
            ax.set_xlabel("Run")
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].set_ylabel("Test MS-SSIM (per batch)")
        fig.tight_layout()
        fig.savefig(out_dir / "main_per_run_distribution.png", dpi=180)
        plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 2: seed consistency — per-run MS-SSIM scatter, coloured by data_seed
    # Shows how consistent each variant is across data and training seeds.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted_dseed_labels: set[int | None] = set()
    for i, variant in enumerate(variant_order):
        for row in sorted(main, key=_sort_key):
            if str(row.get("variant_key")) != variant:
                continue
            val = _safe_float(row.get("test_ms_ssim"))
            if val is None:
                continue
            dseed = _extract_data_seed_from_condition(str(row.get("condition", "")))
            color  = data_seed_colors.get(dseed, data_seed_colors[None])
            marker = data_seed_markers.get(dseed, "x")  # type: ignore[call-overload]
            label = f"data_seed={dseed}" if dseed not in plotted_dseed_labels else None
            ax.scatter(i + (0.1 * list(data_seed_markers.keys()).index(dseed) if dseed in data_seed_markers else 0),
                       val, color=color, marker=marker, s=60, zorder=3, label=label)
            if dseed not in plotted_dseed_labels:
                plotted_dseed_labels.add(dseed)
    ax.set_xticks(range(len(variant_order)))
    ax.set_xticklabels(variant_order)
    ax.set_xlabel("Variant")
    ax.set_ylabel("Test MS-SSIM")
    ax.set_title("Seed consistency: per-run MS-SSIM for main runs\n(coloured by data seed, shape by data seed)")
    ax.grid(True, axis="y", alpha=0.25)
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "main_seed_consistency.png", dpi=180)
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Plot 3: per-data-seed grouped bar chart
    # One group per data_seed; bars are 2D / 3ch / 5ch within each group.
    # Shows whether a particular data split is harder or easier for all variants.
    # -----------------------------------------------------------------------
    by_dseed: dict[int | None, dict[str, list[float]]] = {}
    for row in main:
        dseed = _extract_data_seed_from_condition(str(row.get("condition", "")))
        val = _safe_float(row.get("test_ms_ssim"))
        if val is None:
            continue
        by_dseed.setdefault(dseed, {}).setdefault(str(row.get("variant_key")), []).append(val)

    known_seeds: list[int | None] = list(sorted(s for s in by_dseed if s is not None))
    if None in by_dseed:
        known_seeds.append(None)

    if known_seeds:
        x = range(len(known_seeds))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(6, len(known_seeds) * 2), 5))
        for vi, variant in enumerate(variant_order):
            means = []
            stds  = []
            for dseed in known_seeds:
                vals = by_dseed.get(dseed, {}).get(variant, [])
                mean, std = _mean_std(vals) if vals else (None, None)
                means.append(mean if mean is not None else 0.0)
                stds.append(std if std is not None else 0.0)
            offset = (vi - 1) * width
            bars = ax.bar([xi + offset for xi in x], means, width,
                          label=variant, color=variant_colors.get(variant, "#888888"),
                          yerr=stds, capsize=4, alpha=0.85)
            _ = bars  # suppress unused warning
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"data_seed={s}" if s is not None else "legacy/unset" for s in known_seeds])
        ax.set_ylabel("Test MS-SSIM (mean ± std across training seeds)")
        ax.set_title("Main result by data seed — are some splits harder?")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "main_per_data_seed_bars.png", dpi=180)
        plt.close(fig)


def _plot_data_efficiency(
    summary: list[dict[str, Any]],
    out_dir: Path,
    group: str = "data_efficiency",
    include_main_reference: bool = True,
    filename: str = "data_efficiency.png",
    title_prefix: str = "",
) -> None:
    rows = [
        r for r in _data_efficiency_rows(summary, include_main_reference, group)
        if r.get("n_vols") is not None and (r.get("test_ms_ssim") is not None or r.get("test_ms_ssim_r") is not None)
    ]
    if not rows:
        print(f"No completed {group} runs found; skipping {filename}.")
        return

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        vk = r.get("variant_key") or "unknown"
        by_variant.setdefault(vk, []).append(r)

    by_variant = {k: v for k, v in by_variant.items() if len(v) >= 2}
    if not by_variant:
        print("Not enough data efficiency runs per variant to plot; skipping.")
        return

    plt = _load_pyplot()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (variant, vrows) in enumerate(sorted(by_variant.items())):
        vrows = sorted(vrows, key=lambda r: r.get("n_vols", 20))
        c = colors[i % len(colors)]
        for ax, key in [(axes[0], "test_ms_ssim"), (axes[1], "test_ms_ssim_r")]:
            metric_rows = [r for r in vrows if r.get(key) is not None]
            if len(metric_rows) < 2:
                continue
            ax.plot(
                [r["n_vols"] for r in metric_rows],
                [r[key] for r in metric_rows],
                marker="o",
                linewidth=1.7,
                markersize=5,
                label=variant,
                color=c,
            )

    all_ms_ssim = [r["test_ms_ssim"] for r in rows if r.get("test_ms_ssim") is not None]
    all_ms_ssim_r = [r["test_ms_ssim_r"] for r in rows if r.get("test_ms_ssim_r") is not None]
    for ax, vals, title, ylabel in [
        (axes[0], all_ms_ssim, f"{title_prefix}Test MS-SSIM vs Training Volumes", "MS-SSIM"),
        (axes[1], all_ms_ssim_r, f"{title_prefix}Test MS-SSIM-R vs Training Volumes", "MS-SSIM-R"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Training volumes")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({int(r["n_vols"]) for r in rows if r.get("n_vols") is not None}))
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        if vals:
            margin = max((max(vals) - min(vals)) * 0.15, 0.005)
            ax.set_ylim(min(vals) - margin, max(vals) + margin)

    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def _plot_ablation_studies(summary: list[dict[str, Any]], out_dir: Path) -> None:
    studies = _ablation_studies(summary)
    plt = _load_pyplot()

    filename_map = {
        "Study A - 2D slice stride": "ablation_study_a_slice_stride.png",
        "Study B - 3ch neighbor stride": "ablation_study_b_neighbor_stride.png",
        "Study C - 3ch crop coverage": "ablation_study_c_crop_coverage.png",
    }
    metrics = [
        ("best_val_ms_ssim", "Best val MS-SSIM"),
        ("test_ms_ssim", "Test MS-SSIM"),
        ("test_ms_ssim_r", "Test MS-SSIM-R"),
        ("test_psnr", "Test PSNR"),
    ]
    for title, rows in studies.items():
        rows = [r for r in rows if r.get("best_val_ms_ssim") is not None or r.get("has_eval")]
        if not rows:
            continue
        labels = [r["display_name"].replace("ABL ", "").replace("DE ", "Baseline ") for r in rows]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (key, metric_title) in zip(axes.flat, metrics):
            metric_rows = [(label, row[key]) for label, row in zip(labels, rows) if row.get(key) is not None]
            ax.set_title(metric_title)
            if not metric_rows:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                continue
            ax.bar([x[0] for x in metric_rows], [x[1] for x in metric_rows])
            ax.tick_params(axis="x", labelrotation=20)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_dir / filename_map[title], dpi=180)
        plt.close(fig)


def _generated_plots(out_dir: Path) -> list[Path]:
    return sorted(path for path in (out_dir / name for name in PLOT_FILENAMES) if path.exists())


def _route_plot_files(index_dir: Path, dirs: dict[str, Path]) -> list[Path]:
    routed: list[Path] = []
    for group, filenames in PLOT_ROUTES.items():
        target_dir = dirs[group]
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            src = index_dir / filename
            if not src.exists():
                continue
            dst = target_dir / filename
            if src.resolve() != dst.resolve():
                src.replace(dst)
            routed.append(dst)
    return sorted(routed)


def _assert_plots_generated(out_dir: Path) -> list[Path]:
    plots = _generated_plots(out_dir)
    if not plots:
        raise RuntimeError(
            f"No PNG plots were generated in {out_dir}. "
            "This usually means matplotlib is missing or no plottable history/evaluation data was found."
        )
    return plots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write tables/report only. By default plots are required and missing matplotlib is an error.",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    summary_root = (args.out_dir or (_experiments_root(project_root) / "summaries")).resolve()
    dirs = _summary_dirs(summary_root)
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    index_dir = dirs["index"]

    runs = _discover_runs(project_root)
    history = _history_rows(project_root, runs)
    all_summary = _summary_rows(project_root, runs, history)
    summary = _included_summary_rows(all_summary)
    included_run_ids = {row["run_id"] for row in summary}
    history = [row for row in history if row["run_id"] in included_run_ids]

    _write_csv(index_dir / "impeccable_training_history.csv", history, HISTORY_FIELDS)
    _write_csv(index_dir / "impeccable_run_summary.csv", summary, SUMMARY_FIELDS)
    _write_split_csvs(dirs, summary)
    stats_rows = _compute_stats_tests(summary)
    if stats_rows:
        _write_csv(
            dirs["main_experiment"] / "main_stats_tests.csv",
            stats_rows,
            ["comparison", "metric", "mean_2d", "mean_variant", "delta",
             "cohens_d", "wilcoxon_stat", "wilcoxon_p", "ttest_stat", "ttest_p", "n_seeds", "note"],
        )
    _write_report(index_dir / "impeccable_report.md", summary)
    plots: list[Path] = []
    if not args.no_plots:
        _plot(history, summary, index_dir)
        _assert_plots_generated(index_dir)
        plots = _route_plot_files(index_dir, dirs)

    print(f"Discovered {len(runs)} run folder(s).")
    print(f"Included {len(summary)} run(s) after excluding non-main control variants.")
    print(f"Wrote summaries to: {summary_root}")
    print(f"  - {index_dir / 'impeccable_run_summary.csv'}")
    print(f"  - {index_dir / 'impeccable_training_history.csv'}")
    print(f"  - {index_dir / 'impeccable_report.md'}")
    print(f"  - {dirs['main_experiment'] / 'main_comparison.csv'}")
    print(f"  - {dirs['main_experiment'] / 'main_replicate_summary.csv'}")
    if stats_rows:
        print(f"  - {dirs['main_experiment'] / 'main_stats_tests.csv'}")
    elif not _SCIPY_AVAILABLE:
        print("  scipy not found — stats tests skipped. Install with: pip install scipy")
    print(f"  - {dirs['data_efficiency'] / 'data_efficiency_summary.csv'}")
    print(f"  - {dirs['data_efficiency_100train'] / 'data_efficiency_100train_summary.csv'}")
    print(f"  - {dirs['data_efficiency_100train'] / 'plateau_decision.csv'}")
    print(f"  - {dirs['data_efficiency_100train_v2'] / 'data_efficiency_100train_v2_summary.csv'}")
    print(f"  - {dirs['data_efficiency_100train_v2'] / 'plateau_decision.csv'}")
    print(f"  - {dirs['data_efficiency_100train_channel_window_v2'] / 'data_efficiency_100train_channel_window_v2_summary.csv'}")
    print(f"  - {dirs['ablations'] / 'ablation_summary.csv'}")
    if args.no_plots:
        print("Plots skipped because --no-plots was set.")
    else:
        print(f"Generated {len(plots)} plot file(s):")
        for plot in plots:
            print(f"  - {plot}")
    if not history:
        print("No per-epoch history found. Copy recovered logs or use future runs with history.csv.")


if __name__ == "__main__":
    main()
