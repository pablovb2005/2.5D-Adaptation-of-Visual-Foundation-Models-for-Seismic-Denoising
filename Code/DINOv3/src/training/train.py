"""Train the DINOv3 denoiser (PEFT + LoRA) on the seismic dataset.

Usage:
    python training/train.py --config configs/dinov3_vits_2d.yaml

The config path is resolved relative to Code/DINOv3/src/.
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Make src/ importable regardless of working directory.
SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from data.dataset import SeismicDataset
from models.pipeline import DINOv3Denoiser, _count_params
from training.loss import DenoisingLoss
from evaluation.common.metrics import compute_all

TIMING_SCHEMA_VERSION = 1
TIMING_FIELDS = [
    "epoch",
    "total_epochs",
    "global_step_start",
    "global_step_end",
    "n_train_batches",
    "n_val_batches",
    "train_step_time_s",
    "validation_time_s",
    "checkpoint_time_s",
    "epoch_total_time_s",
    "timing_schema_version",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Linear warm-up then cosine decay to zero.

    warmup_steps and total_steps are derived from training.warmup_epochs and
    training.epochs in the run config multiplied by steps-per-epoch.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


def _resolve_path(cfg_dir: Path, rel: str) -> Path:
    return (cfg_dir / rel).resolve()


def _int_cfg(cfg: dict, key: str, default: int) -> int:
    value = cfg.get(key)
    return default if value is None else int(value)


def _bool_cfg(cfg: dict, key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the LR from the optimizer state, avoiding scheduler state quirks."""
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0].get("lr", float("nan")))


def _validate_model_data_channels(in_chans: int, *datasets: object) -> list[int] | None:
    offsets_seen: list[int] | None = None
    for ds in datasets:
        offsets = getattr(ds, "offsets", None)
        if offsets is None:
            continue
        offsets_list = list(offsets)
        if offsets_seen is None:
            offsets_seen = offsets_list
        elif offsets_seen != offsets_list:
            raise ValueError(
                "Image Impeccable train/validation datasets produced different "
                f"offsets: {offsets_seen} vs {offsets_list}"
            )
        if in_chans != len(offsets_list):
            mode = getattr(ds, "mode", "<unknown>")
            raise ValueError(
                f"model.in_chans={in_chans} does not match data.mode={mode!r}, "
                f"which produces {len(offsets_list)} channel(s) with offsets {offsets_list}. "
                f"Set model.in_chans to {len(offsets_list)}."
            )
    return offsets_seen


def _atomic_torch_save(obj: object, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _sync_cuda_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _trim_csv_after_epoch(csv_path: Path, max_epoch: int) -> list[str] | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        rows = []
        for row in reader:
            try:
                epoch = int(row.get("epoch", ""))
            except ValueError:
                continue
            if epoch <= max_epoch:
                rows.append(row)

    tmp_path = csv_path.with_name(f".{csv_path.name}.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, csv_path)
    return list(reader.fieldnames)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(cfg: dict, cfg_path: Path):
    cfg_dir = cfg_path.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Paths -------------------------------------------------------------------
    project_root = SRC.parent  # Code/DINOv3/
    repo_dir = project_root / "external" / "dinov3"
    weights = _resolve_path(cfg_dir, cfg["model"]["weights"])
    ckpt_dir = _resolve_path(cfg_dir, cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, ckpt_dir / "config.yaml")

    # Data --------------------------------------------------------------------
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "sfm")

    if source == "image_impeccable":
        from data.image_impeccable import ThinkOnwardDataset
        ii_root     = _resolve_path(cfg_dir, data_cfg["root_dir"])
        ii_mode     = str(data_cfg["mode"])
        ii_n_train  = int(data_cfg.get("n_train", 20))
        ii_n_val    = int(data_cfg.get("n_val", 5))
        ii_n_test   = int(data_cfg.get("n_test", 5))
        ii_stride   = int(data_cfg.get("slice_stride", 5))
        ii_train_stride = _int_cfg(data_cfg, "train_slice_stride", ii_stride)
        ii_eval_stride = _int_cfg(data_cfg, "eval_slice_stride", ii_stride)
        ii_val_stride = _int_cfg(data_cfg, "val_slice_stride", ii_eval_stride)
        ii_crop     = int(data_cfg.get("crop_size", 224))
        ii_seed     = int(data_cfg["seed"])
        _tsn = data_cfg.get("train_subset_n", None)
        ii_train_subset_n = int(_tsn) if _tsn is not None else None
        ii_neighbor_stride   = int(data_cfg.get("neighbor_stride", 1))
        ii_train_crop_mode   = str(data_cfg.get("train_crop_mode", data_cfg.get("crop_mode", "center")))
        ii_eval_crop_mode    = str(data_cfg.get("eval_crop_mode", "center"))
        ii_crop_seed         = int(data_cfg.get("crop_seed", data_cfg.get("seed", 42)))
        ii_shuffle_neighbors = bool(data_cfg.get("shuffle_neighbors", False))
        ii_repeat_center     = bool(data_cfg.get("repeat_center", False))
        ii_cache_volumes     = bool(data_cfg.get("cache_volumes", False))
        ii_slice_orientation = str(data_cfg.get("slice_orientation", "auto"))
        train_ds = ThinkOnwardDataset(
            root_dir=ii_root, mode=ii_mode, split="train",
            n_train=ii_n_train, n_val=ii_n_val, n_test=ii_n_test,
            slice_stride=ii_train_stride, crop_size=ii_crop, seed=ii_seed,
            train_subset_n=ii_train_subset_n,
            neighbor_stride=ii_neighbor_stride,
            crop_mode=ii_train_crop_mode,
            crop_seed=ii_crop_seed,
            shuffle_neighbors=ii_shuffle_neighbors,
            repeat_center=ii_repeat_center,
            cache_volumes=ii_cache_volumes,
            slice_orientation=ii_slice_orientation,
        )
        val_ds = ThinkOnwardDataset(
            root_dir=ii_root, mode=ii_mode, split="val",
            n_train=ii_n_train, n_val=ii_n_val, n_test=ii_n_test,
            slice_stride=ii_val_stride, crop_size=ii_crop, seed=ii_seed,
            neighbor_stride=ii_neighbor_stride,
            crop_mode=ii_eval_crop_mode,
            crop_seed=ii_crop_seed,
            shuffle_neighbors=ii_shuffle_neighbors,
            repeat_center=ii_repeat_center,
            cache_volumes=ii_cache_volumes,
            slice_orientation=ii_slice_orientation,
        )
    else:
        label_dir   = _resolve_path(cfg_dir, data_cfg["label_dir"])
        seismic_dir = _resolve_path(cfg_dir, data_cfg["seismic_dir"])
        train_ds = SeismicDataset(
            seismic_dir, label_dir,
            val_split=data_cfg["val_split"], train=True,
            seed=data_cfg["seed"], mode=data_cfg["mode"],
        )
        val_ds = SeismicDataset(
            seismic_dir, label_dir,
            val_split=data_cfg["val_split"], train=False,
            seed=data_cfg["seed"], mode=data_cfg["mode"],
        )

    m_cfg = cfg["model"]
    model_in_chans = int(m_cfg.get("in_chans", 3))
    data_offsets = None
    if source == "image_impeccable":
        data_offsets = _validate_model_data_channels(model_in_chans, train_ds, val_ds)

    tr_cfg = cfg["training"]
    total_epochs = int(tr_cfg["epochs"])
    val_interval = max(1, int(tr_cfg.get("val_interval", tr_cfg.get("validation_interval", 1))))
    resume = _bool_cfg(tr_cfg, "resume", False)
    max_runtime_minutes = tr_cfg.get("max_runtime_minutes")
    max_runtime_seconds = (
        float(max_runtime_minutes) * 60.0 if max_runtime_minutes is not None else None
    )
    train_seed = int(tr_cfg.get("seed", data_cfg.get("seed", 42)))
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    nw = int(tr_cfg.get("num_workers", 4))
    pin_memory = _bool_cfg(tr_cfg, "pin_memory", torch.cuda.is_available())
    persistent_workers = _bool_cfg(tr_cfg, "persistent_workers", False) and nw > 0
    if source == "image_impeccable" and ii_shuffle_neighbors and persistent_workers:
        print("shuffle_neighbors needs fresh workers each epoch; forcing persistent_workers=False")
        persistent_workers = False
    prefetch_factor = tr_cfg.get("prefetch_factor")
    log_interval_batches = int(tr_cfg.get("log_interval_batches", 0) or 0)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(train_seed)
    train_loader_kwargs = dict(
        batch_size=tr_cfg["batch_size"],
        shuffle=True,
        num_workers=nw,
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    val_loader_kwargs = dict(
        batch_size=tr_cfg["batch_size"],
        shuffle=False,
        num_workers=nw,
        pin_memory=pin_memory,
    )
    if nw > 0:
        train_loader_kwargs["persistent_workers"] = persistent_workers
        val_loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            train_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
            val_loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    print(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")
    if source == "image_impeccable":
        print(
            "Image Impeccable sampling: "
            f"train_slice_stride={ii_train_stride}, "
            f"val_slice_stride={ii_val_stride}, "
            f"val_interval={val_interval}"
        )

    # Model -------------------------------------------------------------------
    model = DINOv3Denoiser(
        repo_dir=repo_dir,
        weights_path=weights,
        model_name=m_cfg["name"],
        in_chans=model_in_chans,
        lora_rank=m_cfg["lora_rank"],
        lora_alpha=m_cfg["lora_alpha"],
        lora_dropout=m_cfg["lora_dropout"],
        lora_targets=m_cfg["lora_targets"],
        patch_emb_init=str(m_cfg.get("patch_emb_init", "mixed")),
        full_finetune=bool(m_cfg.get("full_finetune", False)),
    ).to(device)

    total, trainable = _count_params(model)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")
    with (ckpt_dir / "run_meta.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "config_path": str(cfg_path),
                "model": {
                    "name": m_cfg["name"],
                    "in_chans": model_in_chans,
                    "lora_rank": int(m_cfg["lora_rank"]),
                    "lora_alpha": int(m_cfg["lora_alpha"]),
                    "lora_dropout": float(m_cfg["lora_dropout"]),
                    "lora_targets": list(m_cfg["lora_targets"]),
                    "patch_emb_init": str(m_cfg.get("patch_emb_init", "mixed")),
                    "full_finetune": bool(m_cfg.get("full_finetune", False)),
                },
                "data": {
                    "source": source,
                    "mode": data_cfg.get("mode"),
                    "seed": data_cfg.get("seed"),
                    "n_train": data_cfg.get("n_train"),
                    "n_val": data_cfg.get("n_val"),
                    "n_test": data_cfg.get("n_test"),
                    "slice_stride": data_cfg.get("slice_stride"),
                    "train_slice_stride": data_cfg.get("train_slice_stride"),
                    "eval_slice_stride": data_cfg.get("eval_slice_stride"),
                    "val_slice_stride": data_cfg.get("val_slice_stride"),
                    "test_slice_stride": data_cfg.get("test_slice_stride"),
                    "neighbor_stride": data_cfg.get("neighbor_stride"),
                    "train_crop_mode": data_cfg.get("train_crop_mode"),
                    "eval_crop_mode": data_cfg.get("eval_crop_mode"),
                    "crop_seed": data_cfg.get("crop_seed"),
                    "shuffle_neighbors": data_cfg.get("shuffle_neighbors"),
                    "repeat_center": data_cfg.get("repeat_center"),
                    "cache_volumes": ii_cache_volumes if source == "image_impeccable" else None,
                    "offsets": data_offsets,
                    "channels": len(data_offsets) if data_offsets is not None else None,
                },
                "training": {
                    "epochs": total_epochs,
                    "val_interval": val_interval,
                    "seed": train_seed,
                    "resume": resume,
                    "max_runtime_minutes": max_runtime_minutes,
                    "batch_size": int(tr_cfg["batch_size"]),
                    "lr": float(tr_cfg["lr"]),
                    "weight_decay": float(tr_cfg["weight_decay"]),
                    "warmup_epochs": int(tr_cfg["warmup_epochs"]),
                    "loss_lambda": float(tr_cfg["loss_lambda"]),
                    "num_workers": nw,
                    "pin_memory": pin_memory,
                    "persistent_workers": persistent_workers,
                    "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else None,
                    "log_interval_batches": log_interval_batches,
                },
                "dataset": {
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                },
                "params": {
                    "total": int(total),
                    "trainable": int(trainable),
                    "trainable_pct": float(100 * trainable / total),
                },
                "timing": {
                    "schema_version": TIMING_SCHEMA_VERSION,
                    "artifact": "training_timing.csv",
                    "primary_metric": "train_step_time_s",
                    "primary_metric_includes": [
                        "DataLoader wait during training batches",
                        "host-to-device transfer for training batches",
                        "forward pass",
                        "loss computation",
                        "backward pass",
                        "gradient clipping",
                        "optimizer step",
                        "scheduler step",
                    ],
                    "primary_metric_excludes": [
                        "dataset/model construction before the epoch loop",
                        "validation",
                        "checkpoint and CSV writes",
                        "SLURM queue time",
                        "temporary environment setup",
                        "post-training test evaluation",
                    ],
                    "cuda_synchronized": device.type == "cuda",
                    "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
                },
            },
            f,
            sort_keys=False,
        )

    # Optimiser + scheduler ---------------------------------------------------
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tr_cfg["lr"],
        weight_decay=tr_cfg["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = tr_cfg["warmup_epochs"] * steps_per_epoch
    scheduler = _cosine_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = DenoisingLoss(lam=tr_cfg["loss_lambda"]).to(device)

    # Training loop -----------------------------------------------------------
    best_val_ms_ssim = -1.0
    global_step = 0
    start_epoch = 1
    best_path = ckpt_dir / "best.pt"
    last_path = ckpt_dir / "last.pt"
    history_path = ckpt_dir / "history.csv"
    timing_path = ckpt_dir / "training_timing.csv"
    resume_epoch: int | None = None
    history_fields = [
        "epoch",
        "total_epochs",
        "train_loss",
        "val_loss",
        "val_ms_ssim",
        "val_ms_ssim_r",
        "lr",
        "epoch_time_s",
    ]

    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=device)
        best_val_ms_ssim = float(best_ckpt.get("val_ms_ssim", best_val_ms_ssim))

    if resume and last_path.exists():
        last_ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(last_ckpt["model"])
        if "optimizer" in last_ckpt:
            optimizer.load_state_dict(last_ckpt["optimizer"])
        if "scheduler" in last_ckpt:
            scheduler.load_state_dict(last_ckpt["scheduler"])
        global_step = int(last_ckpt.get("global_step", 0))
        best_val_ms_ssim = float(last_ckpt.get("best_val_ms_ssim", best_val_ms_ssim))
        resume_epoch = int(last_ckpt["epoch"])
        start_epoch = resume_epoch + 1
        print(f"Resuming from {last_path} at epoch {start_epoch}/{total_epochs}")
    elif last_path.exists():
        print(f"Existing {last_path} found, but training.resume is false; starting from epoch 1")

    if start_epoch > total_epochs:
        print(f"Run already reached epoch {total_epochs}; nothing to train.")
        print(f"Checkpoints saved to: {ckpt_dir}")
        return

    write_history_header = True
    if resume and history_path.exists() and resume_epoch is not None:
        _existing_fields = _trim_csv_after_epoch(history_path, resume_epoch)
        if _existing_fields:
            history_fields = _existing_fields
            write_history_header = False
    if write_history_header:
        with history_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields)
            writer.writeheader()

    timing_fields = list(TIMING_FIELDS)
    write_timing_header = True
    if resume and timing_path.exists() and resume_epoch is not None:
        _existing_timing_fields = _trim_csv_after_epoch(timing_path, resume_epoch)
        if _existing_timing_fields and set(TIMING_FIELDS).issubset(_existing_timing_fields):
            timing_fields = _existing_timing_fields
            write_timing_header = False
    if write_timing_header:
        with timing_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timing_fields)
            writer.writeheader()

    run_start_time = time.perf_counter()
    epoch_durations = []
    last_finished_epoch = start_epoch - 1

    for epoch in range(start_epoch, total_epochs + 1):
        _sync_cuda_if_needed(device)
        epoch_start_time = time.perf_counter()
        global_step_start = global_step
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        _sync_cuda_if_needed(device)
        train_step_start_time = time.perf_counter()
        for batch_idx, (noisy, clean) in enumerate(train_loader, start=1):
            noisy, clean = noisy.to(device), clean.to(device)
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            global_step += 1
            if log_interval_batches and (
                batch_idx % log_interval_batches == 0 or batch_idx == len(train_loader)
            ):
                _sync_cuda_if_needed(device)
                elapsed_s = time.perf_counter() - train_step_start_time
                print(
                    f"  epoch {epoch}/{total_epochs} batch {batch_idx}/{len(train_loader)} | "
                    f"train_loss_avg={train_loss / batch_idx:.4f} | "
                    f"elapsed={elapsed_s / 60:.1f} min | "
                    f"avg_batch={elapsed_s / batch_idx:.2f}s",
                    flush=True,
                )

        _sync_cuda_if_needed(device)
        train_step_time_s = time.perf_counter() - train_step_start_time
        train_loss /= len(train_loader)
        current_lr = _current_lr(optimizer)

        # Validation ----------------------------------------------------------
        should_validate = val_interval == 1 or epoch == 1 or epoch == total_epochs or epoch % val_interval == 0
        val_loss = None
        val_ms_ssim = None
        val_ms_ssim_r = None
        validation_time_s = 0.0
        n_val_batches = 0

        if should_validate:
            _sync_cuda_if_needed(device)
            validation_start_time = time.perf_counter()
            model.eval()
            val_loss_sum = 0.0
            ms_ssim_sum = 0.0
            ms_ssim_r_sum = 0.0

            with torch.no_grad():
                for noisy, clean in val_loader:
                    n_val_batches += 1
                    noisy, clean = noisy.to(device), clean.to(device)
                    pred = model(noisy)
                    val_loss_sum += loss_fn(pred, clean).item()
                    m = compute_all(pred.cpu(), clean.cpu(), noisy.cpu())
                    ms_ssim_sum += m["ms_ssim"]
                    ms_ssim_r_sum += m["ms_ssim_r"]

            _sync_cuda_if_needed(device)
            validation_time_s = time.perf_counter() - validation_start_time
            val_loss = val_loss_sum / len(val_loader)
            val_ms_ssim = ms_ssim_sum / len(val_loader)
            val_ms_ssim_r = ms_ssim_r_sum / len(val_loader)

            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_MS-SSIM={val_ms_ssim:.4f} | val_MS-SSIM-R={val_ms_ssim_r:.4f} | "
                f"lr={current_lr:.2e}"
            )
        else:
            print(
                f"Epoch {epoch:3d}/{total_epochs} | "
                f"train_loss={train_loss:.4f} | validation=skipped "
                f"(val_interval={val_interval}) | lr={current_lr:.2e}"
            )

        epoch_elapsed_s = time.perf_counter() - epoch_start_time
        checkpoint_start_time = time.perf_counter()
        with history_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history_fields, extrasaction="ignore")
            writer.writerow(
                {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_ms_ssim": val_ms_ssim,
                    "val_ms_ssim_r": val_ms_ssim_r,
                    "lr": current_lr,
                    "epoch_time_s": epoch_elapsed_s,
                }
            )

        # Checkpointing -------------------------------------------------------
        last_checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "val_ms_ssim": val_ms_ssim,
            "val_ms_ssim_r": val_ms_ssim_r,
            "validated": should_validate,
            "best_val_ms_ssim": best_val_ms_ssim,
        }

        if val_ms_ssim is not None and val_ms_ssim > best_val_ms_ssim:
            best_val_ms_ssim = val_ms_ssim
            _atomic_torch_save(
                {"epoch": epoch, "model": model.state_dict(), "val_ms_ssim": val_ms_ssim, "val_ms_ssim_r": val_ms_ssim_r},
                best_path,
            )
            print(f"  New best val MS-SSIM: {best_val_ms_ssim:.4f} (saved)")

        last_checkpoint["best_val_ms_ssim"] = best_val_ms_ssim
        _atomic_torch_save(last_checkpoint, last_path)
        last_finished_epoch = epoch
        _sync_cuda_if_needed(device)
        checkpoint_time_s = time.perf_counter() - checkpoint_start_time
        epoch_total_time_s = time.perf_counter() - epoch_start_time
        epoch_durations.append(epoch_total_time_s)

        with timing_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=timing_fields, extrasaction="ignore")
            writer.writerow(
                {
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "global_step_start": global_step_start,
                    "global_step_end": global_step,
                    "n_train_batches": len(train_loader),
                    "n_val_batches": n_val_batches,
                    "train_step_time_s": train_step_time_s,
                    "validation_time_s": validation_time_s,
                    "checkpoint_time_s": checkpoint_time_s,
                    "epoch_total_time_s": epoch_total_time_s,
                    "timing_schema_version": TIMING_SCHEMA_VERSION,
                }
            )

        if max_runtime_seconds is not None and epoch < total_epochs:
            elapsed = time.perf_counter() - run_start_time
            avg_epoch = sum(epoch_durations) / len(epoch_durations)
            remaining_budget = max_runtime_seconds - elapsed
            if remaining_budget < avg_epoch * 1.2:
                print(
                    "Stopping cleanly before walltime: "
                    f"elapsed={elapsed / 60:.1f} min, "
                    f"avg_epoch={avg_epoch / 60:.1f} min, "
                    f"next_epoch={epoch + 1}"
                )
                break

    best_val_text = f"{best_val_ms_ssim:.4f}" if best_val_ms_ssim >= 0 else "NA"
    print(
        f"\nTraining stopped at epoch {last_finished_epoch}/{total_epochs}. "
        f"Best val MS-SSIM: {best_val_text}"
    )
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config (relative to Code/DINOv3/src/)")
    args = parser.parse_args()

    cfg_path = (SRC / args.config).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train(cfg, cfg_path)
