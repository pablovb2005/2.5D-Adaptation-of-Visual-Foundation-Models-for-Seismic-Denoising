"""Benchmark inference latency and throughput for trained checkpoints.

Measures wall-clock time per slice (ms) and throughput (slices/s) for each
checkpoint using synthetic tensors of the correct shape. No dataset I/O —
this isolates model compute.

Usage (on DAIC, use $PY310 not the system python):
    $PY310 evaluation/benchmark_inference.py \\
        --config configs/dinov3_vits_2d_impeccable_..._daic.yaml \\
        --checkpoint /path/to/best.pt

    $PY310 evaluation/benchmark_inference.py --all-main-runs

Outputs:
    experiments/summaries/timing/inference_benchmark.csv
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

import torch
import yaml

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from models.pipeline import DINOv3Denoiser

PROJECT_ROOT = default_project_root(__file__)  # Code/DINOv3/src -> Code/DINOv3 -> Code -> RP
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "summaries" / "timing"

# Main replicate run directories relative to experiments/runs/
MAIN_RUNS = [
    ("2d",  "impeccable_repeated_stride5_lora_r16",              "seed42_run01", "configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed42_run01_daic.yaml"),
    ("2d",  "impeccable_repeated_stride5_lora_r16",              "seed43_run02", "configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed43_run02_daic.yaml"),
    ("2d",  "impeccable_repeated_stride5_lora_r16",              "seed44_run03", "configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed44_run03_daic.yaml"),
    ("3ch", "impeccable_neighbors3_stride5_lora_r16",            "seed42_run01", "configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_seed42_run01_daic.yaml"),
    ("3ch", "impeccable_neighbors3_stride5_lora_r16",            "seed43_run02", "configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_seed43_run02_daic.yaml"),
    ("3ch", "impeccable_neighbors3_stride5_lora_r16",            "seed44_run03", "configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_seed44_run03_daic.yaml"),
    ("5ch", "impeccable_neighbors5_stride5_patch_emb_lora_r16",  "seed42_run01", "configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_seed42_run01_daic.yaml"),
    ("5ch", "impeccable_neighbors5_stride5_patch_emb_lora_r16",  "seed43_run02", "configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_seed43_run02_daic.yaml"),
    ("5ch", "impeccable_neighbors5_stride5_patch_emb_lora_r16",  "seed44_run03", "configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_seed44_run03_daic.yaml"),
]

OUTPUT_FIELDS = [
    "variant", "run_id", "job_id",
    "batch_size",
    "n_warmup", "n_iters",
    "latency_mean_ms", "latency_std_ms", "latency_p50_ms", "latency_p95_ms",
    "throughput_slices_per_s",
    "gpu_memory_allocated_mb", "gpu_memory_reserved_mb",
    "device", "in_chans", "total_params", "trainable_params",
]


def _load_config(cfg_path: Path) -> dict:
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def _load_model(cfg: dict, cfg_path: Path, ckpt_path: Path, device: torch.device) -> DINOv3Denoiser:
    project_root = SRC.parent
    repo_dir = project_root / "external" / "dinov3"
    weights_file = (cfg_path.parent / cfg["model"]["weights"]).resolve()
    m_cfg = cfg["model"]

    model = DINOv3Denoiser(
        repo_dir=repo_dir,
        weights_path=weights_file,
        model_name=m_cfg["name"],
        in_chans=int(m_cfg.get("in_chans", 3)),
        lora_rank=m_cfg["lora_rank"],
        lora_alpha=m_cfg["lora_alpha"],
        lora_dropout=m_cfg["lora_dropout"],
        lora_targets=m_cfg["lora_targets"],
        patch_emb_init=str(m_cfg.get("patch_emb_init", "mixed")),
        full_finetune=bool(m_cfg.get("full_finetune", False)),
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _resolve_main_run_dir(runs_root: Path, family: str, variant: str, run_id: str) -> Path:
    """Support the current runs/ layout and the legacy main-run layout."""
    relative_run = Path(family) / variant / run_id
    candidates = [runs_root / relative_run]
    if runs_root.name == "runs":
        candidates.append(runs_root.parent / relative_run)
    else:
        candidates.append(runs_root / "runs" / relative_run)

    for candidate in candidates:
        if (candidate / "best.pt").exists():
            return candidate
    return candidates[0]


def benchmark_model(
    model: torch.nn.Module,
    in_chans: int,
    batch_size: int,
    n_warmup: int,
    n_iters: int,
    device: torch.device,
) -> dict:
    import numpy as np

    model.eval()
    x = torch.randn(batch_size, in_chans, 224, 224, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    # Reset memory stats after warm-up
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Timed iterations
    batch_times: list[float] = []
    with torch.no_grad():
        for _ in range(n_iters):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_times.append(time.perf_counter() - t0)

    per_slice_ms = np.array(batch_times) * 1000.0 / batch_size

    mem_alloc_mb = mem_reserved_mb = 0.0
    if device.type == "cuda":
        mem_alloc_mb   = torch.cuda.max_memory_allocated(device)  / 1e6
        mem_reserved_mb = torch.cuda.max_memory_reserved(device)  / 1e6

    return {
        "latency_mean_ms":         round(float(per_slice_ms.mean()),    3),
        "latency_std_ms":          round(float(per_slice_ms.std()),     3),
        "latency_p50_ms":          round(float(np.percentile(per_slice_ms, 50)), 3),
        "latency_p95_ms":          round(float(np.percentile(per_slice_ms, 95)), 3),
        "throughput_slices_per_s": round(batch_size / float(np.mean(batch_times)), 1),
        "gpu_memory_allocated_mb": round(mem_alloc_mb,   1),
        "gpu_memory_reserved_mb":  round(mem_reserved_mb, 1),
    }


def run_single(
    cfg_path: Path,
    ckpt_path: Path,
    batch_sizes: list[int],
    n_warmup: int,
    n_iters: int,
    out_dir: Path,
    variant: str = "",
    run_id: str = "",
    append: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Config: {cfg_path}")
    print(f"Checkpoint: {ckpt_path}")

    cfg = _load_config(cfg_path)
    in_chans = int(cfg["model"].get("in_chans", 3))

    model = _load_model(cfg, cfg_path, ckpt_path, device)
    total_params, trainable_params = _count_params(model)

    rows = []
    for bs in batch_sizes:
        print(f"  Benchmarking batch_size={bs}, {n_warmup} warm-up + {n_iters} timed iters ...", flush=True)
        stats = benchmark_model(model, in_chans, bs, n_warmup, n_iters, device)
        row = {
            "variant":                  variant or cfg_path.stem,
            "run_id":                   run_id,
            "job_id":                   ckpt_path.parent.name,
            "batch_size":               bs,
            "n_warmup":                 n_warmup,
            "n_iters":                  n_iters,
            "device":                   str(device),
            "in_chans":                 in_chans,
            "total_params":             total_params,
            "trainable_params":         trainable_params,
            **stats,
        }
        rows.append(row)
        print(
            f"    latency={stats['latency_mean_ms']:.2f}±{stats['latency_std_ms']:.2f} ms/slice  "
            f"throughput={stats['throughput_slices_per_s']:.0f} slices/s  "
            f"GPU_alloc={stats['gpu_memory_allocated_mb']:.0f} MB"
        )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "inference_benchmark.csv"
    mode = "a" if append and out_csv.exists() else "w"
    with out_csv.open(mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=Path, help="Path to YAML config (relative to src/)")
    parser.add_argument("--checkpoint", type=Path, help="Path to best.pt checkpoint")
    parser.add_argument(
        "--all-main-runs", action="store_true",
        help="Benchmark all 9 main replicates (2D/3ch/5ch × seed42/43/44)"
    )
    parser.add_argument(
        "--batch-sizes", default="1,16",
        help="Comma-separated batch sizes to benchmark (default: 1,16)"
    )
    parser.add_argument("--n-warmup", type=int, default=50, help="Warm-up iterations (default: 50)")
    parser.add_argument("--n-iters",  type=int, default=200, help="Timed iterations (default: 200)")
    parser.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--runs-root", type=Path, default=None,
        help="Root of experiments/runs/ directory (default: PROJECT_ROOT/experiments/runs). "
             "Override on DAIC to point at staff-bulk: $STUDENT_DIR/experiments/runs"
    )
    args = parser.parse_args()

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    if args.all_main_runs:
        runs_root = args.runs_root or (PROJECT_ROOT / "experiments" / "runs")
        first = True
        measured_runs = 0
        for family, variant, run_id, cfg_rel in MAIN_RUNS:
            run_dir = _resolve_main_run_dir(runs_root, family, variant, run_id)
            ckpt = run_dir / "best.pt"
            cfg_path = (SRC / cfg_rel).resolve()

            if not ckpt.exists():
                print(f"Skipping {run_dir}: best.pt not found")
                continue
            if not cfg_path.exists():
                print(f"Skipping {run_dir}: config not found at {cfg_path}")
                continue

            run_single(
                cfg_path, ckpt, batch_sizes,
                args.n_warmup, args.n_iters, args.out_dir,
                variant=family, run_id=run_id,
                append=not first,
            )
            first = False
            measured_runs += 1

        if measured_runs == 0:
            raise SystemExit(
                "No main runs were benchmarked. Check --runs-root and the remote checkpoint layout."
            )
        print(f"\nResults saved to {args.out_dir / 'inference_benchmark.csv'}")

    elif args.config and args.checkpoint:
        cfg_path = (SRC / args.config).resolve() if not args.config.is_absolute() else args.config
        run_single(
            cfg_path, args.checkpoint.resolve(), batch_sizes,
            args.n_warmup, args.n_iters, args.out_dir,
        )
        print(f"\nResults saved to {args.out_dir / 'inference_benchmark.csv'}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
