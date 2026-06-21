"""Generate side-by-side qualitative comparison panels for the paper/poster.

Loads best.pt for 2D, 3ch, and 5ch, runs inference once on the shared test split,
then generates N panels under three subdirectories:

  by_gap/    — hardest examples: largest Δ(5ch − 2D MS-SSIM). These tend to have
               low absolute MS-SSIM (~0.3) because context helps most when a single
               slice is ambiguous. Use for the "where does 2.5D help" narrative.
  by_best/   — highest absolute 5ch MS-SSIM. Clean, representative high-quality
               examples. Use for the "what good denoising looks like" figure.
  by_median/ — samples nearest to the median 5ch MS-SSIM. Typical quality examples
               that best represent the reported mean metric.

Usage (local):
    python evaluation/generate_comparison_panels.py \\
        --project-root C:/UNI/Y3/RP \\
        --out-dir C:/UNI/Y3/RP/experiments/summaries/comparison_panels \\
        [--data-seed 101] [--training-seed 42] [--n-panels 4] [--stride 5]

PowerShell wrapper: src/evaluation/runners/generate_comparison_panels.ps1
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path, project_root as default_project_root

import numpy as np
import torch
import yaml

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from data.image_impeccable import ThinkOnwardDataset, _zscore
from models.pipeline import DINOv3Denoiser
from evaluation.common.metrics import compute_mse, compute_ms_ssim, compute_psnr


# ---------------------------------------------------------------------------
# Variant definitions — path fragments match the DAIC/local run layout
# ---------------------------------------------------------------------------
_VARIANTS = [
    {
        "key": "2D",
        "mode": "2d",
        "in_chans": 3,
        "family": "2d",
        "variant_dir": "impeccable_repeated_stride5_lora_r16",
    },
    {
        "key": "3ch",
        "mode": "2.5d_3ch",
        "in_chans": 3,
        "family": "3ch",
        "variant_dir": "impeccable_neighbors3_stride5_lora_r16",
    },
    {
        "key": "5ch",
        "mode": "2.5d_5ch",
        "in_chans": 5,
        "family": "5ch",
        "variant_dir": "impeccable_neighbors5_stride5_patch_emb_lora_r16",
    },
]

_RUN_CANDIDATES = ["seed42_run01", "seed43_run02", "seed44_run03"]

_MAIN_DATA_SEEDS = [101, 202, 303]


def _resolve_checkpoint(
    project_root: Path,
    training_seed: int,
    variant: dict,
    data_seed: int = 101,
    strict: bool = False,
) -> Path | None:
    """Resolve the best.pt path for a given variant/data_seed/training_seed.

    With strict=True (used by the multi-seed pooled flow), only the new
    main_multidata/<family>/<vdir>/data_seed<N>/ layout is searched and a
    RuntimeError is raised if the checkpoint is not found.  This prevents the
    multidata results from being silently contaminated by legacy checkpoints.
    """
    exp_root = project_root / "experiments" / "runs"
    fallback_root = project_root / "experiments"
    family = variant["family"]
    vdir = variant["variant_dir"]
    seed_prefix = f"seed{training_seed}"

    # 1. Correct new-protocol layout: main_multidata/<family>/<vdir>/data_seed<N>/<run_id>/best.pt
    vdir_ds_dir = exp_root / "main_multidata" / family / vdir / f"data_seed{data_seed}"
    if vdir_ds_dir.exists():
        candidates = sorted(vdir_ds_dir.iterdir()) if vdir_ds_dir.is_dir() else []
        for run_dir in candidates:
            if seed_prefix in run_dir.name and (run_dir / "best.pt").exists():
                return run_dir / "best.pt"
        for run_dir in candidates:
            if (run_dir / "best.pt").exists():
                print(f"  [{variant['key']}] training_seed={training_seed} not found in "
                      f"{vdir_ds_dir.name}, using {run_dir.name}")
                return run_dir / "best.pt"

    if strict:
        raise RuntimeError(
            f"[{variant['key']}] Checkpoint not found for data_seed={data_seed}, "
            f"training_seed={training_seed}. Checked: {vdir_ds_dir}. "
            f"Strict mode: no fallback to other data_seeds or legacy layouts."
        )

    # Non-strict fallbacks: try new vdir layout for other data seeds
    for ds in [s for s in _MAIN_DATA_SEEDS if s != data_seed]:
        other_dir = exp_root / "main_multidata" / family / vdir / f"data_seed{ds}"
        if other_dir.exists():
            candidates = sorted(other_dir.iterdir()) if other_dir.is_dir() else []
            for run_dir in candidates:
                if seed_prefix in run_dir.name and (run_dir / "best.pt").exists():
                    print(f"  [{variant['key']}] data_seed={data_seed} not found (vdir), using data_seed={ds}")
                    return run_dir / "best.pt"

    # 2. Old protocol without vdir: main_multidata/<family>/data_seed<N>/<run_id>/best.pt
    for ds in [data_seed] + [s for s in _MAIN_DATA_SEEDS if s != data_seed]:
        ds_dir = exp_root / "main_multidata" / family / f"data_seed{ds}"
        if ds_dir.exists():
            candidates = sorted(ds_dir.iterdir()) if ds_dir.is_dir() else []
            for run_dir in candidates:
                if seed_prefix in run_dir.name and (run_dir / "best.pt").exists():
                    if ds != data_seed:
                        print(f"  [{variant['key']}] data_seed={data_seed} not found, using data_seed={ds}")
                    return run_dir / "best.pt"
            for run_dir in candidates:
                if (run_dir / "best.pt").exists():
                    print(f"  [{variant['key']}] training_seed={training_seed} not found, using {run_dir.name}")
                    return run_dir / "best.pt"

    # 3. Legacy single-seed layout: <family>/<variant_dir>/<run_id>/best.pt
    for root in [exp_root, fallback_root]:
        for run_id in _RUN_CANDIDATES:
            if not run_id.startswith(seed_prefix):
                continue
            candidate = root / family / vdir / run_id / "best.pt"
            if candidate.exists():
                return candidate
        for run_id in _RUN_CANDIDATES:
            candidate = root / family / vdir / run_id / "best.pt"
            if candidate.exists():
                print(f"  [{variant['key']}] seed{training_seed} not found, falling back to {run_id}")
                return candidate
    return None


def _load_model(ckpt_path: Path, variant: dict, device: torch.device) -> DINOv3Denoiser:
    cfg_path = ckpt_path.parent / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found next to checkpoint: {ckpt_path}")
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    cfg_dir = cfg_path.parent
    m_cfg = cfg["model"]
    repo_dir = SRC.parent / "external" / "dinov3"

    # Weights path in config may be absolute (DAIC) or relative; resolve relative to cfg_dir.
    weights_raw = m_cfg["weights"]
    weights = Path(weights_raw)
    if not weights.is_absolute():
        weights = (cfg_dir / weights_raw).resolve()
    # If still missing, try relative to SRC
    if not weights.exists():
        weights = (SRC / "configs" / weights_raw).resolve()
    if not weights.exists():
        weights = SRC.parent / "weights" / Path(weights_raw).name
    if not weights.exists():
        raise FileNotFoundError(f"Cannot locate weights file: {weights_raw}")

    model = DINOv3Denoiser(
        repo_dir=repo_dir,
        weights_path=weights,
        model_name=m_cfg["name"],
        in_chans=variant["in_chans"],
        lora_rank=m_cfg["lora_rank"],
        lora_alpha=m_cfg["lora_alpha"],
        lora_dropout=m_cfg["lora_dropout"],
        lora_targets=m_cfg["lora_targets"],
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  [{variant['key']}] loaded epoch {ckpt.get('epoch', '?')} "
          f"(val MS-SSIM={ckpt.get('val_ms_ssim', float('nan')):.4f}) from {ckpt_path.name}")
    return model


def _build_shared_dataset(project_root: Path, stride: int, data_seed: int = 42) -> ThinkOnwardDataset:
    """Test split dataset using mode='2d' to build the shared sample index."""
    # Find the dataset root
    candidates = [
        project_root / "Code" / "Dataset" / "ThinkOnwards" / "training_data" / "extracted",
        project_root / "Dataset" / "ThinkOnwards" / "training_data" / "extracted",
    ]
    root = next((c for c in candidates if c.exists()), None)
    if root is None:
        raise FileNotFoundError(
            f"Image Impeccable dataset not found. Checked: {[str(c) for c in candidates]}"
        )
    return ThinkOnwardDataset(
        root_dir=root,
        mode="2d",
        split="test",
        slice_stride=stride,
        crop_mode="center",
        seed=data_seed,
    )


def _build_variant_dataset(ds_2d: ThinkOnwardDataset, mode: str) -> ThinkOnwardDataset:
    """Dataset for a specific mode, sharing the same root/stride/seed as the 2D dataset."""
    return ThinkOnwardDataset(
        root_dir=ds_2d.root_dir,
        mode=mode,
        split="test",
        slice_stride=ds_2d.slice_stride,
        crop_mode="center",
        seed=ds_2d.seed,
    )


def _load_sample_for_mode(
    ds: ThinkOnwardDataset,
    noisy_path: Path,
    clean_path: Path,
    ax: int,
    t: int,
    oh: int,
    ow: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one sample using ds.offsets but at the given (ax, t, oh, ow) coordinates.

    This lets all three modes share the same anchor (t, oh, ow) from the most
    restrictive dataset (5ch, context_radius=2), so 2D/3ch/5ch are always
    compared on exactly the same central slice.
    """
    cs = ds.crop_size
    noisy_vol = np.load(noisy_path, mmap_mode="r", allow_pickle=True)
    clean_vol = np.load(clean_path, mmap_mode="r", allow_pickle=True)
    noisy_slices = [
        np.take(noisy_vol, t + dt, axis=ax)[oh:oh + cs, ow:ow + cs].astype(np.float32)
        for dt in ds.offsets
    ]
    clean_slice = np.take(clean_vol, t, axis=ax)[oh:oh + cs, ow:ow + cs].astype(np.float32)
    noisy_slices = [_zscore(s) for s in noisy_slices]
    clean_slice = _zscore(clean_slice)
    noisy = torch.from_numpy(np.stack(noisy_slices, axis=0))   # [C, H, W]
    clean = torch.from_numpy(clean_slice[None])                  # [1, H, W]
    return noisy, clean


@torch.no_grad()
def _run_inference(
    models: list[tuple[str, DINOv3Denoiser, int]],
    datasets: list[ThinkOnwardDataset],
    device: torch.device,
) -> list[dict]:
    """Run all models on a shared slice index, collecting per-sample predictions.

    Uses the most restrictive dataset (largest context_radius, i.e. 5ch) as the
    anchor so every sample tuple is valid for all modes.  Slices for 2D and 3ch are
    loaded directly from the raw volumes at the same (t, oh, ow) coordinates.

    Returns list of dicts with keys:
      sample_idx, vol_id, slice_t, noisy_center, clean, preds (dict key→tensor)
    """
    anchor_idx = max(range(len(datasets)), key=lambda i: datasets[i].context_radius)
    anchor_ds = datasets[anchor_idx]
    n = len(anchor_ds)

    results = []
    for idx in range(n):
        noisy_path, clean_path, ax, t, oh, ow = anchor_ds.samples[idx]
        # oh/ow are None only for random-crop mode, which we never use here (center crop).
        oh = oh or 0
        ow = ow or 0
        vol_id = int(noisy_path.stem.split("_")[-1])

        preds: dict[str, torch.Tensor] = {}
        noisy_center: torch.Tensor | None = None
        clean_tensor: torch.Tensor | None = None

        for (key, model, in_chans), ds in zip(models, datasets):
            noisy, clean = _load_sample_for_mode(ds, noisy_path, clean_path, ax, t, oh, ow)
            noisy_t = noisy.unsqueeze(0).to(device)
            pred = model(noisy_t).cpu()[0, 0]  # [H, W]
            preds[key] = pred
            if noisy_center is None:
                noisy_center = noisy[in_chans // 2]  # central channel [H, W]
                clean_tensor = clean[0]               # [H, W]

        results.append({
            "sample_idx": idx,
            "vol_id": vol_id,
            "slice_t": t,
            "noisy_path": noisy_path,
            "vol_axis": ax,
            "noisy_center": noisy_center,
            "clean": clean_tensor,
            "preds": preds,
        })

        if (idx + 1) % 20 == 0:
            print(f"  Inference: {idx + 1}/{n} samples", end="\r")

    print(f"  Inference: {n}/{n} samples done.   ")
    return results


def _per_sample_ms_ssim(pred: torch.Tensor, clean: torch.Tensor) -> float:
    return float(compute_ms_ssim(pred.unsqueeze(0).unsqueeze(0), clean.unsqueeze(0).unsqueeze(0)))


def _input_quality_metrics(noisy_center: torch.Tensor, clean: torch.Tensor) -> dict[str, float]:
    noisy_b = noisy_center.unsqueeze(0).unsqueeze(0)
    clean_b = clean.unsqueeze(0).unsqueeze(0)
    return {
        "input_ms_ssim": float(compute_ms_ssim(noisy_b, clean_b)),
        "input_mse": float(compute_mse(noisy_b, clean_b)),
        "input_psnr": float(compute_psnr(noisy_b, clean_b)),
    }


@torch.no_grad()
def _run_metrics_for_seed(
    project_root: Path,
    training_seed: int,
    data_seed: int,
    stride: int,
    device: torch.device,
    mid_slice_filter: bool = True,
) -> dict:
    """Run inference for one (data_seed, training_seed) pair without storing predictions.

    Uses strict checkpoint resolution so multidata results are never silently
    mixed with legacy checkpoints.

    Returns dict mapping sample_idx -> {vol_id, slice_t, vol_depth, input_quality, ms_ssim}.
    """
    ds_2d = _build_shared_dataset(project_root, stride, data_seed=data_seed)
    datasets = [
        _build_variant_dataset(ds_2d, v["mode"]) if v["key"] != "2D" else ds_2d
        for v in _VARIANTS
    ]

    models_list = []
    for v in _VARIANTS:
        ckpt = _resolve_checkpoint(project_root, training_seed, v, data_seed=data_seed, strict=True)
        if ckpt is None:
            raise RuntimeError(f"[{v['key']}] unexpected None checkpoint in strict mode")
        models_list.append((v["key"], _load_model(ckpt, v, device), v["in_chans"]))

    results = _run_inference(models_list, datasets, device)

    _vol_depth_cache: dict = {}
    for r in results:
        cache_key = (str(r["noisy_path"]), r["vol_axis"])
        if cache_key not in _vol_depth_cache:
            vol = np.load(r["noisy_path"], mmap_mode="r", allow_pickle=True)
            _vol_depth_cache[cache_key] = int(vol.shape[r["vol_axis"]])
        r["vol_depth"] = _vol_depth_cache[cache_key]

    if mid_slice_filter:
        before = len(results)
        results = [
            r for r in results
            if 0.20 * r["vol_depth"] <= r["slice_t"] <= 0.80 * r["vol_depth"]
        ]
        print(f"  Middle-slice filter: {before} -> {len(results)} samples")

    per_sample = {}
    for r in results:
        idx = r["sample_idx"]
        per_sample[idx] = {
            "vol_id": r["vol_id"],
            "slice_t": r["slice_t"],
            "vol_depth": r["vol_depth"],
            "input_quality": _input_quality_metrics(r["noisy_center"], r["clean"]),
            "ms_ssim": {k: _per_sample_ms_ssim(r["preds"][k], r["clean"]) for k in r["preds"]},
        }

    del models_list
    return per_sample


def _average_seed_metrics(per_seed_list: list) -> dict:
    """Average per-sample MS-SSIM across training seeds. Input quality taken from the first seed."""
    if len(per_seed_list) == 1:
        return per_seed_list[0]
    reference = per_seed_list[0]
    result = {}
    for idx in reference:
        variant_keys = list(reference[idx]["ms_ssim"].keys())
        avg_ms = {}
        for vk in variant_keys:
            vals = [
                s[idx]["ms_ssim"][vk]
                for s in per_seed_list
                if idx in s and vk in s[idx]["ms_ssim"] and np.isfinite(s[idx]["ms_ssim"][vk])
            ]
            avg_ms[vk] = float(np.mean(vals)) if vals else float("nan")
        result[idx] = {
            "vol_id": reference[idx]["vol_id"],
            "slice_t": reference[idx]["slice_t"],
            "vol_depth": reference[idx]["vol_depth"],
            "input_quality": reference[idx]["input_quality"],
            "ms_ssim": avg_ms,
        }
    return result


def _metrics_to_csv_rows(metrics: dict, data_seed: int | None = None) -> list:
    """Convert per-sample metrics dict to CSV rows sorted by delta_5ch_vs_2d descending."""
    rows = []
    for idx, m in metrics.items():
        ms = m["ms_ssim"]
        fallback = ms[list(ms)[0]]
        ms2d  = ms.get("2D",  fallback)
        ms3ch = ms.get("3ch", fallback)
        ms5ch = ms.get("5ch", fallback)
        iq = m["input_quality"]
        vdepth = m["vol_depth"]
        mid_pct = (m["slice_t"] / vdepth * 100) if vdepth else 0.0
        row: dict = {
            "vol_id":          m["vol_id"],
            "slice_t":         m["slice_t"],
            "vol_depth":       vdepth,
            "mid_slice_pct":   f"{mid_pct:.1f}",
            "sample_idx":      idx,
            "input_ms_ssim":   f"{iq['input_ms_ssim']:.6f}",
            "input_mse":       f"{iq['input_mse']:.6f}",
            "input_psnr":      f"{iq['input_psnr']:.6f}",
            "ms_ssim_2d":      f"{ms2d:.6f}",
            "ms_ssim_3ch":     f"{ms3ch:.6f}",
            "ms_ssim_5ch":     f"{ms5ch:.6f}",
            "delta_3ch_vs_2d": f"{ms3ch - ms2d:+.6f}",
            "delta_5ch_vs_2d": f"{ms5ch - ms2d:+.6f}",
        }
        if data_seed is not None:
            row["data_seed"] = data_seed
        rows.append(row)
    rows.sort(key=lambda r: float(r["delta_5ch_vs_2d"]), reverse=True)
    for rank, row in enumerate(rows):
        row["rank_by_gap"] = rank
    return rows


def _write_csv_rows(rows: list, out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # rank_by_gap first, then the rest
    first = ["rank_by_gap"] if "rank_by_gap" in rows[0] else []
    rest = [k for k in rows[0] if k not in first]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=first + rest)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {out_path} ({len(rows)} rows)")


def _save_panel(sample: dict, rank: int, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("matplotlib is required: pip install matplotlib")

    noisy = sample["noisy_center"].numpy()
    clean = sample["clean"].numpy()
    preds = sample["preds"]
    variant_keys = ["2D", "3ch", "5ch"]
    ms_ssim_vals = sample["ms_ssim"]  # precomputed in main()

    kw = dict(cmap="seismic", aspect="auto", vmin=-2, vmax=2)

    fig, axes = plt.subplots(2, 5, figsize=(21, 9.2))
    fig.suptitle(
        "Denoising Results on the Image Impeccable Dataset",
        fontsize=28,
        fontweight="bold",
    )

    # Row 0: noisy | 2D | 3ch | 5ch | clean
    row0_imgs = [noisy] + [preds[k].numpy() for k in variant_keys if k in preds] + [clean]
    row0_titles = ["Noisy input"] + [
        f"{k}\nMS-SSIM={ms_ssim_vals[k]:.4f}" for k in variant_keys if k in preds
    ] + ["Clean (GT)"]

    for ax, img, title in zip(axes[0], row0_imgs, row0_titles):
        ax.imshow(img, **kw)
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.axis("off")

    # Row 1: empty | residual 2D | residual 3ch | residual 5ch | empty
    axes[1, 0].axis("off")
    axes[1, 4].axis("off")
    for col, k in enumerate(variant_keys):
        if k not in preds:
            axes[1, col + 1].axis("off")
            continue
        residual = noisy - preds[k].numpy()
        axes[1, col + 1].imshow(residual, **kw)
        axes[1, col + 1].set_title(f"Residual {k}", fontsize=22, fontweight="bold")
        axes[1, col + 1].axis("off")

    plt.tight_layout()
    out_path = out_dir / f"panel_{rank:02d}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.parent.name}/{out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2D/3ch/5ch comparison panels.")
    parser.add_argument("--project-root", type=Path, default=default_project_root(__file__))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--data-seed", type=int, default=101,
                        help="Data seed for checkpoint selection (single-seed mode)")
    parser.add_argument("--training-seed", type=int, default=42,
                        help="Training seed for checkpoint selection (single-seed mode)")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=None,
                        help="Multiple training seeds; averaged per-sample before CSV write. "
                             "Activates multi-seed mode (no panels generated).")
    parser.add_argument("--pool-data-seeds", action="store_true",
                        help="Run all three data seeds (101, 202, 303), write per-split CSVs "
                             "plus a pooled CSV. Implies multi-seed mode.")
    parser.add_argument("--seed", type=int, default=None,
                        help="[Legacy] Alias for --training-seed; overrides --training-seed if set")
    parser.add_argument("--n-panels", type=int, default=4, help="Number of panels per selection strategy")
    parser.add_argument("--stride", type=int, default=5, help="Test slice stride (should match eval protocol)")
    parser.add_argument("--no-mid-slice-filter", action="store_true",
                        help="Disable middle-slice filtering (by default, slices in the outer 20%% "
                             "of each volume are excluded because they are predominantly noise).")
    args = parser.parse_args()

    # --seed is a legacy alias; if provided, it overrides --training-seed
    training_seed = args.seed if args.seed is not None else args.training_seed
    mid_slice_filter = not args.no_mid_slice_filter

    project_root = args.project_root.resolve()
    out_dir = args.out_dir or (project_root / "experiments" / "summaries" / "comparison_panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Multi-seed / pooled mode ---------------------------------------------------
    # Activated by --training-seeds (list) and/or --pool-data-seeds.
    # Generates per-split CSVs (ds101/, ds202/, ds303/) and a pooled CSV, but no panels.
    training_seeds_multi = args.training_seeds
    if args.pool_data_seeds or (training_seeds_multi and len(training_seeds_multi) > 1):
        if training_seeds_multi is None:
            training_seeds_multi = [training_seed]
        data_seeds_to_run = _MAIN_DATA_SEEDS if args.pool_data_seeds else [args.data_seed]
        print(f"\nMulti-seed mode: data_seeds={data_seeds_to_run}, training_seeds={training_seeds_multi}")
        print("Panels will NOT be generated in this mode.\n")

        all_pooled_rows: list = []
        global_sample_offset = 0

        for ds in data_seeds_to_run:
            print(f"\n=== Data seed {ds} (averaging {len(training_seeds_multi)} training seeds) ===")
            per_seed_list = []
            for ts in training_seeds_multi:
                print(f"\n  Training seed {ts}...")
                seed_metrics = _run_metrics_for_seed(
                    project_root, ts, ds, args.stride, device, mid_slice_filter,
                )
                per_seed_list.append(seed_metrics)

            averaged = _average_seed_metrics(per_seed_list)

            # Per-split CSV: original local sample_idx, no data_seed column
            ds_rows = _metrics_to_csv_rows(averaged, data_seed=None)
            _write_csv_rows(ds_rows, out_dir / f"ds{ds}" / "comparison_metadata.csv")

            # Pooled: globally unique sample_idx + data_seed column
            pooled_ds_rows = _metrics_to_csv_rows(averaged, data_seed=ds)
            local_max_idx = max(averaged.keys()) if averaged else 0
            for row in pooled_ds_rows:
                row["sample_idx"] = int(row["sample_idx"]) + global_sample_offset
            all_pooled_rows.extend(pooled_ds_rows)
            global_sample_offset += local_max_idx + 1

        # Re-rank pooled rows by gap and write combined CSV
        all_pooled_rows.sort(key=lambda r: float(r["delta_5ch_vs_2d"]), reverse=True)
        for rank, row in enumerate(all_pooled_rows):
            row["rank_by_gap"] = rank
        pooled_out = out_dir / "pooled" / "comparison_metadata.csv"
        _write_csv_rows(all_pooled_rows, pooled_out)

        print(f"\nMulti-seed run complete. {len(all_pooled_rows)} total pooled rows.")
        print("Run analyze_mechanism.ps1 --Metadata pointing to each per-split or pooled CSV.")
        return

    # Resolve checkpoints
    print(f"\nResolving checkpoints (data_seed={args.data_seed}, training_seed={training_seed})...")
    ckpts: dict[str, Path] = {}
    for v in _VARIANTS:
        ckpt = _resolve_checkpoint(project_root, training_seed, v, data_seed=args.data_seed)
        if ckpt is None:
            print(f"  [{v['key']}] WARNING: no checkpoint found - skipping this variant")
        else:
            ckpts[v["key"]] = ckpt

    if len(ckpts) < 2:
        print("ERROR: Need at least 2 variant checkpoints to generate comparison panels.")
        sys.exit(1)

    # Load models and datasets
    active_variants = [v for v in _VARIANTS if v["key"] in ckpts]
    print("\nLoading models...")
    models_list = [
        (v["key"], _load_model(ckpts[v["key"]], v, device), v["in_chans"])
        for v in active_variants
    ]

    print("\nBuilding test datasets...")
    ds_2d = _build_shared_dataset(project_root, args.stride, data_seed=args.data_seed)
    datasets = [
        _build_variant_dataset(ds_2d, v["mode"]) if v["key"] != "2D" else ds_2d
        for v in active_variants
    ]
    print(f"Test split: {len(ds_2d)} samples")

    # Run inference
    print("\nRunning inference (this may take a few minutes on CPU)...")
    results = _run_inference(models_list, datasets, device)

    # Compute vol_depth for each sample from its volume shape along the slice axis.
    # This is needed for middle-slice filtering and is cheap (mmap read, no data load).
    print("\nComputing volume depths and per-sample metrics...")
    _vol_depth_cache: dict[tuple, int] = {}
    for r in results:
        key = (str(r["noisy_path"]), r["vol_axis"])
        if key not in _vol_depth_cache:
            vol = np.load(r["noisy_path"], mmap_mode="r", allow_pickle=True)
            _vol_depth_cache[key] = int(vol.shape[r["vol_axis"]])
        r["vol_depth"] = _vol_depth_cache[key]

    # Middle-slice filter: exclude slices in the bottom 20% or top 20% of a volume.
    # The first and last slices of seismic volumes are predominantly noise and are
    # not representative of denoising quality.
    if mid_slice_filter:
        before = len(results)
        results = [
            r for r in results
            if 0.20 * r["vol_depth"] <= r["slice_t"] <= 0.80 * r["vol_depth"]
        ]
        print(f"  Middle-slice filter: {before} → {len(results)} samples (kept 20%–80% of each volume)")
    else:
        print(f"  Middle-slice filter: disabled ({len(results)} samples)")

    # Precompute per-sample MS-SSIM for all variants (avoids redundant forward calls)
    has_both = "2D" in ckpts and "5ch" in ckpts
    for r in results:
        r["ms_ssim"] = {
            k: _per_sample_ms_ssim(r["preds"][k], r["clean"])
            for k in r["preds"]
        }
        r["input_quality"] = _input_quality_metrics(r["noisy_center"], r["clean"])
        ms_2d   = r["ms_ssim"].get("2D",    0.0)
        ms_5ch = r["ms_ssim"].get("5ch", 0.0)
        r["gap"] = ms_5ch - ms_2d if has_both else ms_5ch

    # --- Three selection strategies ---
    # by_gap:    hardest examples — largest Δ(5ch − 2D), reveals where context helps most
    # by_best:   highest absolute 5ch MS-SSIM — clean, high-quality representative examples
    # by_median: samples nearest to the median 5ch MS-SSIM — typical quality
    by_gap = sorted(results, key=lambda r: r["gap"], reverse=True)[: args.n_panels]

    by_best = sorted(
        results,
        key=lambda r: r["ms_ssim"].get("5ch", r["ms_ssim"].get("2D", 0.0)),
        reverse=True,
    )[: args.n_panels]

    all_ms5ch = [r["ms_ssim"].get("5ch", r["ms_ssim"].get("2D", 0.0)) for r in results]
    median_val = float(np.median(all_ms5ch))
    by_median = sorted(
        results,
        key=lambda r: abs(r["ms_ssim"].get("5ch", r["ms_ssim"].get("2D", 0.0)) - median_val),
    )[: args.n_panels]

    selections = [
        ("by_gap",    by_gap,    "Largest delta - hardest examples (where context helps most)"),
        ("by_best",   by_best,   "Highest absolute 5ch MS-SSIM - best quality examples"),
        ("by_median", by_median, f"Nearest to median 5ch MS-SSIM ({median_val:.4f}) - typical quality"),
    ]

    for sel_name, sel_samples, sel_desc in selections:
        sel_dir = out_dir / sel_name
        sel_dir.mkdir(exist_ok=True)
        print(f"\nSaving {args.n_panels} panels [{sel_desc}] -> {sel_name}/")
        for rank, sample in enumerate(sel_samples):
            _save_panel(sample, rank, sel_dir)

    # Write full metadata CSV (sorted by gap, all samples)
    results_by_gap = sorted(results, key=lambda r: r["gap"], reverse=True)
    meta_path = out_dir / "comparison_metadata.csv"
    with meta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank_by_gap", "vol_id", "slice_t", "vol_depth", "mid_slice_pct", "sample_idx",
            "input_ms_ssim", "input_mse", "input_psnr",
            "ms_ssim_2d", "ms_ssim_3ch", "ms_ssim_5ch",
            "delta_3ch_vs_2d", "delta_5ch_vs_2d",
        ])
        writer.writeheader()
        for rank, r in enumerate(results_by_gap):
            ms = r["ms_ssim"]
            fallback = ms[list(ms)[0]]
            ms2d  = ms.get("2D",    fallback)
            ms3ch = ms.get("3ch",   fallback)
            ms5  = ms.get("5ch", fallback)
            input_quality = r["input_quality"]
            vdepth = r.get("vol_depth", 0)
            mid_pct = (r["slice_t"] / vdepth * 100) if vdepth else 0.0
            writer.writerow({
                "rank_by_gap":   rank,
                "vol_id":        r["vol_id"],
                "slice_t":       r["slice_t"],
                "vol_depth":     vdepth,
                "mid_slice_pct": f"{mid_pct:.1f}",
                "sample_idx":    r["sample_idx"],
                "input_ms_ssim":     f"{input_quality['input_ms_ssim']:.6f}",
                "input_mse":         f"{input_quality['input_mse']:.6f}",
                "input_psnr":        f"{input_quality['input_psnr']:.6f}",
                "ms_ssim_2d":        f"{ms2d:.6f}",
                "ms_ssim_3ch":       f"{ms3ch:.6f}",
                "ms_ssim_5ch":       f"{ms5:.6f}",
                "delta_3ch_vs_2d":   f"{ms3ch - ms2d:+.6f}",
                "delta_5ch_vs_2d":   f"{ms5  - ms2d:+.6f}",
            })

    print(f"\nSaved full metadata: {meta_path.name} ({len(results)} rows)")
    total_panels = args.n_panels * len(selections)
    print(f"Done. {total_panels} panels written across {len(selections)} subdirectories in {out_dir}")


if __name__ == "__main__":
    main()
