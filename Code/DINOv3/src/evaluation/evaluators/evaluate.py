"""Evaluate a trained checkpoint on val split and field test set.

Usage:
    python evaluation/evaluate.py --config configs/dinov3_vits_2d.yaml --checkpoint path/to/best.pt
"""

import argparse
import csv
import sys
from pathlib import Path
from evaluation.common.paths import ensure_src_on_path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

SRC = ensure_src_on_path(__file__)
sys.path.insert(0, str(SRC))

from data.dataset import FieldDataset, SeismicDataset
from models.pipeline import DINOv3Denoiser
from evaluation.common.metrics import compute_all, compute_ms_ssim_r


def _resolve(cfg_dir: Path, rel: str) -> Path:
    return (cfg_dir / rel).resolve()


def _int_cfg(cfg: dict, key: str, default: int) -> int:
    value = cfg.get(key)
    return default if value is None else int(value)


def _validate_model_data_channels(in_chans: int, dataset: object) -> None:
    offsets = getattr(dataset, "offsets", None)
    if offsets is None:
        return
    offsets = list(offsets)
    if in_chans != len(offsets):
        mode = getattr(dataset, "mode", "<unknown>")
        raise ValueError(
            f"model.in_chans={in_chans} does not match data.mode={mode!r}, "
            f"which produces {len(offsets)} channel(s) with offsets {offsets}."
        )


def _load_model(cfg: dict, cfg_dir: Path, ckpt_path: Path, device: torch.device):
    project_root = SRC.parent
    repo_dir = project_root / "external" / "dinov3"
    weights = _resolve(cfg_dir, cfg["model"]["weights"])
    m_cfg = cfg["model"]

    model = DINOv3Denoiser(
        repo_dir=repo_dir,
        weights_path=weights,
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
    val_score = ckpt.get("val_ms_ssim")
    val_text = f"{float(val_score):.4f}" if isinstance(val_score, (int, float)) else "N/A"
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val MS-SSIM={val_text})")
    return model


def _save_figure(noisy, denoised, clean, path: Path):
    """Save side-by-side visualisation: noisy | denoised | clean | residual."""
    residual = noisy - denoised
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["Noisy input", "Denoised", "Clean (GT)", "Residual"]
    imgs = [noisy, denoised, clean, residual]
    for ax, title, img in zip(axes, titles, imgs):
        ax.imshow(img.squeeze(), cmap="seismic", aspect="auto",
                  vmin=-2, vmax=2)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _image_input_flags(data_cfg: dict, input_condition: str) -> tuple[bool, bool]:
    if input_condition == "aligned":
        return False, False
    if input_condition == "train_config":
        return bool(data_cfg.get("repeat_center", False)), bool(data_cfg.get("shuffle_neighbors", False))
    if input_condition == "repeat_center":
        return True, False
    if input_condition == "shuffle_neighbors":
        return False, True
    raise ValueError(f"Unknown input condition: {input_condition}")


def evaluate(cfg: dict, cfg_path: Path, ckpt_path: Path, input_condition: str = "aligned"):
    cfg_dir = cfg_path.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _load_model(cfg, cfg_dir, ckpt_path, device)

    data_cfg = cfg["data"]
    source = data_cfg.get("source", "sfm")
    in_chans = int(cfg["model"].get("in_chans", 3))

    out_name = "eval_results" if input_condition == "aligned" else f"eval_results_{input_condition}"
    out_dir = ckpt_path.parent / out_name
    out_dir.mkdir(exist_ok=True)

    rows = []

    if source == "image_impeccable":
        from data.image_impeccable import ThinkOnwardDataset

        ii_root   = _resolve(cfg_dir, data_cfg["root_dir"])
        ii_mode   = str(data_cfg["mode"])
        repeat_center, shuffle_neighbors = _image_input_flags(data_cfg, input_condition)
        if repeat_center and ii_mode != "2.5d_5ch":
            raise ValueError(
                f"input_condition={input_condition!r} requires mode='2.5d_5ch', got {ii_mode!r}"
            )
        base_stride = int(data_cfg.get("slice_stride", 5))
        eval_stride = _int_cfg(data_cfg, "eval_slice_stride", base_stride)
        test_stride = _int_cfg(data_cfg, "test_slice_stride", eval_stride)
        ii_kwargs = dict(
            root_dir=ii_root, mode=ii_mode,
            n_train=int(data_cfg.get("n_train", 20)),
            n_val=int(data_cfg.get("n_val", 5)),
            n_test=int(data_cfg.get("n_test", 5)),
            slice_stride=test_stride,
            crop_size=int(data_cfg.get("crop_size", 224)),
            seed=int(data_cfg["seed"]),
            neighbor_stride=int(data_cfg.get("neighbor_stride", 1)),
            crop_mode=str(data_cfg.get("eval_crop_mode", data_cfg.get("crop_mode", "center"))),
            crop_seed=int(data_cfg.get("crop_seed", data_cfg.get("seed", 42))),
            repeat_center=repeat_center,
            shuffle_neighbors=shuffle_neighbors,
            cache_volumes=bool(data_cfg.get("cache_volumes", False)),
            slice_orientation=str(data_cfg.get("slice_orientation", "auto")),
        )

        # ------------------------------------------------------------------
        # Test split (held-out, clean labels → all 4 metrics)
        # ------------------------------------------------------------------
        print("\n--- Test set (Image Impeccable) ---")
        print(f"  test_slice_stride={test_stride}")
        print(f"  input_condition={input_condition}")
        test_ds = ThinkOnwardDataset(split="test", **ii_kwargs)
        _validate_model_data_channels(in_chans, test_ds)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

        agg = {"ms_ssim": 0.0, "ms_ssim_r": 0.0, "mse": 0.0, "psnr": 0.0}
        n_batches = 0
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(test_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                m = compute_all(pred.cpu(), clean.cpu(), noisy.cpu())
                for k in agg:
                    agg[k] += m[k]
                n_batches += 1
                rows.append({**m, "split": "test", "batch": i})

        for k in agg:
            agg[k] /= n_batches
        print(f"  MS-SSIM:   {agg['ms_ssim']:.4f}")
        print(f"  MS-SSIM-R: {agg['ms_ssim_r']:.4f}")
        print(f"  MSE:       {agg['mse']:.6f}")
        print(f"  PSNR:      {agg['psnr']:.2f} dB")

        # Save one example figure from the middle of the dataset.
        # Index 0 is always a near-edge slice (predominantly noise); pick a
        # sample roughly 40% into the test set instead.
        ex_idx = max(0, min(len(test_ds) - 1, len(test_ds) * 2 // 5))
        with torch.no_grad():
            noisy_ex, clean_ex = test_ds[ex_idx]
            pred_ex = model(noisy_ex.unsqueeze(0).to(device)).cpu()
        c_idx = in_chans // 2
        _save_figure(noisy_ex[c_idx].numpy(), pred_ex[0, 0].numpy(),
                     clean_ex[0].numpy(), out_dir / "test_example.png")

    else:
        if input_condition != "aligned":
            raise ValueError("input_condition is only supported for Image Impeccable configs")
        # ------------------------------------------------------------------
        # SFM dataset: validation split (all 4 metrics)
        # ------------------------------------------------------------------
        label_dir   = _resolve(cfg_dir, data_cfg["label_dir"])
        seismic_dir = _resolve(cfg_dir, data_cfg["seismic_dir"])
        field_dir   = _resolve(cfg_dir, data_cfg["field_dir"])

        print("\n--- Validation set ---")
        val_ds = SeismicDataset(
            seismic_dir, label_dir,
            val_split=data_cfg["val_split"], train=False,
            seed=data_cfg["seed"], mode=data_cfg["mode"],
        )
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2)

        agg = {"ms_ssim": 0.0, "ms_ssim_r": 0.0, "mse": 0.0, "psnr": 0.0}
        n_batches = 0
        with torch.no_grad():
            for i, (noisy, clean) in enumerate(val_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                pred = model(noisy)
                m = compute_all(pred.cpu(), clean.cpu(), noisy.cpu())
                for k in agg:
                    agg[k] += m[k]
                n_batches += 1
                rows.append({**m, "split": "val", "batch": i})

        for k in agg:
            agg[k] /= n_batches
        print(f"  MS-SSIM:   {agg['ms_ssim']:.4f}")
        print(f"  MS-SSIM-R: {agg['ms_ssim_r']:.4f}")
        print(f"  MSE:       {agg['mse']:.6f}")
        print(f"  PSNR:      {agg['psnr']:.2f} dB")

        with torch.no_grad():
            noisy_ex, clean_ex = val_ds[0]
            pred_ex = model(noisy_ex.unsqueeze(0).to(device)).cpu()
        _save_figure(noisy_ex[0].numpy(), pred_ex[0, 0].numpy(),
                     clean_ex[0].numpy(), out_dir / "val_example.png")

        # ------------------------------------------------------------------
        # Field test evaluation (no clean labels → MS-SSIM-R only)
        # ------------------------------------------------------------------
        print("\n--- Field test set ---")
        field_ds = FieldDataset(field_dir)
        field_loader = DataLoader(field_ds, batch_size=8, shuffle=False, num_workers=2)

        ms_ssim_r_sum = 0.0
        n_field = 0
        with torch.no_grad():
            for noisy_f, _stems in field_loader:
                noisy_f = noisy_f.to(device)
                pred_f = model(noisy_f)
                ms_ssim_r_sum += compute_ms_ssim_r(pred_f.cpu(), noisy_f.cpu())
                n_field += 1
                rows.append({"ms_ssim": None, "ms_ssim_r": ms_ssim_r_sum / n_field,
                             "mse": None, "psnr": None, "split": "field",
                             "batch": n_field - 1})

        print(f"  MS-SSIM-R: {ms_ssim_r_sum / max(1, n_field):.4f}")

    # Save results CSV
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "batch", "ms_ssim",
                                               "ms_ssim_r", "mse", "psnr"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--input-condition",
        choices=["aligned", "train_config", "repeat_center", "shuffle_neighbors"],
        default="aligned",
        help="Input condition for Image Impeccable evaluation.",
    )
    args = parser.parse_args()

    cfg_path = (SRC / args.config).resolve()
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, cfg_path, Path(args.checkpoint).resolve(), args.input_condition)
