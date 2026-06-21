# 2.5D Adaptation of Visual Foundation Models for Seismic Denoising

BSc Research Project — CSE3000 | TU Delft | Pablo Varela Bernal

## Overview

This repository contains the code for a parameter-efficient adaptation study comparing **2D** and **2.5D** input strategies for seismic denoising using a DINOv3-based visual foundation model fine-tuned with LoRA (PEFT).

The core scientific comparison pits three input variants:

| Variant | Description |
|---------|-------------|
| `2D` | Single seismic slice, repeated across 3 channels (`[t, t, t]`) |
| `3ch` | Three adjacent inline slices stacked as RGB |
| `5ch` | Five adjacent inline slices; patch embedding adapted to 5 channels |

All variants use the same volume-level train/validation/test split, the same LoRA targets (`qkv` and `proj`), and the same evaluation protocol (MS-SSIM on the held-out test set).

Primary dataset: **ThinkOnward Image Impeccable** (parts 1–2, 30 paired 3D synthetic volumes).  
Robustness transfer: **F3 Netherlands** (unlabelled field data — no ground-truth accuracy).

## Repository Structure

```
Code/
  DINOv3/
    src/
      configs/          # YAML experiment configs (seeds, splits, hyperparams)
      data/             # Dataset loaders and input-mode logic
      models/           # DINOv3 denoiser pipeline and decoder
      training/         # Training loop and loss
      evaluation/       # Metrics, summarisation, and figure scripts
      dinov3_denoiser.py
    external/           # NOT tracked — see Dependencies below
    weights/            # NOT tracked — see Dependencies below
  DAIC/                 # SLURM submission, evaluation, and environment scripts
```

## Dependencies

### Backbone

This project uses [DINOv3 ViT-S/16](https://github.com/facebookresearch/dinov3) from Meta AI Research as the frozen backbone.

Clone it into `Code/DINOv3/external/dinov3/` and check out the exact commit used:

```bash
git clone https://github.com/facebookresearch/dinov3.git Code/DINOv3/external/dinov3
git -C Code/DINOv3/external/dinov3 checkout 31703e4cbf1ccb7c4a72daa1350405f86754b6d1
```

### Pre-trained weights

Download the DINOv3 ViT-S/16 pre-trained weights from the [Meta DINOv3 model card](https://github.com/facebookresearch/dinov3) and place them at:

```
Code/DINOv3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

### Python environment (DAIC / SLURM)

The project runs on Python 3.10 with CUDA 11.8. One-time environment setup on DAIC:

```bash
sbatch Code/DAIC/setup_env_py310.sh
```

Key packages: `torch==2.6.0+cu118`, `torchvision==0.21.0+cu118`, `peft`, `torchmetrics`, `timm`, `transformers`, `einops`, `pyyaml`

## Data

The **Image Impeccable** dataset (parts 1–2) is available from [ThinkOnward](https://thinkonward.com/app/c/challenges/image-impeccable). Download and place volumes under `Code/Dataset/`.

The **F3 Netherlands** block is available from the [SEG Open Data](https://wiki.seg.org/wiki/F3_Netherlands). Prepare it with `Code/DINOv3/src/evaluation/prepare_f3.py`.

Neither dataset is included in this repository due to licensing constraints.

## Reproducing Main Results

All main experiments are configured via YAML files in `Code/DINOv3/src/configs/`.

Submit the main multi-data-seed rerun (data seeds 101/202/303, training seeds 42/43/44) on DAIC:

```bash
bash Code/DAIC/submit_main_replicates.sh
```

Evaluate after training completes:

```bash
bash Code/DAIC/evaluate_impeccable_all.sh
```

Summarise results:

```bash
python Code/DINOv3/src/evaluation/summarize_impeccable_runs.py
```

## Reproducibility Notes

- Volume-level splits are seeded via `data.seed` in each config; slices from one volume are never split across train/val/test.
- All main variant comparisons use identical protocol, split, and evaluation policy.
- Training seeds 42/43/44 are crossed with data seeds 101/202/303 for the new main rerun.
- F3 results are unlabelled field transfer diagnostics only; they are not denoising accuracy metrics.

## License

Code in `Code/DINOv3/src/` and `Code/DAIC/` is released under the [MIT License](LICENSE).  
The DINOv3 backbone code and weights are subject to Meta's [Apache 2.0 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE).  
Image Impeccable and F3 data are subject to their respective third-party licenses.
