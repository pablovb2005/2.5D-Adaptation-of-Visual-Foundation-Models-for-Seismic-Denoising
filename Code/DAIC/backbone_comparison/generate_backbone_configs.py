from __future__ import annotations

import csv
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "Code" / "DINOv3" / "src" / "configs" / "backbone_comparison"
MATRIX_PATH = PROJECT_ROOT / "Code" / "DAIC" / "backbone_comparison" / "matrix.csv"

STUDENT_DIR = "/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal"
DATA_ROOT = (
    f"{STUDENT_DIR}/Dataset/ThinkOnwards/training_data/extracted"
)

DATA_SEEDS = (101, 202, 303)
TRAINING_SEEDS = (42, 43, 44)
RUN_IDS = {42: "01", 43: "02", 44: "03"}

BACKBONES = {
    "sfm_vit_base_patch16": {
        "model_name": "sfm_vit_base_patch16",
        "weights": "../../../../SFM/SFM-Base.pth",
        "patch_emb_init": "center_preserve",
        "variants": {
            "2d": {
                "family": "2d_native_stride5_lora_r16",
                "mode": "2d_1ch",
                "in_chans": 1,
            },
            "3ch": {
                "family": "neighbors3_stride5_lora_r16",
                "mode": "2.5d_3ch",
                "in_chans": 3,
            },
            "5ch": {
                "family": "neighbors5_stride5_patch_emb_lora_r16",
                "mode": "2.5d_5ch",
                "in_chans": 5,
            },
        },
    },
    "swin_v2_t": {
        "model_name": "swin_v2_t",
        "weights": "../../../weights/swin_v2_t-b137f0e2.pth",
        "patch_emb_init": "mixed",
        "variants": {
            "2d": {
                "family": "2d_repeated_stride5_lora_r16",
                "mode": "2d",
                "in_chans": 3,
            },
            "3ch": {
                "family": "neighbors3_stride5_lora_r16",
                "mode": "2.5d_3ch",
                "in_chans": 3,
            },
            "5ch": {
                "family": "neighbors5_stride5_patch_emb_lora_r16",
                "mode": "2.5d_5ch",
                "in_chans": 5,
            },
        },
    },
}


def _config(
    backbone_key: str,
    backbone: dict,
    variant_key: str,
    variant: dict,
    data_seed: int,
    training_seed: int,
) -> dict:
    run_id = RUN_IDS[training_seed]
    checkpoint_dir = (
        f"{STUDENT_DIR}/experiments/runs/backbone_comparison/"
        f"{backbone_key}/{variant['family']}/data_seed{data_seed}/"
        f"seed{training_seed}_run{run_id}"
    )
    return {
        "model": {
            "name": backbone["model_name"],
            "weights": backbone["weights"],
            "in_chans": variant["in_chans"],
            "lora_rank": 16,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "lora_targets": ["qkv", "proj"],
            "patch_emb_init": backbone["patch_emb_init"],
        },
        "data": {
            "source": "image_impeccable",
            "root_dir": DATA_ROOT,
            "mode": variant["mode"],
            "n_train": 20,
            "n_val": 5,
            "n_test": 5,
            "slice_stride": 5,
            "crop_size": 224,
            "seed": data_seed,
            # 5ch tiles all slices (~13 GB for 30 vols); safe with --mem=24GB.
            "cache_volumes": True,
        },
        "training": {
            "epochs": 50,
            "seed": training_seed,
            "batch_size": 16,
            "lr": 1.0e-4,
            "weight_decay": 0.01,
            "warmup_epochs": 5,
            "loss_lambda": 0.5,
            "num_workers": 0,
            "persistent_workers": True,
            "log_interval_batches": 25,
            "resume": True,
            "max_runtime_minutes": 700,
        },
        "output": {
            "checkpoint_dir": checkpoint_dir,
        },
    }


def _config_name(
    backbone_key: str,
    variant_key: str,
    data_seed: int,
    training_seed: int,
) -> str:
    run_id = RUN_IDS[training_seed]
    return (
        f"{backbone_key}_{variant_key}_impeccable_stride5_lora_r16_"
        f"data{data_seed}_seed{training_seed}_run{run_id}_daic.yaml"
    )


def _validate_local_paths(rows: list[dict[str, object]]) -> None:
    missing: list[str] = []
    for row in rows:
        config_rel = str(row["config"])
        config_path = PROJECT_ROOT / "Code" / "DINOv3" / "src" / config_rel
        if not config_path.exists():
            missing.append(f"config missing: {config_path}")
            continue
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        weights_path = (config_path.parent / str(cfg["model"]["weights"])).resolve()
        if not weights_path.exists():
            missing.append(f"weights missing for {config_rel}: {weights_path}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Generated configs reference missing files:\n{details}")


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    task_id = 0
    for backbone_key, backbone in BACKBONES.items():
        for variant_key, variant in backbone["variants"].items():
            for data_seed in DATA_SEEDS:
                for training_seed in TRAINING_SEEDS:
                    cfg = _config(
                        backbone_key,
                        backbone,
                        variant_key,
                        variant,
                        data_seed,
                        training_seed,
                    )
                    filename = _config_name(
                        backbone_key,
                        variant_key,
                        data_seed,
                        training_seed,
                    )
                    config_path = CONFIG_DIR / filename
                    with config_path.open("w", encoding="utf-8") as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)

                    rows.append(
                        {
                            "task_id": task_id,
                            "backbone": backbone_key,
                            "variant": variant_key,
                            "data_seed": data_seed,
                            "training_seed": training_seed,
                            "config": f"configs/backbone_comparison/{filename}",
                            "checkpoint_dir": cfg["output"]["checkpoint_dir"],
                        }
                    )
                    task_id += 1

    with MATRIX_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "backbone",
                "variant",
                "data_seed",
                "training_seed",
                "config",
                "checkpoint_dir",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    _validate_local_paths(rows)

    print(f"Wrote {len(rows)} configs to {CONFIG_DIR}")
    print(f"Wrote matrix to {MATRIX_PATH}")
    print("Pilot task IDs: 0,9,18,27,36,45")


if __name__ == "__main__":
    main()
