#!/bin/bash
# Extended ablation studies — 36-job SLURM array at n=20 volumes, 3 seeds.
# All ablations use the main-experiment split (data.seed=42, n_train=20).
# Baselines are the already-completed main replicates (2D/3ch/5ch, seeds 42-44).
#
# Submit all:   sbatch ~/RP/Code/DAIC/ablations/submit_ablations_extended.sh
# Resubmit one: sbatch --array=<INDEX> ~/RP/Code/DAIC/ablations/submit_ablations_extended.sh
#
# Index mapping:
#   Study A-ext: 3ch Slice Stride
#     0  3ch stride=3 epochs=30 seed42
#     1  3ch stride=3 epochs=30 seed43
#     2  3ch stride=3 epochs=30 seed44
#     3  3ch stride=1 epochs=10 seed42
#     4  3ch stride=1 epochs=10 seed43
#     5  3ch stride=1 epochs=10 seed44
#
#   Study B-ext: 5ch Neighbor Stride
#     6  5ch ns=2 seed42
#     7  5ch ns=2 seed43
#     8  5ch ns=2 seed44
#     9  5ch ns=3 seed42
#    10  5ch ns=3 seed43
#    11  5ch ns=3 seed44
#
#   Study C-ext: 5ch Grid4 Crop
#    12  5ch grid4 epochs=13 seed42
#    13  5ch grid4 epochs=13 seed43
#    14  5ch grid4 epochs=13 seed44
#
#   Study E: 3ch LoRA Rank
#    15  3ch r=4  alpha=16  seed42
#    16  3ch r=4  alpha=16  seed43
#    17  3ch r=4  alpha=16  seed44
#    18  3ch r=8  alpha=32  seed42
#    19  3ch r=8  alpha=32  seed43
#    20  3ch r=8  alpha=32  seed44
#    21  3ch r=32 alpha=128 seed42
#    22  3ch r=32 alpha=128 seed43
#    23  3ch r=32 alpha=128 seed44
#
#   Study F: 3ch Loss Weight
#    24  3ch lambda=0.0 seed42
#    25  3ch lambda=0.0 seed43
#    26  3ch lambda=0.0 seed44
#    27  3ch lambda=1.0 seed42
#    28  3ch lambda=1.0 seed43
#    29  3ch lambda=1.0 seed44
#
#   Study G: 5ch Patch Embedding Init
#    30  5ch random_outer seed42
#    31  5ch random_outer seed43
#    32  5ch random_outer seed44
#    33  5ch all_mean seed42
#    34  5ch all_mean seed43
#    35  5ch all_mean seed44

#SBATCH --job-name=ablex_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%35
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/ablations/slurm_ext_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/ablations/slurm_ext_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal

CONFIGS=(
  # Study A-ext: 3ch Slice Stride
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride3_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride3_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride3_lora_r16_n20vols_seed44_run03_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride1_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride1_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride1_lora_r16_n20vols_seed44_run03_daic.yaml
  # Study B-ext: 5ch Neighbor Stride
  configs/dinov3_vits_2d5_5ch_impeccable_ns2_stride5_patch_emb_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_ns2_stride5_patch_emb_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_ns2_stride5_patch_emb_lora_r16_n20vols_seed44_run03_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_ns3_stride5_patch_emb_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_ns3_stride5_patch_emb_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_ns3_stride5_patch_emb_lora_r16_n20vols_seed44_run03_daic.yaml
  # Study C-ext: 5ch Grid4 Crop
  configs/dinov3_vits_2d5_5ch_impeccable_grid4_stride5_patch_emb_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_grid4_stride5_patch_emb_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_grid4_stride5_patch_emb_lora_r16_n20vols_seed44_run03_daic.yaml
  # Study E: 3ch LoRA Rank
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r4_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r4_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r4_n20vols_seed44_run03_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r8_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r8_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r8_n20vols_seed44_run03_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r32_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r32_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r32_n20vols_seed44_run03_daic.yaml
  # Study F: 3ch Loss Weight
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_lambda0_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_lambda0_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_lambda0_n20vols_seed44_run03_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_lambda1_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_lambda1_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_lambda1_n20vols_seed44_run03_daic.yaml
  # Study G: 5ch Patch Embedding Init
  configs/dinov3_vits_2d5_5ch_impeccable_rand_outer_stride5_patch_emb_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_rand_outer_stride5_patch_emb_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_rand_outer_stride5_patch_emb_lora_r16_n20vols_seed44_run03_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_all_mean_stride5_patch_emb_lora_r16_n20vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_all_mean_stride5_patch_emb_lora_r16_n20vols_seed43_run02_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_all_mean_stride5_patch_emb_lora_r16_n20vols_seed44_run03_daic.yaml
)

EXP_DIRS=(
  # Study A-ext: 3ch Slice Stride
  $STUDENT_DIR/experiments/runs/ablations/3ch/stride3_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/stride3_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/stride3_n20vols/seed44_run03
  $STUDENT_DIR/experiments/runs/ablations/3ch/stride1_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/stride1_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/stride1_n20vols/seed44_run03
  # Study B-ext: 5ch Neighbor Stride
  $STUDENT_DIR/experiments/runs/ablations/5ch/ns2_stride5_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/5ch/ns2_stride5_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/5ch/ns2_stride5_n20vols/seed44_run03
  $STUDENT_DIR/experiments/runs/ablations/5ch/ns3_stride5_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/5ch/ns3_stride5_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/5ch/ns3_stride5_n20vols/seed44_run03
  # Study C-ext: 5ch Grid4 Crop
  $STUDENT_DIR/experiments/runs/ablations/5ch/grid4_stride5_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/5ch/grid4_stride5_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/5ch/grid4_stride5_n20vols/seed44_run03
  # Study E: 3ch LoRA Rank
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r4_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r4_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r4_n20vols/seed44_run03
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r8_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r8_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r8_n20vols/seed44_run03
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r32_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r32_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/lora_r32_n20vols/seed44_run03
  # Study F: 3ch Loss Weight
  $STUDENT_DIR/experiments/runs/ablations/3ch/lambda0_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/lambda0_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/lambda0_n20vols/seed44_run03
  $STUDENT_DIR/experiments/runs/ablations/3ch/lambda1_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/lambda1_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/3ch/lambda1_n20vols/seed44_run03
  # Study G: 5ch Patch Embedding Init
  $STUDENT_DIR/experiments/runs/ablations/5ch/rand_outer_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/5ch/rand_outer_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/5ch/rand_outer_n20vols/seed44_run03
  $STUDENT_DIR/experiments/runs/ablations/5ch/all_mean_n20vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/5ch/all_mean_n20vols/seed43_run02
  $STUDENT_DIR/experiments/runs/ablations/5ch/all_mean_n20vols/seed44_run03
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
EXP_DIR=${EXP_DIRS[$SLURM_ARRAY_TASK_ID]}
LOG_DIR=$EXP_DIR/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "CONFIG: $CONFIG"
echo "EXP_DIR: $EXP_DIR"

CODE_ROOT="$HOME/RP/Code"
PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310
VENV=/tmp/dinov3_py310_venv_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR"' EXIT

"$PY310" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"

python -m pip install --no-index --find-links="$WHEELS" \
    "torch==2.6.0+cu118" "torchvision==0.21.0+cu118"

python -m pip install --no-index --find-links="$WHEELS" \
    torchmetrics peft numpy matplotlib pyyaml termcolor einops timm submitit \
    transformers accelerate safetensors huggingface_hub

echo "=== GPU info ==="
nvidia-smi

echo "=== Checking imports ==="
python -u -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert sys.version_info >= (3, 10)"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"

cd "$CODE_ROOT/DINOv3/src"
echo "=== Starting training ==="

srun python -u training/train.py --config "$CONFIG"

bash "$CODE_ROOT/DAIC/evaluate_if_complete.sh" "$CONFIG" "$EXP_DIR" "$CODE_ROOT"

echo "Job finished at $(date)"
