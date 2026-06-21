#!/bin/bash
# Data efficiency experiment — 9-job SLURM array (2D + 3ch + 5ch, n_vols in {5,10,15}).
# Submit all:   sbatch ~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh
# Resubmit one: sbatch --array=<INDEX> ~/RP/Code/DAIC/data_efficiency/submit_data_efficiency.sh
#
# Index mapping:
#   0  2D   n=5       3  3ch  n=5       6  5ch n=5
#   1  2D   n=10      4  3ch  n=10      7  5ch n=10
#   2  2D   n=15      5  3ch  n=15      8  5ch n=15

#SBATCH --job-name=de_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal

CONFIGS=(
  configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_n10vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_n15vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_n10vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_n15vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_n10vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_n15vols_seed42_run01_daic.yaml
)

EXP_DIRS=(
  $STUDENT_DIR/experiments/runs/2d/impeccable_repeated_stride5_lora_r16_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/2d/impeccable_repeated_stride5_lora_r16_n10vols/seed42_run01
  $STUDENT_DIR/experiments/runs/2d/impeccable_repeated_stride5_lora_r16_n15vols/seed42_run01
  $STUDENT_DIR/experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16_n10vols/seed42_run01
  $STUDENT_DIR/experiments/runs/3ch/impeccable_neighbors3_stride5_lora_r16_n15vols/seed42_run01
  $STUDENT_DIR/experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n10vols/seed42_run01
  $STUDENT_DIR/experiments/runs/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n15vols/seed42_run01
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
