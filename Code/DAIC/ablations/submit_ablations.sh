#!/bin/bash
# Ablation studies — 5-job SLURM array.
# Submit all:   sbatch ~/RP/Code/DAIC/ablations/submit_ablations.sh
# Resubmit one: sbatch --array=<INDEX> ~/RP/Code/DAIC/ablations/submit_ablations.sh
#
# Index mapping:
#   0  Study A1 — 2D  stride=3  epochs=30  (same total batches as stride=5 / 50ep)
#   1  Study A2 — 2D  stride=1  epochs=10  (same total batches as stride=5 / 50ep)
#   2  Study B1 — 3ch neighbor_stride=2  epochs=50
#   3  Study B2 — 3ch neighbor_stride=3  epochs=50
#   4  Study C1 — 3ch grid4 crop  epochs=13  (same total batches as center / 50ep)
#
# Baselines (free from data_efficiency array):
#   Study A baseline — 2D  stride=5  epochs=50  (array index 0 of data_efficiency)
#   Study B+C baseline — 3ch stride=5 center epochs=50  (array index 3 of data_efficiency)

#SBATCH --job-name=abl_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-4%3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/ablations/slurm_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/ablations/slurm_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal

CONFIGS=(
  configs/dinov3_vits_2d_impeccable_repeated_stride3_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d_impeccable_repeated_stride1_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_ns2_stride5_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_ns3_stride5_lora_r16_n05vols_seed42_run01_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_grid4_stride5_lora_r16_n05vols_seed42_run01_daic.yaml
)

EXP_DIRS=(
  $STUDENT_DIR/experiments/runs/ablations/2d/stride3_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/2d/stride1_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/ns2_stride5_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/ns3_stride5_n05vols/seed42_run01
  $STUDENT_DIR/experiments/runs/ablations/3ch/grid4_stride5_n05vols/seed42_run01
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
