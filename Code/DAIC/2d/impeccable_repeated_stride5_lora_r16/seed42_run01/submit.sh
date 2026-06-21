#!/bin/bash
# Image Impeccable 2D repeated-channel baseline.

#SBATCH --job-name=2d_s5_r16_s42_r01
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/2d/impeccable_repeated_stride5_lora_r16/seed42_run01/slurm_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/2d/impeccable_repeated_stride5_lora_r16/seed42_run01/slurm_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
EXP_DIR=$STUDENT_DIR/experiments/runs/2d/impeccable_repeated_stride5_lora_r16/seed42_run01
CONFIG=configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed42_run01_daic.yaml
LOG_DIR=$EXP_DIR/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

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
