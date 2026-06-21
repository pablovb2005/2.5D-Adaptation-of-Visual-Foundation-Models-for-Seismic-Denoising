#!/bin/bash
# Evaluate one Image Impeccable run.
#
# Usage:
#   sbatch ~/RP/Code/DAIC/evaluate_impeccable_run.sh \
#     configs/<config>_daic.yaml \
#     /tudelft.net/.../experiments/runs/<family>/<variant>/<run_id>/best.pt
#
# The eval CSV and example PNG are written next to the checkpoint under eval_results/.

#SBATCH --job-name=eval_imp_run
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/eval_run_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/eval_run_%j.err

set -euo pipefail

CONFIG="${CONFIG:-${1:-}}"
CHECKPOINT="${CHECKPOINT:-${2:-}}"

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ]; then
    echo "Usage: sbatch evaluate_impeccable_run.sh <config-relative-to-src> <checkpoint-path>"
    exit 2
fi

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
EXP_DIR=$(dirname "$CHECKPOINT")
LOG_DIR=$EXP_DIR/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/eval_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/eval_${SLURM_JOB_ID}.err" >&2)

echo "Evaluation job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CONFIG: $CONFIG"
echo "CHECKPOINT: $CHECKPOINT"

CODE_ROOT="$HOME/RP/Code"
PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310
VENV=/tmp/dinov3_py310_eval_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_eval_tmp_${SLURM_JOB_ID}
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

srun python -u evaluation/evaluate.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"

echo "Evaluation job finished at $(date)"
