#!/bin/bash
# Filtered-reference evaluation for one checkpoint on F3.
#
# Usage:
#   sbatch ~/RP/Code/DAIC/evaluate_filtered_ref_run.sh \
#     <config-path>          (absolute or relative to Code/DINOv3/src/) \
#     <checkpoint-path>      (absolute path to best.pt) \
#     <data-root>            (absolute path to Dataset/F3) \
#     <ref-npy>              (absolute path to f3_filtered_ref.npy) \
#     <out-dir>              (absolute path under experiments/runs/robustness/f3_filtered_ref/...)
#
# Environment overrides (alternative to positional args):
#   ROB_CONFIG, ROB_CHECKPOINT, ROB_DATA_ROOT, ROB_REF_NPY, ROB_OUT_DIR
#   MAX_SAMPLES, ORIENTATION, SAMPLE_COUNT, COMMON_CONTEXT_RADIUS

#SBATCH --job-name=filt_ref_eval
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/filt_ref_eval_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/filt_ref_eval_%j.err

set -euo pipefail

CONFIG="${ROB_CONFIG:-${1:-}}"
CHECKPOINT="${ROB_CHECKPOINT:-${2:-}}"
DATA_ROOT="${ROB_DATA_ROOT:-${3:-}}"
REF_NPY="${ROB_REF_NPY:-${4:-}}"
OUT_DIR="${ROB_OUT_DIR:-${5:-}}"

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$DATA_ROOT" ] || [ -z "$REF_NPY" ] || [ -z "$OUT_DIR" ]; then
    echo "Usage: sbatch evaluate_filtered_ref_run.sh <config> <checkpoint> <data-root> <ref-npy> <out-dir>"
    exit 2
fi

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
mkdir -p "$OUT_DIR/logs"
exec > >(tee -a "$OUT_DIR/logs/filt_ref_eval_${SLURM_JOB_ID}.out") \
     2> >(tee -a "$OUT_DIR/logs/filt_ref_eval_${SLURM_JOB_ID}.err" >&2)

echo "Filtered-reference eval job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CONFIG:       $CONFIG"
echo "CHECKPOINT:   $CHECKPOINT"
echo "DATA_ROOT:    $DATA_ROOT"
echo "REF_NPY:      $REF_NPY"
echo "OUT_DIR:      $OUT_DIR"
echo "ORIENTATION:  ${ORIENTATION:-both}"
echo "SAMPLE_COUNT: ${SAMPLE_COUNT:-32}"
echo "COMMON_CR:    ${COMMON_CONTEXT_RADIUS:-<variant-specific>}"

CODE_ROOT="$HOME/RP/Code"
PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310
VENV=/tmp/dinov3_py310_filt_ref_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_filt_ref_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

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
python -u -c "
import sys, torch
print('python:', sys.version)
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
assert sys.version_info >= (3, 10)
"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"
cd "$CODE_ROOT/DINOv3/src"

echo "=== Starting filtered-reference evaluation ==="
EXTRA_ARGS=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [ -n "${ORIENTATION:-}" ]; then
    EXTRA_ARGS+=(--orientation "$ORIENTATION")
fi
if [ -n "${SAMPLE_COUNT:-}" ]; then
    EXTRA_ARGS+=(--sample-count "$SAMPLE_COUNT")
fi
if [ -n "${COMMON_CONTEXT_RADIUS:-}" ]; then
    EXTRA_ARGS+=(--common-context-radius "$COMMON_CONTEXT_RADIUS")
fi

srun python -u evaluation/evaluate_filtered_reference.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --ref-npy "$REF_NPY" \
    --out-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

echo "Filtered-reference eval job finished at $(date)"
