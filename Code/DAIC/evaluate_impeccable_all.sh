#!/bin/bash
# Evaluate saved best checkpoints for all Image Impeccable runs.
# Results are written under each experiment folder's eval_results/ directory.

#SBATCH --job-name=eval_impeccable
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/eval_impeccable_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/eval_impeccable_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
LOG_DIR=$STUDENT_DIR/experiments/runs/system/evaluation_jobs/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/eval_impeccable_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/eval_impeccable_${SLURM_JOB_ID}.err" >&2)

echo "Evaluation job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

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

CONFIGS=(
    "configs/dinov3_vits_2d_impeccable_daic.yaml"
    "configs/dinov3_vits_2d5_3ch_impeccable_daic.yaml"
    "configs/dinov3_vits_2d5_5ch_a_impeccable_daic.yaml"
    "configs/dinov3_vits_2d5_5ch_b_impeccable_daic.yaml"
)

CHECKPOINTS=(
    "$STUDENT_DIR/experiments/runs/2d/impeccable_v1/best.pt"
    "$STUDENT_DIR/experiments/runs/3ch/impeccable_v1/best.pt"
    "$STUDENT_DIR/experiments/runs/5ch/impeccable_a_patch_emb_head_v1/best.pt"
    "$STUDENT_DIR/experiments/runs/5ch/impeccable_b_patch_emb_lora_v1/best.pt"
)

LABELS=(
    "2D repeated-channel"
    "2.5D 3-channel"
    "2.5D 5-channel A PatchEmb+decoder"
    "2.5D 5-channel B PatchEmb+LoRA"
)

for i in "${!CONFIGS[@]}"; do
    echo
    echo "=== Evaluating ${LABELS[$i]} ==="
    echo "Config:     ${CONFIGS[$i]}"
    echo "Checkpoint: ${CHECKPOINTS[$i]}"

    if [ ! -f "${CHECKPOINTS[$i]}" ]; then
        echo "Missing checkpoint, skipping: ${CHECKPOINTS[$i]}"
        continue
    fi

    srun python -u evaluation/evaluate.py \
        --config "${CONFIGS[$i]}" \
        --checkpoint "${CHECKPOINTS[$i]}"
done

echo "Evaluation job finished at $(date)"
