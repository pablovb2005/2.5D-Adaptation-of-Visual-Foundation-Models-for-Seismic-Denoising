#!/bin/bash
# Run stitched overlap evaluation on the 9 main checkpoints (2D/3ch/5ch × seeds 42/43/44).
# Results are written under each checkpoint directory's stitched_eval_results/ folder.
# After all runs complete, run summarize_stitched.py to aggregate.
#
# Usage:
#   sbatch ~/RP/Code/DAIC/evaluate_stitched_all.sh

#SBATCH --job-name=eval_stitched
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/eval_stitched_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/eval_stitched_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
# All fixed-split main checkpoints (seeds 42/43/44) live under the canonical
# experiments/runs/{2d,3ch,5ch}/ layout, matching the EXP_DIR in each submit.sh.
EXP_BASE=$STUDENT_DIR/experiments/runs
LOG_DIR=$STUDENT_DIR/experiments/runs/system/stitched_eval/logs
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/eval_stitched_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/eval_stitched_${SLURM_JOB_ID}.err" >&2)

echo "Stitched evaluation job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

CODE_ROOT="$HOME/RP/Code"
PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310
VENV=/tmp/dinov3_py310_stitched_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_stitched_tmp_${SLURM_JOB_ID}
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

# 9 main checkpoints: 2D/3ch/5ch × seeds 42/43/44 (data.seed=42 fixed-split)
# Configs from configs/ (relative to $CODE_ROOT/DINOv3/src) so that the relative
# weights path in each config resolves correctly via ../../weights/.
declare -a CONFIGS=(
    "configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed42_run01_daic.yaml"
    "configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed43_run02_daic.yaml"
    "configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_seed44_run03_daic.yaml"
    "configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_seed42_run01_daic.yaml"
    "configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_seed43_run02_daic.yaml"
    "configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_seed44_run03_daic.yaml"
    "configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_seed42_run01_daic.yaml"
    "configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_seed43_run02_daic.yaml"
    "configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_seed44_run03_daic.yaml"
)

declare -a RUNS=(
    "2d/impeccable_repeated_stride5_lora_r16/seed42_run01"
    "2d/impeccable_repeated_stride5_lora_r16/seed43_run02"
    "2d/impeccable_repeated_stride5_lora_r16/seed44_run03"
    "3ch/impeccable_neighbors3_stride5_lora_r16/seed42_run01"
    "3ch/impeccable_neighbors3_stride5_lora_r16/seed43_run02"
    "3ch/impeccable_neighbors3_stride5_lora_r16/seed44_run03"
    "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed42_run01"
    "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed43_run02"
    "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed44_run03"
)

for i in "${!RUNS[@]}"; do
    RUN="${RUNS[$i]}"
    CONFIG="${CONFIGS[$i]}"
    CKPT="$EXP_BASE/$RUN/best.pt"

    echo
    echo "=== Stitched eval: $RUN ==="

    if [ ! -f "$CKPT" ]; then
        echo "Missing checkpoint, skipping: $CKPT"
        continue
    fi

    RESULTS="$EXP_BASE/$RUN/stitched_eval_results/results.csv"
    EXAMPLE_META="$EXP_BASE/$RUN/stitched_eval_results/stitched_example_meta.json"

    if [ -f "$RESULTS" ] && [ -f "$EXAMPLE_META" ]; then
        echo "Already done, skipping: $RUN"
        continue
    fi

    if [ -f "$RESULTS" ]; then
        echo "Metrics already exist; refreshing representative example only."
        python -u evaluation/evaluate_stitched.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            --example-only
    else
        python -u evaluation/evaluate_stitched.py \
            --config "$CONFIG" \
            --checkpoint "$CKPT"
    fi
done

echo
echo "=== All stitched evaluations done at $(date) ==="
echo "Run summarize_stitched.py to aggregate results:"
echo "  python evaluation/summarize_stitched.py --project-root $STUDENT_DIR"
