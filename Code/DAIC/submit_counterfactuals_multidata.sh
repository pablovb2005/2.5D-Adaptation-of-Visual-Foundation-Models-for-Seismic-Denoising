#!/bin/bash
# Context counterfactual evaluation for all 9 (data_seed × training_seed) combinations.
# Runs analyze_context_counterfactuals.py 9 times sequentially (venv set up once).
# Uses --full-test to evaluate on all test samples (not just 64).
#
# Submit:
#   sbatch ~/RP/Code/DAIC/submit_counterfactuals_multidata.sh

#SBATCH --job-name=cf_multidata
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/slurm_cf_multidata_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/slurm_cf_multidata_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
PROJECT_ROOT="$HOME/RP"
SUMMARIES_ROOT="$STUDENT_DIR/experiments/summaries"

DATA_SEEDS=(101 202 303)
TRAINING_SEEDS=(42 43 44)

echo "Context counterfactual multidata eval started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

if [ ! -x "$PY310" ]; then
    echo "ERROR: Python 3.10 not found: $PY310" >&2
    exit 127
fi

VENV=/tmp/dinov3_py310_cf_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_cf_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

echo "=== Setting up venv ==="
"$PY310" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"
python -m pip install --no-index --find-links="$WHEELS" \
    "torch==2.6.0+cu118" "torchvision==0.21.0+cu118"
python -m pip install --no-index --find-links="$WHEELS" \
    torchmetrics peft numpy matplotlib pyyaml termcolor einops timm submitit \
    transformers accelerate safetensors huggingface_hub
echo "Venv ready."

echo "=== GPU info ===" && nvidia-smi

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"
cd "$CODE_ROOT/DINOv3/src"

n_failed=0

echo ""
echo "=== Running 9 context counterfactual evaluations ==="
for data_seed in "${DATA_SEEDS[@]}"; do
    for tseed in "${TRAINING_SEEDS[@]}"; do
        out_dir="$SUMMARIES_ROOT/mechanism_analysis/context_counterfactuals_full/ds${data_seed}_ts${tseed}"
        echo ""
        echo "--- [data_seed=${data_seed} | tseed=${tseed}] -> $out_dir ---"
        python -u evaluation/analyze_context_counterfactuals.py \
            --project-root "$PROJECT_ROOT" \
            --data-seed "$data_seed" \
            --seed "$tseed" \
            --out-dir "$out_dir" \
            --full-test \
            --stride 5 \
            || { echo "  [WARN] failed — continuing"; n_failed=$((n_failed + 1)); }
        echo "  [OK] data_seed=${data_seed} tseed=${tseed}"
    done
done

echo ""
echo "=== Counterfactual eval complete: $n_failed run(s) failed ==="
echo "Finished at $(date)"
