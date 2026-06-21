#!/bin/bash
# Batch F3 robustness evaluation for all full_ft_multidata runs.
# Sets up venv ONCE, then runs all 27 evals sequentially.
# Uses --sample-count all (1594 sections) and --common-context-radius 2
# for full inline+crossline evaluation comparable across variants.
#
# Submit:
#   sbatch ~/RP/Code/DAIC/full_ft/submit_full_ft_f3_batch.sh

#SBATCH --job-name=full_ft_f3_batch
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/full_ft/slurm_f3batch_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/full_ft/slurm_f3batch_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
F3_DATA_ROOT="$STUDENT_DIR/Dataset/F3"

EXP_ROOT="$STUDENT_DIR/experiments/runs/full_ft_multidata"
ROB_ROOT="$STUDENT_DIR/experiments/runs/robustness/f3_allsections/full_ft_multidata"

DATA_SEEDS=(101 202 303)
TRAINING_SEEDS=(42 43 44)
VARIANTS=(2d 3ch 5ch)

echo "Batch F3 eval started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "EXP_ROOT:     $EXP_ROOT"
echo "ROB_ROOT:     $ROB_ROOT"

# NFS (staff-bulk) can take a moment to stabilize at job start.
# Retry up to 3 times (90 s total) before giving up.
for _nfs_try in 1 2 3; do
    [ -x "$PY310" ] && break
    echo "PY310 not accessible (attempt $_nfs_try/3), waiting 30s for NFS..." >&2
    sleep 30
done
if [ ! -x "$PY310" ]; then
    echo "ERROR: Python 3.10 interpreter not found after retries: $PY310" >&2
    exit 127
fi

if [ ! -f "$F3_DATA_ROOT/processed/f3_original.npy" ]; then
    echo "ERROR: F3 data not found at $F3_DATA_ROOT/processed/" >&2
    exit 2
fi

# One-time venv setup
VENV=/tmp/dinov3_py310_f3batch_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_f3batch_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

echo "=== Setting up venv (once for all 27 evals) ==="
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

run_one_eval() {
    local variant="$1" data_seed="$2" tseed="$3"
    local family run_base run_id

    case "$variant" in
        2d)  family=2d;  run_base=impeccable_repeated_stride5_full_ft ;;
        3ch) family=3ch; run_base=impeccable_neighbors3_stride5_full_ft ;;
        5ch) family=5ch; run_base=impeccable_neighbors5_stride5_patch_emb_full_ft ;;
        *)   echo "  [ERROR] unknown variant: $variant"; return 1 ;;
    esac

    case "$tseed" in
        42) run_id=seed42_run01 ;;
        43) run_id=seed43_run02 ;;
        44) run_id=seed44_run03 ;;
        *)  run_id="seed${tseed}_run01" ;;
    esac

    local exp_dir="$EXP_ROOT/$family/$run_base/data_seed${data_seed}/$run_id"
    local out_dir="$ROB_ROOT/$family/$run_base/data_seed${data_seed}/$run_id"
    local ckpt="$exp_dir/best.pt"
    local config="$exp_dir/runtime_config.yaml"
    local results="$out_dir/f3_metrics.csv"

    if [ ! -f "$ckpt" ]; then
        echo "  [SKIP] no checkpoint: $ckpt"
        return 0
    fi
    if [ ! -f "$config" ]; then
        echo "  [SKIP] no config: $config"
        return 0
    fi
    if [ -f "$results" ] && [ "$results" -nt "$ckpt" ]; then
        echo "  [DONE] already evaluated: $variant / data_seed${data_seed} / $run_id"
        return 0
    fi

    mkdir -p "$out_dir/logs"
    echo "  [RUN ] $variant / data_seed${data_seed} / $run_id"

    python -u evaluation/evaluate_robustness.py \
        --config "$config" \
        --checkpoint "$ckpt" \
        --dataset f3 \
        --data-root "$F3_DATA_ROOT" \
        --out-dir "$out_dir" \
        --sample-count all \
        --common-context-radius 2

    echo "  [OK  ] $variant / data_seed${data_seed} / $run_id → $out_dir/f3_metrics.csv"
}

echo ""
echo "=== Starting 27 F3 evaluations (sequential, venv shared) ==="
for variant in "${VARIANTS[@]}"; do
    for data_seed in "${DATA_SEEDS[@]}"; do
        for tseed in "${TRAINING_SEEDS[@]}"; do
            echo ""
            echo "--- [$variant | data_seed${data_seed} | tseed${tseed}] ---"
            run_one_eval "$variant" "$data_seed" "$tseed" \
                || { echo "  [WARN] failed — continuing with remaining evals"; n_failed=$((n_failed + 1)); }
        done
    done
done

echo ""
echo "=== Batch complete: $n_failed eval(s) failed ==="
echo "Finished at $(date)"
