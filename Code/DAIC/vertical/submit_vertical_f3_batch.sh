#!/bin/bash
# F3 field-transfer evaluation for the VERTICAL-TRAINED Image Impeccable models.
#
# The vertical-orientation control retrains 2d/3ch/5ch on vertical inline
# Image Impeccable sections (experiments/runs/vertical_orientation/). Those
# checkpoints had only the held-out II test set auto-evaluated; they were never
# applied to F3. This job evaluates each of the 27 vertical-trained checkpoints
# on F3 in BOTH orientations:
#
#   - both      : vertical inline/crossline sections. This is the orientation
#                 that MATCHES the vertical training, and is the key diagnostic.
#   - timeslice : horizontal time slices (mismatched to vertical training,
#                 skipping the shallow no-data zone via --section-min 50).
#
# 27 checkpoints x 2 orientations = 54 evaluations.
#
# Outputs are written to fresh subtrees so they do NOT collide with the
# horizontally-trained F3 results:
#   experiments/runs/robustness/f3_vertical_trained/both/...
#   experiments/runs/robustness/f3_vertical_trained/timeslice/...
#
# Idempotent: a run whose metrics CSV is newer than its checkpoint is skipped,
# so re-submitting after a mid-job failure resumes cleanly.
#
# Submit:
#   sbatch ~/RP/Code/DAIC/vertical/submit_vertical_f3_batch.sh

#SBATCH --job-name=vert_f3_batch
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/vertical/slurm_vertf3_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/vertical/slurm_vertf3_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
F3_DATA_ROOT="$STUDENT_DIR/Dataset/F3"

EXP_ROOT="$STUDENT_DIR/experiments/runs/vertical_orientation"
ROB_ROOT="$STUDENT_DIR/experiments/runs/robustness/f3_vertical_trained"

DATA_SEEDS=(101 202 303)
TRAINING_SEEDS=(42 43 44)
VARIANTS=(2d 3ch 5ch)

echo "Vertical-trained F3 batch started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "EXP_ROOT:     $EXP_ROOT"
echo "ROB_ROOT:     $ROB_ROOT"

# NFS (staff-bulk) can take a moment to stabilize at job start.
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

# One-time venv setup.
VENV=/tmp/dinov3_py310_vertf3_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_vertf3_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

echo "=== Setting up venv (once for all 54 evals) ==="
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

n_done=0
n_alreadydone=0
n_missing=0
n_failed=0

run_base_for() {  # variant -> run_base (LoRA, matches vertical_orientation layout)
    case "$1" in
        2d)  echo impeccable_repeated_stride5_lora_r16 ;;
        3ch) echo impeccable_neighbors3_stride5_lora_r16 ;;
        5ch) echo impeccable_neighbors5_stride5_patch_emb_lora_r16 ;;
    esac
}
run_id_for() {  # training seed -> run_id
    case "$1" in
        42) echo seed42_run01 ;;
        43) echo seed43_run02 ;;
        44) echo seed44_run03 ;;
        *)  echo "seed${1}_run01" ;;
    esac
}

run_one_eval() {
    local orientation="$1" variant="$2" data_seed="$3" tseed="$4"
    local run_base run_id
    run_base="$(run_base_for "$variant")"
    run_id="$(run_id_for "$tseed")"

    local exp_dir="$EXP_ROOT/$variant/$run_base/data_seed${data_seed}/$run_id"
    local out_dir="$ROB_ROOT/$orientation/$variant/$run_base/data_seed${data_seed}/$run_id"
    local ckpt="$exp_dir/best.pt"
    local config="$exp_dir/runtime_config.yaml"
    local results="$out_dir/f3_metrics.csv"

    if [ ! -f "$ckpt" ];   then echo "  [MISS] $orientation/$variant/data${data_seed}/$run_id — no checkpoint: $ckpt"; n_missing=$((n_missing+1)); return 0; fi
    if [ ! -f "$config" ]; then echo "  [MISS] $orientation/$variant/data${data_seed}/$run_id — no config: $config";  n_missing=$((n_missing+1)); return 0; fi
    if [ -f "$results" ] && [ "$results" -nt "$ckpt" ]; then
        echo "  [DONE] $orientation/$variant/data${data_seed}/$run_id — already evaluated"; n_alreadydone=$((n_alreadydone+1)); return 0
    fi

    # Orientation-specific args. timeslice skips the shallow F3 no-data zone.
    local orient_args
    if [ "$orientation" = "timeslice" ]; then
        orient_args=(--orientation timeslice --sample-count all --common-context-radius 2 --section-min 50)
    else
        orient_args=(--orientation both --sample-count all --common-context-radius 2)
    fi

    mkdir -p "$out_dir/logs"
    echo "  [RUN ] $orientation/$variant/data${data_seed}/$run_id"
    if python -u evaluation/evaluate_robustness.py \
            --config "$config" --checkpoint "$ckpt" --dataset f3 \
            --data-root "$F3_DATA_ROOT" --out-dir "$out_dir" "${orient_args[@]}"; then
        echo "  [OK  ] $orientation/$variant/data${data_seed}/$run_id -> $results"; n_done=$((n_done+1)); return 0
    else
        echo "  [FAIL] $orientation/$variant/data${data_seed}/$run_id"; n_failed=$((n_failed+1)); return 1
    fi
}

for orientation in both timeslice; do
    echo ""
    echo "############## ORIENTATION: $orientation ##############"
    for variant in "${VARIANTS[@]}"; do
        for data_seed in "${DATA_SEEDS[@]}"; do
            for tseed in "${TRAINING_SEEDS[@]}"; do
                run_one_eval "$orientation" "$variant" "$data_seed" "$tseed" || true
            done
        done
    done
done

n_total=$((n_done + n_alreadydone + n_missing + n_failed))
echo ""
echo "=== Vertical-trained F3 batch complete ==="
echo "  expected:        54"
echo "  evaluated (RUN): $n_done"
echo "  already done:    $n_alreadydone"
echo "  missing inputs:  $n_missing"
echo "  failed:          $n_failed"
echo "  accounted for:   $n_total"
echo "Finished at $(date)"
if [ "$n_missing" -gt 0 ]; then
    echo "WARNING: $n_missing run(s) had missing checkpoint/config — check the [MISS] lines above." >&2
fi
if [ "$n_failed" -gt 0 ]; then exit 1; fi
if [ "$((n_done + n_alreadydone))" -eq 0 ]; then
    echo "ERROR: no evaluations ran and none were already done — paths are likely wrong." >&2
    exit 3
fi
exit 0
