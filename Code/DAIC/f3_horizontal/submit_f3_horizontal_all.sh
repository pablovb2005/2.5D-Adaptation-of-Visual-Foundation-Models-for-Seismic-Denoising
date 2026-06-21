#!/bin/bash
# Consolidated F3 HORIZONTAL (time-slice) re-evaluation — ALL models in ONE job.
#
# Why: the models train on horizontal depth/time slices of Image Impeccable
# (vol[:, :, k] planes), but F3 was previously evaluated on vertical
# inline/crossline sections. This job re-evaluates F3 with the matching
# horizontal orientation (--orientation timeslice), skipping the shallow F3
# no-data zone via --section-min 50.
#
# Covers, sequentially, in a single venv (queue-friendly):
#   - main_multidata     : 27 (2d/3ch/5ch x data{101,202,303} x seed{42,43,44})
#   - full_ft_multidata  : 27
#   - backbone_comparison: 54 (sfm_vit_base_patch16 + swin_v2_t, from matrix.csv)
#   - filtered_reference :  9 (2d/3ch/5ch x seed{42,43,44}, old fixed-split LoRA)
#   = 117 evaluations
#
# Idempotent: a run whose results CSV is newer than its checkpoint is skipped,
# so re-submitting after a mid-job failure resumes cleanly.
#
# Submit:
#   sbatch ~/RP/Code/DAIC/f3_horizontal/submit_f3_horizontal_all.sh

#SBATCH --job-name=f3_horiz_all
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/f3_horizontal/slurm_f3horiz_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/f3_horizontal/slurm_f3horiz_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
F3_DATA_ROOT="$STUDENT_DIR/Dataset/F3"
F3_REF_NPY="$F3_DATA_ROOT/processed/f3_filtered_ref.npy"

RUNS_ROOT="$STUDENT_DIR/experiments/runs"
# New horizontal output subtrees (old inline/crossline results are left untouched).
ROB_HORIZ="$RUNS_ROOT/robustness/f3_horizontal"
ROB_FILT_HORIZ="$RUNS_ROOT/robustness/f3_filtered_ref_horizontal"

MATRIX="$CODE_ROOT/DAIC/backbone_comparison/matrix.csv"

# Common horizontal-evaluation arguments for every run.
COMMON_ARGS=(--orientation timeslice --sample-count all --common-context-radius 2 --section-min 50)

DATA_SEEDS=(101 202 303)
TRAINING_SEEDS=(42 43 44)
VARIANTS=(2d 3ch 5ch)

echo "F3 horizontal batch started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "ROB_HORIZ:        $ROB_HORIZ"
echo "ROB_FILT_HORIZ:   $ROB_FILT_HORIZ"
echo "COMMON_ARGS:      ${COMMON_ARGS[*]}"

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
if [ ! -f "$F3_REF_NPY" ]; then
    echo "WARN: filtered reference not found at $F3_REF_NPY; filtered-ref block will skip." >&2
fi
if [ ! -f "$MATRIX" ]; then
    echo "WARN: backbone matrix.csv not found at $MATRIX; backbone block will skip." >&2
fi

# One-time venv setup.
VENV=/tmp/dinov3_py310_f3horiz_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_f3horiz_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

echo "=== Setting up venv (once for all evaluations) ==="
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

# Counters kept separate so the log/e-mail distinguishes "nothing to do (resume)"
# from "paths are wrong / inputs missing" — an all-missing run must NOT look like success.
n_done=0
n_alreadydone=0
n_missing=0
n_failed=0

# --- Generic no-reference F3 eval (main / full_ft / backbone) ---------------
run_noref() {
    local label="$1" config="$2" ckpt="$3" out_dir="$4"
    local results="$out_dir/f3_metrics.csv"

    if [ ! -f "$ckpt" ];   then echo "  [MISS] $label — no checkpoint: $ckpt"; n_missing=$((n_missing+1)); return 0; fi
    if [ ! -f "$config" ]; then echo "  [MISS] $label — no config: $config";  n_missing=$((n_missing+1)); return 0; fi
    if [ -f "$results" ] && [ "$results" -nt "$ckpt" ]; then
        echo "  [DONE] $label — already evaluated"; n_alreadydone=$((n_alreadydone+1)); return 0
    fi

    mkdir -p "$out_dir/logs"
    echo "  [RUN ] $label"
    if python -u evaluation/evaluate_robustness.py \
            --config "$config" --checkpoint "$ckpt" --dataset f3 \
            --data-root "$F3_DATA_ROOT" --out-dir "$out_dir" "${COMMON_ARGS[@]}"; then
        echo "  [OK  ] $label -> $results"; n_done=$((n_done+1)); return 0
    else
        echo "  [FAIL] $label"; n_failed=$((n_failed+1)); return 1
    fi
}

# --- Filtered-reference F3 eval --------------------------------------------
run_filtref() {
    local label="$1" config="$2" ckpt="$3" out_dir="$4"
    local results="$out_dir/f3_filtered_ref_metrics.csv"

    if [ ! -f "$F3_REF_NPY" ]; then echo "  [MISS] $label — no filtered ref"; n_missing=$((n_missing+1)); return 0; fi
    if [ ! -f "$ckpt" ];   then echo "  [MISS] $label — no checkpoint: $ckpt"; n_missing=$((n_missing+1)); return 0; fi
    if [ ! -f "$config" ]; then echo "  [MISS] $label — no config: $config";  n_missing=$((n_missing+1)); return 0; fi
    if [ -f "$results" ] && [ "$results" -nt "$ckpt" ]; then
        echo "  [DONE] $label — already evaluated"; n_alreadydone=$((n_alreadydone+1)); return 0
    fi

    mkdir -p "$out_dir/logs"
    echo "  [RUN ] $label"
    if python -u evaluation/evaluate_filtered_reference.py \
            --config "$config" --checkpoint "$ckpt" \
            --data-root "$F3_DATA_ROOT" --ref-npy "$F3_REF_NPY" \
            --out-dir "$out_dir" "${COMMON_ARGS[@]}"; then
        echo "  [OK  ] $label -> $results"; n_done=$((n_done+1)); return 0
    else
        echo "  [FAIL] $label"; n_failed=$((n_failed+1)); return 1
    fi
}

run_base_for() {  # variant -> run_base (LoRA main_multidata)
    case "$1" in
        2d)  echo impeccable_repeated_stride5_lora_r16 ;;
        3ch) echo impeccable_neighbors3_stride5_lora_r16 ;;
        5ch) echo impeccable_neighbors5_stride5_patch_emb_lora_r16 ;;
    esac
}
run_base_full_ft_for() {  # variant -> run_base (full fine-tuning)
    case "$1" in
        2d)  echo impeccable_repeated_stride5_full_ft ;;
        3ch) echo impeccable_neighbors3_stride5_full_ft ;;
        5ch) echo impeccable_neighbors5_stride5_patch_emb_full_ft ;;
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

# ===========================================================================
echo ""
echo "############## BLOCK 1/4: main_multidata (27) ##############"
for variant in "${VARIANTS[@]}"; do
    run_base="$(run_base_for "$variant")"
    for ds in "${DATA_SEEDS[@]}"; do
        for ts in "${TRAINING_SEEDS[@]}"; do
            rid="$(run_id_for "$ts")"
            exp="$RUNS_ROOT/main_multidata/$variant/$run_base/data_seed${ds}/$rid"
            out="$ROB_HORIZ/main_multidata/$variant/$run_base/data_seed${ds}/$rid"
            run_noref "main/$variant/data${ds}/$rid" "$exp/runtime_config.yaml" "$exp/best.pt" "$out" || true
        done
    done
done

echo ""
echo "############## BLOCK 2/4: full_ft_multidata (27) ##############"
for variant in "${VARIANTS[@]}"; do
    run_base="$(run_base_full_ft_for "$variant")"
    for ds in "${DATA_SEEDS[@]}"; do
        for ts in "${TRAINING_SEEDS[@]}"; do
            rid="$(run_id_for "$ts")"
            exp="$RUNS_ROOT/full_ft_multidata/$variant/$run_base/data_seed${ds}/$rid"
            out="$ROB_HORIZ/full_ft_multidata/$variant/$run_base/data_seed${ds}/$rid"
            run_noref "full_ft/$variant/data${ds}/$rid" "$exp/runtime_config.yaml" "$exp/best.pt" "$out" || true
        done
    done
done

echo ""
echo "############## BLOCK 3/4: backbone_comparison (54) ##############"
if [ -f "$MATRIX" ]; then
    BB_ROOT="$RUNS_ROOT/backbone_comparison"
    BB_HORIZ="$ROB_HORIZ/backbone_comparison"
    while IFS=, read -r TASK_ID BACKBONE VARIANT DATA_SEED TRAINING_SEED CONFIG EXP_DIR; do
        [ "$TASK_ID" = "task_id" ] && continue
        [ -z "${TASK_ID:-}" ] && continue
        EXP_DIR="${EXP_DIR%$'\r'}"; CONFIG="${CONFIG%$'\r'}"
        CONFIG_ABS="$CODE_ROOT/DINOv3/src/$CONFIG"
        OUT_DIR="${EXP_DIR/$BB_ROOT/$BB_HORIZ}"
        run_noref "backbone/$BACKBONE/$VARIANT/data${DATA_SEED}/seed${TRAINING_SEED}" \
            "$CONFIG_ABS" "$EXP_DIR/best.pt" "$OUT_DIR" || true
    done < "$MATRIX"
else
    echo "  [SKIP] no matrix.csv"
fi

echo ""
echo "############## BLOCK 4/4: filtered_reference (9) ##############"
# Old fixed-split main LoRA checkpoints live under runs/<family> or experiments/<family>.
resolve_family_dir() {  # family -> existing dir or empty
    local family="$1"
    for root in "$RUNS_ROOT" "$STUDENT_DIR/experiments"; do
        if [ -d "$root/$family" ]; then echo "$root/$family"; return 0; fi
    done
    echo ""
}
for variant in "${VARIANTS[@]}"; do
    run_base="$(run_base_for "$variant")"
    fam_dir="$(resolve_family_dir "$variant")"
    if [ -z "$fam_dir" ]; then echo "  [SKIP] no family dir for $variant"; continue; fi
    for ts in "${TRAINING_SEEDS[@]}"; do
        rid="$(run_id_for "$ts")"
        exp="$fam_dir/$run_base/$rid"
        out="$ROB_FILT_HORIZ/$variant/$run_base/$rid"
        run_filtref "filt/$variant/$rid" "$exp/config.yaml" "$exp/best.pt" "$out" || true
    done
done

n_total=$((n_done + n_alreadydone + n_missing + n_failed))
echo ""
echo "=== F3 horizontal batch complete ==="
echo "  expected:        117"
echo "  evaluated (RUN): $n_done"
echo "  already done:    $n_alreadydone"
echo "  missing inputs:  $n_missing"
echo "  failed:          $n_failed"
echo "  accounted for:   $n_total"
echo "Finished at $(date)"
if [ "$n_missing" -gt 0 ]; then
    echo "WARNING: $n_missing run(s) had missing checkpoint/config/ref — check the [MISS] lines above." >&2
fi
# Fail the job if any eval errored, or if nothing actually ran AND nothing was
# already done (i.e. every path was wrong) so an all-skip run is not a silent success.
if [ "$n_failed" -gt 0 ]; then exit 1; fi
if [ "$((n_done + n_alreadydone))" -eq 0 ]; then
    echo "ERROR: no evaluations ran and none were already done — paths are likely wrong." >&2
    exit 3
fi
exit 0
