#!/bin/bash
# Submit filtered-reference evaluation jobs for all main checkpoints.
#
# Discovers best.pt under experiments/runs/2d, experiments/runs/3ch, experiments/runs/5ch
# on staff-bulk, then submits one filtered-reference evaluation job per main checkpoint.
# Only main runs (2D/3ch/5ch × seed42/43/44) are submitted.
#
# Prerequisites:
#   1. f3_filtered_ref.npy must exist at $F3_DATA_ROOT/processed/f3_filtered_ref.npy
#   2. Run prepare_filtered_ref.py on the dip-steered median-filter volume first.
#
# Usage (dry-run — prints sbatch commands without submitting):
#   bash ~/RP/Code/DAIC/evaluate_filtered_ref_all.sh
#
# Usage (submit):
#   SUBMIT=1 bash ~/RP/Code/DAIC/evaluate_filtered_ref_all.sh
#
# Environment variables:
#   SUBMIT=1            Actually submit jobs (default: dry-run only).
#   F3_REF_NPY          Override path to f3_filtered_ref.npy.
#                       Default: $STUDENT_DIR/Dataset/F3/processed/f3_filtered_ref.npy
#   ONLY_FAMILY         If set, only process this family (e.g. "2d").
#   MAX_SAMPLES         Limit sections for smoke testing.
#   ORIENTATION         F3 orientation: inline, crossline, or both (default: both).
#   SAMPLE_COUNT        Sections per orientation, or "all" (default: evaluator default 32).
#   COMMON_CONTEXT_RADIUS
#                       Shared valid-center radius, e.g. 2 for comparable 2D/3ch/5ch.

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
EXP_ROOT=$STUDENT_DIR/experiments/runs
LEGACY_EXP_ROOT=$STUDENT_DIR/experiments
ROB_ROOT=$STUDENT_DIR/experiments/runs/robustness
CODE_ROOT=$HOME/RP/Code
SUBMIT="${SUBMIT:-0}"
ONLY_FAMILY="${ONLY_FAMILY:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
ORIENTATION="${ORIENTATION:-}"
SAMPLE_COUNT="${SAMPLE_COUNT:-}"
COMMON_CONTEXT_RADIUS="${COMMON_CONTEXT_RADIUS:-}"

F3_DATA_ROOT="$STUDENT_DIR/Dataset/F3"
F3_REF_NPY="${F3_REF_NPY:-$F3_DATA_ROOT/processed/f3_filtered_ref.npy}"

SCRIPT="$CODE_ROOT/DAIC/evaluate_filtered_ref_run.sh"

echo "=== Filtered-reference evaluation batch submission ==="
echo "EXP_ROOT:        $EXP_ROOT"
echo "LEGACY_EXP_ROOT: $LEGACY_EXP_ROOT"
echo "ROB_ROOT:        $ROB_ROOT"
echo "F3_REF_NPY:      $F3_REF_NPY"
echo "SUBMIT:          $SUBMIT"
if [ -n "$MAX_SAMPLES" ]; then
    echo "MAX_SAMPLES:     $MAX_SAMPLES (smoke-test mode)"
fi
if [ -n "$ORIENTATION" ]; then
    echo "ORIENTATION:     $ORIENTATION"
fi
if [ -n "$SAMPLE_COUNT" ]; then
    echo "SAMPLE_COUNT:    $SAMPLE_COUNT"
fi
if [ -n "$COMMON_CONTEXT_RADIUS" ]; then
    echo "COMMON_CR:       $COMMON_CONTEXT_RADIUS"
fi
echo ""

# Check that the filtered reference exists before doing anything else.
if [ ! -f "$F3_REF_NPY" ]; then
    echo "ERROR: Filtered reference not found at $F3_REF_NPY"
    echo "Run prepare_filtered_ref.py first:"
    echo "  python Code/DINOv3/src/data/prepare_filtered_ref.py \\"
    echo "    --input <f3_filtered_ref.segy> \\"
    echo "    --reference-npy $F3_DATA_ROOT/processed/f3_original.npy \\"
    echo "    --output $F3_DATA_ROOT/processed/"
    exit 1
fi

n_submitted=0
n_skipped=0

is_main_run() {
    local family="$1"
    local variant="$2"
    local run_id="$3"

    case "$family:$variant:$run_id" in
        2d:impeccable_repeated_stride5_lora_r16:seed42_run01|\
2d:impeccable_repeated_stride5_lora_r16:seed43_run02|\
2d:impeccable_repeated_stride5_lora_r16:seed44_run03|\
3ch:impeccable_neighbors3_stride5_lora_r16:seed42_run01|\
3ch:impeccable_neighbors3_stride5_lora_r16:seed43_run02|\
3ch:impeccable_neighbors3_stride5_lora_r16:seed44_run03|\
5ch:impeccable_neighbors5_stride5_patch_emb_lora_r16:seed42_run01|\
5ch:impeccable_neighbors5_stride5_patch_emb_lora_r16:seed43_run02|\
5ch:impeccable_neighbors5_stride5_patch_emb_lora_r16:seed44_run03)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

for family in 2d 3ch 5ch; do
    if [ -n "$ONLY_FAMILY" ] && [ "$family" != "$ONLY_FAMILY" ]; then
        continue
    fi

    family_dir=""
    for candidate_root in "$EXP_ROOT" "$LEGACY_EXP_ROOT"; do
        if [ -d "$candidate_root/$family" ]; then
            family_dir="$candidate_root/$family"
            break
        fi
    done
    if [ -z "$family_dir" ]; then
        echo "[$family] No experiment directory found; skipping."
        continue
    fi
    echo "[$family] Discovering checkpoints under $family_dir"

    while IFS= read -r -d '' ckpt_path; do
        exp_dir="$(dirname "$ckpt_path")"
        config_in_dir="$exp_dir/config.yaml"

        if [ ! -f "$config_in_dir" ]; then
            echo "  [SKIP] No config.yaml next to $ckpt_path; skipping."
            n_skipped=$((n_skipped + 1))
            continue
        fi

        rel="${exp_dir#$family_dir/}"
        variant="$(dirname "$rel")"
        run_id="$(basename "$rel")"

        if ! is_main_run "$family" "$variant" "$run_id"; then
            echo "  [SKIP] $family/$variant/$run_id is not a canonical main run."
            n_skipped=$((n_skipped + 1))
            continue
        fi

        out_dir="$ROB_ROOT/f3_filtered_ref/$family/$variant/$run_id"
        results_csv="$out_dir/f3_filtered_ref_metrics.csv"
        if [ -f "$results_csv" ] && [ "$results_csv" -nt "$ckpt_path" ]; then
            echo "  [DONE] $family/$variant/$run_id — results already exist."
            n_skipped=$((n_skipped + 1))
            continue
        fi

        extra_args=""
        if [ -n "$MAX_SAMPLES" ]; then
            extra_args="MAX_SAMPLES=$MAX_SAMPLES "
        fi
        if [ -n "$ORIENTATION" ]; then
            extra_args="${extra_args}ORIENTATION=$ORIENTATION "
        fi
        if [ -n "$SAMPLE_COUNT" ]; then
            extra_args="${extra_args}SAMPLE_COUNT=$SAMPLE_COUNT "
        fi
        if [ -n "$COMMON_CONTEXT_RADIUS" ]; then
            extra_args="${extra_args}COMMON_CONTEXT_RADIUS=$COMMON_CONTEXT_RADIUS "
        fi

        cmd="${extra_args}ROB_CONFIG=$config_in_dir ROB_CHECKPOINT=$ckpt_path ROB_DATA_ROOT=$F3_DATA_ROOT ROB_REF_NPY=$F3_REF_NPY ROB_OUT_DIR=$out_dir sbatch $SCRIPT"

        echo "  [SUBMIT] $family/$variant/$run_id"
        echo "           $cmd"

        if [ "$SUBMIT" = "1" ]; then
            ROB_CONFIG="$config_in_dir" \
            ROB_CHECKPOINT="$ckpt_path" \
            ROB_DATA_ROOT="$F3_DATA_ROOT" \
            ROB_REF_NPY="$F3_REF_NPY" \
            ROB_OUT_DIR="$out_dir" \
            MAX_SAMPLES="$MAX_SAMPLES" \
            ORIENTATION="$ORIENTATION" \
            SAMPLE_COUNT="$SAMPLE_COUNT" \
            COMMON_CONTEXT_RADIUS="$COMMON_CONTEXT_RADIUS" \
            sbatch "$SCRIPT"
            n_submitted=$((n_submitted + 1))
            sleep 0.5
        fi

    done < <(find "$family_dir" -name "best.pt" -print0 2>/dev/null | sort -z)
done

echo ""
echo "=== Summary ==="
echo "  Submitted: $n_submitted"
echo "  Skipped:   $n_skipped"
if [ "$SUBMIT" != "1" ]; then
    echo ""
    echo "Dry-run complete. Set SUBMIT=1 to actually submit jobs."
    echo ""
    echo "Remember: f3_filtered_ref.npy must exist at:"
    echo "  $F3_REF_NPY"
fi
