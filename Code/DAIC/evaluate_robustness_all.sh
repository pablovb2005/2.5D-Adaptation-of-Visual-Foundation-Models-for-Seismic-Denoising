#!/bin/bash
# Submit robustness evaluation jobs for all completed main checkpoints.
#
# Discovers best.pt under experiments/runs/2d, experiments/runs/3ch, experiments/runs/5ch
# on staff-bulk, then submits one F3 evaluation job per checkpoint.
#
# Usage (dry-run — prints sbatch commands without submitting):
#   bash ~/RP/Code/DAIC/evaluate_robustness_all.sh
#
# Usage (submit):
#   SUBMIT=1 bash ~/RP/Code/DAIC/evaluate_robustness_all.sh
#
# Environment variables:
#   SUBMIT=1            Actually submit jobs (default: dry-run only).
#   DATASETS            Space-separated list of datasets to run (default: "f3").
#   ONLY_FAMILY         If set, only process this family (e.g. "2d").
#   MAX_SAMPLES         If set, pass --max-samples to evaluate_robustness.py
#                       (smoke testing: SUBMIT=1 MAX_SAMPLES=2 bash ...).
#   ORIENTATION         F3 orientation: inline, crossline, or both (default: both).
#   SAMPLE_COUNT        Sections per orientation, or "all" (default: evaluator default 32).
#   COMMON_CONTEXT_RADIUS
#                       Shared valid-center radius, e.g. 2 for comparable 2D/3ch/5ch
#                       all-section F3 evaluation.
#   MAIN_ONLY           If 1, submit only the canonical 2D/3ch/5ch main runs.
#                       Defaults to 1 for SAMPLE_COUNT=all, otherwise 0.

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
EXP_ROOT=$STUDENT_DIR/experiments/runs
LEGACY_EXP_ROOT=$STUDENT_DIR/experiments
ROB_ROOT=$STUDENT_DIR/experiments/runs/robustness
CODE_ROOT=$HOME/RP/Code
SUBMIT="${SUBMIT:-0}"
DATASETS="${DATASETS:-f3}"
ONLY_FAMILY="${ONLY_FAMILY:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
ORIENTATION="${ORIENTATION:-}"
SAMPLE_COUNT="${SAMPLE_COUNT:-}"
COMMON_CONTEXT_RADIUS="${COMMON_CONTEXT_RADIUS:-}"
if [ -z "${MAIN_ONLY:-}" ]; then
    if [ "$SAMPLE_COUNT" = "all" ]; then
        MAIN_ONLY=1
    else
        MAIN_ONLY=0
    fi
fi

SCRIPT="$CODE_ROOT/DAIC/evaluate_robustness_run.sh"

F3_DATA_ROOT="$STUDENT_DIR/Dataset/F3"

echo "=== Robustness evaluation batch submission ==="
echo "EXP_ROOT:       $EXP_ROOT"
echo "LEGACY_EXP_ROOT:$LEGACY_EXP_ROOT"
echo "ROB_ROOT:       $ROB_ROOT"
echo "DATASETS:       $DATASETS"
echo "SUBMIT:         $SUBMIT"
echo "MAIN_ONLY:      $MAIN_ONLY"
if [ -n "$MAX_SAMPLES" ]; then
    echo "MAX_SAMPLES:    $MAX_SAMPLES (smoke-test mode)"
fi
if [ -n "$ORIENTATION" ]; then
    echo "ORIENTATION:    $ORIENTATION"
fi
if [ -n "$SAMPLE_COUNT" ]; then
    echo "SAMPLE_COUNT:   $SAMPLE_COUNT"
fi
if [ -n "$COMMON_CONTEXT_RADIUS" ]; then
    echo "COMMON_CR:      $COMMON_CONTEXT_RADIUS"
fi
echo ""

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

# Iterate over main experiment families.
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
        echo "[$family] No experiment directory found under $EXP_ROOT or $LEGACY_EXP_ROOT; skipping."
        continue
    fi
    echo "[$family] Discovering checkpoints under $family_dir"

    # Walk variant/run_id directories looking for best.pt.
    while IFS= read -r -d '' ckpt_path; do
        exp_dir="$(dirname "$ckpt_path")"
        config_in_dir="$exp_dir/config.yaml"

        if [ ! -f "$config_in_dir" ]; then
            echo "  [SKIP] No config.yaml next to $ckpt_path; skipping."
            n_skipped=$((n_skipped + 1))
            continue
        fi

        # Derive variant and run_id from the path relative to the family dir.
        rel="${exp_dir#$family_dir/}"          # e.g. impeccable_neighbors3.../seed42_run01
        variant="$(dirname "$rel")"
        run_id="$(basename "$rel")"

        if [ "$MAIN_ONLY" = "1" ] && ! is_main_run "$family" "$variant" "$run_id"; then
            echo "  [SKIP] $family/$variant/$run_id is not a canonical main run."
            n_skipped=$((n_skipped + 1))
            continue
        fi

        for dataset in $DATASETS; do
            if [ "$dataset" = "f3" ]; then
                data_root="$F3_DATA_ROOT"
                if [ ! -f "$data_root/processed/f3_original.npy" ]; then
                    echo "  [SKIP F3] f3_original.npy not found at $data_root/processed/; skipping."
                    n_skipped=$((n_skipped + 1))
                    continue
                fi
            else
                echo "  [ERROR] Unknown dataset: $dataset"
                continue
            fi

            out_dataset="$dataset"
            if [ "$dataset" = "f3" ] && [ "$SAMPLE_COUNT" = "all" ]; then
                out_dataset="f3_allsections"
            fi
            out_dir="$ROB_ROOT/$out_dataset/$family/$variant/$run_id"

            # Skip if results already exist and are newer than the checkpoint.
            results_csv="$out_dir/${dataset}_metrics.csv"
            if [ -f "$results_csv" ] && [ "$results_csv" -nt "$ckpt_path" ]; then
                echo "  [DONE] $family/$variant/$run_id ($dataset) — results already exist."
                n_skipped=$((n_skipped + 1))
                continue
            fi

            # Build the sbatch command.
            extra_args=""
            if [ -n "$MAX_SAMPLES" ]; then
                # Pass via environment so evaluate_robustness_run.sh can forward it.
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

            cmd="${extra_args}ROB_CONFIG=$config_in_dir ROB_CHECKPOINT=$ckpt_path ROB_DATASET=$dataset ROB_DATA_ROOT=$data_root ROB_OUT_DIR=$out_dir sbatch $SCRIPT"

            echo "  [SUBMIT] $family/$variant/$run_id ($dataset)"
            echo "           $cmd"

            if [ "$SUBMIT" = "1" ]; then
                ROB_CONFIG="$config_in_dir" \
                ROB_CHECKPOINT="$ckpt_path" \
                ROB_DATASET="$dataset" \
                ROB_DATA_ROOT="$data_root" \
                ROB_OUT_DIR="$out_dir" \
                MAX_SAMPLES="$MAX_SAMPLES" \
                ORIENTATION="$ORIENTATION" \
                SAMPLE_COUNT="$SAMPLE_COUNT" \
                COMMON_CONTEXT_RADIUS="$COMMON_CONTEXT_RADIUS" \
                sbatch "$SCRIPT"
                n_submitted=$((n_submitted + 1))
                sleep 0.5
            fi
        done

    done < <(find "$family_dir" -name "best.pt" -print0 2>/dev/null | sort -z)
done

echo ""
echo "=== Summary ==="
echo "  Submitted: $n_submitted"
echo "  Skipped:   $n_skipped"
if [ "$SUBMIT" != "1" ]; then
    echo ""
    echo "Dry-run complete. Set SUBMIT=1 to actually submit jobs."
fi
