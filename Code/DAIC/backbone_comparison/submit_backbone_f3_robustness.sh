#!/bin/bash
# Submit F3 robustness evaluation for all backbone comparison checkpoints.
#
# Reads matrix.csv and submits one evaluate_robustness_run.sh job per row
# that has a completed best.pt. Skips rows where f3_metrics.csv already exists.
#
# Usage (dry-run — prints commands without submitting):
#   bash ~/RP/Code/DAIC/backbone_comparison/submit_backbone_f3_robustness.sh
#
# Usage (submit):
#   SUBMIT=1 bash ~/RP/Code/DAIC/backbone_comparison/submit_backbone_f3_robustness.sh
#
# Output root:
#   $STUDENT_DIR/experiments/runs/robustness/f3/backbone_comparison/<backbone>/<variant>/...

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
MATRIX="$CODE_ROOT/DAIC/backbone_comparison/matrix.csv"
ROB_SCRIPT="$CODE_ROOT/DAIC/evaluate_robustness_run.sh"
F3_DATA_ROOT="$STUDENT_DIR/Dataset/F3"
SUBMIT="${SUBMIT:-0}"

if [ ! -f "$MATRIX" ]; then
    echo "ERROR: matrix.csv not found at $MATRIX"
    exit 2
fi

if [ ! -f "$F3_DATA_ROOT/processed/f3_original.npy" ]; then
    echo "ERROR: F3 data not found at $F3_DATA_ROOT/processed/f3_original.npy"
    exit 2
fi

echo "=== Backbone comparison F3 robustness batch submission ==="
echo "MATRIX:        $MATRIX"
echo "ROB_SCRIPT:    $ROB_SCRIPT"
echo "F3_DATA_ROOT:  $F3_DATA_ROOT"
echo "SUBMIT:        $SUBMIT"
echo ""

n_submitted=0
n_skipped=0
n_missing=0

while IFS=, read -r TASK_ID BACKBONE VARIANT DATA_SEED TRAINING_SEED CONFIG EXP_DIR; do
    # Skip header row
    [ "$TASK_ID" = "task_id" ] && continue

    # Strip Windows CRLF
    EXP_DIR="${EXP_DIR%$'\r'}"
    CONFIG="${CONFIG%$'\r'}"

    CKPT="$EXP_DIR/best.pt"
    CONFIG_ABS="$CODE_ROOT/DINOv3/src/$CONFIG"

    if [ ! -f "$CKPT" ]; then
        echo "  [MISSING] task $TASK_ID ($BACKBONE / $VARIANT / data$DATA_SEED / seed$TRAINING_SEED)"
        n_missing=$((n_missing + 1))
        continue
    fi

    if [ ! -f "$CONFIG_ABS" ]; then
        echo "  [NO CONFIG] task $TASK_ID: $CONFIG_ABS not found; skipping."
        n_missing=$((n_missing + 1))
        continue
    fi

    # Derive output dir by replacing .../runs/backbone_comparison/...
    # with .../runs/robustness/f3/backbone_comparison/...
    BB_ROOT="$STUDENT_DIR/experiments/runs/backbone_comparison"
    ROB_ROOT="$STUDENT_DIR/experiments/runs/robustness/f3/backbone_comparison"
    OUT_DIR="${EXP_DIR/$BB_ROOT/$ROB_ROOT}"

    if [ -f "$OUT_DIR/f3_metrics.csv" ]; then
        echo "  [DONE]    task $TASK_ID ($BACKBONE / $VARIANT / data$DATA_SEED / seed$TRAINING_SEED)"
        n_skipped=$((n_skipped + 1))
        continue
    fi

    echo "  [SUBMIT]  task $TASK_ID ($BACKBONE / $VARIANT / data$DATA_SEED / seed$TRAINING_SEED)"
    if [ "$SUBMIT" = "1" ]; then
        ROB_CONFIG="$CONFIG_ABS" \
        ROB_CHECKPOINT="$CKPT" \
        ROB_DATASET="f3" \
        ROB_DATA_ROOT="$F3_DATA_ROOT" \
        ROB_OUT_DIR="$OUT_DIR" \
        sbatch "$ROB_SCRIPT"
        n_submitted=$((n_submitted + 1))
        sleep 0.3
    fi

done < "$MATRIX"

echo ""
echo "=== Summary ==="
echo "  Submitted:       $n_submitted"
echo "  Already done:    $n_skipped"
echo "  Missing best.pt: $n_missing"
if [ "$SUBMIT" != "1" ]; then
    echo ""
    echo "Dry-run complete. Set SUBMIT=1 to actually submit jobs."
fi
