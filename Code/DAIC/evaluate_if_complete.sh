#!/bin/bash
# Evaluate best.pt only after a logical training run reaches its target epoch.
#
# Usage:
#   bash evaluate_if_complete.sh <config-relative-to-src> <experiment-dir> [code-root]
#
# Intended to be called from DAIC training submit scripts after training/train.py.
# Partial resume chunks exit successfully without touching eval_results/.

set -euo pipefail

CONFIG="${1:-}"
EXP_DIR="${2:-}"
CODE_ROOT="${3:-${CODE_ROOT:-$HOME/RP/Code}}"
FORCE_EVAL="${FORCE_EVAL:-0}"

if [ -z "$CONFIG" ] || [ -z "$EXP_DIR" ]; then
    echo "Auto-eval usage: evaluate_if_complete.sh <config> <experiment-dir> [code-root]"
    exit 2
fi

HISTORY="$EXP_DIR/history.csv"
BEST="$EXP_DIR/best.pt"
RESULTS="$EXP_DIR/eval_results/results.csv"

echo "=== Auto-eval completion check ==="
echo "CONFIG: $CONFIG"
echo "EXP_DIR: $EXP_DIR"

if [ ! -f "$HISTORY" ]; then
    echo "Skipping test evaluation: missing history.csv at $HISTORY"
    exit 0
fi

if [ ! -f "$BEST" ]; then
    echo "Skipping test evaluation: missing best.pt at $BEST"
    exit 0
fi

if ! epoch_total=$(awk -F, '
    NR == 1 {
        for (i = 1; i <= NF; i++) {
            gsub(/\r/, "", $i)
            if ($i == "epoch") epoch_col = i
            if ($i == "total_epochs") total_col = i
        }
        next
    }
    epoch_col && total_col && $epoch_col != "" {
        e = $epoch_col
        t = $total_col
        gsub(/\r/, "", e)
        gsub(/\r/, "", t)
    }
    END {
        if (e == "" || t == "") exit 2
        print e " " t
    }
' "$HISTORY"); then
    echo "Skipping test evaluation: could not parse final epoch from $HISTORY"
    exit 0
fi

read -r LAST_EPOCH TOTAL_EPOCHS <<EOF
$epoch_total
EOF

if [ "$LAST_EPOCH" -lt "$TOTAL_EPOCHS" ]; then
    echo "Run is partial: epoch $LAST_EPOCH/$TOTAL_EPOCHS; skipping test evaluation until completion."
    exit 0
fi

if [ "$FORCE_EVAL" != "1" ] && [ -f "$RESULTS" ] && [ "$RESULTS" -nt "$BEST" ]; then
    echo "Skipping test evaluation: $RESULTS is newer than $BEST."
    echo "Set FORCE_EVAL=1 to re-run evaluation."
    exit 0
fi

echo "Run is complete: epoch $LAST_EPOCH/$TOTAL_EPOCHS."
echo "Evaluating best checkpoint: $BEST"

cd "$CODE_ROOT/DINOv3/src"

if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    srun python -u evaluation/evaluate.py --config "$CONFIG" --checkpoint "$BEST"
else
    python -u evaluation/evaluate.py --config "$CONFIG" --checkpoint "$BEST"
fi

echo "Auto-eval finished at $(date)"
