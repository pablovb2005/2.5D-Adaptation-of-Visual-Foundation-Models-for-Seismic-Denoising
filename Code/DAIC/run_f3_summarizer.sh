#!/bin/bash
# Run F3 allsections robustness summarizer for all completed experiment sets.
# Creates a lightweight venv (numpy + matplotlib + pyyaml; no PyTorch).
# Execute on DAIC login node: bash ~/RP/Code/DAIC/run_f3_summarizer.sh
# Or invoke via the local PowerShell runner: .\Code\DAIC\summarize_f3_robustness.ps1
set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310
CODE_ROOT=$HOME/RP/Code

ROB_ROOT=$STUDENT_DIR/experiments/runs/robustness
RESULT_DATASET=f3_allsections
SUMMARY_ROOT=$STUDENT_DIR/experiments/summaries

VENV=/tmp/dinov3_summarizer_venv_$$
TMPDIR=/tmp/dinov3_summarizer_tmp_$$
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

echo "Setting up lightweight venv..."
"$PY310" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip --no-index --find-links="$WHEELS"
python -m pip install --quiet --no-index --find-links="$WHEELS" numpy matplotlib pyyaml

SUMMARIZER="$CODE_ROOT/DINOv3/src/evaluation/summarize_robustness.py"

run_summarizer() {
    local exp_sets="$1"
    local out_label="$2"
    local out_dir="$SUMMARY_ROOT/f3_allsections_${out_label}"

    echo ""
    echo "=== Summarizing: $exp_sets → $out_dir ==="
    PYTHONPATH=$CODE_ROOT/DINOv3/src \
      python "$SUMMARIZER" \
        --robustness-root "$ROB_ROOT" \
        --result-dataset "$RESULT_DATASET" \
        --experiment-sets "$exp_sets" \
        --output-dir "$out_dir"
}

# main_multidata (always run — expected to be present)
run_summarizer "main_multidata" "main_multidata"

# full_ft_multidata (skip gracefully if not yet evaluated)
FULL_FT_DIR=$ROB_ROOT/$RESULT_DATASET/full_ft_multidata
if [ -d "$FULL_FT_DIR" ] && find "$FULL_FT_DIR" -name f3_metrics.csv -quit 2>/dev/null; then
    run_summarizer "full_ft_multidata" "full_ft_multidata"
else
    echo ""
    echo "=== full_ft_multidata: no results yet, skipping ==="
fi

echo ""
echo "=== F3 summarization complete ==="
