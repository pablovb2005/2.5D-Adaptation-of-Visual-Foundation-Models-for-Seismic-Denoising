#!/bin/bash
# Submit SFM/SwinV2 backbone-comparison PEFT runs.
#
# Full matrix:
#   sbatch --array=0-53%18 ~/RP/Code/DAIC/backbone_comparison/submit_backbone_comparison.sh
#
# Pilot matrix:
#   sbatch --array=0,9,18,27,36,45 ~/RP/Code/DAIC/backbone_comparison/submit_backbone_comparison.sh

#SBATCH --job-name=bb_cmp
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/backbone_comparison/slurm_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/backbone_comparison/slurm_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
MATRIX="$CODE_ROOT/DAIC/backbone_comparison/matrix.csv"
TASK_ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"

if [ ! -f "$MATRIX" ]; then
    echo "Missing matrix: $MATRIX"
    echo "Run Code/DAIC/backbone_comparison/generate_backbone_configs.py before submission."
    exit 2
fi

ROW=$(awk -F, -v id="$TASK_ID" 'NR > 1 && $1 == id {print; exit}' "$MATRIX")
if [ -z "$ROW" ]; then
    echo "No matrix row for task ID $TASK_ID"
    exit 2
fi
ROW="${ROW%$'\r'}"

IFS=, read -r MATRIX_ID BACKBONE VARIANT DATA_SEED TRAINING_SEED CONFIG EXP_DIR <<EOF
$ROW
EOF
EXP_DIR="${EXP_DIR%$'\r'}"

LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_ID}.out") \
    2> >(tee -a "$LOG_DIR/slurm_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID: $TASK_ID"
echo "BACKBONE: $BACKBONE"
echo "VARIANT: $VARIANT"
echo "DATA_SEED: $DATA_SEED"
echo "TRAINING_SEED: $TRAINING_SEED"
echo "CONFIG: $CONFIG"
echo "EXP_DIR: $EXP_DIR"

PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310

if [ ! -x "$PY310" ]; then
    echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
    exit 127
fi

CONFIG_ABS="$CODE_ROOT/DINOv3/src/$CONFIG"
if [ ! -f "$CONFIG_ABS" ]; then
    echo "ERROR: Config not found: $CONFIG_ABS" >&2
    exit 2
fi

export CONFIG_ABS
WEIGHTS_REL=$("$PY310" - <<'PY'
import os
from pathlib import Path

config = Path(os.environ["CONFIG_ABS"])
for line in config.read_text().splitlines():
    stripped = line.strip()
    if stripped.startswith("weights:"):
        print(stripped.split(":", 1)[1].strip().strip('"').strip("'"))
        break
else:
    raise SystemExit("ERROR: no model.weights entry found")
PY
)
export WEIGHTS_REL
WEIGHTS_ABS=$("$PY310" - <<'PY'
import os
from pathlib import Path

print((Path(os.environ["CONFIG_ABS"]).parent / os.environ["WEIGHTS_REL"]).resolve())
PY
)

if [ ! -f "$WEIGHTS_ABS" ]; then
    echo "ERROR: Model weights not found." >&2
    echo "CONFIG_ABS: $CONFIG_ABS" >&2
    echo "WEIGHTS_REL: $WEIGHTS_REL" >&2
    echo "WEIGHTS_ABS: $WEIGHTS_ABS" >&2
    exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is unavailable on $(hostname); expected a GPU allocation." >&2
    exit 3
fi
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "ERROR: no visible NVIDIA GPU on $(hostname); expected --gres=gpu:1 allocation." >&2
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
    exit 3
fi

echo "CONFIG_ABS: $CONFIG_ABS"
echo "WEIGHTS_ABS: $WEIGHTS_ABS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

VENV=/tmp/dinov3_py310_venv_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_tmp_${SLURM_JOB_ID}
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
python -u -c "import sys, torch, torchvision; print('python:', sys.version); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); assert sys.version_info >= (3, 10)"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"

cd "$CODE_ROOT/DINOv3/src"
echo "=== Starting training ==="

srun python -u training/train.py --config "$CONFIG"

bash "$CODE_ROOT/DAIC/evaluate_if_complete.sh" "$CONFIG" "$EXP_DIR" "$CODE_ROOT"

echo "Job finished at $(date)"
