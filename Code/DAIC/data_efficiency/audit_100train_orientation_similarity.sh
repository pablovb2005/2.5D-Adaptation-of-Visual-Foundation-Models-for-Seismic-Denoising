#!/bin/bash
# Test whether low raw similarity in the 100-train canonical root is explained
# by a simple clean-target orientation transform.
#
# Submit:
#   sbatch ~/RP/Code/DAIC/data_efficiency/audit_100train_orientation_similarity.sh

#SBATCH --job-name=ii100_orient
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_orient100_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_orient100_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
DATASET_NAME="${DATASET_NAME:-data_efficiency_100train_canonical_v2}"
SUMMARY_STUDY_NAME="${SUMMARY_STUDY_NAME:-data_efficiency_100train_v2}"
DATA_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/$DATASET_NAME/extracted"
DATA_SEED="${DATA_SEED:-101}"
OUT_ROOT="$STUDENT_DIR/experiments/summaries/$SUMMARY_STUDY_NAME/orientation_similarity/data_seed${DATA_SEED}"
SPLITS="${SPLITS:-val test}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "Job started on $(hostname) at $(date)"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUT_ROOT: $OUT_ROOT"
echo "DATASET_NAME: $DATASET_NAME"
echo "DATA_SEED: $DATA_SEED"

if [ ! -x "$PY310" ]; then
  echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
  exit 127
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: canonical dataset root not found: $DATA_ROOT" >&2
  exit 2
fi

VENV=/tmp/ii100_orient_venv_${SLURM_JOB_ID:-manual}_$$
TMPDIR=/tmp/ii100_orient_tmp_${SLURM_JOB_ID:-manual}_$$
mkdir -p "$TMPDIR" "$OUT_ROOT"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

"$PY310" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"
python -m pip install --quiet --no-index --find-links="$WHEELS" \
  numpy pyyaml torchmetrics "torch==2.6.0+cu118" "torchvision==0.21.0+cu118"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"
cd "$CODE_ROOT/DINOv3/src"

python -u evaluation/audit_raw_orientation_impeccable.py \
  --root-dir "$DATA_ROOT" \
  --output-dir "$OUT_ROOT" \
  --mode 2d \
  --splits $SPLITS \
  --n-train 100 \
  --n-val 10 \
  --n-test 10 \
  --seed "$DATA_SEED" \
  --slice-stride 5 \
  --crop-size 224 \
  --batch-size "$BATCH_SIZE"

echo "=== Wrote orientation audit files ==="
find "$OUT_ROOT" -maxdepth 1 -type f | sort
echo "Job finished at $(date)"
