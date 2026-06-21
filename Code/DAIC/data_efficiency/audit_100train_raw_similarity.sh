#!/bin/bash
# Audit raw noisy-center-vs-clean similarity on the 120-pair canonical dataset.
#
# This is a dataset sanity check for the data-efficiency study, not model
# evaluation. It reuses the exact Image Impeccable split settings used by the
# 100-train sweep and writes per-batch, per-volume, and per-source CSVs.
#
# Submit default data_seed=101 held-out audit:
#   sbatch ~/RP/Code/DAIC/data_efficiency/audit_100train_raw_similarity.sh
#
# Override split seed or modes:
#   DATA_SEED=202 sbatch ~/RP/Code/DAIC/data_efficiency/audit_100train_raw_similarity.sh
#   MODES="2d 2.5d_5ch" sbatch ~/RP/Code/DAIC/data_efficiency/audit_100train_raw_similarity.sh

#SBATCH --job-name=ii100_raw
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_raw100_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_raw100_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
DATASET_NAME="${DATASET_NAME:-data_efficiency_100train_canonical_v2}"
SUMMARY_STUDY_NAME="${SUMMARY_STUDY_NAME:-data_efficiency_100train_v2}"
DATA_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/$DATASET_NAME/extracted"
DATA_SEED="${DATA_SEED:-101}"
OUT_ROOT="$STUDENT_DIR/experiments/summaries/$SUMMARY_STUDY_NAME/raw_similarity/data_seed${DATA_SEED}"
MODES="${MODES:-2d 2.5d_3ch 2.5d_5ch}"
SPLITS="${SPLITS:-val test}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-manual}"
echo "DATA_ROOT: $DATA_ROOT"
echo "OUT_ROOT: $OUT_ROOT"
echo "DATASET_NAME: $DATASET_NAME"
echo "DATA_SEED: $DATA_SEED"
echo "MODES: $MODES"
echo "SPLITS: $SPLITS"

if [ ! -x "$PY310" ]; then
  echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
  exit 127
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "ERROR: canonical dataset root not found: $DATA_ROOT" >&2
  exit 2
fi

VENV=/tmp/ii100_raw_venv_${SLURM_JOB_ID:-manual}_$$
TMPDIR=/tmp/ii100_raw_tmp_${SLURM_JOB_ID:-manual}_$$
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

for MODE in $MODES; do
  case "$MODE" in
    2d) LABEL=2d ;;
    2.5d_3ch) LABEL=3ch ;;
    2.5d_5ch) LABEL=5ch ;;
    *)
      echo "ERROR: unsupported MODE=$MODE" >&2
      exit 2
      ;;
  esac

  echo "=== Auditing mode=$MODE at $(date) ==="
  python -u evaluation/audit_raw_image_impeccable.py \
    --root-dir "$DATA_ROOT" \
    --output-dir "$OUT_ROOT/$LABEL" \
    --mode "$MODE" \
    --splits $SPLITS \
    --n-train 100 \
    --n-val 10 \
    --n-test 10 \
    --seed "$DATA_SEED" \
    --slice-stride 5 \
    --crop-size 224 \
    --batch-size "$BATCH_SIZE"
done

echo "=== Wrote audit files ==="
find "$OUT_ROOT" -maxdepth 2 -type f | sort
echo "Job finished at $(date)"
