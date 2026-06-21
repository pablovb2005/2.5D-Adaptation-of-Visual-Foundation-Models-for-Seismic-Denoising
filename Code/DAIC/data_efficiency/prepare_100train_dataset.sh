#!/bin/bash
# Prepare a repaired smaller canonical ThinkOnward dataset for the large data-efficiency sweep.
#
# Default target:
#   - existing repaired parts 1-2: 30 pairs
#   - official parts 3-8: 90 pairs
#   - total: 120 pairs with a 100/10/10 volume-level split
#   - clean-target orientation selected automatically from noisy/clean alignment
#
# Submit:
#   sbatch ~/RP/Code/DAIC/data_efficiency/prepare_100train_dataset.sh
#
# Resume after timeout:
#   sbatch ~/RP/Code/DAIC/data_efficiency/prepare_100train_dataset.sh

#SBATCH --job-name=ii100_prep
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_prep100_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_prep100_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
TOOL="$CODE_ROOT/DAIC/tools/prepare_thinkonward_full.py"
CURRENT_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/extracted"
DATASET_NAME="${DATASET_NAME:-data_efficiency_100train_canonical_v2}"
FULL_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/$DATASET_NAME/extracted"
STAGING_DIR="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/${DATASET_NAME}_staging"
PARTS="${PARTS:-3 4 5 6 7 8}"
TARGET_PAIRS="${TARGET_PAIRS:-120}"
N_TRAIN="${N_TRAIN:-100}"
N_VAL="${N_VAL:-10}"
N_TEST="${N_TEST:-10}"
MIN_INITIAL_FREE_GB="${MIN_INITIAL_FREE_GB:-180}"
BASE_URL=https://xeek-public-287031953319-eb80.s3.amazonaws.com/image-impeccable

download_archive() {
  local url="$1"
  local partial="$2"
  local archive="$3"

  if [ -f "$archive" ]; then
    echo "Using existing staged archive: $archive"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
      --output "$partial" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --tries=5 --waitretry=10 --output-document="$partial" "$url"
  else
    echo "ERROR: neither curl nor wget is available on this node." >&2
    exit 127
  fi
  mv "$partial" "$archive"
}

expected_total=30
for PART in $PARTS; do
  if [ "$PART" = "17" ]; then
    expected_total=$((expected_total + 10))
  else
    expected_total=$((expected_total + 15))
  fi
done

if [ "$expected_total" -ne "$TARGET_PAIRS" ]; then
  echo "ERROR: PARTS='$PARTS' gives $expected_total pairs, but TARGET_PAIRS=$TARGET_PAIRS." >&2
  echo "       Adjust PARTS or TARGET_PAIRS so the audit target matches the imported archives." >&2
  exit 2
fi

if [ $((N_TRAIN + N_VAL + N_TEST)) -ne "$TARGET_PAIRS" ]; then
  echo "ERROR: split $N_TRAIN/$N_VAL/$N_TEST does not sum to TARGET_PAIRS=$TARGET_PAIRS." >&2
  exit 2
fi

mkdir -p "$FULL_ROOT" "$STAGING_DIR"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DATASET_NAME: $DATASET_NAME"
echo "FULL_ROOT: $FULL_ROOT"
echo "STAGING_DIR: $STAGING_DIR"
echo "PARTS: $PARTS"
echo "TARGET_PAIRS: $TARGET_PAIRS"
echo "SPLIT: train=$N_TRAIN val=$N_VAL test=$N_TEST"
df -h "$STUDENT_DIR" "$FULL_ROOT" "$STAGING_DIR" || true

if [ ! -x "$PY310" ]; then
  echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
  exit 127
fi

VENV=/tmp/ii100_prep_venv_${SLURM_JOB_ID}
TMPDIR=/tmp/ii100_prep_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

"$PY310" -m venv "$VENV"
PY="$VENV/bin/python"
"$PY" -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"
"$PY" -m pip install --quiet --no-index --find-links="$WHEELS" numpy
"$PY" -c "import numpy; print('numpy:', numpy.__version__)"

"$PY" "$TOOL" check-space \
  --path "$STUDENT_DIR" \
  --marker "$STAGING_DIR/.initial_${MIN_INITIAL_FREE_GB}gb_space_gate_passed" \
  --min-free-gb "$MIN_INITIAL_FREE_GB"

# Reuse the already-repaired first 30 pairs rather than downloading parts 1-2.
"$PY" "$TOOL" import-dir \
  --input "$CURRENT_ROOT" \
  --output "$FULL_ROOT" \
  --part-name existing_parts_01_02 \
  --expected-pairs 30 \
  --min-free-gb 20 \
  --clean-orientation-policy auto

for PART in $PARTS; do
  ARCHIVE="$STAGING_DIR/image-impeccable-train-data-part${PART}.zip"
  PARTIAL="$ARCHIVE.part"
  URL="$BASE_URL/image-impeccable-train-data-part${PART}.zip"
  REPORT="$FULL_ROOT/_manifests/official_part_$(printf '%02d' "$PART").json"
  EXPECTED_PAIRS=15
  if [ "$PART" = "17" ]; then
    EXPECTED_PAIRS=10
  fi

  if [ -f "$REPORT" ]; then
    echo "=== Checking completed official archive part $PART before skipping ==="
    "$PY" "$TOOL" check-part \
      --output "$FULL_ROOT" \
      --part-name "official_part_$(printf '%02d' "$PART")"
    continue
  fi

  echo "=== Downloading official archive part $PART at $(date) ==="
  download_archive "$URL" "$PARTIAL" "$ARCHIVE"

  "$PY" "$TOOL" build-archive \
    --archive "$ARCHIVE" \
    --output "$FULL_ROOT" \
    --part-name "official_part_$(printf '%02d' "$PART")" \
    --expected-pairs "$EXPECTED_PAIRS" \
    --min-free-gb 20 \
    --clean-orientation-policy auto

  case "$ARCHIVE" in
    "$STAGING_DIR"/*)
      rm -f -- "$ARCHIVE"
      ;;
    *)
      echo "Refusing to remove archive outside staging directory: $ARCHIVE" >&2
      exit 2
      ;;
  esac
done

echo "=== Canonical audit at $(date) ==="
"$PY" "$TOOL" audit \
  --output "$FULL_ROOT" \
  --expected-pairs "$TARGET_PAIRS" \
  --seed 42 \
  --n-train "$N_TRAIN" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST"

"$PY" "$TOOL" check-ready --output "$FULL_ROOT" --expected-pairs "$TARGET_PAIRS"

echo "Dataset preparation finished at $(date)"
