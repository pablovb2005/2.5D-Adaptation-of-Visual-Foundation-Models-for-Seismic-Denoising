#!/bin/bash
# Build the canonical 250-pair ThinkOnward dataset archive-by-archive.
#
# Submit:
#   sbatch ~/RP/Code/DAIC/full250/prepare_dataset.sh
#
# Resume after timeout:
#   sbatch ~/RP/Code/DAIC/full250/prepare_dataset.sh
#
# Restrict a recovery job to selected official archive parts:
#   PARTS="7 8 9" sbatch ~/RP/Code/DAIC/full250/prepare_dataset.sh

#SBATCH --job-name=ii_full250_prep
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/full250/slurm_prep_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/full250/slurm_prep_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
TOOL="$CODE_ROOT/DAIC/tools/prepare_thinkonward_full.py"
CURRENT_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/extracted"
FULL_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/full250_canonical/extracted"
STAGING_DIR="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/full250_staging"
PARTS="${PARTS:-3 4 5 6 7 8 9 10 11 12 13 14 15 16 17}"
BASE_URL=https://xeek-public-287031953319-eb80.s3.amazonaws.com/image-impeccable

mkdir -p "$FULL_ROOT" "$STAGING_DIR"

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "FULL_ROOT: $FULL_ROOT"
echo "STAGING_DIR: $STAGING_DIR"
echo "PARTS: $PARTS"
df -h "$STUDENT_DIR" "$FULL_ROOT" "$STAGING_DIR" || true

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is not available on this node; cannot download official archives." >&2
  exit 127
fi
if [ ! -x "$PY310" ]; then
  echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
  exit 127
fi

VENV=/tmp/ii_full250_prep_venv_${SLURM_JOB_ID}
TMPDIR=/tmp/ii_full250_prep_tmp_${SLURM_JOB_ID}
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
  --marker "$STAGING_DIR/.initial_350gb_space_gate_passed" \
  --min-free-gb 350

# Reuse the already-repaired first 30 pairs rather than downloading parts 1-2.
"$PY" "$TOOL" import-dir \
  --input "$CURRENT_ROOT" \
  --output "$FULL_ROOT" \
  --part-name existing_parts_01_02 \
  --expected-pairs 30 \
  --min-free-gb 20

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
  if [ ! -f "$ARCHIVE" ]; then
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
      --output "$PARTIAL" "$URL"
    mv "$PARTIAL" "$ARCHIVE"
  else
    echo "Using existing staged archive: $ARCHIVE"
  fi

  "$PY" "$TOOL" build-archive \
    --archive "$ARCHIVE" \
    --output "$FULL_ROOT" \
    --part-name "official_part_$(printf '%02d' "$PART")" \
    --expected-pairs "$EXPECTED_PAIRS" \
    --min-free-gb 20

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

echo "=== Full canonical audit at $(date) ==="
"$PY" "$TOOL" audit \
  --output "$FULL_ROOT" \
  --expected-pairs 250 \
  --seed 42 \
  --n-train 200 \
  --n-val 25 \
  --n-test 25

echo "Dataset preparation finished at $(date)"
