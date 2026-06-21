#!/bin/bash
# Large data-efficiency sweep: 3 variants × 5 n-train sizes × 3 training seeds × 3 data seeds = 135 jobs.
#
# Index layout (outer → inner): data_seed → training_seed → variant → n_train
#   data_seed_idx    = TASK_ID / 45          (0→101, 1→202, 2→303)
#   training_seed_idx = (TASK_ID % 45) / 15  (0→42,  1→43,  2→44)
#   variant_idx      = (TASK_ID % 15) / 5    (0→2d,  1→3ch, 2→5ch)
#   size_idx         = TASK_ID % 5           (0→20,  1→35,  2→50,  3→75,  4→100)
#
# Quick index reference (data_seed=101, training_seed=42):
#   0  2D  n=20    5  3ch n=20   10  5ch n=20
#   1  2D  n=35    6  3ch n=35   11  5ch n=35
#   2  2D  n=50    7  3ch n=50   12  5ch n=50
#   3  2D  n=75    8  3ch n=75   13  5ch n=75
#   4  2D  n=100   9  3ch n=100  14  5ch n=100
# Add 45 for training_seed=43, 90 for training_seed=44.
# Add 15 for data_seed=202 block, 30 for data_seed=303 block (within each training-seed group).
#
# Submit all 135 jobs:
#   sbatch ~/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh
# Resubmit one:
#   sbatch --array=<INDEX> ~/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh
# Resubmit a range:
#   sbatch --array=<A>-<B> ~/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh

#SBATCH --job-name=de100_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-134
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_100train_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_100train_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
DATASET_NAME="${DATASET_NAME:-data_efficiency_100train_canonical_v2}"
STUDY_NAME="${STUDY_NAME:-data_efficiency_100train_v2}"
RUN_SUFFIX="${RUN_SUFFIX:-100train_v2}"
DATA_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/$DATASET_NAME/extracted"
TOOL="$CODE_ROOT/DAIC/tools/prepare_thinkonward_full.py"

wait_for_staff_bulk() {
  local attempt
  for ((attempt = 1; attempt <= 6; attempt++)); do
    if ls "$STUDENT_DIR" >/dev/null 2>&1; then
      return 0
    fi
    echo "Waiting for staff-bulk access to $STUDENT_DIR (attempt $attempt/6)..." >&2
    sleep 10
  done

  echo "ERROR: staff-bulk path is not accessible from this job: $STUDENT_DIR" >&2
  ls -ld /tudelft.net /tudelft.net/staff-bulk /tudelft.net/staff-bulk/ewi /tudelft.net/staff-bulk/ewi/insy 2>&1 || true
  exit 1
}

make_log_dir() {
  local attempt
  wait_for_staff_bulk
  for ((attempt = 1; attempt <= 6; attempt++)); do
    if mkdir -p "$LOG_DIR"; then
      return 0
    fi
    echo "Waiting to create run log directory $LOG_DIR (attempt $attempt/6)..." >&2
    sleep 10
    wait_for_staff_bulk
  done

  echo "ERROR: could not create run log directory: $LOG_DIR" >&2
  exit 1
}

_csv_to_array() {
  local raw="$1"
  raw="${raw// /}"
  if [ -z "$raw" ]; then
    return 1
  fi
  IFS=',' read -r -a CSV_VALUES <<< "$raw"
}

_require_nonempty_values() {
  local label="$1"
  shift
  if [ "$#" -eq 0 ]; then
    echo "ERROR: $label did not produce any values." >&2
    exit 2
  fi
  local value
  for value in "$@"; do
    if [ -z "$value" ]; then
      echo "ERROR: $label contains an empty value." >&2
      exit 2
    fi
  done
}

_require_integer_values() {
  local label="$1"
  shift
  local value
  for value in "$@"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "ERROR: $label contains a non-integer value: $value" >&2
      exit 2
    fi
  done
}

_csv_to_array "${DATA_SEEDS_CSV:-101,202,303}" || { echo "ERROR: DATA_SEEDS_CSV is empty." >&2; exit 2; }
DATA_SEEDS=("${CSV_VALUES[@]}")
_csv_to_array "${TRAINING_SEEDS_CSV:-42,43,44}" || { echo "ERROR: TRAINING_SEEDS_CSV is empty." >&2; exit 2; }
TRAINING_SEEDS=("${CSV_VALUES[@]}")
_csv_to_array "${TRAIN_SIZES_CSV:-20,35,50,75,100}" || { echo "ERROR: TRAIN_SIZES_CSV is empty." >&2; exit 2; }
TRAIN_SIZES=("${CSV_VALUES[@]}")
_csv_to_array "${VARIANTS_CSV:-2d,3ch,5ch}" || { echo "ERROR: VARIANTS_CSV is empty." >&2; exit 2; }
VARIANTS=("${CSV_VALUES[@]}")

_require_nonempty_values DATA_SEEDS_CSV "${DATA_SEEDS[@]}"
_require_nonempty_values TRAINING_SEEDS_CSV "${TRAINING_SEEDS[@]}"
_require_nonempty_values TRAIN_SIZES_CSV "${TRAIN_SIZES[@]}"
_require_nonempty_values VARIANTS_CSV "${VARIANTS[@]}"
_require_integer_values DATA_SEEDS_CSV "${DATA_SEEDS[@]}"
_require_integer_values TRAINING_SEEDS_CSV "${TRAINING_SEEDS[@]}"
_require_integer_values TRAIN_SIZES_CSV "${TRAIN_SIZES[@]}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SIZES="${#TRAIN_SIZES[@]}"
N_VARIANTS="${#VARIANTS[@]}"
N_TSEEDS="${#TRAINING_SEEDS[@]}"
N_DSEEDS="${#DATA_SEEDS[@]}"
TOTAL_TASKS=$((N_DSEEDS * N_TSEEDS * N_VARIANTS * N_SIZES))

if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$TASK_ID is outside configured task range 0-$((TOTAL_TASKS - 1))." >&2
  echo "Configured matrix: DATA_SEEDS=${DATA_SEEDS[*]} TRAINING_SEEDS=${TRAINING_SEEDS[*]} VARIANTS=${VARIANTS[*]} TRAIN_SIZES=${TRAIN_SIZES[*]}" >&2
  exit 2
fi

DATA_SEED_IDX=$((TASK_ID / (N_TSEEDS * N_VARIANTS * N_SIZES)))
REM=$((TASK_ID % (N_TSEEDS * N_VARIANTS * N_SIZES)))
TSEED_IDX=$((REM / (N_VARIANTS * N_SIZES)))
REM2=$((REM % (N_VARIANTS * N_SIZES)))
VARIANT_IDX=$((REM2 / N_SIZES))
SIZE_IDX=$((REM2 % N_SIZES))

DATA_SEED="${DATA_SEEDS[$DATA_SEED_IDX]}"
TSEED="${TRAINING_SEEDS[$TSEED_IDX]}"
TRAIN_N="${TRAIN_SIZES[$SIZE_IDX]}"
VARIANT="${VARIANTS[$VARIANT_IDX]}"
VARIANT="${VARIANT,,}"

case "$TSEED" in
  42) RUN_ID="seed42_run01" ;;
  43) RUN_ID="seed43_run02" ;;
  44) RUN_ID="seed44_run03" ;;
   *) RUN_ID="seed${TSEED}_run01" ;;
esac

case "$VARIANT" in
  2d)
    MODE="2d"
    FAMILY="2d"
    RUN_BASE="impeccable_repeated_stride5_lora_r16_n${TRAIN_N}vols_${RUN_SUFFIX}"
    MODEL_IN_CHANS_LINE=""
    ;;
  3ch)
    MODE="2.5d_3ch"
    FAMILY="3ch"
    RUN_BASE="impeccable_neighbors3_stride5_lora_r16_n${TRAIN_N}vols_${RUN_SUFFIX}"
    MODEL_IN_CHANS_LINE=""
    ;;
  5ch)
    MODE="2.5d_5ch"
    FAMILY="5ch"
    RUN_BASE="impeccable_neighbors5_stride5_patch_emb_lora_r16_n${TRAIN_N}vols_${RUN_SUFFIX}"
    MODEL_IN_CHANS_LINE="  in_chans: 5"
    ;;
  7ch)
    MODE="2.5d_7ch"
    FAMILY="7ch"
    RUN_BASE="impeccable_neighbors7_stride5_patch_emb_lora_r16_n${TRAIN_N}vols_${RUN_SUFFIX}"
    MODEL_IN_CHANS_LINE=$'  in_chans: 7\n  patch_emb_init: mixed'
    ;;
  9ch)
    MODE="2.5d_9ch"
    FAMILY="9ch"
    RUN_BASE="impeccable_neighbors9_stride5_patch_emb_lora_r16_n${TRAIN_N}vols_${RUN_SUFFIX}"
    MODEL_IN_CHANS_LINE=$'  in_chans: 9\n  patch_emb_init: mixed'
    ;;
  *)
    echo "ERROR: unsupported variant: $VARIANT" >&2
    exit 2
    ;;
esac

EXP_DIR="$STUDENT_DIR/experiments/runs/$STUDY_NAME/$FAMILY/$RUN_BASE/data_seed${DATA_SEED}/$RUN_ID"
LOG_DIR="$EXP_DIR/logs"
CONFIG_RUNTIME="$EXP_DIR/runtime_config.yaml"
make_log_dir
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  ARRAY_TASK_ID: $TASK_ID"
echo "VARIANT: $VARIANT  TRAIN_N: $TRAIN_N  DATA_SEED: $DATA_SEED  TRAINING_SEED: $TSEED"
echo "CONFIGURED_TASKS: $TOTAL_TASKS  DATA_SEEDS: ${DATA_SEEDS[*]}  TRAINING_SEEDS: ${TRAINING_SEEDS[*]}  VARIANTS: ${VARIANTS[*]}  TRAIN_SIZES: ${TRAIN_SIZES[*]}"
echo "DATASET_NAME: $DATASET_NAME  STUDY_NAME: $STUDY_NAME  RUN_SUFFIX: $RUN_SUFFIX"
echo "DATA_ROOT: $DATA_ROOT"
echo "EXP_DIR: $EXP_DIR"

TMPDIR=/tmp/dinov3_py310_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
RUNTIME_BUILD_DIR=""
cleanup() {
  rm -rf "$TMPDIR"
  if [ -n "${RUNTIME_BUILD_DIR:-}" ]; then
    rm -rf "$RUNTIME_BUILD_DIR"
  fi
}
trap cleanup EXIT

if [ ! -x "$PY310" ]; then
  echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
  exit 127
fi

RUNTIME_VENV="${RUNTIME_VENV:-$STUDENT_DIR/venvs/dinov3_py310_runtime_torch260_cu118_v1}"
RUNTIME_VENV_STAMP="dinov3-py310-torch2.6.0-cu118-runtime-v1"
RUNTIME_VENV_READY="$RUNTIME_VENV/.dinov3_runtime_ready"
RUNTIME_VENV_LOCK="$STUDENT_DIR/venvs/.dinov3_py310_runtime_torch260_cu118_v1.lock"

runtime_venv_ready() {
  [ -x "$RUNTIME_VENV/bin/python" ] &&
    [ -f "$RUNTIME_VENV_READY" ] &&
    grep -qx "$RUNTIME_VENV_STAMP" "$RUNTIME_VENV_READY"
}

build_runtime_venv() {
  local build_dir="${RUNTIME_VENV}.build.${SLURM_JOB_ID}"
  RUNTIME_BUILD_DIR="$build_dir"
  echo "Building shared runtime venv: $RUNTIME_VENV"
  rm -rf "$build_dir"
  "$PY310" -m venv "$build_dir"
  "$build_dir/bin/python" -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"
  "$build_dir/bin/python" -m pip install --no-index --find-links="$WHEELS" \
    "torch==2.6.0+cu118" "torchvision==0.21.0+cu118"
  "$build_dir/bin/python" -m pip install --no-index --find-links="$WHEELS" \
    numpy torchmetrics peft matplotlib pyyaml termcolor einops timm submitit \
    transformers accelerate safetensors huggingface_hub
  "$build_dir/bin/python" - <<'PY'
import sys
import torch
import torchvision
import numpy
import torchmetrics
import peft
import yaml
import matplotlib
import transformers
import accelerate
import safetensors
import huggingface_hub
import timm
import submitit
import termcolor
import einops

assert sys.version_info >= (3, 10)
assert torch.__version__.startswith("2.6.0")
assert torchvision.__version__.startswith("0.21.0")
PY
  printf '%s\n' "$RUNTIME_VENV_STAMP" > "$build_dir/.dinov3_runtime_ready"
  rm -rf "$RUNTIME_VENV"
  mv "$build_dir" "$RUNTIME_VENV"
  RUNTIME_BUILD_DIR=""
}

ensure_runtime_venv() {
  mkdir -p "$(dirname "$RUNTIME_VENV")"
  if runtime_venv_ready; then
    echo "Reusing shared runtime venv: $RUNTIME_VENV"
    return 0
  fi

  if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to safely build the shared runtime venv." >&2
    exit 127
  fi

  exec 9>"$RUNTIME_VENV_LOCK"
  flock -x 9
  if runtime_venv_ready; then
    echo "Reusing shared runtime venv after lock wait: $RUNTIME_VENV"
  else
    build_runtime_venv
  fi
  flock -u 9

  if ! runtime_venv_ready; then
    echo "ERROR: shared runtime venv is not ready after build: $RUNTIME_VENV" >&2
    exit 1
  fi
}

ensure_runtime_venv
source "$RUNTIME_VENV/bin/activate"
python "$TOOL" check-ready --output "$DATA_ROOT" --expected-pairs 120

# For small training subsets the main process can cache mmap objects safely
# (num_workers=0 avoids the fork+NFS EFAULT seen in the 2026-05-28 incident).
# 5ch tiles all slices (~13 GB for 30 vols) so requires --mem=24GB; safe here.
# For n>=50, RAM uncertainty is too high; fall back to standard NFS mmap mode.
if [ "$TRAIN_N" -le 35 ]; then
  CACHE_VOLS=true
  DATA_WORKERS=0
  PERSISTENT_WORKERS=false
else
  CACHE_VOLS=false
  DATA_WORKERS=4
  PERSISTENT_WORKERS=true
fi

cat > "$CONFIG_RUNTIME" <<EOF
model:
  name: dinov3_vits16
  weights: $CODE_ROOT/DINOv3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
$MODEL_IN_CHANS_LINE
  lora_rank: 16
  lora_alpha: 64
  lora_dropout: 0.1
  lora_targets: [qkv, proj]

data:
  source: image_impeccable
  root_dir: $DATA_ROOT
  mode: "$MODE"
  n_train: 100
  n_val: 10
  n_test: 10
  train_subset_n: $TRAIN_N
  slice_stride: 5
  crop_size: 224
  seed: $DATA_SEED
  cache_volumes: $CACHE_VOLS

training:
  epochs: 50
  val_interval: 5
  seed: $TSEED
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 0.01
  warmup_epochs: 5
  loss_lambda: 0.5
  num_workers: $DATA_WORKERS
  persistent_workers: $PERSISTENT_WORKERS
  log_interval_batches: 25
  resume: true
  max_runtime_minutes: 700

output:
  checkpoint_dir: $EXP_DIR
EOF

echo "Runtime config written to: $CONFIG_RUNTIME"

echo "=== GPU info ==="
nvidia-smi

echo "=== Checking imports ==="
python -u -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert sys.version_info >= (3, 10)"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"

cd "$CODE_ROOT/DINOv3/src"
echo "=== Starting training ==="

srun python -u training/train.py --config "$CONFIG_RUNTIME"

bash "$CODE_ROOT/DAIC/evaluate_if_complete.sh" "$CONFIG_RUNTIME" "$EXP_DIR" "$CODE_ROOT"

echo "Job finished at $(date)"
