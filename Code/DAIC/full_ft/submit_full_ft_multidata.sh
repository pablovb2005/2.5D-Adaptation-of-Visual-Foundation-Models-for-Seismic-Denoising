#!/bin/bash
# Full fine-tuning multi-data-seed reruns: 3 variants x 3 data seeds x 3 training seeds = 27 jobs.
#
# Index layout (outer -> inner): data_seed -> training_seed -> variant
#   data_seed_idx = TASK_ID / 9         (0->101, 1->202, 2->303)
#   tseed_idx     = (TASK_ID % 9) / 3   (0->42,  1->43,  2->44)
#   variant_idx   = TASK_ID % 3         (0->2d,  1->3ch, 2->5ch)
#
# Quick index reference:
#   data_seed 101, tseed 42: 0=2d  1=3ch  2=5ch
#   data_seed 101, tseed 43: 3=2d  4=3ch  5=5ch
#   data_seed 101, tseed 44: 6=2d  7=3ch  8=5ch
#   data_seed 202, tseed 42: 9=2d 10=3ch 11=5ch
#   data_seed 202, tseed 43:12=2d 13=3ch 14=5ch
#   data_seed 202, tseed 44:15=2d 16=3ch 17=5ch
#   data_seed 303, tseed 42:18=2d 19=3ch 20=5ch
#   data_seed 303, tseed 43:21=2d 22=3ch 23=5ch
#   data_seed 303, tseed 44:24=2d 25=3ch 26=5ch
#
# Submit all 27 jobs:
#   sbatch ~/RP/Code/DAIC/full_ft/submit_full_ft_multidata.sh
# Resubmit one job (e.g. task 0):
#   sbatch --array=0 ~/RP/Code/DAIC/full_ft/submit_full_ft_multidata.sh

#SBATCH --job-name=full_ft_md_%a
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-26
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/full_ft/slurm_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/full_ft/slurm_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
DATA_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/extracted"

DATA_SEEDS=(101 202 303)
TRAINING_SEEDS=(42 43 44)
VARIANTS=(2d 3ch 5ch)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

DATA_SEED_IDX=$((TASK_ID / 9))
TSEED_IDX=$(( (TASK_ID % 9) / 3 ))
VARIANT_IDX=$((TASK_ID % 3))

DATA_SEED="${DATA_SEEDS[$DATA_SEED_IDX]}"
TSEED="${TRAINING_SEEDS[$TSEED_IDX]}"
VARIANT="${VARIANTS[$VARIANT_IDX]}"

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
    RUN_BASE="impeccable_repeated_stride5_full_ft"
    MODEL_IN_CHANS_LINE=""
    ;;
  3ch)
    MODE="2.5d_3ch"
    FAMILY="3ch"
    RUN_BASE="impeccable_neighbors3_stride5_full_ft"
    MODEL_IN_CHANS_LINE=""
    ;;
  5ch)
    MODE="2.5d_5ch"
    FAMILY="5ch"
    RUN_BASE="impeccable_neighbors5_stride5_patch_emb_full_ft"
    MODEL_IN_CHANS_LINE="  in_chans: 5"
    ;;
  *)
    echo "ERROR: unsupported variant: $VARIANT" >&2
    exit 2
    ;;
esac

EXP_DIR="$STUDENT_DIR/experiments/runs/full_ft_multidata/$FAMILY/$RUN_BASE/data_seed${DATA_SEED}/$RUN_ID"
LOG_DIR="$EXP_DIR/logs"
CONFIG_RUNTIME="$EXP_DIR/runtime_config.yaml"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  ARRAY_TASK_ID: $TASK_ID"
echo "VARIANT: $VARIANT  DATA_SEED: $DATA_SEED  TRAINING_SEED: $TSEED"
echo "EXP_DIR: $EXP_DIR"

if [ ! -x "$PY310" ]; then
  echo "ERROR: Python 3.10 interpreter not found or not executable: $PY310" >&2
  exit 127
fi

VENV=/tmp/dinov3_py310_venv_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR" "$VENV"' EXIT

"$PY310" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"

python -m pip install --no-index --find-links="$WHEELS" \
  "torch==2.6.0+cu118" "torchvision==0.21.0+cu118"

python -m pip install --no-index --find-links="$WHEELS" \
  torchmetrics peft numpy matplotlib pyyaml termcolor einops timm submitit \
  transformers accelerate safetensors huggingface_hub

cat > "$CONFIG_RUNTIME" <<EOF
model:
  name: dinov3_vits16
  weights: $CODE_ROOT/DINOv3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
$MODEL_IN_CHANS_LINE
  lora_rank: 0
  lora_alpha: 64
  lora_dropout: 0.1
  lora_targets: [qkv, proj]
  full_finetune: true

data:
  source: image_impeccable
  root_dir: $DATA_ROOT
  mode: "$MODE"
  n_train: 20
  n_val: 5
  n_test: 5
  slice_stride: 5
  crop_size: 224
  seed: $DATA_SEED
  cache_volumes: true

training:
  epochs: 50
  seed: $TSEED
  batch_size: 8
  lr: 1.0e-5
  weight_decay: 0.01
  warmup_epochs: 5
  loss_lambda: 0.5
  num_workers: 0
  persistent_workers: true
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
echo "=== Starting training (full fine-tuning) ==="

srun python -u training/train.py --config "$CONFIG_RUNTIME"

bash "$CODE_ROOT/DAIC/evaluate_if_complete.sh" "$CONFIG_RUNTIME" "$EXP_DIR" "$CODE_ROOT"

echo "Job finished at $(date)"
