#!/bin/bash
# Trained controls multi-data-seed reruns: 3 controls x 3 data seeds x 3 training seeds = 27 jobs.
#
# Controls:
#   3ch_shuffled       -- 3ch LoRA, neighbor channels from a different volume during training
#   5ch_repeated_center -- 5ch LoRA, input [t,t,t,t,t] (capacity control, no neighbor content)
#   5ch_shuffled       -- 5ch LoRA, neighbor channels from a different volume during training
#
# Index layout (outer -> inner): data_seed -> training_seed -> control
#   data_seed_idx = TASK_ID / 9         (0->101, 1->202, 2->303)
#   tseed_idx     = (TASK_ID % 9) / 3   (0->42,  1->43,  2->44)
#   control_idx   = TASK_ID % 3         (0->3ch_shuffled, 1->5ch_repeated_center, 2->5ch_shuffled)
#
# Quick index reference:
#   data_seed 101, tseed 42:  0=3ch_shuf  1=5ch_rep  2=5ch_shuf
#   data_seed 101, tseed 43:  3=3ch_shuf  4=5ch_rep  5=5ch_shuf
#   data_seed 101, tseed 44:  6=3ch_shuf  7=5ch_rep  8=5ch_shuf
#   data_seed 202, tseed 42:  9=3ch_shuf 10=5ch_rep 11=5ch_shuf
#   data_seed 202, tseed 43: 12=3ch_shuf 13=5ch_rep 14=5ch_shuf
#   data_seed 202, tseed 44: 15=3ch_shuf 16=5ch_rep 17=5ch_shuf
#   data_seed 303, tseed 42: 18=3ch_shuf 19=5ch_rep 20=5ch_shuf
#   data_seed 303, tseed 43: 21=3ch_shuf 22=5ch_rep 23=5ch_shuf
#   data_seed 303, tseed 44: 24=3ch_shuf 25=5ch_rep 26=5ch_shuf
#
# Submit all 27 jobs:
#   sbatch ~/RP/Code/DAIC/submit_controls_multidata.sh
# Resubmit one job (e.g. task 0):
#   sbatch --array=0 ~/RP/Code/DAIC/submit_controls_multidata.sh

#SBATCH --job-name=ctrl_md_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-26
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/slurm_ctrl_md_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/slurm_ctrl_md_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
DATA_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/extracted"

DATA_SEEDS=(101 202 303)
TRAINING_SEEDS=(42 43 44)
CONTROLS=(3ch_shuffled 5ch_repeated_center 5ch_shuffled)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

DATA_SEED_IDX=$((TASK_ID / 9))
TSEED_IDX=$(( (TASK_ID % 9) / 3 ))
CONTROL_IDX=$((TASK_ID % 3))

DATA_SEED="${DATA_SEEDS[$DATA_SEED_IDX]}"
TSEED="${TRAINING_SEEDS[$TSEED_IDX]}"
CONTROL="${CONTROLS[$CONTROL_IDX]}"

case "$TSEED" in
  42) RUN_ID="seed42_run01" ;;
  43) RUN_ID="seed43_run02" ;;
  44) RUN_ID="seed44_run03" ;;
   *) RUN_ID="seed${TSEED}_run01" ;;
esac

case "$CONTROL" in
  3ch_shuffled)
    MODE="2.5d_3ch"
    RUN_BASE="impeccable_shuffled3_stride5_lora_r16"
    MODEL_IN_CHANS_LINE=""
    DATA_EXTRA_LINE="  shuffle_neighbors: true"
    NUM_WORKERS=0
    PERSISTENT_WORKERS="false"
    ;;
  5ch_repeated_center)
    MODE="2.5d_5ch"
    RUN_BASE="impeccable_repeated_center_stride5_patch_emb_lora_r16"
    MODEL_IN_CHANS_LINE="  in_chans: 5"
    DATA_EXTRA_LINE="  repeat_center: true"
    NUM_WORKERS=0
    PERSISTENT_WORKERS="false"
    ;;
  5ch_shuffled)
    MODE="2.5d_5ch"
    RUN_BASE="impeccable_shuffled5_stride5_patch_emb_lora_r16"
    MODEL_IN_CHANS_LINE="  in_chans: 5"
    DATA_EXTRA_LINE="  shuffle_neighbors: true"
    NUM_WORKERS=0
    PERSISTENT_WORKERS="false"
    ;;
  *)
    echo "ERROR: unsupported control: $CONTROL" >&2
    exit 2
    ;;
esac

EXP_DIR="$STUDENT_DIR/experiments/runs/controls_multidata/$CONTROL/$RUN_BASE/data_seed${DATA_SEED}/$RUN_ID"
LOG_DIR="$EXP_DIR/logs"
CONFIG_RUNTIME="$EXP_DIR/runtime_config.yaml"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  ARRAY_TASK_ID: $TASK_ID"
echo "CONTROL: $CONTROL  DATA_SEED: $DATA_SEED  TRAINING_SEED: $TSEED"
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
  lora_rank: 16
  lora_alpha: 64
  lora_dropout: 0.1
  lora_targets: [qkv, proj]

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
$DATA_EXTRA_LINE

training:
  epochs: 50
  seed: $TSEED
  batch_size: 16
  lr: 1.0e-4
  weight_decay: 0.01
  warmup_epochs: 5
  loss_lambda: 0.5
  num_workers: $NUM_WORKERS
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
echo "=== Starting training ($CONTROL) ==="

srun python -u training/train.py --config "$CONFIG_RUNTIME"

bash "$CODE_ROOT/DAIC/evaluate_if_complete.sh" "$CONFIG_RUNTIME" "$EXP_DIR" "$CODE_ROOT"

echo "Job finished at $(date)"
