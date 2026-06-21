#!/bin/bash
# Full-250 targeted PEFT extension: 2D, 3ch, and 5ch with three seeds.
#
# Submit the seed42 pilot first:
#   sbatch --array=0,3,6 ~/RP/Code/DAIC/full250/submit_peft_core.sh
#
# Submit replication only after the pilot passes the runtime gate:
#   sbatch --array=1,2,4,5,7,8 ~/RP/Code/DAIC/full250/submit_peft_core.sh
#
# Resubmit one cleanly resumable run:
#   sbatch --array=<INDEX> ~/RP/Code/DAIC/full250/submit_peft_core.sh

#SBATCH --job-name=ii250_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%9
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/full250/slurm_peft_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/full250/slurm_peft_%A_%a.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310="$STUDENT_DIR/conda/envs/py310/bin/python"
WHEELS="$STUDENT_DIR/wheels_py310"
DATA_ROOT="$STUDENT_DIR/Dataset/ThinkOnwards/training_data/full250_canonical/extracted"
TOOL="$CODE_ROOT/DAIC/tools/prepare_thinkonward_full.py"

CONFIGS=(
  configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_n200vols_seed42_run01_full250_daic.yaml
  configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_n200vols_seed43_run02_full250_daic.yaml
  configs/dinov3_vits_2d_impeccable_repeated_stride5_lora_r16_n200vols_seed44_run03_full250_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_n200vols_seed42_run01_full250_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_n200vols_seed43_run02_full250_daic.yaml
  configs/dinov3_vits_2d5_3ch_impeccable_neighbors3_stride5_lora_r16_n200vols_seed44_run03_full250_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_n200vols_seed42_run01_full250_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_n200vols_seed43_run02_full250_daic.yaml
  configs/dinov3_vits_2d5_5ch_impeccable_neighbors5_stride5_patch_emb_lora_r16_n200vols_seed44_run03_full250_daic.yaml
)

EXP_DIRS=(
  $STUDENT_DIR/experiments/runs/full250/2d/impeccable_repeated_stride5_lora_r16_n200vols/seed42_run01
  $STUDENT_DIR/experiments/runs/full250/2d/impeccable_repeated_stride5_lora_r16_n200vols/seed43_run02
  $STUDENT_DIR/experiments/runs/full250/2d/impeccable_repeated_stride5_lora_r16_n200vols/seed44_run03
  $STUDENT_DIR/experiments/runs/full250/3ch/impeccable_neighbors3_stride5_lora_r16_n200vols/seed42_run01
  $STUDENT_DIR/experiments/runs/full250/3ch/impeccable_neighbors3_stride5_lora_r16_n200vols/seed43_run02
  $STUDENT_DIR/experiments/runs/full250/3ch/impeccable_neighbors3_stride5_lora_r16_n200vols/seed44_run03
  $STUDENT_DIR/experiments/runs/full250/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n200vols/seed42_run01
  $STUDENT_DIR/experiments/runs/full250/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n200vols/seed43_run02
  $STUDENT_DIR/experiments/runs/full250/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16_n200vols/seed44_run03
)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
EXP_DIR=${EXP_DIRS[$SLURM_ARRAY_TASK_ID]}
LOG_DIR="$EXP_DIR/logs"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/slurm_${SLURM_JOB_ID}.err" >&2)

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID  ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "CONFIG: $CONFIG"
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

# prepare_thinkonward_full.py imports numpy even for check-ready, while the
# base DAIC py310 environment intentionally does not guarantee project deps.
python -m pip install --quiet --no-index --find-links="$WHEELS" numpy
python "$TOOL" check-ready --output "$DATA_ROOT" --expected-pairs 250

python -m pip install --no-index --find-links="$WHEELS" \
  "torch==2.6.0+cu118" "torchvision==0.21.0+cu118"

python -m pip install --no-index --find-links="$WHEELS" \
  torchmetrics peft matplotlib pyyaml termcolor einops timm submitit \
  transformers accelerate safetensors huggingface_hub

echo "=== GPU info ==="
nvidia-smi

echo "=== Checking imports ==="
python -u -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert sys.version_info >= (3, 10)"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"

cd "$CODE_ROOT/DINOv3/src"
echo "=== Starting training ==="

srun python -u training/train.py --config "$CONFIG"

bash "$CODE_ROOT/DAIC/evaluate_if_complete.sh" "$CONFIG" "$EXP_DIR" "$CODE_ROOT"

echo "Job finished at $(date)"
