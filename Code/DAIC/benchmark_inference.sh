#!/bin/bash
# Benchmark inference latency for all 9 main replicates (2D / 3ch / 5ch).
# Run once on a GPU node to produce experiments/summaries/timing/inference_benchmark.csv

#SBATCH --job-name=bench_infer
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/benchmark_inference_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/benchmark_inference_%j.err

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
CODE_ROOT="$HOME/RP/Code"
PY310=$STUDENT_DIR/conda/envs/py310/bin/python
WHEELS=$STUDENT_DIR/wheels_py310
VENV=/tmp/dinov3_py310_venv_${SLURM_JOB_ID}
TMPDIR=/tmp/dinov3_py310_tmp_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1
trap 'rm -rf "$TMPDIR"' EXIT

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

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
python -u -c "import sys, torch; print('python:', sys.version); print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); assert sys.version_info >= (3, 10)"

export PYTHONPATH="$CODE_ROOT/DINOv3/src:${PYTHONPATH:-}"

# Checkpoints and configs are on staff-bulk / home; output goes to staff-bulk summaries
OUT_DIR=$STUDENT_DIR/experiments/summaries/timing

cd "$CODE_ROOT/DINOv3/src"
echo "=== Starting inference benchmark ==="

srun python -u evaluation/benchmark_inference.py \
    --all-main-runs \
    --runs-root "$STUDENT_DIR/experiments/runs" \
    --batch-sizes 1,16 \
    --n-warmup 50 \
    --n-iters 200 \
    --out-dir "$OUT_DIR"

echo "Job finished at $(date)"
