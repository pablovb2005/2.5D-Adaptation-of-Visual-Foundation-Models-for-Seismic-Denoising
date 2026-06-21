#!/bin/bash
#SBATCH --job-name=setup_py310
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/setup_py310_%j.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/setup_py310_%j.err

# One-time setup for Python 3.10 on DAIC.
# Creates a small Python 3.10 conda prefix on staff-bulk and downloads py310 wheels.

set -euo pipefail
trap 'echo "ERROR: setup_env_py310.sh failed at line $LINENO at $(date)" >&2' ERR

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
PY310_PREFIX=$STUDENT_DIR/conda/envs/py310
CONDA_PKGS=$STUDENT_DIR/conda/pkgs
WHEELS=$STUDENT_DIR/wheels_py310
LOG_DIR=$STUDENT_DIR/experiments/runs/system/setup_py310/logs
TMPDIR=/tmp/dinov3_py310_setup_tmp_${SLURM_JOB_ID}
LOCAL_WHEELS=$TMPDIR/wheels
TORCH_REQ=$TMPDIR/torch_py310_requirements.txt
DEPS_REQ=$TMPDIR/dinov3_py310_requirements.txt
CONSTRAINTS=$TMPDIR/constraints_py310.txt

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/setup_py310_${SLURM_JOB_ID}.out") 2> >(tee -a "$LOG_DIR/setup_py310_${SLURM_JOB_ID}.err" >&2)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Python 3.10 prefix: $PY310_PREFIX"
echo "Conda package cache: $CONDA_PKGS"
echo "Wheel cache: $WHEELS"
echo "Local temp dir: $TMPDIR"
echo "Local wheel staging dir: $LOCAL_WHEELS"

mkdir -p "$CONDA_PKGS" "$WHEELS"

export CONDA_PKGS_DIRS="$CONDA_PKGS"
mkdir -p "$TMPDIR" "$LOCAL_WHEELS"
export TMPDIR
export PYTHONNOUSERSITE=1
export PIP_NO_CACHE_DIR=1

cat > "$TORCH_REQ" <<'EOF'
torch==2.6.0+cu118
torchvision==0.21.0+cu118
EOF

cat > "$DEPS_REQ" <<'EOF'
pip
setuptools
wheel
torchmetrics
peft
numpy
matplotlib
pyyaml
termcolor
einops
timm
submitit
transformers
accelerate
safetensors
huggingface_hub
EOF

cat > "$CONSTRAINTS" <<'EOF'
torch==2.6.0+cu118
torchvision==0.21.0+cu118
EOF

echo "=== Environment diagnostics ==="
uname -a
df -h "$STUDENT_DIR" "$TMPDIR" /tmp || true
du -sh "$WHEELS" 2>/dev/null || true
~/miniconda3/bin/conda --version

NEED_CREATE=0
if [ ! -x "$PY310_PREFIX/bin/python" ]; then
    NEED_CREATE=1
else
    if ! "$PY310_PREFIX/bin/python" - <<'PY'
import sys
assert sys.version_info[:2] == (3, 10), sys.version
import pip
print("Existing prefix OK:", sys.version)
PY
    then
        echo "=== Existing prefix is not a valid Python 3.10 + pip environment ==="
        NEED_CREATE=1
    fi
fi

if [ "$NEED_CREATE" -eq 1 ]; then
    if [ -d "$PY310_PREFIX" ]; then
        echo "=== Removing incomplete Python 3.10 prefix ==="
        rm -rf "$PY310_PREFIX"
    fi
    echo "=== Creating Python 3.10 conda prefix ==="
    ~/miniconda3/bin/conda create -y -p "$PY310_PREFIX" python=3.10 pip --override-channels -c defaults
else
    echo "=== Python 3.10 prefix already exists ==="
fi

"$PY310_PREFIX/bin/python" --version
"$PY310_PREFIX/bin/python" -m pip --version

copy_wheels_to_staff_bulk() {
    echo "=== Copying staged wheels to staff-bulk without chmod/copymode ==="
    for src in "$LOCAL_WHEELS"/*.whl; do
        [ -e "$src" ] || continue
        dest="$WHEELS/$(basename "$src")"
        if [ -f "$dest" ] && cmp -s "$src" "$dest"; then
            echo "Already present: $(basename "$src")"
            continue
        fi
        echo "Writing: $(basename "$src")"
        cat "$src" > "$dest"
    done
}

echo "=== Downloading PyTorch py310 wheels (cu118) ==="
"$PY310_PREFIX/bin/python" -m pip download \
    --dest="$LOCAL_WHEELS" \
    --no-cache-dir \
    --requirement "$TORCH_REQ" \
    --index-url https://download.pytorch.org/whl/cu118

echo "=== Downloading other py310 dependency wheels ==="
"$PY310_PREFIX/bin/python" -m pip download \
    --dest="$LOCAL_WHEELS" \
    --no-cache-dir \
    --only-binary=:all: \
    --find-links="$LOCAL_WHEELS" \
    --constraint "$CONSTRAINTS" \
    --requirement "$DEPS_REQ"

copy_wheels_to_staff_bulk

echo "=== Wheel files saved ==="
ls "$WHEELS"/*.whl 2>/dev/null | wc -l
du -sh "$WHEELS"
ls -lh "$WHEELS" | sed -n '1,80p'

echo "=== Testing py310 /tmp venv install from wheel cache ==="
TEST_VENV=/tmp/dinov3_py310_setup_test_${SLURM_JOB_ID}
rm -rf "$TEST_VENV"
"$PY310_PREFIX/bin/python" -m venv "$TEST_VENV"
source "$TEST_VENV/bin/activate"
python -m pip install --upgrade pip --quiet --no-index --find-links="$WHEELS"
python -m pip install --no-index --find-links="$WHEELS" \
    --requirement "$TORCH_REQ"
python -m pip install --no-index --find-links="$WHEELS" \
    --constraint "$CONSTRAINTS" \
    --requirement "$DEPS_REQ"
python -u -c "import sys, torch, torchvision, torchmetrics, peft, timm, transformers; print('python:', sys.version); print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda available:', torch.cuda.is_available()); assert sys.version_info >= (3, 10); assert torch.__version__.startswith('2.6.0')"
deactivate
rm -rf "$TEST_VENV"

echo "Python 3.10 setup complete. Job finished at $(date)"
