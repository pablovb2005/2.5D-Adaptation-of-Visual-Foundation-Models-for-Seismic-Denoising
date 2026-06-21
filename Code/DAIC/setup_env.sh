#!/bin/bash
#SBATCH --job-name=download_wheels
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=setup_pkgs_%j.out
#SBATCH --error=setup_pkgs_%j.err

# One-time setup: downloads all required wheel files to staff-bulk.
# Staff-bulk NFS blocks pip install --target (atomic .tmp writes fail),
# but writing plain .whl zip files works fine.
#
# After this job completes, each training job runs:
#   WHEELS=/tudelft.net/staff-bulk/.../wheels
#   VENV=/tmp/dinov3_venv
#   python3.9 -m venv $VENV && source $VENV/bin/activate
#   pip install --no-index --find-links=$WHEELS torch torchvision ...
# which takes ~2 minutes instead of 15.

set -euo pipefail

STUDENT_DIR=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal
WHEELS=$STUDENT_DIR/wheels

echo "Job started on $(hostname) at $(date)"
echo "Downloading wheels to: $WHEELS"

mkdir -p $WHEELS

# Use staff-bulk itself as TMPDIR so downloads stay on the same filesystem.
export TMPDIR=$WHEELS

echo "=== Downloading PyTorch wheels (cu118) ==="
~/miniconda3/bin/pip download \
    --dest=$WHEELS \
    --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu118

echo "=== Downloading other dependency wheels ==="
~/miniconda3/bin/pip download \
    --dest=$WHEELS \
    --no-cache-dir \
    torchmetrics peft numpy matplotlib pyyaml termcolor einops timm submitit

echo "=== Wheel files saved ==="
ls $WHEELS/*.whl 2>/dev/null | wc -l
du -sh $WHEELS

echo "Download complete. Job finished at $(date)"
