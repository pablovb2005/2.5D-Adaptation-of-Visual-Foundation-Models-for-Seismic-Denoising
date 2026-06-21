#!/bin/bash
# Full 5-variant channel-window data-efficiency sweep.
#
# Variants:  2D / 3ch / 5ch / 7ch / 9ch
# n-train:   5, 10, 15, 20, 35, 50, 75, 100
# Data seeds:     101, 202, 303
# Training seeds: 42, 43, 44
# Total jobs: 8 x 5 x 3 x 3 = 360
#
# 25-job pilot (data_seed=101, tseed=42, n=20-100) is already running under
# study data_efficiency_100train_channel_window_v2. This script submits the
# remaining 335 jobs in two sbatch calls:
#
#   Call 1: continuation of pilot for n=20-100 (indices 25-224, 200 jobs).
#   Call 2: new n=5/10/15 extension (indices 0-134, 135 jobs).
#
# cache_volumes and num_workers are set automatically by the inner script:
#   n <= 35 -> cache_volumes=true, num_workers=0
#   n >= 50 -> cache_volumes=false, num_workers=4
#
# Run on DAIC:
#   bash ~/RP/Code/DAIC/data_efficiency/submit_channel_window_full.sh

set -euo pipefail

SCRIPT="$HOME/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: inner submit script not found at $SCRIPT" >&2
  exit 1
fi

# Shared env for both calls.
export DATA_SEEDS_CSV="101,202,303"
export TRAINING_SEEDS_CSV="42,43,44"
export VARIANTS_CSV="2d,3ch,5ch,7ch,9ch"
export STUDY_NAME="data_efficiency_100train_channel_window_v2"
export RUN_SUFFIX="100train_channel_window_v2"

# -----------------------------------------------------------------------
# Call 1: continue pilot for n=20-100, all remaining seeds (200 jobs).
# Indices 0-24 are the pilot (data_seed=101, tseed=42). Skip them.
# -----------------------------------------------------------------------
echo "=== Submitting n=20-100 continuation (array 25-224, 200 jobs) ==="
export TRAIN_SIZES_CSV="20,35,50,75,100"
sbatch --export=ALL --array=25-224 "$SCRIPT"

# -----------------------------------------------------------------------
# Call 2: new n=5/10/15 extension (135 jobs).
# Fresh array 0-134 with only the three new training sizes.
# -----------------------------------------------------------------------
echo "=== Submitting n=5/10/15 extension (array 0-134, 135 jobs) ==="
export TRAIN_SIZES_CSV="5,10,15"
sbatch --qos=short --time=01:30:00 --export=ALL --array=0-134 "$SCRIPT"

echo ""
echo "Submitted 335 new jobs across two array batches."
echo "With the 25-job pilot already running, the full 360-job matrix is covered:"
echo "  n=5,10,15,20,35,50,75,100 x 2D/3ch/5ch/7ch/9ch x 3 data seeds x 3 training seeds"
echo "Study: $STUDY_NAME  Run suffix: $RUN_SUFFIX"
