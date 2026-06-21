#!/bin/bash
# n=5/10/15 extension for the 5-variant channel-window sweep.
#
# Adds the three smallest training sizes to data_efficiency_100train_channel_window_v2.
# All n<=35 jobs use cache_volumes=true + num_workers=0 (fast, safe on NFS).
#
# Variants:  2D / 3ch / 5ch / 7ch / 9ch
# n-train:   5, 10, 15
# Data seeds:     101, 202, 303
# Training seeds: 42, 43, 44
# Total:     3 x 5 x 3 x 3 = 135 jobs (array 0-134)
#
# Submit after queue space opens (typically once full_ft and/or controls jobs complete):
#   bash ~/RP/Code/DAIC/data_efficiency/submit_channel_window_small_n.sh

set -euo pipefail

SCRIPT="$HOME/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "ERROR: inner submit script not found at $SCRIPT" >&2
  exit 1
fi

echo "=== Submitting n=5/10/15 extension (array 0-134, 135 jobs) ==="

DATA_SEEDS_CSV=101,202,303 \
TRAINING_SEEDS_CSV=42,43,44 \
TRAIN_SIZES_CSV=5,10,15 \
VARIANTS_CSV=2d,3ch,5ch,7ch,9ch \
STUDY_NAME=data_efficiency_100train_channel_window_v2 \
RUN_SUFFIX=100train_channel_window_v2 \
sbatch --qos=short --time=01:30:00 --export=ALL --array=0-134 "$SCRIPT"

echo "Submitted 135 jobs (n=5, 10, 15 x 5 variants x 3 data seeds x 3 training seeds)."
echo "Study: data_efficiency_100train_channel_window_v2"
