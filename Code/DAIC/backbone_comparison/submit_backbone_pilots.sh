#!/bin/bash
# Submit the six backbone-comparison pilot jobs:
# SFM and SwinV2-T x 2D/3ch/5ch at data_seed=101, training_seed=42.

set -euo pipefail

ROOT="$HOME/RP/Code/DAIC/backbone_comparison"
sbatch --array=0,9,18,27,36,45 "$ROOT/submit_backbone_comparison.sh"
