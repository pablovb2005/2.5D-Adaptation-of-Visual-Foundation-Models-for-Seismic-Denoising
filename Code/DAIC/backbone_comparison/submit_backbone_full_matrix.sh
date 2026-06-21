#!/bin/bash
# Submit the full SFM/SwinV2-T backbone-comparison matrix.

set -euo pipefail

ROOT="$HOME/RP/Code/DAIC/backbone_comparison"
sbatch --array=0-53%18 "$ROOT/submit_backbone_comparison.sh"
