#!/bin/bash
# Submit all 9 mechanism control training runs.
#
# Controls:
#   3ch_shuffled    — 3ch architecture, neighbor channels from a different volume during training
#   5ch_repeated_center — 5ch architecture, input [t,t,t,t,t] (capacity control, no neighbor info)
#   5ch_shuffled    — 5ch architecture, neighbor channels from a different volume during training
#
# After runs complete, evaluate all with:
#   pwsh Code/DINOv3/src/evaluation/runners/evaluate_mechanism_controls.ps1

set -euo pipefail

ROOT="$HOME/RP/Code/DAIC"

echo "Submitting 3ch_shuffled controls..."
sbatch "$ROOT/3ch_shuffled/impeccable_shuffled3_stride5_lora_r16/seed42_run01/submit.sh"
sbatch "$ROOT/3ch_shuffled/impeccable_shuffled3_stride5_lora_r16/seed43_run02/submit.sh"
sbatch "$ROOT/3ch_shuffled/impeccable_shuffled3_stride5_lora_r16/seed44_run03/submit.sh"

echo "Submitting 5ch_repeated_center controls..."
sbatch "$ROOT/5ch_repeated_center/impeccable_repeated_center_stride5_patch_emb_lora_r16/seed42_run01/submit.sh"
sbatch "$ROOT/5ch_repeated_center/impeccable_repeated_center_stride5_patch_emb_lora_r16/seed43_run02/submit.sh"
sbatch "$ROOT/5ch_repeated_center/impeccable_repeated_center_stride5_patch_emb_lora_r16/seed44_run03/submit.sh"

echo "Submitting 5ch_shuffled controls..."
sbatch "$ROOT/5ch_shuffled/impeccable_shuffled5_stride5_patch_emb_lora_r16/seed42_run01/submit.sh"
sbatch "$ROOT/5ch_shuffled/impeccable_shuffled5_stride5_patch_emb_lora_r16/seed43_run02/submit.sh"
sbatch "$ROOT/5ch_shuffled/impeccable_shuffled5_stride5_patch_emb_lora_r16/seed44_run03/submit.sh"

echo "All 9 mechanism control jobs submitted."
echo "Monitor with: squeue -u pvarelabernal"
