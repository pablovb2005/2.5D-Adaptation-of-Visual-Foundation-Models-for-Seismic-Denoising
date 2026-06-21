#!/bin/bash
# Submit the two extra training-seed replicates for the core main comparison.

set -euo pipefail

ROOT="$HOME/RP/Code/DAIC"

sbatch "$ROOT/2d/impeccable_repeated_stride5_lora_r16/seed43_run02/submit.sh"
sbatch "$ROOT/2d/impeccable_repeated_stride5_lora_r16/seed44_run03/submit.sh"
sbatch "$ROOT/3ch/impeccable_neighbors3_stride5_lora_r16/seed43_run02/submit.sh"
sbatch "$ROOT/3ch/impeccable_neighbors3_stride5_lora_r16/seed44_run03/submit.sh"
sbatch "$ROOT/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed43_run02/submit.sh"
sbatch "$ROOT/5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16/seed44_run03/submit.sh"
