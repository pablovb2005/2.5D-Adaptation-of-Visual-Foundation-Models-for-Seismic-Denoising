#!/bin/bash
# Fast pilot for the repaired 100-train root:
# 1 data seed x 1 training seed x 5 variants x 5 train sizes = 25 jobs.

#SBATCH --job-name=de100pilot_%a
#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pablovb2005@gmail.com
#SBATCH --output=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_100train_channel_pilot_%A_%a.out
#SBATCH --error=/home/nfs/pvarelabernal/RP/Code/DAIC/data_efficiency/slurm_100train_channel_pilot_%A_%a.err

set -euo pipefail

export DATA_SEEDS_CSV="${DATA_SEEDS_CSV:-101}"
export TRAINING_SEEDS_CSV="${TRAINING_SEEDS_CSV:-42}"
export TRAIN_SIZES_CSV="${TRAIN_SIZES_CSV:-20,35,50,75,100}"
export VARIANTS_CSV="${VARIANTS_CSV:-2d,3ch,5ch,7ch,9ch}"

export DATASET_NAME="${DATASET_NAME:-data_efficiency_100train_canonical_v2}"
export STUDY_NAME="${STUDY_NAME:-data_efficiency_100train_channel_window_v2}"
export RUN_SUFFIX="${RUN_SUFFIX:-100train_channel_window_v2}"

exec bash "$HOME/RP/Code/DAIC/data_efficiency/submit_100train_data_efficiency.sh"
