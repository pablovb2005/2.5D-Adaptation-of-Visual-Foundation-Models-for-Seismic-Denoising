#!/usr/bin/env bash
# Retrieve DAIC experiment artifacts into the local experiments/runs layout.
#
# Run from local Git Bash/WSL, for example:
#   bash Code/DAIC/retrieve_daic_results.sh
#   bash Code/DAIC/retrieve_daic_results.sh --checkpoints --family full_ft
#
# By default this copies lightweight result artifacts only:
# history/config metadata, logs, and eval_results. It deliberately excludes
# checkpoints unless --checkpoints is passed.

set -euo pipefail

REMOTE_USER="${REMOTE_USER:-pvarelabernal}"
REMOTE_HOST="${REMOTE_HOST:-login.daic.tudelft.nl}"
REMOTE_STUDENT_DIR="${REMOTE_STUDENT_DIR:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

CHECKPOINTS=0
DRY_RUN=0
PULL_SUMMARIES=0
FAMILIES=()

usage() {
    cat <<'EOF'
Usage: bash Code/DAIC/retrieve_daic_results.sh [options]

Options:
  --checkpoints       Also copy best.pt and last.pt files.
  --summaries         Also copy experiments/summaries if present on DAIC.
  --family NAME       Only copy one runs subfolder, e.g. full_ft or ablations.
                      May be repeated. Default: copy all runs subfolders.
  --dry-run           Show what would be copied without transferring files.
  -h, --help          Show this help.

Environment overrides:
  REMOTE_USER, REMOTE_HOST, REMOTE_STUDENT_DIR, PROJECT_ROOT

Default remote:
  pvarelabernal@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal/experiments

Notes:
  Known legacy main folders under remote experiments/{2d,3ch,5ch}/ are also
  pulled into the local experiments/runs layout when no matching remote
  experiments/runs/<family>/<variant>/ directory exists.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoints)
            CHECKPOINTS=1
            shift
            ;;
        --summaries)
            PULL_SUMMARIES=1
            shift
            ;;
        --family)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --family" >&2
                exit 2
            fi
            FAMILIES+=("$2")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required. Run this from Git Bash/WSL or install rsync." >&2
    exit 1
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"
REMOTE_RUNS="${REMOTE_STUDENT_DIR}/experiments/runs"
LOCAL_RUNS="${PROJECT_ROOT}/experiments/runs"

mkdir -p "$LOCAL_RUNS"

RSYNC_OPTS=(-av --prune-empty-dirs)
if [[ "$DRY_RUN" -eq 1 ]]; then
    RSYNC_OPTS+=(--dry-run)
fi

RUN_FILTERS=(
    --include='*/'
    --include='history.csv'
    --include='training_timing.csv'
    --include='config.yaml'
    --include='run_meta.yaml'
    --include='*.out'
    --include='*.err'
    --include='eval_results/***'
    --include='f3_metrics.csv'
    --include='f3_panel_*.png'
    --include='f3_filtered_ref_metrics.csv'
    --include='f3_filt_panel_*.png'
)
if [[ "$CHECKPOINTS" -eq 1 ]]; then
    RUN_FILTERS+=(--include='best.pt' --include='last.pt')
else
    RUN_FILTERS+=(--exclude='*.pt')
fi
RUN_FILTERS+=(--exclude='*')

LEGACY_MAIN_RUNS=(
    "2d/impeccable_repeated_stride5_lora_r16"
    "3ch/impeccable_neighbors3_stride5_lora_r16"
    "5ch/impeccable_neighbors5_stride5_patch_emb_lora_r16"
)

should_pull_legacy_family() {
    local rel="$1"
    local family="${rel%%/*}"

    if [[ "${#FAMILIES[@]}" -eq 0 ]]; then
        return 0
    fi

    for requested in "${FAMILIES[@]}"; do
        if [[ "$requested" == "$family" || "$requested" == "$rel" ]]; then
            return 0
        fi
    done
    return 1
}

pull_legacy_main_runs() {
    local rel
    for rel in "${LEGACY_MAIN_RUNS[@]}"; do
        if ! should_pull_legacy_family "$rel"; then
            continue
        fi

        local remote_canonical="${REMOTE_RUNS}/${rel}"
        local remote_legacy="${REMOTE_STUDENT_DIR}/experiments/${rel}"
        local local_target="${LOCAL_RUNS}/${rel}"

        if ssh "$REMOTE" "test -d '${remote_canonical}'"; then
            echo "Skip legacy ${rel}; remote canonical directory exists."
            continue
        fi
        if ! ssh "$REMOTE" "test -d '${remote_legacy}'"; then
            continue
        fi

        echo "Remote legacy main: ${REMOTE}:${remote_legacy}/"
        echo "Local normalized:   ${local_target}/"
        mkdir -p "$local_target"
        rsync "${RSYNC_OPTS[@]}" "${RUN_FILTERS[@]}" \
            "${REMOTE}:${remote_legacy}/" \
            "${local_target}/"
    done
}

echo "Remote runs: ${REMOTE}:${REMOTE_RUNS}/"
echo "Local runs:  ${LOCAL_RUNS}/"
if [[ "$CHECKPOINTS" -eq 0 ]]; then
    echo "Mode: lightweight artifacts only; checkpoints excluded."
else
    echo "Mode: including best.pt and last.pt checkpoints."
fi

if [[ "${#FAMILIES[@]}" -eq 0 ]]; then
    rsync "${RSYNC_OPTS[@]}" "${RUN_FILTERS[@]}" \
        "${REMOTE}:${REMOTE_RUNS}/" \
        "${LOCAL_RUNS}/"
else
    for family in "${FAMILIES[@]}"; do
        mkdir -p "${LOCAL_RUNS}/${family}"
        remote_family="${REMOTE_RUNS}/${family}"
        if ssh "$REMOTE" "test -d '${remote_family}'"; then
            rsync "${RSYNC_OPTS[@]}" "${RUN_FILTERS[@]}" \
                "${REMOTE}:${remote_family}/" \
                "${LOCAL_RUNS}/${family}/"
        else
            echo "Remote canonical family not found, skipping: ${REMOTE}:${remote_family}/"
        fi
    done
fi

pull_legacy_main_runs

if [[ "$PULL_SUMMARIES" -eq 1 ]]; then
    LOCAL_SUMMARIES="${PROJECT_ROOT}/experiments/summaries"
    REMOTE_SUMMARIES="${REMOTE_STUDENT_DIR}/experiments/summaries"
    mkdir -p "$LOCAL_SUMMARIES"
    SUMMARY_FILTERS=(
        --include='*/'
        --include='*.csv'
        --include='*.md'
        --include='*.png'
        --include='*.json'
        --include='*.txt'
        --exclude='*'
    )
    echo "Remote summaries: ${REMOTE}:${REMOTE_SUMMARIES}/"
    echo "Local summaries:  ${LOCAL_SUMMARIES}/"
    rsync "${RSYNC_OPTS[@]}" "${SUMMARY_FILTERS[@]}" \
        "${REMOTE}:${REMOTE_SUMMARIES}/" \
        "${LOCAL_SUMMARIES}/"
fi

echo "Retrieval finished."
