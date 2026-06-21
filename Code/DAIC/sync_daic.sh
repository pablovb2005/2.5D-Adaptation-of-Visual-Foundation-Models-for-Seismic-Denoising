#!/usr/bin/env bash
# One-command local <-> DAIC sync for this project.
#
# Default workflow:
#   1. Push local Code/DINOv3 and Code/DAIC to DAIC.
#   2. Pull staff-bulk experiments/runs and experiments/summaries to local.
#
# NOTE: Any rsync that runs ON DAIC itself (server-side, same NFS) must use
# --inplace; the staff-bulk NFS blocks mkstemp (the default rsync temp file).
# This script runs cross-network (local → DAIC via SSH) and does NOT need --inplace.
#
# Run from Git Bash/WSL at the repo root:
#   bash Code/DAIC/sync_daic.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

REMOTE="${REMOTE:-daic}"
REMOTE_STUDENT_DIR="${REMOTE_STUDENT_DIR:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/pvarelabernal}"
REMOTE_CODE_ROOT="${REMOTE_CODE_ROOT:-~/RP/Code}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/daic_sync_%r_%h_%p}"

DRY_RUN=0
PUSH=1
PULL=1
CHECKPOINTS=1

usage() {
    cat <<'EOF'
Usage: bash Code/DAIC/sync_daic.sh [options]

Default:
  Push local Code/DINOv3 and Code/DAIC to DAIC, then pull DAIC staff-bulk
  experiments/runs and experiments/summaries back to local experiments/.
  All run families (2d, 3ch, 5ch, ablations, robustness, ...) are now under
  the canonical experiments/runs/ layout on DAIC after the 2026-06-11 migration.

Options:
  --dry-run           Show rsync changes without transferring files.
  --push-only         Only push Code/DINOv3 and Code/DAIC to DAIC.
  --pull-only         Only pull experiments from DAIC to local.
  --no-checkpoints    Pull results without best.pt and last.pt.
  --remote HOST       SSH target or alias. Default: daic.
  -h, --help          Show this help.

Environment overrides:
  PROJECT_ROOT        Local project root. Default: inferred from this script.
  REMOTE              SSH target/alias. Default: daic.
  REMOTE_STUDENT_DIR  Staff-bulk root. Default: Pablo's PRLab staff-bulk dir.
  REMOTE_CODE_ROOT    Remote code root. Default: ~/RP/Code.

Required SSH alias in the same environment you run this script from:
  Host daic
    HostName login.daic.tudelft.nl
    User pvarelabernal

Then use the default command:
  bash Code/DAIC/sync_daic.sh

The script enables SSH multiplexing for this run, so after the first successful
password prompt the remaining rsync/ssh calls should reuse the same connection.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --push-only)
            PUSH=1
            PULL=0
            shift
            ;;
        --pull-only)
            PUSH=0
            PULL=1
            shift
            ;;
        --no-checkpoints)
            CHECKPOINTS=0
            shift
            ;;
        --remote)
            if [[ $# -lt 2 ]]; then
                echo "Missing value for --remote" >&2
                exit 2
            fi
            REMOTE="$2"
            shift 2
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

RSYNC_OPTS=(-av)
if [[ "$DRY_RUN" -eq 1 ]]; then
    RSYNC_OPTS+=(--dry-run)
fi
SSH_OPTS=(
    -o ControlMaster=auto
    -o ControlPersist=10m
    -o "ControlPath=${SSH_CONTROL_PATH}"
)
RSYNC_OPTS+=(-e "ssh ${SSH_OPTS[*]}")

CODE_EXCLUDES=(
    --exclude='.venv'
    --exclude='__pycache__'
    --exclude='*.pyc'
)

RUN_FILTERS=(
    --prune-empty-dirs
    --include='*/'
    --include='history.csv'
    --include='training_timing.csv'
    --include='config.yaml'
    --include='run_meta.yaml'
    --include='*.out'
    --include='*.err'
    --include='eval_results/***'
    --include='stitched_eval_results/***'
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

push_code() {
    echo "== Push code to ${REMOTE}:${REMOTE_CODE_ROOT} =="
    rsync "${RSYNC_OPTS[@]}" "${CODE_EXCLUDES[@]}" \
        "${PROJECT_ROOT}/Code/DINOv3/" \
        "${REMOTE}:${REMOTE_CODE_ROOT}/DINOv3/"

    rsync "${RSYNC_OPTS[@]}" "${CODE_EXCLUDES[@]}" \
        "${PROJECT_ROOT}/Code/DAIC/" \
        "${REMOTE}:${REMOTE_CODE_ROOT}/DAIC/"

    if [[ -d "${PROJECT_ROOT}/Code/SFM" ]]; then
        rsync "${RSYNC_OPTS[@]}" \
            "${PROJECT_ROOT}/Code/SFM/" \
            "${REMOTE}:${REMOTE_CODE_ROOT}/SFM/"
    fi
}

pull_experiments() {
    local local_runs="${PROJECT_ROOT}/experiments/runs"
    local local_summaries="${PROJECT_ROOT}/experiments/summaries"
    local remote_experiments="${REMOTE_STUDENT_DIR}/experiments"

    mkdir -p "$local_runs" "$local_summaries"

    # Trigger the NFS automount for staff-bulk before rsync.
    # rsync spawns a non-interactive subprocess on the login node; the automounter
    # only wakes up /tudelft.net/staff-bulk/ on demand, so without this ls the
    # rsync server gets EACCES (Permission denied 13) even though the path exists.
    echo "== Triggering staff-bulk automount =="
    if ! ssh "${SSH_OPTS[@]}" "$REMOTE" "ls '${REMOTE_STUDENT_DIR}' >/dev/null 2>&1"; then
        echo "Warning: could not list ${REMOTE_STUDENT_DIR} — rsync may still fail." >&2
    fi

    if [[ "$CHECKPOINTS" -eq 1 ]]; then
        echo "== Pull runs from ${REMOTE}:${remote_experiments}/runs/ with best.pt and last.pt =="
    else
        echo "== Pull runs from ${REMOTE}:${remote_experiments}/runs/ without checkpoints =="
    fi
    rsync "${RSYNC_OPTS[@]}" "${RUN_FILTERS[@]}" \
        "${REMOTE}:${remote_experiments}/runs/" \
        "${local_runs}/"

    if ssh "${SSH_OPTS[@]}" "$REMOTE" "test -d '${remote_experiments}/summaries'"; then
        echo "== Pull summaries from ${REMOTE}:${remote_experiments}/summaries/ =="
        rsync "${RSYNC_OPTS[@]}" \
            "${REMOTE}:${remote_experiments}/summaries/" \
            "${local_summaries}/"
    else
        echo "== Remote summaries directory not found; skipping summaries =="
    fi
}

echo "Project root: $PROJECT_ROOT"
echo "Remote:       $REMOTE"
echo "Dry run:      $DRY_RUN"

if [[ "$PUSH" -eq 1 ]]; then
    bash "${SCRIPT_DIR}/audit_daic_scripts.sh"
    push_code
fi

if [[ "$PULL" -eq 1 ]]; then
    pull_experiments
fi

echo "DAIC sync finished."
