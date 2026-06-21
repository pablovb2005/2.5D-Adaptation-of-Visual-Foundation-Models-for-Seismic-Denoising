#!/usr/bin/env bash
# Pre-submit checks for DAIC wrapper scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAIC_DIR="${1:-$SCRIPT_DIR}"

unsafe_check_ready_pattern='("\$PY310"[[:space:]]+"\$TOOL"[[:space:]]+check-ready|\$PY310[^[:cntrl:]]*prepare_thinkonward_full\.py[[:space:]]+check-ready)'

echo "== Audit DAIC scripts: $DAIC_DIR =="

if grep -R -n -E --include='*.sh' --exclude-dir='__pycache__' \
    "$unsafe_check_ready_pattern" "$DAIC_DIR"; then
    cat >&2 <<'EOF'
ERROR: Unsafe DAIC dataset readiness check found.

prepare_thinkonward_full.py imports NumPy at module import time, but the base
DAIC py310 environment is not guaranteed to have NumPy. Create the /tmp job venv
first, install NumPy from $WHEELS, and run check-ready with the venv Python.
EOF
    exit 1
fi

echo "DAIC script audit passed."
