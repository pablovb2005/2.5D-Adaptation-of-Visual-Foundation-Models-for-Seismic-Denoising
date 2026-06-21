#!/usr/bin/env bash
# Persistent SSH multiplexing helper for DAIC access from WSL/Git Bash.
#
# Password authentication on login.daic.tudelft.nl is interactive. This helper
# lets Pablo authenticate once, then lets later non-interactive agents reuse the
# same OpenSSH control socket.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

REMOTE="${REMOTE:-daic}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-$HOME/.ssh/daic_mux_%r_%h_%p}"
SSH_CONTROL_PERSIST="${SSH_CONTROL_PERSIST:-8h}"

usage() {
    cat <<EOF
Usage: bash Code/DAIC/daic_mux.sh <command> [args]

Commands:
  start           Start the persistent DAIC SSH master connection.
  check           Verify that the master connection can run a command.
  status          Ask ssh whether the master connection is alive.
  stop            Close the master connection.
  cmd <command>   Run a remote command through the master connection.
  shell           Open an interactive DAIC shell using the master connection.
  sync [args]     Run Code/DAIC/sync_daic.sh through the master connection.
  path            Print the configured OpenSSH ControlPath.

Environment:
  REMOTE                SSH target/alias. Default: daic.
  SSH_CONTROL_PATH      Control socket path. Default: ~/.ssh/daic_mux_%r_%h_%p.
  SSH_CONTROL_PERSIST   How long to keep the master open. Default: 8h.

Examples:
  bash Code/DAIC/daic_mux.sh start
  bash Code/DAIC/daic_mux.sh check
  bash Code/DAIC/daic_mux.sh cmd 'squeue -u pvarelabernal'
  bash Code/DAIC/daic_mux.sh sync --dry-run
EOF
}

ssh_mux_opts=(
    -o BatchMode=yes
    -o ControlMaster=auto
    -o "ControlPersist=${SSH_CONTROL_PERSIST}"
    -o "ControlPath=${SSH_CONTROL_PATH}"
)

ssh_check_opts=(
    -o BatchMode=yes
    -o "ControlPath=${SSH_CONTROL_PATH}"
)

ensure_ssh_dir() {
    mkdir -p "$HOME/.ssh"
    chmod 700 "$HOME/.ssh"
}

is_active() {
    ssh "${ssh_check_opts[@]}" "$REMOTE" true >/dev/null 2>&1
}

require_active() {
    if ! is_active; then
        cat >&2 <<EOF
DAIC SSH master is not active.

Start it from an interactive terminal first:
  bash Code/DAIC/daic_mux.sh start

EOF
        exit 1
    fi
}

start_master() {
    ensure_ssh_dir

    if is_active; then
        echo "DAIC SSH master is already active."
        ssh "${ssh_check_opts[@]}" "$REMOTE" hostname
        return 0
    fi

    cat <<EOF
Starting DAIC SSH master for ${REMOTE}.

Enter your DAIC password if prompted. The password is handled by ssh directly
and is not stored by this script. On success, later commands can reuse:
  ${SSH_CONTROL_PATH}

EOF

    ssh -MNf \
        -o ControlMaster=yes \
        -o "ControlPersist=${SSH_CONTROL_PERSIST}" \
        -o "ControlPath=${SSH_CONTROL_PATH}" \
        "$REMOTE"

    echo "DAIC SSH master started."
    ssh "${ssh_check_opts[@]}" "$REMOTE" hostname
}

check_master() {
    ssh "${ssh_check_opts[@]}" "$REMOTE" hostname
}

status_master() {
    ssh -O check "${ssh_check_opts[@]}" "$REMOTE"
}

stop_master() {
    ssh -O exit "${ssh_check_opts[@]}" "$REMOTE" || true
}

run_remote_command() {
    if [[ $# -lt 1 ]]; then
        echo "Missing remote command for 'cmd'." >&2
        usage >&2
        exit 2
    fi
    require_active
    ssh "${ssh_mux_opts[@]}" "$REMOTE" "$@"
}

run_shell() {
    require_active
    ssh "${ssh_mux_opts[@]}" "$REMOTE"
}

run_sync() {
    require_active
    cd "$PROJECT_ROOT"
    SSH_CONTROL_PATH="$SSH_CONTROL_PATH" REMOTE="$REMOTE" \
        bash Code/DAIC/sync_daic.sh "$@"
}

command="${1:-}"
if [[ -z "$command" ]]; then
    usage
    exit 2
fi
shift || true

case "$command" in
    start)
        start_master
        ;;
    check)
        check_master
        ;;
    status)
        status_master
        ;;
    stop)
        stop_master
        ;;
    cmd)
        run_remote_command "$@"
        ;;
    shell)
        run_shell
        ;;
    sync)
        run_sync "$@"
        ;;
    path)
        echo "$SSH_CONTROL_PATH"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac
