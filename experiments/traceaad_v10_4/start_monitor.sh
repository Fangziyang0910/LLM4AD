#!/usr/bin/env bash
# Start the TraceAAD v10.4 real-time web monitor in a detached tmux session.
# Usage:
#   ./start_monitor.sh [PORT] [--foreground] [--restart]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SESSION_NAME="v104_monitor"
PORT="${1:-8765}"
FOREGROUND=0
RESTART=0

for arg in "$@"; do
    case "$arg" in
        --foreground|-f)
            FOREGROUND=1
            ;;
        --restart|-r)
            RESTART=1
            ;;
        [0-9]*)
            PORT="$arg"
            ;;
    esac
done

cd "${REPO_ROOT}"

# If port is occupied or monitor is running and restart is requested
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    if [ "$RESTART" -eq 1 ]; then
        echo "Restarting existing tmux session: ${SESSION_NAME}..."
        tmux kill-session -t "${SESSION_NAME}"
        sleep 1
    else
        echo "Monitor is already running in tmux session: ${SESSION_NAME}"
        echo "  URL: http://127.0.0.1:${PORT}"
        echo "  Attach: tmux attach -t ${SESSION_NAME}"
        echo "  Logs: tmux capture-pane -pt ${SESSION_NAME}:0"
        echo "  Use --restart to restart."
        exit 0
    fi
fi

# Also check if v103_monitor is occupying the port and restart was requested
if tmux has-session -t "v103_monitor" 2>/dev/null; then
    if [ "$RESTART" -eq 1 ]; then
        echo "Stopping previous v103_monitor session..."
        tmux kill-session -t "v103_monitor"
        sleep 1
    fi
fi

CMD="uv run python -m experiments.traceaad_v10_4.monitor --port ${PORT}"

if [ "$FOREGROUND" -eq 1 ]; then
    echo "Starting monitor in foreground on port ${PORT}..."
    exec ${CMD}
fi

echo "Starting monitor in background tmux session: ${SESSION_NAME} (port ${PORT})..."
tmux new-session -d -s "${SESSION_NAME}" "${CMD}"

# Wait a moment and check status
sleep 2
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "Monitor started successfully!"
    echo "  Web UI: http://127.0.0.1:${PORT}"
    echo "  Attach: tmux attach -t ${SESSION_NAME}"
    echo "  Stop:   tmux kill-session -t ${SESSION_NAME}"
else
    echo "ERROR: Monitor failed to start. Recent output:"
    tmux capture-pane -pt "${SESSION_NAME}:0" 2>/dev/null || true
    exit 1
fi
