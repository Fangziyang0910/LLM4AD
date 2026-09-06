#!/usr/bin/env bash
# Start the TraceAAD v10.6 real-time web monitor in a detached tmux session.
# Usage:
#   ./start_monitor.sh [PORT] [--foreground] [--restart]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SESSION_NAME="v106_monitor"
PORT="8765"
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

# If restart requested, kill existing v106_monitor
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    if [ "$RESTART" -eq 1 ]; then
        echo "Restarting existing tmux session: ${SESSION_NAME}..."
        tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
        sleep 0.5
    else
        echo "Monitor is already running in tmux session: ${SESSION_NAME}"
        echo "  URL: http://127.0.0.1:${PORT}"
        echo "  Attach: tmux attach -t ${SESSION_NAME}"
        echo "  Logs: tmux capture-pane -pt ${SESSION_NAME}:0"
        echo "  Use --restart to restart."
        exit 0
    fi
fi

# Stop other monitor sessions to prevent port conflict on default port
for other_sess in v105_monitor v104_monitor v103_monitor v102_monitor; do
    if tmux has-session -t "${other_sess}" 2>/dev/null; then
        echo "Stopping previous monitor session ${other_sess} to free port ${PORT}..."
        tmux kill-session -t "${other_sess}" 2>/dev/null || true
        sleep 0.5
    fi
done

# If port is still occupied by any stale process, free it
if command -v fuser >/dev/null 2>&1; then
    if fuser "${PORT}/tcp" 2>/dev/null; then
        echo "Port ${PORT} is occupied. Freeing port ${PORT}..."
        fuser -k -TERM "${PORT}/tcp" 2>/dev/null || true
        sleep 0.5
    fi
fi

CMD="uv run python -m experiments.traceaad_v10_6.monitor --port ${PORT}"

if [ "$FOREGROUND" -eq 1 ]; then
    echo "Starting monitor in foreground on port ${PORT}..."
    exec ${CMD}
fi

echo "Starting TraceAAD V10.6 monitor in background tmux session: ${SESSION_NAME} (port ${PORT})..."
tmux new-session -d -s "${SESSION_NAME}" "${CMD}"

# Wait a moment and check status
sleep 2
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "=========================================================="
    echo "🚀 TraceAAD V10.6 可视化监控启动成功!"
    echo "  Web UI: http://127.0.0.1:${PORT}"
    echo "  Attach: tmux attach -t ${SESSION_NAME}"
    echo "  Stop:   tmux kill-session -t ${SESSION_NAME}"
    echo "=========================================================="
else
    echo "ERROR: Monitor failed to start. Recent output:"
    tmux capture-pane -pt "${SESSION_NAME}:0" 2>/dev/null || true
    exit 1
fi
