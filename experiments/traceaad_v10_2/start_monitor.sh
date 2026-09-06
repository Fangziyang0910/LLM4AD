#!/usr/bin/env bash
set -euo pipefail

SESSION="v102_monitor"
PORT="8765"
RESTART=false

for arg in "$@"; do
    if [ "$arg" = "--restart" ]; then
        RESTART=true
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        PORT="$arg"
    fi
done

if [ "$RESTART" = true ] && tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Stopping existing $SESSION for restart..."
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    sleep 0.5
fi

# If v102_monitor is already running, notify and exit
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION is already running."
    echo "Visit: http://127.0.0.1:$PORT"
    exit 0
fi

# If other monitor sessions are running on the default port, stop them to prevent address conflict
for other in v11_monitor v103_monitor; do
for other in v103_monitor; do
    if tmux has-session -t "$other" 2>/dev/null; then
        echo "Stopping other $other session to free port $PORT..."
        tmux kill-session -t "$other" 2>/dev/null || true
        sleep 0.5
    fi
done

# If port is still occupied by any stale python process, gracefully clean it
if command -v fuser >/dev/null 2>&1; then
    if fuser "$PORT/tcp" 2>/dev/null; then
        echo "Port $PORT is occupied. Freeing port $PORT..."
        fuser -k -TERM "$PORT/tcp" 2>/dev/null || true
        sleep 0.5
    fi
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DIR/../.." && pwd)"

echo "Starting TraceAAD V10.2 Monitor on port $PORT in tmux session $SESSION..."
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" "uv run python -m experiments.traceaad_v10_2.monitor --port $PORT"
sleep 1.5

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "=========================================================="
    echo " TraceAAD V10.2 可视化监控已启动"
    echo " 本地访问: http://127.0.0.1:$PORT"
    echo " tmux 会话: tmux attach -t $SESSION"
    echo "=========================================================="
else
    echo "启动失败，请检查日志或手动执行:"
    echo "  uv run python -m experiments.traceaad_v10_2.monitor --port $PORT"
    exit 1
fi
