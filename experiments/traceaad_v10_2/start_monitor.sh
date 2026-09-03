#!/usr/bin/env bash
SESSION="v102_monitor"
PORT="${1:-8765}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session $SESSION is already running."
    echo "Visit: http://127.0.0.1:$PORT"
    exit 0
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
    echo "启动失败，请检查日志。"
    exit 1
fi
