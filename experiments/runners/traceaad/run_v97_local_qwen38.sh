#!/usr/bin/env bash
# 最终正确版：12路 V9.7 Qwen3.8 + resume已跑700轮
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BATCH="${BATCH:-v9_7_qwen38_20260815_010107}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
POLL_SEC="${POLL_SEC:-30}"
LOG_DIR="${REPO_ROOT}/experiments/_logs"
mkdir -p "$LOG_DIR"
BATCH_LOG="${LOG_DIR}/${BATCH}_sequential.log"

TASKS=(tsp_construct online_bin_packing op_aco cvrp)
TASK_SHORT=(tsp obp op cvrp)

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$BATCH_LOG"
}

session_name() {
    local task="$1" rep="$2"
    printf 'v97q38_%s_r%s' "${task:0:3}" "$rep"
}

run_dir_for() {
    local task="$1" rep="$2"
    printf '%s/experiments/%s/traceaad_v9_7/%s_%s_rep%s' \
        "$REPO_ROOT" "$task" "$BATCH" "$task" "$rep"
}

is_running() {
    tmux has-session -t "=$1" 2>/dev/null
}

launch_one() {
    local task="$1" rep="$2"
    local run_name="${BATCH}_${task}_rep${rep}"
    local run_dir session cmd
    run_dir="$(run_dir_for "$task" "$rep")"
    session="$(session_name "$task" "$rep")"

    if is_running "$session"; then
        log "resume $run_name (already running)"
        return 0
    fi

    if [[ -d "$run_dir" ]]; then
        log "resume $run_name (checkpoint found)"
        cmd="uv run python -m experiments.runners.traceaad.run --task $(printf '%q' "$task") --version v9_7 --backend local --model Qwen3.8-27B --budget 1000 --repeat $rep --seed $rep --resume-from $run_dir"
    else
        log "start $run_name"
        cmd="uv run python -m experiments.runners.traceaad.run --task $(printf '%q' "$task") --version v9_7 --backend local --model Qwen3.8-27B --budget 1000 --repeat $rep --seed $rep --run-name $run_name"
    fi

    tmux new-session -d -s "$session" -c "$REPO_ROOT" "$cmd"
    log "launched/resumed $run_name"
}

log "=== Starting 12路 V9.7 on Qwen3.8 ==="

while true; do
    running_n=0
    pending=()

    for task in "${TASKS[@]}"; do
        for rep in 1 2 3; do
            session="$(session_name "$task" "$rep")"
            run_dir="$(run_dir_for "$task" "$rep")"

            if is_running "$session"; then
                running_n=$((running_n + 1))
                continue
            fi

            if [[ -d "$run_dir" ]]; then
                pending+=("${task}:${rep}")
            fi
        done
    done

    log "running=${running_n} pending=${#pending[@]}"

    for spec in "${pending[@]}"; do
        if [[ "$running_n" -ge "$MAX_CONCURRENT" ]]; then break; fi
        task="${spec%%:*}"
        rep="${spec##*:}"
        launch_one "$task" "$rep"
        running_n=$((running_n + 1))
    done

    sleep "$POLL_SEC"
done
