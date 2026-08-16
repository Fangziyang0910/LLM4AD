#!/usr/bin/env bash
# V9.7 Qwen3.8：TSP / OBP / OP 各三路，OP 结束后退出，不启动 CVRP。
# CVRP 与远程 V9.8/V9.9 抢本机评价核，本批不再续跑。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BATCH="${BATCH:-v9_7_qwen38_20260815_010107}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
POLL_SEC="${POLL_SEC:-60}"
LOG_DIR="${REPO_ROOT}/experiments/_logs"
mkdir -p "$LOG_DIR"
BATCH_LOG="${LOG_DIR}/${BATCH}_sequential.log"

TASKS=(tsp_construct online_bin_packing op_aco)
TOTAL=$(( ${#TASKS[@]} * 3 ))

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

is_done() {
    local run_dir="$1"
    local summary ckpt n_eval
    summary="${run_dir}/logs/run_summary.json"
    ckpt="${run_dir}/checkpoints/latest.json"
    if [[ -f "$summary" ]]; then
        if grep -q '"status": *"finished"' "$summary"; then
            return 0
        fi
    fi
    if [[ -f "$ckpt" ]]; then
        n_eval=$(python3 - "$ckpt" <<'PY'
import json, sys
try:
    data = json.loads(open(sys.argv[1]).read())
    print(data.get("n_eval") or 0)
except Exception:
    print(0)
PY
)
        [[ "$n_eval" -ge 1000 ]] && return 0
    fi
    return 1
}

launch_one() {
    local task="$1" rep="$2"
    local run_name="${BATCH}_${task}_rep${rep}"
    local run_dir session cmd
    run_dir="$(run_dir_for "$task" "$rep")"
    session="$(session_name "$task" "$rep")"

    if is_running "$session"; then
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

log "=== Starting ${TOTAL}-run V9.7 on Qwen3.8 (no CVRP, POLL=${POLL_SEC}s) ==="

while true; do
    running_n=0
    done_n=0
    pending=()

    for task in "${TASKS[@]}"; do
        for rep in 1 2 3; do
            session="$(session_name "$task" "$rep")"
            run_dir="$(run_dir_for "$task" "$rep")"

            if is_running "$session"; then
                running_n=$((running_n + 1))
                continue
            fi

            if is_done "$run_dir"; then
                done_n=$((done_n + 1))
                continue
            fi

            pending+=("${task}:${rep}")
        done
    done

    log "done=${done_n}/${TOTAL} running=${running_n} pending=${#pending[@]}"

    if (( done_n == TOTAL && running_n == 0 && ${#pending[@]} == 0 )); then
        log "TSP/OBP/OP finished; CVRP skipped. launcher exiting."
        exit 0
    fi

    for spec in "${pending[@]}"; do
        if [[ "$running_n" -ge "$MAX_CONCURRENT" ]]; then break; fi
        task="${spec%%:*}"
        rep="${spec##*:}"
        launch_one "$task" "$rep"
        running_n=$((running_n + 1))
    done

    sleep "$POLL_SEC"
done
