#!/usr/bin/env bash
# V9.7 Qwen3.8 CVRP 补跑守护：等待本机评价 CPU 窗口后自动启动三路。
#
# 背景：同批次 `v9_7_qwen38_20260815_010107` 的 TSP / OBP 已完成、OP 三路仍在跑，
# CVRP 因与远程 V9.8 / V9.9 批次共用本机评价核而搁置。直接启动会超卖 32 核。
#
# 启动条件（连续 N 次轮询都满足才启动，避免瞬时尖峰误判）：
#   1) 本机 .venv 评价进程总 CPU < 23 核（为 CVRP 3 路 × 2 worker ≈ 6 核留余量）
#   2) 本批次 OP 三路已 done，或评价总 CPU < 15 核（模型与 CPU 均极空闲）
#
# CVRP 以 --eval-workers 2 启动，峰值约 6 核；seeded 分数不随 worker 数变化。
# 全部三路完成后守护退出。重启本脚本可安全 resume。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

BATCH="${BATCH:-v9_7_qwen38_20260815_010107}"
EVAL_CPU_HARD="${EVAL_CPU_HARD:-2300}"   # 23 核：CVRP 可启动的硬上限
EVAL_CPU_EASY="${EVAL_CPU_EASY:-1500}"   # 15 核：OP 未 done 时也允许启动
REQUIRED_CONFIRM="${REQUIRED_CONFIRM:-3}" # 连续确认轮数
POLL_SEC="${POLL_SEC:-60}"
LOG_DIR="${REPO_ROOT}/experiments/_logs"
mkdir -p "$LOG_DIR"
BATCH_LOG="${LOG_DIR}/${BATCH}_cvrp_wait.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$BATCH_LOG"
}

run_dir_for() {
    printf '%s/experiments/%s/traceaad_v9_7/%s_%s_rep%s' \
        "$REPO_ROOT" "$1" "$BATCH" "$1" "$2"
}

is_running() {
    tmux has-session -t "=$1" 2>/dev/null
}

is_done() {
    local run_dir="$1"
    local summary ckpt n_eval
    summary="${run_dir}/logs/run_summary.json"
    ckpt="${run_dir}/checkpoints/latest.json"
    if [[ -f "$summary" ]] && grep -q '"status": *"finished"' "$summary"; then
        return 0
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

eval_cpu_pct() {
    # 仅统计本仓库评价进程（.venv python 子进程），不含 llama-server 与编辑器。
    ps -u "$(whoami)" -o %cpu=,args= | awk '/\.venv\/bin\/python/ {s += $1} END {print int(s)}'
}

op_done_count() {
    local done=0 rep
    for rep in 1 2 3; do
        if is_done "$(run_dir_for op_aco "$rep")"; then
            done=$((done + 1))
        fi
    done
    echo "$done"
}

launch_one() {
    local task="$1" rep="$2"
    local run_name run_dir session cmd
    run_name="${BATCH}_${task}_rep${rep}"
    run_dir="$(run_dir_for "$task" "$rep")"
    session="v97q38_${task:0:3}_r${rep}"

    if is_running "$session" || is_done "$run_dir"; then
        return 0
    fi

    if [[ -d "$run_dir" ]]; then
        log "resume $run_name (checkpoint found)"
        cmd="uv run python -m experiments.runners.traceaad.run --task ${task} --version v9_7 --backend local --model Qwen3.8-27B --budget 1000 --eval-workers 2 --repeat ${rep} --seed ${rep} --resume-from ${run_dir}"
    else
        log "start $run_name"
        cmd="uv run python -m experiments.runners.traceaad.run --task ${task} --version v9_7 --backend local --model Qwen3.8-27B --budget 1000 --eval-workers 2 --repeat ${rep} --seed ${rep} --run-name ${run_name}"
    fi

    tmux new-session -d -s "$session" -c "$REPO_ROOT" "$cmd"
    log "launched/resumed $run_name"
}

log "=== V9.7 Qwen3.8 CVRP wait-and-launch (batch ${BATCH}, hard=${EVAL_CPU_HARD} easy=${EVAL_CPU_EASY}, confirm=${REQUIRED_CONFIRM}) ==="

confirm=0
while true; do
    running=0
    done_n=0
    for rep in 1 2 3; do
        session="v97q38_cvr_r${rep}"
        run_dir="$(run_dir_for cvrp_aco "$rep")"
        if is_running "$session"; then
            running=$((running + 1))
        elif is_done "$run_dir"; then
            done_n=$((done_n + 1))
        fi
    done

    if (( done_n == 3 && running == 0 )); then
        log "all three CVRP runs finished; launcher exiting"
        exit 0
    fi

    cpu=$(eval_cpu_pct)
    op_done=$(op_done_count)
    if (( cpu < EVAL_CPU_HARD )) && (( op_done == 3 || cpu < EVAL_CPU_EASY )); then
        confirm=$((confirm + 1))
        log "window open: eval_cpu=${cpu}/100 cores% (hard ${EVAL_CPU_HARD}), op_done=${op_done}/3, confirm ${confirm}/${REQUIRED_CONFIRM}, cvrp_done=${done_n}/3 running=${running}"
    else
        confirm=0
        log "waiting: eval_cpu=${cpu}/100 cores% op_done=${op_done}/3 cvrp_done=${done_n}/3 running=${running}"
    fi

    if (( confirm >= REQUIRED_CONFIRM )); then
        log "window confirmed; launching CVRP runs"
        for rep in 1 2 3; do
            launch_one cvrp_aco "$rep"
        done
        confirm=0
    fi

    sleep "$POLL_SEC"
done