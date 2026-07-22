#!/usr/bin/env bash
# Launch 9 Knapsack Construct runs: 3 methods × 3 repeats across 3 LLM sources.
# Usage: bash experiments/knapsack_construct/launch_nine_tmux.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

BATCH_TS="${BATCH_TS:-$(date +%Y%m%d_%H%M%S)}"
NO_PROXY_ALL="183.36.243.124,222.201.145.8,localhost,127.0.0.1,::1"

# rep -> GPU source
declare -A BASE_URL MODEL API_KEY SOURCE_NAME
BASE_URL[1]="http://183.36.243.124:9000/v1"
MODEL[1]="/home/fzy/models/Qwen3.6-27B-NVFP4"
API_KEY[1]="${ZHONG_API_KEY:?set ZHONG_API_KEY to zhong api key}"
SOURCE_NAME[1]="zhong"

BASE_URL[2]="http://222.201.145.8:8080/v1"
MODEL[2]="qwen3.6-27b-awq"
API_KEY[2]="EMPTY"
SOURCE_NAME[2]="server1"

BASE_URL[3]="http://127.0.0.1:8001/v1"
MODEL[3]="Qwen3.6-27B"
API_KEY[3]="EMPTY"
SOURCE_NAME[3]="fang"

METHODS=(mcts_ahd pathwise traceaad)

LAUNCH_LOG="$ROOT/experiments/knapsack_construct/launch_${BATCH_TS}.log"
mkdir -p "$ROOT/experiments/knapsack_construct"
{
  echo "batch_ts=$BATCH_TS"
  echo "train_seed=2024 (from generated_data_config)"
  echo "task=knapsack_construct n_instance=32 n_items=50 capacity=100"
} | tee "$LAUNCH_LOG"

start_one() {
  local method="$1"
  local rep="$2"
  local session="kp_${method}_rep${rep}"
  local run_ts="${BATCH_TS}_kp_rep${rep}"
  local script="$ROOT/experiments/knapsack_construct/${method}/run_experiment.py"
  local cmd

  if tmux has-session -t "$session" 2>/dev/null; then
    echo "SKIP existing session $session" | tee -a "$LAUNCH_LOG"
    return
  fi

  cmd="cd '$ROOT' && \
export NO_PROXY='$NO_PROXY_ALL' no_proxy='$NO_PROXY_ALL' \
LLM_BASE_URL='${BASE_URL[$rep]}' LLM_MODEL='${MODEL[$rep]}' LLM_API_KEY='${API_KEY[$rep]}' \
RUN_TIMESTAMP='$run_ts' && \
echo starting method=$method rep=$rep source=${SOURCE_NAME[$rep]} run_ts=$run_ts && \
uv run python '$script'; echo EXIT:\$?; exec bash"

  tmux new-session -d -s "$session" "$cmd"
  echo "STARTED $session method=$method rep=$rep source=${SOURCE_NAME[$rep]} model=${MODEL[$rep]} run_ts=$run_ts" \
    | tee -a "$LAUNCH_LOG"
}

for rep in 1 2 3; do
  for method in "${METHODS[@]}"; do
    start_one "$method" "$rep"
    sleep 3
  done
done

echo "---- tmux sessions ----" | tee -a "$LAUNCH_LOG"
tmux ls | tee -a "$LAUNCH_LOG" || true
echo "launch_log=$LAUNCH_LOG"
