#!/usr/bin/env bash
# Wait for V9.9 CVRP rep3 to finish (process exit + 1000 evaluations), then run
# the combined 3-repeat held-out evaluation with 32 workers.
set -u
cd /home/fang/code/LLM4AD/LLM4AD
REP3=experiments/cvrp_aco/traceaad_v9_9/v9_9_20260816_154200_cvrp_aco_rep3
LOG=experiments/cvrp_aco/traceaad_v9_9/eval_best_20260817_v99_complete_run.log
MAX_WAIT_SEC=10800
waited=0
while true; do
  if ! pgrep -f "v9_9_20260816_154200_cvrp_aco_rep3" >/dev/null; then
    n=$(wc -l < "$REP3/evaluations.csv")
    if [ "$n" -ge 1001 ]; then
      echo "[$(date '+%F %T')] rep3 finished ($n csv lines); launching combined eval" | tee -a "$LOG"
      uv run python experiments/evaluate_best.py \
        experiments/cvrp_aco/traceaad_v9_9/v9_9_20260816_154200_cvrp_aco_rep1 \
        experiments/cvrp_aco/traceaad_v9_9/v9_9_20260816_154200_cvrp_aco_rep2 \
        "$REP3" \
        --output-dir experiments/cvrp_aco/traceaad_v9_9/eval_best_20260817_v99_complete \
        --workers 32 2>&1 | tee -a "$LOG"
    else
      echo "[$(date '+%F %T')] rep3 process ended but evaluations.csv has only $n lines; NOT launching eval" | tee -a "$LOG"
    fi
    break
  fi
  sleep 60
  waited=$((waited + 60))
  if [ "$waited" -ge "$MAX_WAIT_SEC" ]; then
    echo "[$(date '+%F %T')] gave up after ${MAX_WAIT_SEC}s; rep3 still running" | tee -a "$LOG"
    break
  fi
done
