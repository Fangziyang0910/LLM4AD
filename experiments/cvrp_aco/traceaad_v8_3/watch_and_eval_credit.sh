#!/usr/bin/env bash
# Wait for V8.3 credit CVRP runs to finish, then run held-out eval.
set -euo pipefail
cd /home/fang/code/LLM4AD/LLM4AD
EVAL_TAG=eval_best_v83_credit_20260807
ROOT=experiments/cvrp_aco/traceaad_v8_3/version8_3
LOG=experiments/cvrp_aco/traceaad_v8_3/${EVAL_TAG}.log

echo "[$(date -Is)] watcher start" | tee -a "$LOG"

while true; do
  done_n=0
  for r in 1 2 3; do
    s="$ROOT/v83_20260806_credit_cvrp_rep$r/logs/summary.json"
    if [[ -f "$s" ]] && python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('status',''))" "$s" | grep -qx finished; then
      done_n=$((done_n+1))
    fi
  done
  echo "[$(date -Is)] cvrp_finished=$done_n/3" | tee -a "$LOG"
  if [[ "$done_n" -eq 3 ]]; then
    break
  fi
  # progress peek
  for r in 1 2 3; do
    p="$ROOT/v83_20260806_credit_cvrp_rep$r/logs/progress.log"
    if [[ -f "$p" ]]; then
      echo "  rep$r $(tail -n 1 "$p")" | tee -a "$LOG"
    fi
  done
  sleep 120
done

echo "[$(date -Is)] launching CVRP held-out" | tee -a "$LOG"
PYTHONUNBUFFERED=1 uv run python -u experiments/cvrp_aco/evaluate_best_on_test.py \
  "$ROOT"/v83_20260806_credit_cvrp_rep{1,2,3} \
  --output-dir "experiments/cvrp_aco/traceaad_v8_3/${EVAL_TAG}" \
  --workers 8 \
  2>&1 | tee -a "$LOG"

echo "[$(date -Is)] CVRP held-out done; marker written" | tee -a "$LOG"
touch "experiments/cvrp_aco/traceaad_v8_3/${EVAL_TAG}.READY"
