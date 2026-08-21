#!/usr/bin/env bash
# Wait for the remaining V9.14 batch runs (tsp rep2, cvrp rep2/rep3) to finish,
# then run the per-task 3-repeat held-out evaluations. Result rows in the docs
# are written by the operator after inspecting results.json.
set -u
cd /home/fzy/code/LLM4AD
LOG=experiments/_logs/eval_v914_waiter.log
TSPEX=experiments/tsp_construct/traceaad_v9_14/eval_best_20260821_v914_complete/results.json
CVRPEX=experiments/cvrp_aco/traceaad_v9_14/eval_best_20260821_v914_complete/results.json
MAX_WAIT_SEC=14400

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

is_finished() {
  uv run python -c "
import json, sys
try:
    s = json.load(open('$1/logs/summary.json'))
except OSError:
    sys.exit(1)
sys.exit(0 if s.get('status') == 'finished' else 1)
"
}

alive() { pgrep -f "$1" >/dev/null; }

if [ ! -f "$TSPEX" ]; then
  TSP_R2=experiments/tsp_construct/traceaad_v9_14/v9_14_20260821_001824_tsp_construct_rep2
  while true; do
    if is_finished "$TSP_R2"; then break; fi
    if ! alive "v9_14_20260821_001824_tsp_construct_rep2"; then
      log "tsp rep2 process ended without finished summary; skipping TSP eval"
      break
    fi
    if [ "$SECONDS" -ge "$MAX_WAIT_SEC" ]; then log "gave up waiting for tsp rep2"; break; fi
    sleep 60
  done
  if is_finished "$TSP_R2"; then
    log "tsp rep2 finished; launching TSP complete eval (workers 8)"
    uv run python experiments/evaluate_best.py \
      experiments/tsp_construct/traceaad_v9_14/v9_14_20260821_001824_tsp_construct_rep1 \
      experiments/tsp_construct/traceaad_v9_14/v9_14_20260821_001824_tsp_construct_rep2 \
      experiments/tsp_construct/traceaad_v9_14/v9_14_20260821_001824_tsp_construct_rep3 \
      --output-dir experiments/tsp_construct/traceaad_v9_14/eval_best_20260821_v914_complete \
      --workers 8 >> "$LOG" 2>&1
    log "TSP eval exit=$?"
  fi
fi

if [ ! -f "$CVRPEX" ]; then
  CVRP_R2=experiments/cvrp_aco/traceaad_v9_14/v9_14_20260821_001824_cvrp_aco_rep2
  CVRP_R3=experiments/cvrp_aco/traceaad_v9_14/v9_14_20260821_001824_cvrp_aco_rep3
  while true; do
    if is_finished "$CVRP_R2" && is_finished "$CVRP_R3"; then break; fi
    if ! alive "v9_14_20260821_001824_cvrp_aco_rep2" && ! is_finished "$CVRP_R2"; then
      log "cvrp rep2 process ended without finished summary; skipping CVRP eval"
      break
    fi
    if ! alive "v9_14_20260821_001824_cvrp_aco_rep3" && ! is_finished "$CVRP_R3"; then
      log "cvrp rep3 process ended without finished summary; skipping CVRP eval"
      break
    fi
    if [ "$SECONDS" -ge "$MAX_WAIT_SEC" ]; then log "gave up waiting for cvrp rep2/rep3"; break; fi
    sleep 60
  done
  if is_finished "$CVRP_R2" && is_finished "$CVRP_R3"; then
    log "cvrp rep2/rep3 finished; launching CVRP complete eval (workers 16)"
    uv run python experiments/evaluate_best.py \
      experiments/cvrp_aco/traceaad_v9_14/v9_14_20260821_001824_cvrp_aco_rep1 \
      experiments/cvrp_aco/traceaad_v9_14/v9_14_20260821_001824_cvrp_aco_rep2 \
      experiments/cvrp_aco/traceaad_v9_14/v9_14_20260821_001824_cvrp_aco_rep3 \
      --output-dir experiments/cvrp_aco/traceaad_v9_14/eval_best_20260821_v914_complete \
      --workers 16 >> "$LOG" 2>&1
    log "CVRP eval exit=$?"
  fi
fi

log "watcher done"
