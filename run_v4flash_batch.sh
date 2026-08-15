#!/usr/bin/env bash
# V4Flash 20-experiment scheduler.
# Methods run serially (one method fully finishes before the next starts);
# the 4 tasks of each method run in parallel, each in its own tmux session.
#
# Usage: run from repo root via tmux:
#   tmux new-session -d -s scheduler "bash run_v4flash_batch.sh"
set -u

REPO=/home/fzy/code/LLM4AD
BASE_URL="https://opencode.ai/zen/go/v1"
MODEL="deepseek-v4-flash"
BATCH="v4flash_$(date +%Y%m%d_%H%M%S)"
SEED=1   # round 2 (round 1 used seed=0)

METHODS=(mcts_ahd pathwise eoh reevo calm)
TASKS=(tsp_construct cvrp_aco op_aco online_bin_packing)
SHORT=(tsp cvrp op obp)
declare -A SHORTMAP
for i in "${!TASKS[@]}"; do SHORTMAP[${TASKS[$i]}]="${SHORT[$i]}"; done

for method in "${METHODS[@]}"; do
  echo "== [$(date '+%F %T')] starting method: $method ==" | tee -a /tmp/v4flash_scheduler.log
  SESSIONS=()
  for task in "${TASKS[@]}"; do
    short=${SHORTMAP[$task]}
    run_name="${BATCH}_${short}_${method}_rep1"
    session="${BATCH}_${method}_${short}"
    if [ "$method" = "eoh" ]; then
      budget_arg="--budget 1000"
    else
      budget_arg="--max-sample-nums 1000"
    fi
    cmd="uv run python -m experiments.runners.${method}.run --task ${task} --base-url ${BASE_URL} --model ${MODEL} --seed ${SEED} --repeat 1 --run-name ${run_name} ${budget_arg}"
    tmux new-session -d -s "$session" -c "$REPO" "$cmd"
    echo "  launched $session : $cmd" | tee -a /tmp/v4flash_scheduler.log
    SESSIONS+=("$session")
  done

  # Wait until every session of this method has finished.
  while :; do
    alive=0
    for s in "${SESSIONS[@]}"; do
      if tmux has-session -t "=$s" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    if [ "$alive" -eq 0 ]; then break; fi
    echo "  [$(date '+%F %T')] $method: $alive session(s) running" | tee -a /tmp/v4flash_scheduler.log
    sleep 120
  done
  echo "== [$(date '+%F %T')] method $method finished ==" | tee -a /tmp/v4flash_scheduler.log
done

echo "ALL 20 EXPERIMENTS DONE at $(date '+%F %T')" | tee -a /tmp/v4flash_scheduler.log
