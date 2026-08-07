#!/usr/bin/env bash
# Evaluate each formal CVRP-ACO run's training best on held-out test_200.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
EVAL="uv run python experiments/cvrp_aco/evaluate_best_on_test.py"
WORKERS="${WORKERS:-16}"
TAG="${TAG:-20260804_test200}"

run_method() {
  local method="$1"
  local out="$2"
  shift 2
  echo "=== ${method} -> ${out} ==="
  mkdir -p "$out"
  $EVAL "$@" --output-dir "$out" --splits test_200 --workers "$WORKERS" \
    >"${out}/eval.log" 2>&1
  python3 - <<PY
import json
from pathlib import Path
p = Path("${out}") / "results.json"
d = json.loads(p.read_text())
s = d["results_by_split"]["test_200"]["summary"]
print(f"${method}: mean={s['mean']:.6f} std={s['sample_std']}")
PY
}

run_method mcts_ahd \
  "experiments/cvrp_aco/mcts_ahd/eval_best_${TAG}" \
  experiments/cvrp_aco/mcts_ahd/20260711_115024 \
  experiments/cvrp_aco/mcts_ahd/20260712_021911 \
  experiments/cvrp_aco/mcts_ahd/20260712_021957

run_method pathwise \
  "experiments/cvrp_aco/pathwise/eval_best_${TAG}" \
  experiments/cvrp_aco/pathwise/20260730_1755_cvrp_pw_rep1 \
  experiments/cvrp_aco/pathwise/20260730_1755_cvrp_pw_rep2 \
  experiments/cvrp_aco/pathwise/20260730_1755_cvrp_pw_rep3

run_method eoh \
  "experiments/cvrp_aco/eoh/eval_best_${TAG}" \
  experiments/cvrp_aco/eoh/eoh_paper_20260729_2350_cvrp_eoh_rep1 \
  experiments/cvrp_aco/eoh/eoh_paper_20260729_2350_cvrp_eoh_rep2 \
  experiments/cvrp_aco/eoh/eoh_paper_20260729_2350_cvrp_eoh_rep3

run_method reevo \
  "experiments/cvrp_aco/reevo/eval_best_${TAG}" \
  experiments/cvrp_aco/reevo/20260730_1755_cvrp_reevo_rep1 \
  experiments/cvrp_aco/reevo/20260730_1755_cvrp_reevo_rep2 \
  experiments/cvrp_aco/reevo/20260730_1755_cvrp_reevo_rep3

run_method shinka_evo \
  "experiments/cvrp_aco/shinka_evo/eval_best_${TAG}" \
  experiments/cvrp_aco/shinka_evo/20260730_1755_cvrp_shinka_rep1 \
  experiments/cvrp_aco/shinka_evo/20260730_1755_cvrp_shinka_rep2_retry2 \
  experiments/cvrp_aco/shinka_evo/20260730_1755_cvrp_shinka_rep3_retry2

run_method traceaad_v4 \
  "experiments/cvrp_aco/traceaad_v4/version4/eval_best_${TAG}" \
  experiments/cvrp_aco/traceaad_v4/version4/20260723_204526_cvrp_v4_rep1 \
  experiments/cvrp_aco/traceaad_v4/version4/20260723_204526_cvrp_v4_rep2 \
  experiments/cvrp_aco/traceaad_v4/version4/20260723_204526_cvrp_v4_rep3

run_method traceaad_v5 \
  "experiments/cvrp_aco/traceaad_v5/eval_best_${TAG}" \
  experiments/cvrp_aco/traceaad_v5/version5/20260728_151736_cvrp_v5_rep1 \
  experiments/cvrp_aco/traceaad_v5/version5/20260728_151736_cvrp_v5_rep2 \
  experiments/cvrp_aco/traceaad_v5/version5/20260728_151736_cvrp_v5_rep3

run_method traceaad_v6 \
  "experiments/cvrp_aco/traceaad_v6/eval_best_${TAG}" \
  experiments/cvrp_aco/traceaad_v6/version6/v6_20260802_170400_cvrp_v6_rep1 \
  experiments/cvrp_aco/traceaad_v6/version6/v6_20260802_170400_cvrp_v6_rep2 \
  experiments/cvrp_aco/traceaad_v6/version6/v6_20260802_170400_cvrp_v6_rep3

run_method traceaad_v7 \
  "experiments/cvrp_aco/traceaad_v7/eval_best_${TAG}" \
  experiments/cvrp_aco/traceaad_v7/version7/v7_20260804_001931_cvrp_v7_rep1 \
  experiments/cvrp_aco/traceaad_v7/version7/v7_20260804_001931_cvrp_v7_rep2 \
  experiments/cvrp_aco/traceaad_v7/version7/v7_20260804_001931_cvrp_v7_rep3

echo "ALL DONE"
