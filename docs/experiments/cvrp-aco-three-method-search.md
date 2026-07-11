# CVRP-ACO three-method search

Three concurrent searches use the same Qwen3.6-27B endpoint and the method
configurations previously used for `tsp_construct`.

| Method | Search budget | Evaluators | Method configuration source |
|---|---:|---:|---|
| MCTS-AHD | 1000 | 4 | `experiments/tsp_construct/mcts_ahd/run_experiment.py` |
| PathWise | 500 | 4 | `experiments/tsp_construct/pathwise/run_experiment.py` |
| TraceAAD | 1000 | 1 | `experiments/tsp_construct/traceaad/run_experiment.py` |

All runs evaluate `CVRPACOEvaluation(split="train")` with 10 fixed CVRP50
instances, 30 ants, 100 ACO iterations, capacity 50, task timeout 120 seconds,
and ACO seed 1234. The three thin runners reuse the TSP method runners and only
override the task and output directory, preventing method-configuration drift.

Entrypoints:

- `LLM4AD/experiments/cvrp_aco/mcts_ahd/run_experiment.py`
- `LLM4AD/experiments/cvrp_aco/pathwise/run_experiment.py`
- `LLM4AD/experiments/cvrp_aco/traceaad/run_experiment.py`

Run directories and the initial health snapshot are appended after launch.

## Launch and initial health snapshot

Launched at 2026-07-11 11:50 CST in detached tmux sessions:

| Method | tmux session | Run directory |
|---|---|---|
| MCTS-AHD | `cvrp_aco_mcts_20260711_115008` | `LLM4AD/experiments/cvrp_aco/mcts_ahd/20260711_115024` |
| PathWise | `cvrp_aco_pathwise_20260711_115008` | `LLM4AD/experiments/cvrp_aco/pathwise/20260711_115024` |
| TraceAAD | `cvrp_aco_traceaad_20260711_115008` | `LLM4AD/experiments/cvrp_aco/traceaad/20260711_115024` |

The remote endpoint reported model `qwen3.6-27b-awq`. All three run configs
contain the intended task and method budgets. The first health check observed:

- MCTS-AHD: 2 evaluated samples, best score `-20.09873084602281`;
- PathWise: 3 evaluated samples, best score `-18.688237314472385`;
- TraceAAD: at least one valid evaluated sample, best score
  `-18.688237314472385`; one later generated candidate was rejected with
  `score=None`, after which the run continued issuing successful HTTP requests.

No traceback, HTTP 4xx/5xx, or actual aborted summary was present. The literal
`_search_aborted: False` configuration line is not an error signal.

## ETA snapshot at 2026-07-11 12:05 CST

After about 15 minutes of wall time:

| Method | Progress | Valid | Best score | Mean rate | Estimated remaining |
|---|---:|---:|---:|---:|---:|
| MCTS-AHD | 23 / 1000 | 23 | `-9.543906976338437` | 1.52 samples/min | 10-12 hours |
| PathWise | 31 / 500 | 29 | `-12.14452070545429` | 2.05 samples/min | 3.5-4.5 hours |
| TraceAAD | 21 / 1000 | 19 | `-13.835640604793852` | 1.39 samples/min | 11-13 hours |

The corresponding approximate completion windows are 15:30-16:30 CST for
PathWise, 22:30-00:30 for MCTS-AHD, and 23:00-01:00 for TraceAAD. These are
early-run estimates: PathWise and TraceAAD use extra policy/reflection calls,
and invalid generated candidates consume sample orders with very short failed
evaluations, so later throughput can move in either direction. Overall, all
three are expected to finish around midnight, with a practical uncertainty of
roughly one hour.
