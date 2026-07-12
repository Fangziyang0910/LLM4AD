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

## Progress snapshot at 2026-07-11 18:24 CST

All three tmux sessions remain alive and their logs are current. No traceback,
HTTP 4xx/5xx, timeout, sample-error, or aborted-summary marker was found.

| Method | Progress | Valid / invalid | Best sample | Best score | Cumulative rate | Updated ETA |
|---|---:|---:|---:|---:|---:|---:|
| MCTS-AHD | 484 / 1000 | 463 / 21 | 450 | `-8.880799274182696` | 1.23/min | 2026-07-12 01:20 |
| PathWise | 442 / 500 | 434 / 8 | 196 | `-9.902781548316613` | 1.12/min | 2026-07-11 19:15 |
| TraceAAD | 428 / 1000 | 412 / 16 | 288 | `-8.80554588266062` | 1.09/min | 2026-07-12 03:10 |

The long-window throughput is lower than the first 15-minute estimate. The
updated ETA uses the full elapsed time since launch and is therefore the more
reliable projection. PathWise's best has not improved since sample 196, while
MCTS-AHD improved through sample 450 and TraceAAD's current best was found at
sample 288.

## PathWise first completion and repeats

The first PathWise CVRP run completed at 2026-07-11 19:19 CST:

- run: `LLM4AD/experiments/cvrp_aco/pathwise/20260711_115024`
- status: `finished`, `num_samples=500`, `evaluate_success_program_num=491`, `evaluate_failed_program_num=9`
- best: sample `196`, operator `world_model`, train score `-9.902781548316613`
- search duration: `26958.63s`; `search_aborted=false`; `error_count=0`

After completion, two independent repeats were started in detached tmux sessions:

| Repeat | tmux session | Run directory |
|---|---|---|
| rep2 | `cvrp_aco_pathwise_rep2_20260711_192005` | `LLM4AD/experiments/cvrp_aco/pathwise/20260711_192005` |
| rep3 | `cvrp_aco_pathwise_rep3_20260711_192010` | `LLM4AD/experiments/cvrp_aco/pathwise/20260711_192010` |

The first run's best program was evaluated on the canonical held-out CVRP
splits with 30 ants, 100 ACO iterations, and `aco_seed=1234`:

| Split | Instances | Objective (mean route length) | Score |
|---|---:|---:|---:|
| `test_50` | 64 | `10.095987786938297` | `-10.095987786938297` |
| `test_100` | 64 | `17.388369383129007` | `-17.388369383129007` |

Evaluation artifact: `LLM4AD/experiments/cvrp_aco/pathwise/eval_20260711_192250_rep1/results.json`.
The reusable evaluator is `experiments/cvrp_aco/pathwise/evaluate_best_on_test.py`.
The final three-run result and search curve will be added under
`docs/results/` only after both repeats finish and are evaluated.
