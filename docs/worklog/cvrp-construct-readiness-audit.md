# CVRP construct task readiness audit

`LLM4AD/llm4ad/task/optimization/cvrp_construct` can execute legal constructive
heuristics, but it is not yet at the same experiment-readiness level as the
current `tsp_construct` workflow.

## Verified

- The task exposes the standard `Evaluation` contract and a six-argument
  `select_next_node` template, so integrated methods can use it.
- The generated-task registry defines deterministic train/eval splits:
  16 instances, 50 customers plus one depot, capacity 40, seeds 2024/2025.
- The bundled heuristic completed a live smoke test. It scored
  `-15.038005452835502` on train and `-14.907857906186855` on eval; the first
  train route visited all 50 customers and used 7 depot visits.
- Train and eval instances differ.

## Gaps before a formal run

- A heuristic that repeatedly returns depot `0` makes no construction progress
  and consumes the full evaluator timeout. The 1-second smoke test returned
  `None` only after 1.016 seconds. There is no route step/progress guard.
- Returned nodes are not explicitly validated against the feasible unvisited
  set before indexing/removal. Invalid generated programs fail indirectly.
- There are no dedicated CVRP task tests for feasibility, capacity, score
  reproducibility, invalid outputs, or timeout behavior.
- There is no formal `experiments/cvrp_construct/<method>/run_experiment.py`
  surface or held-out multi-size evaluator comparable to TSP.
- The current generated protocol has only 16 synthetic train and 16 synthetic
  eval instances. It does not define a final benchmark set, best-known values,
  gaps, or a capacity policy for larger CVRP sizes.
- The MCTS-AHD and PathWise papers evaluate CVRP under ACO, not this
  step-by-step `cvrp_construct` task. Therefore this module is a platform task,
  not a drop-in reproduction of their published CVRP protocol.

## Readiness decision

Use it only for a small smoke search after adding evaluator safeguards and task
tests. Before a full repeated comparison, add a timestamped runner, held-out
evaluation script, explicit multi-size capacity/demand protocol, and a result
metric (raw distance versus gap to a solver/reference).

The separate paper-aligned ACO task was subsequently added under
`llm4ad.task.optimization.cvrp_aco`; the limitations above still apply to the
step-by-step `cvrp_construct` task itself.
