# CVRP-ACO task integration

## Mechanism contract

The native task is `llm4ad.task.optimization.cvrp_aco.CVRPACOEvaluation`.
Candidate programs implement:

```python
heuristics(distance_matrix, coordinates, demands, capacity) -> edge_prior
```

`edge_prior` is an `n x n` matrix combined with ACO pheromone values. The ACO
mechanism matches the shared ReEvo, MCTS-AHD, HSEvo, and PathWise reference
implementation:

- 30 ants, 100 iterations;
- pheromone decay 0.9;
- alpha = beta = 1;
- capacity-aware customer masks and revisitable depot;
- negative/zero heuristic entries clipped to `1e-9`;
- score is negative mean best route length because LLM4AD maximizes fitness.

## Instance protocol

All instances have a depot at `[0.5, 0.5]`, customer coordinates sampled
uniformly from `[0, 1]^2`, integer demands sampled uniformly from 1 through 9,
and vehicle capacity 50.

| Split | Role | Customers | Instances | Seed |
|---|---|---:|---:|---:|
| `train` | search | 50 | 10 | 1234 |
| `val_20`, `val_50`, `val_100` | validation | 20/50/100 | 64 | 1234 continuous stream |
| `test_50`, `test_100` | canonical test | 50/100 | 64 | 1234 continuous stream |
| `test_20` | compatibility test | 20 | 64 | 3200 |
| `paper_test_50`, `paper_test_100` | PathWise-size report | 50/100 | 250 | 4500/4100 |

The canonical 10/64-instance splits replay the original `RandomState(1234)`
stream in its published generation order: train, val20/50/100, then
test50/100. They are byte-for-byte equal after loading to the corresponding
MCTS-AHD reference arrays; the shared train array also matches PathWise. The
reference directory contains a legacy `test20` array that is not emitted by its
current generator and would alter the test50/100 RNG stream if inserted. The
LLM4AD `test_20` compatibility split therefore uses its own declared seed and
is not part of the paper comparison protocol.
PathWise publishes the 250-instance test count but its released repository does
not expose the generation seed; the `paper_test_*` splits therefore match its
distribution, size, and count, but are explicitly LLM4AD-owned fixed instances
rather than claims of byte-identical paper data.

## Evidence

- Dedicated task tests: 14 passed.
- Default inverse-distance heuristic on the full training protocol:
  score `-18.688237314472385`, elapsed `14.82s` on the current host.
- Invalid shapes and non-finite matrices return evaluation failure.
- ACO sampling is deterministic for a fixed `aco_seed`, so all methods receive
  the same stochastic evaluation stream.

For formal comparisons, every method must use the same task split, `aco_seed`,
`n_ants`, and `n_iterations`. Search results should report three independent
method runs; final test tables should report route length per split and may add
relative gaps once a common solver/reference baseline is fixed.
