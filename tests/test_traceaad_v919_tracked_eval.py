"""Bit-identity of the tracked evaluations against the benchmark evaluators.

The tracked classes replicate the benchmark solving loops call-for-call, so
the fitness must match exactly (same instances, same rng draw order, same
aggregation) while the trajectory comes out of the same pass.
"""

from __future__ import annotations

import numpy as np

from llm4ad.method.traceaad_v9_19.tracked_eval import (
    TRACKED_EVALUATIONS,
    TrackedCVRPACOEvaluation,
    TrackedOBPEvaluation,
    TrackedOPACOEvaluation,
    TrackedTSPEvaluation,
    TrackedVRPTWEvaluation,
    prefix_states,
)
from llm4ad.task.optimization.cvrp_aco.evaluation import CVRPACOEvaluation
from llm4ad.task.optimization.online_bin_packing.evaluation import OBPEvaluation
from llm4ad.task.optimization.op_aco.evaluation import OPACOEvaluation
from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation


def tsp_nearest(current_node, destination_node, unvisited_nodes, distance_matrix):
    distances = distance_matrix[current_node][unvisited_nodes]
    return int(unvisited_nodes[int(np.argmin(distances))])


def vrptw_nearest(
    current_node, depot, unvisited_nodes, rest_capacity, current_time,
    demands, distance_matrix, time_windows,
):
    distances = distance_matrix[current_node][unvisited_nodes]
    return int(unvisited_nodes[int(np.argmin(distances))])


def obp_best_fit(item, valid_bins):
    return -valid_bins.astype(np.float64)


def op_heuristic(prize, distance, maxlen):
    return prize[:, None] / distance


def cvrp_heuristic(distance_matrix, coordinates, demands, capacity):
    return 1.0 / (distance_matrix + 1.0)


def test_tracked_tsp_matches_benchmark() -> None:
    kwargs = {"n_instance": 3, "problem_size": 8, "seed": 7}
    benchmark = TSPEvaluation(timeout_seconds=30, **kwargs)
    tracked = TrackedTSPEvaluation(timeout_seconds=30, **kwargs)
    result = tracked.evaluate_program("_", tsp_nearest)
    assert result.fitness == benchmark.evaluate_program("_", tsp_nearest)
    assert len(result.trajectories) == 3
    for trajectory in result.trajectories:
        assert len(trajectory) <= 12
        lengths = [len(state) for state in trajectory]
        assert lengths == sorted(lengths)
        assert trajectory[0] == [0]
        assert trajectory[1][0] == 0 and len(trajectory[1]) == 2


def test_tracked_obp_matches_benchmark() -> None:
    kwargs = {"n_instances": 2, "n_items": 60, "capacity": 100, "seed": 11}
    benchmark = OBPEvaluation(timeout_seconds=30, **kwargs)
    tracked = TrackedOBPEvaluation(timeout_seconds=30, **kwargs)
    result = tracked.evaluate_program("_", obp_best_fit)
    assert result.fitness == benchmark.evaluate_program("_", obp_best_fit)
    assert len(result.trajectories) == 2
    for trajectory in result.trajectories:
        assert len(trajectory) <= 12


def test_tracked_vrptw_matches_benchmark() -> None:
    kwargs = {"n_instance": 3, "problem_size": 8, "seed": 5}
    benchmark = VRPTWEvaluation(timeout_seconds=30, **kwargs)
    tracked = TrackedVRPTWEvaluation(timeout_seconds=30, **kwargs)
    result = tracked.evaluate_program("_", vrptw_nearest)
    assert result.fitness == benchmark.evaluate_program("_", vrptw_nearest)
    assert len(result.trajectories) == 3
    for trajectory in result.trajectories:
        assert trajectory[0] == [0]


def test_tracked_op_aco_matches_benchmark_serial_and_parallel() -> None:
    for workers in (1, 2):
        kwargs = {"n_ants": 4, "n_iterations": 5, "aco_seed": 99, "n_workers": workers}
        benchmark = OPACOEvaluation(timeout_seconds=60, split="train", **kwargs)
        tracked = TrackedOPACOEvaluation(timeout_seconds=60, split="train", **kwargs)
        result = tracked.evaluate_program("_", op_heuristic)
        assert result.fitness == benchmark.evaluate_program("_", op_heuristic)
        assert len(result.trajectories) == benchmark.n_instance
        for trajectory in result.trajectories:
            assert len(trajectory) <= 5


def test_tracked_cvrp_aco_matches_benchmark_serial_and_parallel() -> None:
    for workers in (1, 2):
        kwargs = {"n_ants": 4, "n_iterations": 5, "aco_seed": 99, "n_workers": workers}
        benchmark = CVRPACOEvaluation(timeout_seconds=120, split="train", **kwargs)
        tracked = TrackedCVRPACOEvaluation(timeout_seconds=120, split="train", **kwargs)
        result = tracked.evaluate_program("_", cvrp_heuristic)
        assert result.fitness == benchmark.evaluate_program("_", cvrp_heuristic)
        assert len(result.trajectories) == benchmark.n_instance
        for trajectory in result.trajectories:
            assert len(trajectory) <= 5


def test_tracked_invalid_candidates_match_benchmark() -> None:
    # a TSP heuristic that revisits a node is invalid in both evaluations
    def broken(current_node, destination_node, unvisited_nodes, distance_matrix):
        return 0

    kwargs = {"n_instance": 2, "problem_size": 8, "seed": 3}
    benchmark = TSPEvaluation(timeout_seconds=30, **kwargs)
    tracked = TrackedTSPEvaluation(timeout_seconds=30, **kwargs)
    assert tracked.evaluate_program("_", broken) is None
    assert benchmark.evaluate_program("_", broken) is None


def test_tracked_evaluations_cover_all_tasks() -> None:
    assert set(TRACKED_EVALUATIONS) == {
        "tsp_construct",
        "online_bin_packing",
        "vrptw_construct",
        "op_aco",
        "cvrp_aco",
    }


def test_prefix_states_retention() -> None:
    states = prefix_states(list(range(100)), 12)
    assert len(states) == 12
    assert states[0] == [0]
    assert states[-1] == list(range(100))
    assert prefix_states([4, 5], 12) == [[4], [4, 5]]
