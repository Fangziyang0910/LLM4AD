"""BehaveSim trajectory store utilities for V9.19.

The behavior trajectory of an algorithm is recorded inside the tracked
training evaluation (``tracked_eval.py``): one execution on the training
instances yields both the benchmark fitness and the per-instance choice
sequence. This module keeps the distance protocol: state retention counts,
DTW over states with normalized Levenshtein local cost, and the mean over
training instances. The kernel definitions match the offline BehaveSim
metric.
"""

from __future__ import annotations

from typing import Any

import numba
import numpy as np

from llm4ad.task.optimization.cvrp_aco.evaluation import CVRPACOEvaluation
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.task.optimization.op_aco.evaluation import OPACOEvaluation
from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

BEHAVESIM_PROTOCOL_ID = "behavesim_v4_train_trajectory"
PREFIX_TASKS = frozenset({"tsp_construct", "online_bin_packing", "vrptw_construct"})
RETENTION_POINTS = {
    "tsp_construct": 12,
    "op_aco": 5,
    "online_bin_packing": 12,
    "cvrp_aco": 5,
    "vrptw_construct": 12,
}

_TASK_EVALUATION_TYPES = (
    (TSPEvaluation, "tsp_construct"),
    (OPACOEvaluation, "op_aco"),
    (CVRPACOEvaluation, "cvrp_aco"),
    (VRPTWEvaluation, "vrptw_construct"),
    (OBPEvaluation, "online_bin_packing"),
)


def detect_task(evaluation: Any) -> str:
    for evaluation_type, task in _TASK_EVALUATION_TYPES:
        if isinstance(evaluation, evaluation_type):
            return task
    raise ValueError(f"unsupported evaluation type: {type(evaluation).__name__}")


def build_protocol(task: str) -> dict[str, Any]:
    """Metadata of the frozen V9.19 trajectory distance protocol."""
    return {
        "protocol_id": BEHAVESIM_PROTOCOL_ID,
        "trajectory_source": "training evaluation (tracked_eval)",
        "retention_points": RETENTION_POINTS[task],
        "distance": "DTW / min trajectory length, Levenshtein local cost",
        "aggregation": "mean over paired training instances",
        "prefix_optimized": task in PREFIX_TASKS,
    }


# ---------------------------------------------------------------------------
# Distance kernels (identical definitions to the offline BehaveSim metric)
# ---------------------------------------------------------------------------


@numba.njit(cache=True)
def _edit_distance_with_workspace(a, len_a, b, len_b, previous, current):
    for index in range(len_b + 1):
        previous[index] = index
    for i in range(1, len_a + 1):
        current[0] = i
        left_value = a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if left_value == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost
            )
        for index in range(len_b + 1):
            previous[index] = current[index]
    return previous[len_b]


@numba.njit(cache=True)
def _prefix_probe_distance(left_states, left_lengths, right_states, right_lengths):
    left_points = 0
    for index in range(len(left_lengths)):
        if left_lengths[index] > 0:
            left_points += 1
    right_points = 0
    for index in range(len(right_lengths)):
        if right_lengths[index] > 0:
            right_points += 1
    left_final_len = left_lengths[left_points - 1]
    right_final_len = right_lengths[right_points - 1]
    edit_dp = np.empty((left_final_len + 1, right_final_len + 1), dtype=np.int32)
    for i in range(left_final_len + 1):
        edit_dp[i, 0] = i
    for j in range(right_final_len + 1):
        edit_dp[0, j] = j
    left_final = left_states[left_points - 1]
    right_final = right_states[right_points - 1]
    for i in range(1, left_final_len + 1):
        for j in range(1, right_final_len + 1):
            cost = 0 if left_final[i - 1] == right_final[j - 1] else 1
            edit_dp[i, j] = min(
                edit_dp[i - 1, j] + 1,
                edit_dp[i, j - 1] + 1,
                edit_dp[i - 1, j - 1] + cost,
            )
    previous = np.full(right_points + 1, np.inf, dtype=np.float64)
    current = np.full(right_points + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, left_points + 1):
        current[0] = np.inf
        len_a = left_lengths[i - 1]
        for j in range(1, right_points + 1):
            len_b = right_lengths[j - 1]
            denominator = max(len_a, len_b)
            local = 0.0 if denominator == 0 else edit_dp[len_a, len_b] / denominator
            current[j] = local + min(previous[j], current[j - 1], previous[j - 1])
        for j in range(right_points + 1):
            previous[j] = current[j]
    return previous[right_points] / min(left_points, right_points)


@numba.njit(cache=True)
def _generic_probe_distance(left_states, left_lengths, right_states, right_lengths):
    left_points = 0
    for index in range(len(left_lengths)):
        if left_lengths[index] > 0:
            left_points += 1
    right_points = 0
    for index in range(len(right_lengths)):
        if right_lengths[index] > 0:
            right_points += 1
    max_state = max(left_states.shape[1], right_states.shape[1])
    edit_previous = np.empty(max_state + 1, dtype=np.int32)
    edit_current = np.empty(max_state + 1, dtype=np.int32)
    dtw_previous = np.full(right_points + 1, np.inf, dtype=np.float64)
    dtw_current = np.full(right_points + 1, np.inf, dtype=np.float64)
    dtw_previous[0] = 0.0
    for i in range(1, left_points + 1):
        dtw_current[0] = np.inf
        len_a = left_lengths[i - 1]
        for j in range(1, right_points + 1):
            len_b = right_lengths[j - 1]
            raw = _edit_distance_with_workspace(
                left_states[i - 1],
                len_a,
                right_states[j - 1],
                len_b,
                edit_previous,
                edit_current,
            )
            denominator = max(len_a, len_b)
            local = 0.0 if denominator == 0 else raw / denominator
            dtw_current[j] = local + min(
                dtw_previous[j], dtw_current[j - 1], dtw_previous[j - 1]
            )
        for j in range(right_points + 1):
            dtw_previous[j] = dtw_current[j]
    return dtw_previous[right_points] / min(left_points, right_points)


@numba.njit(cache=True)
def _one_vs_many_panel(new_states, new_lengths, old_states, old_lengths, prefix_mode):
    """Serial trajectory distance from one new profile to every stored one.

    The incremental search path calls this once per new node against at most
    a few hundred stored profiles, so a serial loop is fast enough and each
    side counts its own retained states.
    """
    n_old = old_states.shape[0]
    n_probes = old_states.shape[1]
    distances = np.empty(n_old, dtype=np.float32)
    for old_index in range(n_old):
        total = 0.0
        for probe_index in range(n_probes):
            if prefix_mode:
                total += _prefix_probe_distance(
                    new_states[probe_index],
                    new_lengths[probe_index],
                    old_states[old_index, probe_index],
                    old_lengths[old_index, probe_index],
                )
            else:
                total += _generic_probe_distance(
                    new_states[probe_index],
                    new_lengths[probe_index],
                    old_states[old_index, probe_index],
                    old_lengths[old_index, probe_index],
                )
        distances[old_index] = total / n_probes
    return distances


def pack_profiles(
    trajectories: list[list[list[list[int]]]],
) -> tuple[np.ndarray, np.ndarray]:
    n_candidates = len(trajectories)
    n_probes = len(trajectories[0])
    max_points = max(
        len(trajectory) for profile in trajectories for trajectory in profile
    )
    max_state = max(
        len(state)
        for profile in trajectories
        for trajectory in profile
        for state in trajectory
    )
    states = np.full(
        (n_candidates, n_probes, max_points, max_state), -1, dtype=np.int32
    )
    lengths = np.zeros((n_candidates, n_probes, max_points), dtype=np.int32)
    for candidate_index, profile in enumerate(trajectories):
        if len(profile) != n_probes:
            raise ValueError("profile instance counts differ")
        for probe_index, trajectory in enumerate(profile):
            for point_index, state in enumerate(trajectory):
                lengths[candidate_index, probe_index, point_index] = len(state)
                states[candidate_index, probe_index, point_index, : len(state)] = state
    return states, lengths


def trajectory_distances(
    new: list[list[list[int]]],
    old: list[list[list[list[int]]]],
    *,
    prefix_mode: bool,
) -> np.ndarray:
    """BehaveSim distances from one new node's profile to every stored node."""
    if not old:
        return np.zeros(0, dtype=np.float32)
    new_states, new_lengths = pack_profiles([new])
    old_states, old_lengths = pack_profiles(old)
    return _one_vs_many_panel(
        new_states[0], new_lengths[0], old_states, old_lengths, prefix_mode
    )


__all__ = [
    "BEHAVESIM_PROTOCOL_ID",
    "PREFIX_TASKS",
    "RETENTION_POINTS",
    "build_protocol",
    "detect_task",
    "pack_profiles",
    "trajectory_distances",
]
