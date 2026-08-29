"""V9.19 tracked training evaluation: one pass yields fitness and trajectory.

Each class replicates the benchmark evaluator's solving loop call-for-call
(same instances, same rng draw order, same aggregation), so the fitness is
bit-identical to the platform evaluation, and records the per-step choices
on the training instances as the behavior trajectory. The trajectory is the
BehaveSim measurement: construct tasks record the growing choice sequence,
ACO tasks record the incumbent route of every iteration.
"""

from __future__ import annotations

import copy
import multiprocessing
from dataclasses import dataclass

import numpy as np

from llm4ad.base import set_kill_with_parent
from llm4ad.task.optimization.cvrp_aco.evaluation import (
    ACO as CVRPACO,
    CVRPACOEvaluation,
)
from llm4ad.task.optimization.op_aco.evaluation import ACO as OPACO
from llm4ad.task.optimization.op_aco.evaluation import OPACOEvaluation
from llm4ad.task.optimization.online_bin_packing.evaluation import OBPEvaluation
from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

TRAJECTORY_POINTS = {
    "tsp_construct": 12,
    "online_bin_packing": 12,
    "vrptw_construct": 12,
    "op_aco": 5,
    "cvrp_aco": 5,
}


@dataclass(slots=True)
class TrackedResult:
    """Benchmark fitness plus one retained trajectory per training instance."""

    fitness: float
    trajectories: list[list[list[int]]]


def _sample_indices(count: int, max_points: int) -> list[int]:
    if count <= max_points:
        return list(range(count))
    indices = np.rint(np.linspace(0, count - 1, max_points)).astype(int)
    unique = list(dict.fromkeys(int(index) for index in indices))
    if len(unique) != max_points:
        raise AssertionError("trajectory sampling produced duplicate points")
    return unique


def prefix_states(route: list[int], max_points: int) -> list[list[int]]:
    """Evenly retained prefixes of one growing choice sequence."""
    return [route[: index + 1] for index in _sample_indices(len(route), max_points)]


def _trim_op_route(route: np.ndarray, dummy_node: int) -> list[int]:
    result = []
    for value in route.tolist():
        node = int(value)
        if node == dummy_node:
            break
        result.append(node)
    return result


def _trim_cvrp_route(route: np.ndarray) -> list[int]:
    result = [int(value) for value in route.tolist()]
    while len(result) > 1 and result[-1] == 0 and result[-2] == 0:
        result.pop()
    return result


# ---------------------------------------------------------------------------
# construct tasks
# ---------------------------------------------------------------------------


class TrackedTSPEvaluation(TSPEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        dis = np.ones(self.n_instance)
        trajectories: list[list[list[int]]] = []
        n_ins = 0
        for instance, distance_matrix in self._datasets:
            neighbor_matrix = self.generate_neighborhood_matrix(instance)
            destination_node = 0
            current_node = 0
            route = np.zeros(self.problem_size)
            for i in range(1, self.problem_size - 1):
                near_nodes = neighbor_matrix[current_node][1:]
                mask = ~np.isin(near_nodes, route[:i])
                unvisited_near_nodes = near_nodes[mask]
                next_node = callable_func(
                    current_node, destination_node, unvisited_near_nodes, distance_matrix
                )
                if next_node in route:
                    return None
                current_node = next_node
                route[i] = current_node
            mask = ~np.isin(np.arange(self.problem_size), route[: self.problem_size - 1])
            last_node = np.arange(self.problem_size)[mask]
            current_node = last_node[0]
            route[self.problem_size - 1] = current_node
            dis[n_ins] = self.tour_cost(instance, route, self.problem_size)
            # ``route`` already starts with the depot sentinel at index zero.
            # Prepending another depot creates a false self-transition in the
            # BehaveSim trajectory while leaving benchmark fitness unchanged.
            flat_route = [int(node) for node in route]
            trajectories.append(prefix_states(flat_route, TRAJECTORY_POINTS["tsp_construct"]))
            n_ins += 1
            if n_ins == self.n_instance:
                break
        return TrackedResult(float(-np.average(dis)), trajectories)


class TrackedOBPEvaluation(OBPEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        num_bins_per_instance = []
        trajectories: list[list[list[int]]] = []
        for name in self._datasets:
            instance = self._datasets[name]
            capacity = instance["capacity"]
            items = instance["items"]
            bins = np.array([capacity for _ in range(instance["num_items"])])
            choices: list[int] = []
            for item in items:
                valid_bin_indices = self.get_valid_bin_indices(item, bins)
                priorities = callable_func(item, bins[valid_bin_indices])
                best_bin = valid_bin_indices[np.argmax(priorities)]
                bins[best_bin] -= item
                choices.append(int(best_bin))
            num_bins_per_instance.append(int((bins != capacity).sum()))
            trajectories.append(
                prefix_states(choices, TRAJECTORY_POINTS["online_bin_packing"])
            )
        return TrackedResult(
            float(-np.mean(num_bins_per_instance)), trajectories
        )


class TrackedVRPTWEvaluation(VRPTWEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        dis = np.ones(self.n_instance)
        trajectories: list[list[list[int]]] = []
        n_ins = 0
        for instance, distance_matrix, demands, vehicle_capacity, time_service, time_windows in (
            self._datasets
        ):
            route = []
            current_load = 0
            current_node = 0
            current_time = 0
            route.append(current_node)
            unvisited_nodes = set(range(1, self.problem_size + 1))

            def feasible_customers():
                return np.array(
                    [
                        node
                        for node in sorted(unvisited_nodes)
                        if current_load + demands[node] <= vehicle_capacity
                        and max(
                            current_time + distance_matrix[current_node, node],
                            time_windows[node, 0],
                        )
                        <= time_windows[node, 1]
                        and max(
                            current_time + distance_matrix[current_node, node],
                            time_windows[node, 0],
                        )
                        + time_service[node]
                        + distance_matrix[node, 0]
                        <= time_windows[0, 1]
                    ],
                    dtype=int,
                )

            feasible_unvisited_nodes = feasible_customers()
            while unvisited_nodes:
                if len(feasible_unvisited_nodes) == 0:
                    if current_node == 0:
                        return None
                    route.append(0)
                    current_load = 0
                    current_time = 0
                    current_node = 0
                    feasible_unvisited_nodes = feasible_customers()
                    continue

                next_node = callable_func(
                    current_node,
                    0,
                    feasible_unvisited_nodes,
                    vehicle_capacity - current_load,
                    current_time,
                    copy.deepcopy(demands),
                    copy.deepcopy(distance_matrix),
                    copy.deepcopy(time_windows),
                )
                if next_node == 0:
                    if current_node == 0:
                        return None
                    route.append(next_node)
                    current_load = 0
                    current_time = 0
                    current_node = 0
                else:
                    if next_node not in feasible_unvisited_nodes:
                        return None
                    travel_time = distance_matrix[current_node][next_node]
                    current_time += travel_time
                    current_time = max(current_time, time_windows[next_node][0])
                    current_time += time_service[next_node]
                    route.append(next_node)
                    current_load += demands[next_node]
                    unvisited_nodes.remove(next_node)
                    current_node = next_node

                feasible_unvisited_nodes = feasible_customers()

            if route[-1] != 0:
                route.append(0)
            if len(set(route)) != self.problem_size + 1:
                return None

            trajectories.append(
                prefix_states([int(v) for v in route], TRAJECTORY_POINTS["vrptw_construct"])
            )
            dis[n_ins] = self.tour_cost(
                distance_matrix, route, time_service, time_windows
            )
            n_ins += 1
            if n_ins == self.n_instance:
                break
        return TrackedResult(float(-np.average(dis)), trajectories)


# ---------------------------------------------------------------------------
# ACO tasks: tracked pool jobs mirror ACO.run() call-for-call
# ---------------------------------------------------------------------------


def _tracked_op_job(job) -> tuple[float, list[list[int]]]:
    """Top-level worker for spawn ProcessPool (must be picklable)."""
    prizes, distances, prior, maxlen, n_ants, n_iterations, aco_seed, instance_index = job
    rng = np.random.default_rng(aco_seed + instance_index)
    aco = OPACO(prizes, distances, maxlen, prior, n_ants=n_ants, rng=rng)
    max_points = TRAJECTORY_POINTS["op_aco"]
    best_score = -float("inf")
    best_route: list[int] | None = None
    states: list[list[int]] = []
    for _ in range(int(n_iterations)):
        solutions = aco._gen_sol()
        objectives = aco._gen_sol_obj(solutions)
        iteration_best = int(np.argmax(objectives))
        iteration_score = float(objectives[iteration_best])
        if iteration_score > best_score:
            best_score = iteration_score
            best_route = _trim_op_route(solutions[:, iteration_best], aco.n)
        states.append(best_route.copy())
        aco.alltime_best_obj = max(aco.alltime_best_obj, iteration_score)
        aco._update_pheromone(solutions.T, objectives)
    indices = _sample_indices(len(states), max_points)
    return best_score, [states[index] for index in indices]


def _tracked_cvrp_job(job) -> tuple[float, list[list[int]]]:
    """Top-level worker for spawn ProcessPool (must be picklable)."""
    distances, demands, prior, capacity, n_ants, n_iterations, aco_seed, instance_index = job
    rng = np.random.default_rng(aco_seed + instance_index)
    aco = CVRPACO(distances, demands, prior, capacity, n_ants=n_ants, rng=rng)
    max_points = TRAJECTORY_POINTS["cvrp_aco"]
    best_cost = float("inf")
    best_route: list[int] | None = None
    states: list[list[int]] = []
    for _ in range(int(n_iterations)):
        paths = aco._generate_paths()
        costs = aco._path_costs(paths)
        iteration_best = int(np.argmin(costs))
        iteration_cost = float(costs[iteration_best])
        if iteration_cost < best_cost:
            best_cost = iteration_cost
            best_route = _trim_cvrp_route(paths[:, iteration_best])
        states.append(best_route.copy())
        aco.lowest_cost = min(aco.lowest_cost, iteration_cost)
        aco._update_pheromone(paths, costs)
    indices = _sample_indices(len(states), max_points)
    return best_cost, [states[index] for index in indices]


class TrackedOPACOEvaluation(OPACOEvaluation):
    def _evaluate_tracked(self, heuristic) -> TrackedResult:
        jobs = []
        for index, coordinates in enumerate(self._datasets):
            prizes, distances, prior = self._build_prior(coordinates, heuristic)
            jobs.append(
                (
                    prizes,
                    distances,
                    prior,
                    self.max_len,
                    self.n_ants,
                    self.n_iterations,
                    self.aco_seed,
                    index,
                )
            )
        if self.n_workers <= 1 or len(jobs) <= 1:
            results = [_tracked_op_job(job) for job in jobs]
        else:
            workers = min(self.n_workers, len(jobs))
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                processes=workers, initializer=set_kill_with_parent
            ) as pool:
                results = pool.map(_tracked_op_job, jobs)
        objs = [result[0] for result in results]
        trajectories = [result[1] for result in results]
        return TrackedResult(float(np.mean(objs)), trajectories)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return self._evaluate_tracked(callable_func)


class TrackedCVRPACOEvaluation(CVRPACOEvaluation):
    def _evaluate_tracked(self, heuristic) -> TrackedResult:
        jobs = []
        for index, instance in enumerate(self._datasets):
            distances, demands, prior = self._build_prior(instance, heuristic)
            jobs.append(
                (
                    distances,
                    demands,
                    prior,
                    self.capacity,
                    self.n_ants,
                    self.n_iterations,
                    self.aco_seed,
                    index,
                )
            )
        if self.n_workers <= 1 or len(jobs) <= 1:
            results = [_tracked_cvrp_job(job) for job in jobs]
        else:
            workers = min(self.n_workers, len(jobs))
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                processes=workers, initializer=set_kill_with_parent
            ) as pool:
                results = pool.map(_tracked_cvrp_job, jobs)
        costs = [result[0] for result in results]
        trajectories = [result[1] for result in results]
        return TrackedResult(float(-np.mean(costs)), trajectories)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return self._evaluate_tracked(callable_func)


TRACKED_EVALUATIONS = {
    "tsp_construct": TrackedTSPEvaluation,
    "online_bin_packing": TrackedOBPEvaluation,
    "vrptw_construct": TrackedVRPTWEvaluation,
    "op_aco": TrackedOPACOEvaluation,
    "cvrp_aco": TrackedCVRPACOEvaluation,
}


__all__ = [
    "TRAJECTORY_POINTS",
    "TRACKED_EVALUATIONS",
    "TrackedResult",
    "TrackedCVRPACOEvaluation",
    "TrackedOBPEvaluation",
    "TrackedOPACOEvaluation",
    "TrackedTSPEvaluation",
    "TrackedVRPTWEvaluation",
    "prefix_states",
]
