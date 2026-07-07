from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.cvrp_aco.dataset import (
    CAPACITY,
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.cvrp_aco.template import template_program, task_description

__all__ = ["CVRPACOEvaluation"]


class ACO:
    """CVRP ant colony optimizer adapted from the MCTS-AHD benchmark."""

    def __init__(
            self,
            distances,
            demand,
            heuristic,
            capacity,
            n_ants=30,
            decay=0.9,
            alpha=1,
            beta=1,
            rng: np.random.Generator | None = None,
    ):
        self.problem_size = len(distances)
        self.distances = np.asarray(distances, dtype=float)
        self.demand = np.asarray(demand, dtype=float)
        self.heuristic = np.asarray(heuristic, dtype=float)
        self.capacity = float(capacity)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone = np.ones_like(self.distances)
        self.shortest_path = None
        self.lowest_cost = float("inf")
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, n_iterations: int) -> float:
        for _ in range(int(n_iterations)):
            paths = self.gen_path()
            costs = self.gen_path_costs(paths)

            best_idx = int(np.argmin(costs))
            best_cost = float(costs[best_idx])
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            self.update_pheronome(paths, costs)

        return float(self.lowest_cost)

    def update_pheronome(self, paths, costs) -> None:
        self.pheromone = self.pheromone * self.decay
        for ant_idx in range(self.n_ants):
            path = paths[:, ant_idx]
            cost = costs[ant_idx]
            next_nodes = np.roll(path, shift=-1)
            self.pheromone[path[:-1], next_nodes[:-1]] += 1.0 / cost
        self.pheromone[self.pheromone < 1e-10] = 1e-10

    def gen_path_costs(self, paths):
        u = paths.T
        v = np.roll(u, shift=-1, axis=1)
        return np.sum(self.distances[u[:, :-1], v[:, :-1]], axis=1)

    def gen_path(self):
        actions = np.zeros(self.n_ants, dtype=int)
        visit_mask = np.ones((self.n_ants, self.problem_size), dtype=float)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = np.zeros(self.n_ants, dtype=float)
        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)

        paths_list = [actions]
        done = self.check_done(visit_mask, actions)
        while not done:
            actions = self.pick_move(actions, visit_mask, capacity_mask)
            paths_list.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            done = self.check_done(visit_mask, actions)

        return np.stack(paths_list)

    def pick_move(self, prev, visit_mask, capacity_mask):
        pheromone = self.pheromone[prev]
        heuristic = self.heuristic[prev]
        weights = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask * capacity_mask)
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("ACO transition probabilities are invalid.")

        actions = np.empty(self.n_ants, dtype=int)
        for ant_idx in range(self.n_ants):
            probs = weights[ant_idx] / row_sums[ant_idx]
            actions[ant_idx] = self.rng.choice(self.problem_size, p=probs)
        return actions

    def update_visit_mask(self, visit_mask, actions):
        visit_mask[np.arange(self.n_ants), actions] = 0
        visit_mask[:, 0] = 1
        has_unvisited_customer = (visit_mask[:, 1:] != 0).any(axis=1)
        visit_mask[(actions == 0) & has_unvisited_customer, 0] = 0
        return visit_mask

    def update_capacity_mask(self, cur_nodes, used_capacity):
        capacity_mask = np.ones((self.n_ants, self.problem_size), dtype=float)
        used_capacity[cur_nodes == 0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        remaining_capacity = self.capacity - used_capacity
        capacity_mask[self.demand[np.newaxis, :] > remaining_capacity[:, np.newaxis]] = 0
        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions):
        return bool((visit_mask[:, 1:] == 0).all() and (actions == 0).all())


def distance_matrix(node_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(node_positions, dtype=float)
    return np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)


def _call_heuristic(
        heuristic: Callable,
        dist_mat: np.ndarray,
        coordinates: np.ndarray,
        demand: np.ndarray,
        capacity: int,
) -> np.ndarray:
    n_args = len(inspect.getfullargspec(heuristic).args)
    if n_args == 4:
        return heuristic(dist_mat.copy(), coordinates.copy(), demand.copy(), capacity)
    if n_args == 2:
        return heuristic(dist_mat.copy(), demand / capacity)
    return heuristic(dist_mat.copy(), coordinates.copy(), demand.copy(), capacity)


def solve(instance: np.ndarray, heuristic: Callable, n_ants: int, n_iterations: int, rng: np.random.Generator) -> float:
    demand = np.asarray(instance[:, 0], dtype=float)
    coordinates = np.asarray(instance[:, 1:], dtype=float)
    dist_mat = distance_matrix(coordinates)
    dist_mat[np.diag_indices_from(dist_mat)] = 1.0

    heu = np.asarray(_call_heuristic(heuristic, dist_mat, coordinates, demand, CAPACITY), dtype=float)
    if heu.shape != dist_mat.shape or not np.all(np.isfinite(heu)):
        return float("inf")
    heu = heu + 1e-9
    heu[heu < 1e-9] = 1e-9

    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=n_ants, rng=rng)
    return float(aco.run(n_iterations))


def _evaluate_cvrp_aco_instance(heuristic: Callable, payload, context) -> float:
    idx, instance = payload
    rng = np.random.default_rng(None if context["seed"] is None else int(context["seed"]) + idx)
    try:
        return solve(
            instance=instance,
            heuristic=heuristic,
            n_ants=context["n_ants"],
            n_iterations=context["n_iterations"],
            rng=rng,
        )
    except Exception:
        return float("inf")


class CVRPACOEvaluation(Evaluation):
    """Evaluator for CVRP ant colony optimization heuristic matrices."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            n_ants: int = 30,
            n_iterations: int = 100,
            seed: int | None = 1234,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._datasets, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.problem_size = int(self.dataset_metadata["problem_size"])
        self.capacity = int(self.dataset_metadata["capacity"])
        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.seed = seed
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            costs = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(enumerate(self._datasets)),
                instance_eval=_evaluate_cvrp_aco_instance,
                context={
                    "n_ants": self.n_ants,
                    "n_iterations": self.n_iterations,
                    "seed": self.seed,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return -float(np.mean(costs))
        except Exception:
            return None

    def evaluate(self, heuristic: Callable) -> float:
        costs = []
        for idx, instance in enumerate(self._datasets):
            rng = np.random.default_rng(None if self.seed is None else int(self.seed) + idx)
            try:
                costs.append(solve(
                    instance=instance,
                    heuristic=heuristic,
                    n_ants=self.n_ants,
                    n_iterations=self.n_iterations,
                    rng=rng,
                ))
            except Exception:
                costs.append(float("inf"))

        return -float(np.mean(costs))
