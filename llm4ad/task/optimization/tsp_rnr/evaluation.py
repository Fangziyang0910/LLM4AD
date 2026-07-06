from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.tsp_rnr.dataset import load_split_instances
from llm4ad.task.optimization.tsp_rnr.template import task_description, template_program

__all__ = ["TSPRnrEvaluation"]


def tour_cost(tour: list[int], dist: np.ndarray) -> float:
    return float(sum(dist[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def nearest_neighbour(dist: np.ndarray, start: int = 0) -> list[int]:
    n = len(dist)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    for _ in range(n - 1):
        current = tour[-1]
        next_node = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: dist[current, j],
        )
        tour.append(next_node)
        visited[next_node] = True
    tour.append(start)
    return tour


def two_opt(tour: list[int], dist: np.ndarray) -> list[int]:
    tour = list(tour)
    n = len(tour) - 1
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                before = dist[tour[i - 1], tour[i]] + dist[tour[j], tour[j + 1]]
                after = dist[tour[i - 1], tour[j]] + dist[tour[i], tour[j + 1]]
                if after < before - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def cheapest_insertion(partial_tour: list[int], removed_nodes: list[int], dist: np.ndarray) -> list[int]:
    tour = list(partial_tour)
    for node in removed_nodes:
        best_delta = np.inf
        best_pos = 1
        n = len(tour)
        for i in range(n):
            j = (i + 1) % n
            delta = dist[tour[i], node] + dist[node, tour[j]] - dist[tour[i], tour[j]]
            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1
        tour.insert(best_pos, int(node))
    return tour


class TSPRnrEvaluation(Evaluation):
    """Evaluator for EoH's TSP ruin-and-recreate destroy-operator task."""

    def __init__(
            self,
            timeout_seconds=40,
            split: str = DEFAULT_SPLIT,
            n_destroy: int | None = None,
            iter_max: int | None = None,
            time_max: float | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_nodes = int(self.dataset_metadata["n_nodes"])
        self.n_destroy = int(n_destroy if n_destroy is not None else self.dataset_metadata["n_destroy"])
        self.iter_max = int(iter_max if iter_max is not None else self.dataset_metadata["iter_max"])
        self.time_max = float(time_max if time_max is not None else self.dataset_metadata["time_max"])
        self.run_seed = int(self.dataset_metadata["run_seed"])

    def _rnr(
            self,
            dist: np.ndarray,
            destroy_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
    ) -> float:
        n = len(dist)
        tour = nearest_neighbour(dist)
        tour = two_opt(tour, dist)
        best_cost = tour_cost(tour, dist)
        best_tour = tour[:]

        import time

        t_end = time.time() + self.time_max
        for _ in range(self.iter_max):
            if time.time() > t_end:
                break

            open_tour = np.array(best_tour[:-1], dtype=int)
            nodes_to_remove = destroy_fn(open_tour.copy(), dist.copy(), self.n_destroy)
            nodes_to_remove = np.asarray(nodes_to_remove, dtype=int).flatten()
            if len(nodes_to_remove) < self.n_destroy:
                continue
            nodes_to_remove = nodes_to_remove[:self.n_destroy]
            if not all(0 <= v < n for v in nodes_to_remove):
                continue

            removed_set = set(nodes_to_remove.tolist())
            partial = [v for v in open_tour if v not in removed_set]
            if len(partial) < 2:
                continue
            new_open = cheapest_insertion(partial, nodes_to_remove.tolist(), dist)
            new_tour = new_open + [new_open[0]]
            new_tour = two_opt(new_tour, dist)
            cost = tour_cost(new_tour, dist)
            if cost < best_cost:
                best_cost = cost
                best_tour = new_tour[:]

        return best_cost

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, destroy_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]) -> float | None:
        try:
            costs = []
            for instance in self._instances:
                np.random.seed(self.run_seed + int(instance["instance_id"]))
                costs.append(self._rnr(instance["distance_matrix"], destroy_fn))
            return -float(np.mean(costs))
        except Exception:
            return None
