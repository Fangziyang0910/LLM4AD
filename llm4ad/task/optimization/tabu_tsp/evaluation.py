from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.tabu_tsp.dataset import load_split_instances
from llm4ad.task.optimization.tabu_tsp.template import task_description, template_program

__all__ = ["TabuTSPEvaluation", "nearest_neighbour", "tour_cost"]


def tour_cost(tour: list[int], distances: np.ndarray) -> float:
    return float(sum(distances[tour[i], tour[i + 1]] for i in range(len(tour) - 1)))


def nearest_neighbour(distances: np.ndarray, start: int = 0) -> list[int]:
    n_nodes = len(distances)
    visited = [False] * n_nodes
    visited[start] = True
    tour = [start]
    for _ in range(n_nodes - 1):
        current = tour[-1]
        next_node = min(
            (idx for idx in range(n_nodes) if not visited[idx]),
            key=lambda idx: distances[current, idx],
        )
        tour.append(next_node)
        visited[next_node] = True
    tour.append(start)
    return tour


def move_pairs_for(n_nodes: int) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(1, n_nodes - 1)
        for j in range(i + 1, n_nodes)
    ]


class TabuTSPEvaluation(Evaluation):
    """Evaluator for EoH's Tabu Search move-scoring task on TSP."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            n_iter: int | None = None,
            tabu_tenure: int | None = None,
            n_runs: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_iter = int(n_iter if n_iter is not None else self.dataset_metadata["n_iter"])
        self.tabu_tenure = int(
            tabu_tenure if tabu_tenure is not None else self.dataset_metadata["tabu_tenure"]
        )
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self._move_pairs = {
            int(instance["n_nodes"]): move_pairs_for(int(instance["n_nodes"]))
            for instance in self._instances
        }

    def _run_tabu(
            self,
            distances: np.ndarray,
            score_fn: Callable[[np.ndarray, np.ndarray, float, float, np.ndarray, int, int], np.ndarray],
            seed: int,
    ) -> float:
        rng = np.random.RandomState(seed)
        n_nodes = len(distances)
        start = int(rng.randint(n_nodes))
        tour = nearest_neighbour(distances, start)
        current_cost = tour_cost(tour, distances)
        best_cost = current_cost

        tabu_expiry: dict[tuple[int, int], int] = {}
        move_pairs = self._move_pairs[n_nodes]
        n_moves = len(move_pairs)

        for iteration in range(self.n_iter):
            delta_costs = np.empty(n_moves, dtype=float)
            is_tabu_mask = np.zeros(n_moves, dtype=bool)
            tabu_ages = np.zeros(n_moves, dtype=int)
            move_keys = []

            for move_idx, (i, j) in enumerate(move_pairs):
                delta = (
                    distances[tour[i - 1], tour[j]]
                    + distances[tour[i], tour[j + 1]]
                    - distances[tour[i - 1], tour[i]]
                    - distances[tour[j], tour[j + 1]]
                )
                delta_costs[move_idx] = delta

                u, v = tour[i - 1], tour[j]
                key = (min(u, v), max(u, v))
                move_keys.append(key)
                expiry = tabu_expiry.get(key, 0)
                if iteration < expiry:
                    is_tabu_mask[move_idx] = True
                    tabu_ages[move_idx] = iteration - (expiry - self.tabu_tenure)

            try:
                scores = score_fn(
                    delta_costs,
                    is_tabu_mask,
                    float(best_cost),
                    float(current_cost),
                    tabu_ages,
                    int(iteration),
                    int(self.n_iter),
                )
                scores = np.asarray(scores, dtype=float)
            except Exception:
                scores = np.full(n_moves, -np.inf)

            finite_mask = np.isfinite(scores)
            if not finite_mask.any():
                continue

            best_move_idx = int(np.argmax(np.where(finite_mask, scores, -np.inf)))
            i, j = move_pairs[best_move_idx]
            tour[i:j + 1] = tour[i:j + 1][::-1]
            current_cost += delta_costs[best_move_idx]
            tabu_expiry[move_keys[best_move_idx]] = iteration + self.tabu_tenure

            if current_cost < best_cost:
                best_cost = current_cost

        return float(best_cost)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, score_fn: Callable) -> float | None:
        try:
            costs_by_size: dict[int, list[float]] = defaultdict(list)
            for instance in self._instances:
                n_nodes = int(instance["n_nodes"])
                distances = instance["distances"]
                for seed in range(self.n_runs):
                    costs_by_size[n_nodes].append(self._run_tabu(distances, score_fn, seed))

            mean_per_node = [
                float(np.mean(costs)) / n_nodes
                for n_nodes, costs in sorted(costs_by_size.items())
            ]
            return -float(np.mean(mean_per_node))
        except Exception:
            return None
