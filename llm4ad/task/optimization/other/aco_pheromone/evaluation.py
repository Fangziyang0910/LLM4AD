from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.other.aco_pheromone.dataset import load_split_instances
from llm4ad.task.optimization.other.aco_pheromone.template import task_description, template_program
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

__all__ = ["ACOPheromoneEvaluation"]


class ACOPheromoneEvaluation(Evaluation):
    """Evaluator for EoH's ACO pheromone-update task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            n_ants: int | None = None,
            iter_max: int | None = None,
            alpha: float | None = None,
            beta: float | None = None,
            rho: float | None = None,
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
        self.n_cities = int(self.dataset_metadata["n_cities"])
        self.n_ants = int(n_ants if n_ants is not None else self.dataset_metadata["n_ants"])
        self.iter_max = int(iter_max if iter_max is not None else self.dataset_metadata["iter_max"])
        self.alpha = float(alpha if alpha is not None else self.dataset_metadata["alpha"])
        self.beta = float(beta if beta is not None else self.dataset_metadata["beta"])
        self.rho = float(rho if rho is not None else self.dataset_metadata["rho"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    @staticmethod
    def _tour_cost(tour: np.ndarray, distances: np.ndarray) -> float:
        n = len(tour)
        return float(sum(distances[tour[i], tour[(i + 1) % n]] for i in range(n)))

    def _construct_tour(self, pheromone: np.ndarray, eta: np.ndarray) -> np.ndarray:
        n = pheromone.shape[0]
        start = np.random.randint(n)
        visited = np.zeros(n, dtype=bool)
        visited[start] = True
        tour = [start]

        for _ in range(n - 1):
            current = tour[-1]
            attract = (pheromone[current] ** self.alpha) * (eta[current] ** self.beta)
            attract[visited] = 0.0
            total = attract.sum()
            if total < 1e-300:
                attract = (~visited).astype(float)
                total = attract.sum()
            probabilities = attract / total
            nxt = int(np.random.choice(n, p=probabilities))
            tour.append(nxt)
            visited[nxt] = True

        return np.array(tour, dtype=int)

    def _run_aco(
            self,
            distances: np.ndarray,
            update_fn: Callable[..., np.ndarray],
    ) -> float:
        n = len(distances)
        with np.errstate(divide="ignore", invalid="ignore"):
            eta = np.where(distances > 0, 1.0 / distances, 0.0)
        np.fill_diagonal(eta, 0.0)

        pheromone = np.ones((n, n), dtype=float)
        best_tour = None
        best_cost = np.inf

        for iteration in range(self.iter_max):
            ant_tours = []
            tour_costs = []
            for _ in range(self.n_ants):
                tour = self._construct_tour(pheromone, eta)
                cost = self._tour_cost(tour, distances)
                ant_tours.append(tour)
                tour_costs.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour.copy()

            if best_tour is None:
                raise ValueError("No ant tour was constructed.")

            new_pheromone = update_fn(
                pheromone.copy(),
                ant_tours,
                np.array(tour_costs, dtype=float),
                best_tour.copy(),
                float(best_cost),
                self.rho,
                iteration,
                self.iter_max,
            )
            new_pheromone = np.asarray(new_pheromone, dtype=float)
            if new_pheromone.shape != (n, n):
                raise ValueError(
                    f"update_pheromone returned shape {new_pheromone.shape}, expected ({n}, {n})."
                )
            if not np.all(np.isfinite(new_pheromone)):
                raise ValueError("update_pheromone returned non-finite values.")
            pheromone = np.maximum(new_pheromone, 1e-10)

        return float(best_cost)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, update_fn: Callable[..., np.ndarray]) -> float | None:
        try:
            costs = []
            for instance in self._instances:
                run_costs = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_costs.append(self._run_aco(instance["distances"], update_fn))
                costs.append(float(np.mean(run_costs)))
            return -float(np.mean(costs))
        except Exception:
            return None
