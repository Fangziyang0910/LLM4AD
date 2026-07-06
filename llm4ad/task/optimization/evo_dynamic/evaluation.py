from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.evo_dynamic.dataset import load_split_instances
from llm4ad.task.optimization.evo_dynamic.template import task_description, template_program

__all__ = ["EvoDynamicEvaluation", "sphere_fitness", "ea_step"]


def sphere_fitness(population: np.ndarray, optimum: np.ndarray) -> np.ndarray:
    return -np.sum((population - optimum[np.newaxis, :]) ** 2, axis=1)


def ea_step(
        population: np.ndarray,
        fitness: np.ndarray,
        bounds: np.ndarray,
        sigma: float = 0.3,
) -> np.ndarray:
    pop_size, n_dims = population.shape
    idx = np.random.randint(0, pop_size, (pop_size, 3))
    winners = idx[np.arange(pop_size), np.argmax(fitness[idx], axis=1)]
    offspring = population[winners] + np.random.normal(0.0, sigma, (pop_size, n_dims))
    return np.clip(offspring, bounds[0], bounds[1])


class EvoDynamicEvaluation(Evaluation):
    """Evaluator for EoH's dynamic-EA response-strategy task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            pop_size: int | None = None,
            k_iter: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.pop_size = int(pop_size if pop_size is not None else self.dataset_metadata["pop_size"])
        self.k_iter = int(k_iter if k_iter is not None else self.dataset_metadata["k_iter"])
        self.run_seed_mode = self.dataset_metadata["run_seed_mode"]

    def _run(
            self,
            instance: dict[str, Any],
            respond_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> float | None:
        trajectory = instance["trajectory"]
        n_dims = int(instance["n_dims"])
        bounds = np.array([[-5.0] * n_dims, [5.0] * n_dims])
        lower, upper = bounds[0], bounds[1]

        population = np.random.uniform(lower, upper, (self.pop_size, n_dims))
        fitness = sphere_fitness(population, trajectory[0])
        best_position = population[int(np.argmax(fitness))].copy()
        total_error = 0.0

        for env_idx, optimum in enumerate(trajectory):
            if env_idx > 0:
                new_population = respond_fn(
                    population.copy(),
                    fitness.copy(),
                    best_position.copy(),
                    bounds.copy(),
                )
                new_population = np.asarray(new_population, dtype=float)
                if new_population.shape != population.shape:
                    return None
                population = np.clip(new_population, lower, upper)
                fitness = sphere_fitness(population, optimum)

            for _ in range(self.k_iter):
                population = ea_step(population, fitness, bounds)
                fitness = sphere_fitness(population, optimum)

            best_position = population[int(np.argmax(fitness))].copy()
            total_error += float(np.linalg.norm(best_position - optimum))

        return total_error / len(trajectory)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, respond_fn: Callable) -> float | None:
        try:
            errors_by_group: dict[str, list[float]] = defaultdict(list)
            if self.run_seed_mode == "global":
                np.random.seed(int(self.dataset_metadata["run_seed"]))

            for instance in self._instances:
                if self.run_seed_mode == "local_id":
                    np.random.seed(int(instance["local_id"]))

                error = self._run(instance, respond_fn)
                if error is None:
                    return None
                errors_by_group[instance["group_label"]].append(error)

            mean_errors = [
                float(np.mean(errors))
                for _, errors in sorted(errors_by_group.items())
            ]
            return -float(np.mean(mean_errors))
        except Exception:
            return None
