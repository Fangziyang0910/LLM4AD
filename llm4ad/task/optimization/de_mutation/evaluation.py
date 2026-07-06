from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.de_mutation.dataset import load_split_instances
from llm4ad.task.optimization.de_mutation.template import task_description, template_program

__all__ = ["DEMutationEvaluation"]


class DEMutationEvaluation(Evaluation):
    """Evaluator for EoH's Differential Evolution mutation task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            pop_size: int | None = None,
            max_evals: int | None = None,
            n_runs: int | None = None,
            F: float | None = None,
            CR: float | None = None,
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
        self.max_evals = int(max_evals if max_evals is not None else self.dataset_metadata["max_evals"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.F = float(F if F is not None else self.dataset_metadata["F"])
        self.CR = float(CR if CR is not None else self.dataset_metadata["CR"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_de(
            self,
            instance: dict[str, Any],
            mutation_fn: Callable[..., np.ndarray],
    ) -> float:
        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])

        population = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        fitness = np.array([func(individual) for individual in population])
        n_evals = self.pop_size
        best_idx = int(np.argmin(fitness))

        while n_evals < self.max_evals:
            for current_idx in range(self.pop_size):
                if n_evals >= self.max_evals:
                    break

                mutant = mutation_fn(
                    population.copy(),
                    current_idx,
                    best_idx,
                    fitness.copy(),
                    self.F,
                    bounds,
                )
                mutant = np.asarray(mutant, dtype=float)
                if mutant.shape != (dim,):
                    raise ValueError(f"mutation returned shape {mutant.shape}, expected ({dim},).")
                if not np.all(np.isfinite(mutant)):
                    raise ValueError("mutation returned non-finite values.")

                cross_mask = np.random.rand(dim) < self.CR
                cross_mask[np.random.randint(dim)] = True
                trial = np.where(cross_mask, mutant, population[current_idx])
                trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

                trial_fitness = func(trial)
                n_evals += 1
                if trial_fitness <= fitness[current_idx]:
                    population[current_idx] = trial
                    fitness[current_idx] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = current_idx

        return float(fitness[best_idx])

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, mutation_fn: Callable[..., np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_de(instance, mutation_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None
