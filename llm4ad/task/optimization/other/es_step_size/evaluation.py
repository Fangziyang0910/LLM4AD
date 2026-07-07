from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.es_step_size.dataset import load_split_instances
from llm4ad.task.optimization.other.es_step_size.template import task_description, template_program

__all__ = ["ESStepSizeEvaluation"]


class ESStepSizeEvaluation(Evaluation):
    """Evaluator for EoH's Evolution Strategy step-size adaptation task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            lam: int | None = None,
            max_evals: int | None = None,
            n_runs: int | None = None,
            ema_alpha: float | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.lam = int(lam if lam is not None else self.dataset_metadata["lam"])
        self.max_evals = int(max_evals if max_evals is not None else self.dataset_metadata["max_evals"])
        self.max_generations = (self.max_evals - 1) // self.lam
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.ema_alpha = float(
            ema_alpha if ema_alpha is not None else self.dataset_metadata["ema_alpha"]
        )
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_es(
            self,
            instance: dict[str, Any],
            adapt_fn: Callable[[float, float, float, np.ndarray, int, int, int], float],
    ) -> float:
        func = instance["func"]
        n = int(instance["dim"])
        lo, hi = instance["bounds"]

        x = lo + (hi - lo) * np.random.rand(n)
        f_x = func(x)
        sigma = (hi - lo) / 4.0
        domain_width = hi - lo

        acceptance_rate = 0.2
        best_f = f_x
        n_evals = 1
        generation = 0

        while n_evals < self.max_evals:
            remaining = self.max_evals - n_evals
            lam_this = min(self.lam, remaining)

            offspring = np.clip(x + sigma * np.random.randn(lam_this, n), lo, hi)
            f_offspring = np.array([func(candidate) for candidate in offspring])
            n_evals += lam_this

            n_accepted = int(np.sum(f_offspring < f_x))
            gen_acceptance = n_accepted / lam_this
            acceptance_rate = (
                (1.0 - self.ema_alpha) * acceptance_rate
                + self.ema_alpha * gen_acceptance
            )

            best_idx = int(np.argmin(f_offspring))
            if f_offspring[best_idx] < f_x:
                x = offspring[best_idx]
                f_x = f_offspring[best_idx]
                best_f = min(best_f, f_x)

            new_sigma = float(adapt_fn(
                float(sigma),
                float(acceptance_rate),
                float(f_x),
                f_offspring.copy(),
                n,
                generation,
                self.max_generations,
            ))
            if not np.isfinite(new_sigma):
                raise ValueError("adapt_step_size returned a non-finite value.")
            sigma = float(np.clip(new_sigma, 1e-12, domain_width))
            generation += 1

        return float(best_f)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(
            self,
            adapt_fn: Callable[[float, float, float, np.ndarray, int, int, int], float],
    ) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_es(instance, adapt_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None
