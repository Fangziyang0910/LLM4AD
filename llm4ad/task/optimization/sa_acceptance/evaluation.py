from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.sa_acceptance.dataset import load_split_instances
from llm4ad.task.optimization.sa_acceptance.template import task_description, template_program

__all__ = ["SAAcceptanceEvaluation"]


class SAAcceptanceEvaluation(Evaluation):
    """Evaluator for EoH's simulated-annealing acceptance-probability task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            max_iter: int | None = None,
            sigma_ratio: float | None = None,
            t_ratio: float | None = None,
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
        self.max_iter = int(max_iter if max_iter is not None else self.dataset_metadata["max_iter"])
        self.sigma_ratio = float(
            sigma_ratio if sigma_ratio is not None else self.dataset_metadata["sigma_ratio"]
        )
        self.t_ratio = float(t_ratio if t_ratio is not None else self.dataset_metadata["t_ratio"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

        for instance in self._instances:
            instance["T0"] = self._calibrate_t0(instance)

    def _calibrate_t0(self, instance: dict[str, Any], n_samples: int = 200) -> float:
        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]
        sigma = self.sigma_ratio * (hi - lo)
        np.random.seed(0)

        deltas = []
        for _ in range(n_samples):
            x = lo + (hi - lo) * np.random.rand(dim)
            x_new = np.clip(x + np.random.normal(0, sigma, dim), lo, hi)
            delta = func(x_new) - func(x)
            if delta > 0:
                deltas.append(delta)

        mean_delta = float(np.mean(deltas)) if deltas else 1.0
        return mean_delta / np.log(2)

    def _run_sa(
            self,
            instance: dict[str, Any],
            acceptance_fn: Callable[[float, float, int, int], float],
    ) -> float:
        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]
        sigma = self.sigma_ratio * (hi - lo)
        temperature = float(instance["T0"])
        cooling = self.t_ratio ** (1.0 / self.max_iter)

        x = lo + (hi - lo) * np.random.rand(dim)
        current_f = func(x)
        best_f = current_f

        for iteration in range(self.max_iter):
            x_new = np.clip(x + np.random.normal(0, sigma, dim), lo, hi)
            new_f = func(x_new)
            delta = new_f - current_f

            if delta < 0:
                x = x_new
                current_f = new_f
            else:
                probability = float(acceptance_fn(delta, temperature, iteration, self.max_iter))
                if not np.isfinite(probability):
                    raise ValueError("acceptance_probability returned a non-finite value.")
                probability = max(0.0, min(1.0, probability))
                if np.random.rand() < probability:
                    x = x_new
                    current_f = new_f

            if current_f < best_f:
                best_f = current_f
            temperature *= cooling

        return float(best_f)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, acceptance_fn: Callable[[float, float, int, int], float]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_sa(instance, acceptance_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None
