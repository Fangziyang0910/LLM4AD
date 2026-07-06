from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bbob_metaheuristic.dataset import load_split_instances
from llm4ad.task.optimization.bbob_metaheuristic.template import task_description, template_program
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

__all__ = ["BBOBMetaheuristicEvaluation"]


class BBOBMetaheuristicEvaluation(Evaluation):
    """Evaluator for EoH's complete black-box metaheuristic design task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            budget: int | None = None,
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
        self.budget = int(budget if budget is not None else self.dataset_metadata["budget"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_one(self, instance: dict[str, Any], solver_callable: Callable, seed: int) -> float:
        np.random.seed(seed)
        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]
        bounds = np.array([np.full(dim, lo), np.full(dim, hi)])

        if isinstance(solver_callable, type):
            solver = solver_callable(func, dim, bounds, self.budget)
            x_best = solver.solve()
        else:
            x_best = solver_callable(func, dim, bounds, self.budget)

        x_best = np.asarray(x_best, dtype=float)
        if x_best.shape != (dim,) or not np.all(np.isfinite(x_best)):
            raise ValueError("metaheuristic returned an invalid solution vector")
        x_best = np.clip(x_best, lo, hi)
        return float(func(x_best))

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, solver_callable: Callable) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    run_bests.append(self._run_one(instance, solver_callable, seed))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None
