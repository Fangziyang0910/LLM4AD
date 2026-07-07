from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.mobbob_metaheuristic.dataset import load_split_instances
from llm4ad.task.optimization.other.mobbob_metaheuristic.template import task_description, template_program

__all__ = ["MoBBOBMetaheuristicEvaluation", "pareto_front_2d", "hypervolume_2d"]


def pareto_front_2d(objectives: np.ndarray) -> np.ndarray:
    if len(objectives) == 0:
        return np.empty((0, 2))
    points = np.array(objectives, dtype=float)
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]
    pareto = []
    min_f2 = np.inf
    for point in points:
        if point[1] < min_f2:
            pareto.append(point)
            min_f2 = point[1]
    return np.array(pareto)


def hypervolume_2d(objectives: np.ndarray, ref_point: np.ndarray) -> float:
    front = pareto_front_2d(objectives)
    if len(front) == 0:
        return 0.0
    front = front[np.all(front < ref_point, axis=1)]
    if len(front) == 0:
        return 0.0
    hv = 0.0
    for i, point in enumerate(front):
        next_f1 = front[i + 1, 0] if i + 1 < len(front) else ref_point[0]
        hv += (next_f1 - point[0]) * (ref_point[1] - point[1])
    return float(hv)


class MoBBOBMetaheuristicEvaluation(Evaluation):
    """Evaluator for EoH's multi-objective black-box metaheuristic task."""

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
        self.dim = int(self.dataset_metadata["dim"])
        self.budget = int(budget if budget is not None else self.dataset_metadata["budget"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_one(self, instance: dict[str, Any], solver_callable: Callable, seed: int) -> float:
        np.random.seed(seed)
        func = instance["func"]
        lower, upper = instance["bounds"]
        bounds = np.array([lower, upper])
        ref_point = instance["ref_point"]
        n_obj = int(instance["n_obj"])

        if isinstance(solver_callable, type):
            solver = solver_callable(func, self.dim, bounds, self.budget, n_obj)
            front_x = solver.solve()
        else:
            front_x = solver_callable(func, self.dim, bounds, self.budget, n_obj)

        front_x = np.asarray(front_x, dtype=float).reshape(-1, self.dim)
        if len(front_x) == 0 or not np.all(np.isfinite(front_x)):
            raise ValueError("multi-objective metaheuristic returned an invalid front")
        front_x = np.clip(front_x, lower, upper)
        front_f = np.array([func(x) for x in front_x])
        return hypervolume_2d(front_f, ref_point)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, solver_callable: Callable) -> float | None:
        try:
            hypervolumes = []
            for instance in self._instances:
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    hypervolumes.append(self._run_one(instance, solver_callable, seed))
            return float(np.mean(hypervolumes))
        except Exception:
            return None
