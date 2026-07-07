from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pymoo.core.crossover import Crossover

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.nsga2_pymoo.dataset import load_split_instances
from llm4ad.task.optimization.other.nsga2_pymoo.template import task_description, template_program

__all__ = ["NSGA2PymooEvaluation"]


class CrossoverAdapter(Crossover):
    def __init__(self, crossover_fn: Callable[[np.ndarray, np.ndarray], tuple]):
        super().__init__(n_parents=2, n_offsprings=2)
        self.crossover_fn = crossover_fn

    def _do(self, problem, X, **kwargs):
        _, n_matings, _ = X.shape
        Y = np.zeros_like(X)
        for i in range(n_matings):
            c1, c2 = self.crossover_fn(X[0, i].copy(), X[1, i].copy())
            c1 = np.asarray(c1, dtype=float)
            c2 = np.asarray(c2, dtype=float)
            if c1.shape != X[0, i].shape or c2.shape != X[1, i].shape:
                raise ValueError("crossover returned invalid offspring shape")
            if not (np.all(np.isfinite(c1)) and np.all(np.isfinite(c2))):
                raise ValueError("crossover returned non-finite offspring")
            Y[0, i] = np.clip(c1, problem.xl, problem.xu)
            Y[1, i] = np.clip(c2, problem.xl, problem.xu)
        return Y


class NSGA2PymooEvaluation(Evaluation):
    """Evaluator for EoH's pymoo-backed NSGA-II crossover task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            pop_size: int | None = None,
            n_gen: int | None = None,
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
        self.pop_size = int(pop_size if pop_size is not None else self.dataset_metadata["pop_size"])
        self.n_gen = int(n_gen if n_gen is not None else self.dataset_metadata["n_gen"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_nsga2(
            self,
            instance: dict[str, Any],
            crossover_fn: Callable[[np.ndarray, np.ndarray], tuple],
            seed: int,
    ) -> float:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.indicators.hv import HV
        from pymoo.operators.mutation.pm import PM
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
        from pymoo.termination import get_termination

        problem = get_problem(instance["name"])
        algorithm = NSGA2(
            pop_size=self.pop_size,
            crossover=CrossoverAdapter(crossover_fn),
            mutation=PM(prob=1.0 / int(instance["n_var"]), eta=20),
            eliminate_duplicates=True,
        )
        result = minimize(
            problem,
            algorithm,
            get_termination("n_gen", self.n_gen),
            seed=seed,
            verbose=False,
        )
        return float(HV(ref_point=instance["ref_point"])(result.opt.get("F")))

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, crossover_fn: Callable[[np.ndarray, np.ndarray], tuple]) -> float | None:
        try:
            hypervolumes = []
            for instance in self._instances:
                runs = [
                    self._run_nsga2(instance, crossover_fn, seed)
                    for seed in range(self.seed_start, self.seed_start + self.n_runs)
                ]
                hypervolumes.append(float(np.mean(runs)))
            return float(np.mean(hypervolumes))
        except Exception:
            return None
