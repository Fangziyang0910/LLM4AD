from __future__ import annotations

from typing import Any, Callable

import nevergrad as ng
import numpy as np
from nevergrad.optimization.optimizerlib import _OnePlusOne

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.one_plus_one.dataset import load_split_instances
from llm4ad.task.optimization.other.one_plus_one.template import task_description, template_program

__all__ = ["OnePlusOneEvaluation"]


class _EolOnePlusOne(_OnePlusOne):
    """Nevergrad _OnePlusOne with only the Gaussian mutation step replaced."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eol_mutation_fn = None
        self._eol_success_window = 20
        self._eol_history: list[int] = []

    def _internal_ask_candidate(self):
        if (
                self._eol_mutation_fn is None
                or not self._num_ask
                or self.mutation != "gaussian"
        ):
            return super()._internal_ask_candidate()

        ref = self.parametrization
        pessimistic = self.current_bests["pessimistic"].parameter.spawn_child()
        current_x = pessimistic.get_standardized_data(reference=ref)
        success_rate = float(np.mean(self._eol_history)) if self._eol_history else 0.2

        noise = self._eol_mutation_fn(
            current_x.copy(),
            float(self._sigma),
            success_rate,
            self.dimension,
            self._num_ask,
            self.budget or 1000,
        )
        noise = np.asarray(noise, dtype=float)
        if noise.shape != (self.dimension,):
            raise ValueError(f"generate_mutation shape {noise.shape} != ({self.dimension},)")

        out = pessimistic.set_standardized_data(noise)
        out._meta["sigma"] = self._sigma
        return out

    def _internal_tell(self, x, loss):
        improved = loss < self._previous_best_loss
        self._eol_history.append(1 if improved else 0)
        if len(self._eol_history) > self._eol_success_window:
            self._eol_history.pop(0)
        super()._internal_tell(x, loss)


class OnePlusOneEvaluation(Evaluation):
    """Evaluator for EoH's Nevergrad OnePlusOne mutation design task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            max_evals: int | None = None,
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
        self.max_evals = int(max_evals if max_evals is not None else self.dataset_metadata["max_evals"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])

    def _run_one_plus_one(
            self,
            instance: dict[str, Any],
            mutation_fn: Callable[[np.ndarray, float, float, int, int, int], np.ndarray],
    ) -> float:
        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]

        param = ng.p.Array(shape=(dim,)).set_bounds(lo, hi)
        opt = _EolOnePlusOne(
            parametrization=param,
            budget=self.max_evals,
            mutation="gaussian",
        )
        opt._eol_mutation_fn = mutation_fn
        recommendation = opt.minimize(func)
        return float(func(recommendation.value))

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, mutation_fn: Callable[[np.ndarray, float, float, int, int, int], np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_one_plus_one(instance, mutation_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None
