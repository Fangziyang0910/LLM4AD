from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.tpe_bandwidth.dataset import load_split_instances
from llm4ad.task.optimization.other.tpe_bandwidth.template import task_description, template_program

__all__ = ["TPEBandwidthEvaluation"]

optuna.logging.set_verbosity(optuna.logging.ERROR)


class TPEBandwidthEvaluation(Evaluation):
    """Evaluator for EoH's Optuna TPE observation-weight design task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            n_startup: int | None = None,
            n_iter: int | None = None,
            n_runs: int | None = None,
            n_ei_candidates: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_startup = int(
            n_startup if n_startup is not None else self.dataset_metadata["n_startup"]
        )
        self.n_iter = int(n_iter if n_iter is not None else self.dataset_metadata["n_iter"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.n_ei_candidates = int(
            n_ei_candidates
            if n_ei_candidates is not None
            else self.dataset_metadata["n_ei_candidates"]
        )

    def _run_tpe(
            self,
            instance: dict[str, Any],
            weights_fn: Callable[[int], np.ndarray],
            seed: int,
    ) -> float:
        func = instance["func"]
        lo, hi = instance["lo"], instance["hi"]

        def objective(trial):
            x = trial.suggest_float("x", lo, hi)
            return float(func(x))

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup,
            n_ei_candidates=self.n_ei_candidates,
            weights=weights_fn,
            seed=seed,
        )
        study = optuna.create_study(sampler=sampler)
        study.optimize(
            objective,
            n_trials=self.n_startup + self.n_iter,
            show_progress_bar=False,
        )
        return float(study.best_value)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, weights_fn: Callable[[int], np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = [
                    self._run_tpe(instance, weights_fn, seed)
                    for seed in range(self.n_runs)
                ]
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None
