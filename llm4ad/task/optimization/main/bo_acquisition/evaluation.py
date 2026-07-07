from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.bo_acquisition.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.bo_acquisition.template import template_program, task_description

__all__ = ["BOAcquisitionEvaluation"]


def _run_bo_instance(instance: dict, acq_fn: Callable, seed: int, context: dict) -> float:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    rng = np.random.default_rng(seed)
    func = instance["func"]
    n_var = int(instance["n_var"])
    f_opt = float(instance["f_opt"])

    x_obs = rng.uniform(0.0, 1.0, (context["n_init"], n_var))
    y_obs = np.array([func(x) for x in x_obs])

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0,
        length_scale_bounds=(1e-2, 10.0),
        nu=2.5,
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=0,
        normalize_y=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(context["n_iter"]):
            gp.fit(x_obs, y_obs)
            x_cand = rng.uniform(0.0, 1.0, (context["n_candidates"], n_var))
            mu, sigma = gp.predict(x_cand, return_std=True)
            sigma = np.maximum(sigma, 1e-9)

            acq_vals = np.asarray(acq_fn(mu, sigma, float(np.min(y_obs))), dtype=float).ravel()
            if acq_vals.shape != (context["n_candidates"],) or not np.all(np.isfinite(acq_vals)):
                raise ValueError("acquisition returned invalid candidate scores.")

            x_next = x_cand[np.argmax(acq_vals)]
            y_obs = np.append(y_obs, func(x_next))
            x_obs = np.vstack([x_obs, x_next])

    simple_regret = float(np.min(y_obs)) - f_opt
    return max(simple_regret, 0.0)


def _evaluate_bo_acquisition_instance(acq_fn: Callable, payload, context) -> float:
    _idx, instance = payload
    regrets = [
        _run_bo_instance(instance, acq_fn, seed, context)
        for seed in range(context["n_runs"])
    ]
    return float(np.log10(np.mean(regrets) + 1e-8))


class BOAcquisitionEvaluation(Evaluation):
    """Evaluator for Bayesian optimization acquisition functions."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            n_init: int = 5,
            n_iter: int = 20,
            n_candidates: int = 256,
            n_runs: int = 5,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._datasets, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_init = int(n_init)
        self.n_iter = int(n_iter)
        self.n_candidates = int(n_candidates)
        self.n_runs = int(n_runs)
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def _run_bo(self, instance: dict, acq_fn: Callable, seed: int) -> float:
        return _run_bo_instance(
            instance,
            acq_fn,
            seed,
            {
                "n_init": self.n_init,
                "n_iter": self.n_iter,
                "n_candidates": self.n_candidates,
                "n_runs": self.n_runs,
            },
        )

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            log_regrets = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(enumerate(self._datasets)),
                instance_eval=_evaluate_bo_acquisition_instance,
                context={
                    "n_init": self.n_init,
                    "n_iter": self.n_iter,
                    "n_candidates": self.n_candidates,
                    "n_runs": self.n_runs,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return -float(np.mean(log_regrets))
        except Exception:
            return None

    def evaluate(self, acq_fn: Callable) -> float | None:
        try:
            log_regrets = []
            for instance in self._datasets:
                regrets = [
                    self._run_bo(instance, acq_fn, seed)
                    for seed in range(self.n_runs)
                ]
                log_regrets.append(float(np.log10(np.mean(regrets) + 1e-8)))
            return -float(np.mean(log_regrets))
        except Exception:
            return None
