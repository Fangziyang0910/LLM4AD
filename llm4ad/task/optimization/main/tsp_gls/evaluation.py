from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.main.tsp_gls.dataset import load_split_instances
from llm4ad.task.optimization.main.tsp_gls.gls import solve_instance
from llm4ad.task.optimization.main.tsp_gls.template import task_description, template_program

__all__ = ["TSPGLSEvaluation"]


def _evaluate_tsp_gls_instance(update_fn: Callable, instance, context) -> float:
    return solve_instance(
        instance["optimal_cost"],
        instance["distance_matrix"],
        context["time_limit"],
        context["ite_max"],
        context["perturbation_moves"],
        update_fn,
    )


class TSPGLSEvaluation(Evaluation):
    """Evaluator for EoH's TSP Guided Local Search edge-update task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            time_limit: float | None = None,
            ite_max: int | None = None,
            perturbation_moves: int | None = None,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.problem_size = int(self.dataset_metadata["problem_size"])
        self.time_limit = float(
            time_limit if time_limit is not None else self.dataset_metadata["time_limit"]
        )
        self.ite_max = int(ite_max if ite_max is not None else self.dataset_metadata["ite_max"])
        self.perturbation_moves = int(
            perturbation_moves
            if perturbation_moves is not None
            else self.dataset_metadata["perturbation_moves"]
        )
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            gaps = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(self._instances),
                instance_eval=_evaluate_tsp_gls_instance,
                context={
                    "time_limit": self.time_limit,
                    "ite_max": self.ite_max,
                    "perturbation_moves": self.perturbation_moves,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return -float(np.mean(gaps))
        except Exception:
            return None

    def evaluate(self, update_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> float | None:
        try:
            gaps = [
                solve_instance(
                    instance["optimal_cost"],
                    instance["distance_matrix"],
                    self.time_limit,
                    self.ite_max,
                    self.perturbation_moves,
                    update_fn,
                )
                for instance in self._instances
            ]
            return -float(np.mean(gaps))
        except Exception:
            return None
