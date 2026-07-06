from __future__ import annotations

import math
from typing import Any, Callable

from llm4ad.base import Evaluation
from llm4ad.task.science_discovery.sine_approx.template import (
    task_description,
    template_program,
)

__all__ = ["SineApproxEvaluation"]

BLOCKED_SOURCE_SNIPPETS = (
    "math.sin",
    "cmath.sin",
    "numpy.sin",
    "np.sin",
    "from math import sin",
    "import sin",
)


def _sample_points() -> list[float]:
    points = []
    for index in range(161):
        base = -math.pi + (2.0 * math.pi * index / 160.0)
        wobble = 0.007 * math.sin(index * 1.61803398875)
        points.append(max(-math.pi, min(math.pi, base + wobble)))
    return points


def _source_violation(source: str) -> str | None:
    lowered = source.lower()
    for blocked in BLOCKED_SOURCE_SNIPPETS:
        if blocked in lowered:
            return blocked
    return None


def evaluate(program_str: str, approximate: Callable[[float], float]) -> float | None:
    if _source_violation(program_str):
        return None

    try:
        abs_errors = []
        for x in _sample_points():
            estimate = float(approximate(x))
            if not math.isfinite(estimate) or abs(estimate) > 10.0:
                return None
            abs_errors.append(abs(estimate - math.sin(x)))

        rmse = math.sqrt(sum(error * error for error in abs_errors) / len(abs_errors))
        max_error = max(abs_errors)
        return float(1.0 / (1.0 + 4.0 * rmse + max_error))
    except Exception:
        return None


class SineApproxEvaluation(Evaluation):
    """Evaluator for ShinkaEvolve's headless sine-approximation example."""

    def __init__(self, timeout_seconds=20, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        self.dataset_metadata = {
            "dataset_id": "sine_approx_headless_fixed_points_v1",
            "task": "sine_approx",
            "split": "fixed",
            "n_instances": 1,
            "n_points": 161,
            "source": "reference_code/ShinkaEvolve/examples/sine_approx_headless",
        }

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return evaluate(program_str, callable_func)
