from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.circle_packing.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.circle_packing.template import (
    task_description,
    template_program,
)

__all__ = ["CirclePackingEvaluation", "validate_packing"]


def _coerce_result(result: Any) -> tuple[np.ndarray, np.ndarray, float | None]:
    if not isinstance(result, tuple) or len(result) not in {2, 3}:
        raise ValueError(
            "construct_packing must return (centers, radii) or "
            "(centers, radii, reported_sum)."
        )

    centers = np.asarray(result[0], dtype=float)
    radii = np.asarray(result[1], dtype=float)
    reported_sum = None if len(result) == 2 else float(result[2])
    return centers, radii, reported_sum


def validate_packing(
        centers: np.ndarray,
        radii: np.ndarray,
        *,
        n_circles: int = 26,
        square_size: float = 1.0,
        atol: float = 0.0,
) -> tuple[bool, str]:
    if centers.shape != (n_circles, 2):
        return False, f"centers shape must be ({n_circles}, 2), got {centers.shape}."
    if radii.shape != (n_circles,):
        return False, f"radii shape must be ({n_circles},), got {radii.shape}."

    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
        return False, "centers and radii must be finite."
    if np.any(radii < 0):
        return False, "radii must be non-negative."

    lower = centers - radii[:, None]
    upper = centers + radii[:, None]
    if np.any(lower < -atol) or np.any(upper > square_size + atol):
        return False, "all circles must be fully contained in the unit square."

    deltas = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    radius_sums = radii[:, None] + radii[None, :]
    overlap = distances < radius_sums - atol
    np.fill_diagonal(overlap, False)
    if np.any(overlap):
        i, j = np.argwhere(overlap)[0]
        return False, f"circles {int(i)} and {int(j)} overlap."

    return True, "valid packing."


class CirclePackingEvaluation(Evaluation):
    """Evaluator for the ShinkaEvolve n=26 circle-packing task."""

    def __init__(
            self,
            timeout_seconds=30,
            split: str = DEFAULT_SPLIT,
            atol: float = 0.0,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.atol = float(atol)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, construct_packing: Callable) -> float | None:
        try:
            result = construct_packing()
            centers, radii, reported_sum = _coerce_result(result)
            instance = self._instances[0]
            valid, _ = validate_packing(
                centers,
                radii,
                n_circles=int(instance["n_circles"]),
                square_size=float(instance["square_size"]),
                atol=self.atol,
            )
            if not valid:
                return None

            score = float(np.sum(radii))
            if reported_sum is not None and not np.isclose(
                    score,
                    reported_sum,
                    atol=self.atol,
                    rtol=0.0,
            ):
                return None
            return score
        except Exception:
            return None
