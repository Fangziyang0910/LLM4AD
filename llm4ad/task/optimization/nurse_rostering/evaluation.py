from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.nurse_rostering.dataset import load_split_instances
from llm4ad.task.optimization.nurse_rostering.template import task_description, template_program

__all__ = ["NurseRosteringEvaluation", "compute_roster_metrics"]


def compute_roster_metrics(result: dict[str, Any]) -> dict[str, float]:
    assignment = result["assignment"]
    workload = result["workload"]
    preferences = result["preferences"]
    max_consecutive = int(result["max_consecutive"])
    n_nurses, n_days = assignment.shape

    workload_std = float(np.std(workload))
    preference_values = [
        preferences[nurse, assignment[nurse, day]]
        for nurse in range(n_nurses)
        for day in range(n_days)
        if assignment[nurse, day] >= 0
    ]
    preference_mean = float(np.mean(preference_values)) if preference_values else 0.0

    consecutive_violations = 0
    for nurse in range(n_nurses):
        run = 0
        for day in range(n_days):
            if assignment[nurse, day] >= 0:
                run += 1
                if run > max_consecutive:
                    consecutive_violations += 1
            else:
                run = 0

    night_morning_violations = sum(
        1
        for nurse in range(n_nurses)
        for day in range(1, n_days)
        if assignment[nurse, day - 1] == 2 and assignment[nurse, day] == 0
    )

    composite = (
        workload_std
        + (1.0 - preference_mean)
        + 0.2 * consecutive_violations
        + 0.3 * night_morning_violations
    )
    return {
        "workload_std": workload_std,
        "preference_mean": preference_mean,
        "consecutive_violations": float(consecutive_violations),
        "night_morning_violations": float(night_morning_violations),
        "composite": float(composite),
    }


class NurseRosteringEvaluation(Evaluation):
    """Evaluator for EoH's greedy nurse-rostering scoring task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])

    def _construct_roster(
            self,
            instance: dict[str, Any],
            score_fn: Callable[[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int], float],
    ) -> dict[str, Any]:
        n_nurses = int(instance["n_nurses"])
        n_days = int(instance["n_days"])
        preferences = instance["preferences"]
        requirements = instance["requirements"]
        n_shift_types = int(instance["n_shift_types"])
        total_per_day = int(requirements.sum())

        assignment = np.full((n_nurses, n_days), -1, dtype=int)
        workload = np.zeros(n_nurses, dtype=float)
        consecutive = np.zeros(n_nurses, dtype=int)
        last_shift = np.full(n_nurses, -1, dtype=int)

        for day in range(n_days):
            target = float((day + 1) * total_per_day) / n_nurses
            assigned_today: set[int] = set()

            for shift_type in range(n_shift_types):
                needed = int(requirements[shift_type])
                eligible = [idx for idx in range(n_nurses) if idx not in assigned_today]
                if len(eligible) < needed:
                    continue

                scored: list[tuple[float, int]] = []
                for nurse_idx in eligible:
                    try:
                        score = float(score_fn(
                            int(nurse_idx),
                            int(shift_type),
                            int(day),
                            workload.copy(),
                            preferences,
                            consecutive.copy(),
                            last_shift.copy(),
                            float(target),
                            int(n_days),
                        ))
                    except Exception:
                        score = 0.0
                    if not np.isfinite(score):
                        score = 0.0
                    scored.append((score, nurse_idx))

                scored.sort(key=lambda item: (-item[0], item[1]))
                for _, nurse_idx in scored[:needed]:
                    assignment[nurse_idx, day] = shift_type
                    workload[nurse_idx] += 1
                    assigned_today.add(nurse_idx)
                    last_shift[nurse_idx] = shift_type

            for nurse_idx in range(n_nurses):
                if nurse_idx in assigned_today:
                    consecutive[nurse_idx] += 1
                else:
                    consecutive[nurse_idx] = 0

        return {
            "assignment": assignment,
            "workload": workload,
            "preferences": preferences,
            "max_consecutive": instance["max_consecutive"],
        }

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, score_fn: Callable) -> float | None:
        try:
            composites = []
            for instance in self._instances:
                result = self._construct_roster(instance, score_fn)
                composites.append(compute_roster_metrics(result)["composite"])
            return -float(np.mean(composites))
        except Exception:
            return None
