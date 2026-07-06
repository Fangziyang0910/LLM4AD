from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.nurse_rostering.dataset import load_split_instances
from llm4ad.task.optimization.nurse_rostering.evaluation import NurseRosteringEvaluation


def linear_score(
        nurse_idx: int,
        shift_type: int,
        day: int,
        nurse_workload: np.ndarray,
        nurse_preferences: np.ndarray,
        consecutive_days: np.ndarray,
        last_shift_type: np.ndarray,
        target_workload: float,
        n_days: int,
) -> float:
    preference = nurse_preferences[nurse_idx, shift_type]
    workload_gap = nurse_workload[nurse_idx] - target_workload
    consecutive_penalty = max(0.0, float(consecutive_days[nurse_idx]) - 4.0)
    night_morning_penalty = 1.0 if (shift_type == 0 and last_shift_type[nurse_idx] == 2) else 0.0
    return float(preference - 0.5 * workload_gap - 2.0 * consecutive_penalty
                 - 5.0 * night_morning_penalty)


def nonfinite_score(
        nurse_idx: int,
        shift_type: int,
        day: int,
        nurse_workload: np.ndarray,
        nurse_preferences: np.ndarray,
        consecutive_days: np.ndarray,
        last_shift_type: np.ndarray,
        target_workload: float,
        n_days: int,
) -> float:
    return float("nan")


def test_nurse_rostering_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "nurse_rostering_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert len(instances) == 5
    assert instances[0]["n_nurses"] == 8
    assert instances[0]["n_days"] == 14
    assert instances[0]["requirements"].tolist() == [2, 2, 1]
    assert instances[0]["preferences"].shape == (8, 3)


def test_nurse_rostering_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 20
    assert len(instances) == 20
    assert {instance["n_days"] for instance in instances} == {14, 21}
    assert [group["seed"] for group in metadata["groups"]] == [200, 300]


def test_nurse_rostering_evaluates_linear_score():
    evaluator = NurseRosteringEvaluation(split="train")

    score = evaluator.evaluate_program("_", linear_score)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score < 0.0


def test_nurse_rostering_handles_nonfinite_scores_like_source():
    evaluator = NurseRosteringEvaluation(split="train")

    score = evaluator.evaluate_program("_", nonfinite_score)

    assert isinstance(score, float)
    assert np.isfinite(score)
