from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.moead_decomposition.dataset import load_split_instances
from llm4ad.task.optimization.other.moead_decomposition.evaluation import MOEADDecompositionEvaluation


def tchebycheff(F: np.ndarray, weights: np.ndarray, ideal_point: np.ndarray) -> np.ndarray:
    return np.max(np.abs(F - ideal_point) * weights, axis=1)


def invalid_shape(F: np.ndarray, weights: np.ndarray, ideal_point: np.ndarray) -> np.ndarray:
    return np.zeros(len(F) + 1, dtype=float)


def invalid_nonfinite(F: np.ndarray, weights: np.ndarray, ideal_point: np.ndarray) -> np.ndarray:
    return np.full(len(F), np.nan, dtype=float)


def test_moead_decomposition_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "moead_decomposition_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 2
    assert metadata["n_gen"] == 100
    assert metadata["n_runs"] == 3
    assert metadata["T"] == 5
    assert metadata["hv_samples"] == 20000
    assert [instance["n_var"] for instance in instances] == [7, 12]
    assert {instance["n_obj"] for instance in instances} == {3}


def test_moead_decomposition_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 4
    assert metadata["n_gen"] == 200
    assert metadata["n_runs"] == 10
    assert metadata["hv_samples"] == 30000
    assert [instance["n_var"] for instance in instances] == [7, 12, 20, 30]


def test_moead_decomposition_evaluates_tchebycheff_quickly():
    evaluator = MOEADDecompositionEvaluation(
        split="train",
        n_gen=2,
        n_runs=1,
        hv_samples=500,
    )

    score = evaluator.evaluate_program("_", tchebycheff)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score >= 0.0


def test_moead_decomposition_rejects_invalid_scores():
    evaluator = MOEADDecompositionEvaluation(
        split="train",
        n_gen=2,
        n_runs=1,
        hv_samples=500,
    )

    assert evaluator.evaluate_program("_", invalid_shape) is None
    assert evaluator.evaluate_program("_", invalid_nonfinite) is None
