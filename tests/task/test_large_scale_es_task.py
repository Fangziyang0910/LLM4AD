from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.large_scale_es.dataset import load_split_instances
from llm4ad.task.optimization.large_scale_es.evaluation import LargeScaleESEvaluation


def separable_rank_update(
        d: np.ndarray,
        p_c: np.ndarray,
        weights: np.ndarray,
        y_k: np.ndarray,
        c1: float,
        cmu: float,
        cc: float,
        hsig: float,
        n: int,
        generation: int,
        max_generations: int,
) -> np.ndarray:
    rank1 = c1 * (p_c ** 2 + (1.0 - hsig) * cc * (2.0 - cc) * d)
    rankmu = cmu * np.einsum("i,ij->j", weights, y_k ** 2)
    return (1.0 - c1 - cmu) * d + rank1 + rankmu


def invalid_diagonal_cov(
        d: np.ndarray,
        p_c: np.ndarray,
        weights: np.ndarray,
        y_k: np.ndarray,
        c1: float,
        cmu: float,
        cc: float,
        hsig: float,
        n: int,
        generation: int,
        max_generations: int,
) -> np.ndarray:
    return np.zeros((n, 1), dtype=float)


def test_large_scale_es_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "large_scale_es_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["max_evals"] == 30000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {100}


def test_large_scale_es_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["max_evals"] == 60000
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {100, 200}


def test_large_scale_es_evaluates_baseline_quickly():
    evaluator = LargeScaleESEvaluation(
        split="train",
        max_evals=20,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", separable_rank_update)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_large_scale_es_rejects_invalid_shape():
    evaluator = LargeScaleESEvaluation(
        split="train",
        max_evals=20,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_diagonal_cov) is None
