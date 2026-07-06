from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.one_plus_one.dataset import load_split_instances
from llm4ad.task.optimization.one_plus_one.evaluation import OnePlusOneEvaluation


def gaussian_mutation(
        current_solution: np.ndarray,
        sigma: float,
        success_rate: float,
        n_dims: int,
        iteration: int,
        max_evals: int,
) -> np.ndarray:
    return sigma * np.random.normal(0.0, 1.0, n_dims)


def invalid_mutation(
        current_solution: np.ndarray,
        sigma: float,
        success_rate: float,
        n_dims: int,
        iteration: int,
        max_evals: int,
) -> np.ndarray:
    return np.zeros(n_dims + 1, dtype=float)


def test_one_plus_one_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "one_plus_one_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["max_evals"] == 1000
    assert metadata["n_runs"] == 3
    assert len(instances) == 5
    assert {instance["name"] for instance in instances} == {
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    }
    assert all(instance["dim"] == 10 for instance in instances)


def test_one_plus_one_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["max_evals"] == 5000
    assert metadata["n_runs"] == 10
    assert len(instances) == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_one_plus_one_evaluates_gaussian_mutation_quickly():
    evaluator = OnePlusOneEvaluation(split="train", max_evals=6, n_runs=1)

    score = evaluator.evaluate_program("_", gaussian_mutation)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 0.0


def test_one_plus_one_rejects_invalid_mutation():
    evaluator = OnePlusOneEvaluation(split="train", max_evals=6, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_mutation) is None
