from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.es_step_size.dataset import load_split_instances
from llm4ad.task.optimization.es_step_size.evaluation import ESStepSizeEvaluation


def one_fifth_rule(
        sigma: float,
        acceptance_rate: float,
        f_parent: float,
        f_offspring: np.ndarray,
        n: int,
        generation: int,
        max_generations: int,
) -> float:
    c = 0.817
    if acceptance_rate > 0.2:
        return sigma / c
    if acceptance_rate < 0.2:
        return sigma * c
    return sigma


def invalid_sigma(
        sigma: float,
        acceptance_rate: float,
        f_parent: float,
        f_offspring: np.ndarray,
        n: int,
        generation: int,
        max_generations: int,
) -> float:
    return float("nan")


def test_es_step_size_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "es_step_size_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["lam"] == 10
    assert metadata["max_evals"] == 3000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {10}


def test_es_step_size_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["lam"] == 10
    assert metadata["max_evals"] == 15000
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_es_step_size_evaluates_baseline_quickly():
    evaluator = ESStepSizeEvaluation(
        split="train",
        lam=3,
        max_evals=20,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", one_fifth_rule)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_es_step_size_rejects_nonfinite_sigma():
    evaluator = ESStepSizeEvaluation(
        split="train",
        lam=3,
        max_evals=20,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_sigma) is None
