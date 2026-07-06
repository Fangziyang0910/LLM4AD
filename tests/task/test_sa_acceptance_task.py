from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.sa_acceptance.dataset import load_split_instances
from llm4ad.task.optimization.sa_acceptance.evaluation import SAAcceptanceEvaluation


def boltzmann(delta_fitness: float, temperature: float, iteration: int, max_iterations: int) -> float:
    return float(np.exp(-delta_fitness / max(temperature, 1e-10)))


def invalid_probability(
        delta_fitness: float,
        temperature: float,
        iteration: int,
        max_iterations: int,
) -> float:
    return float("nan")


def test_sa_acceptance_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "sa_acceptance_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["max_iter"] == 5000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {10}


def test_sa_acceptance_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_sa_acceptance_evaluates_boltzmann_quickly():
    evaluator = SAAcceptanceEvaluation(
        split="train",
        max_iter=20,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", boltzmann)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_sa_acceptance_rejects_nonfinite_probability():
    evaluator = SAAcceptanceEvaluation(
        split="train",
        max_iter=20,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_probability) is None
