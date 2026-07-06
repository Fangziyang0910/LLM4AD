from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.de_mutation.dataset import load_split_instances
from llm4ad.task.optimization.de_mutation.evaluation import DEMutationEvaluation


def de_rand_1(
        population: np.ndarray,
        current_idx: int,
        best_idx: int,
        fitness: np.ndarray,
        F: float,
        bounds: np.ndarray,
) -> np.ndarray:
    pop_size, dim = population.shape
    candidates = [i for i in range(pop_size) if i != current_idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])


def invalid_mutation(
        population: np.ndarray,
        current_idx: int,
        best_idx: int,
        fitness: np.ndarray,
        F: float,
        bounds: np.ndarray,
) -> np.ndarray:
    return np.zeros((population.shape[1], 1), dtype=float)


def test_de_mutation_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "de_mutation_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["pop_size"] == 20
    assert metadata["max_evals"] == 5000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {10}


def test_de_mutation_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["pop_size"] == 50
    assert metadata["max_evals"] == 20000
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_de_mutation_evaluates_de_rand_1_quickly():
    evaluator = DEMutationEvaluation(
        split="train",
        pop_size=5,
        max_evals=20,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", de_rand_1)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_de_mutation_rejects_invalid_shape():
    evaluator = DEMutationEvaluation(
        split="train",
        pop_size=5,
        max_evals=20,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_mutation) is None
