from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.de_crossover_100d.dataset import load_split_instances
from llm4ad.task.optimization.de_crossover_100d.evaluation import DECrossover100DEvaluation


def binomial_crossover(
        target: np.ndarray,
        mutant: np.ndarray,
        CR: float,
        generation: int,
        max_generations: int,
        fitness_target: float,
        fitness_best: float,
) -> np.ndarray:
    dim = len(target)
    mask = np.random.rand(dim) < CR
    mask[np.random.randint(dim)] = True
    return np.where(mask, mutant, target)


def invalid_crossover(
        target: np.ndarray,
        mutant: np.ndarray,
        CR: float,
        generation: int,
        max_generations: int,
        fitness_target: float,
        fitness_best: float,
) -> np.ndarray:
    return np.zeros((len(target), 1), dtype=float)


def test_de_crossover_100d_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "de_crossover_100d_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["pop_size"] == 20
    assert metadata["max_evals"] == 20000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {100}


def test_de_crossover_100d_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 15
    assert metadata["pop_size"] == 50
    assert metadata["max_evals"] == 100000
    assert metadata["n_runs"] == 20
    assert {instance["dim"] for instance in instances} == {50, 100, 200}


def test_de_crossover_100d_evaluates_binomial_quickly():
    evaluator = DECrossover100DEvaluation(
        split="train",
        pop_size=5,
        max_evals=20,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", binomial_crossover)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_de_crossover_100d_rejects_invalid_shape():
    evaluator = DECrossover100DEvaluation(
        split="train",
        pop_size=5,
        max_evals=20,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_crossover) is None
