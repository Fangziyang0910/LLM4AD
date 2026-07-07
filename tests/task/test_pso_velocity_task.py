from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.pso_velocity.dataset import load_split_instances
from llm4ad.task.optimization.other.pso_velocity.evaluation import PSOVelocityEvaluation


def standard_pso_update(
        velocities: np.ndarray,
        positions: np.ndarray,
        pbest_positions: np.ndarray,
        pbest_fitness: np.ndarray,
        gbest_position: np.ndarray,
        gbest_fitness: float,
        w: float,
        c1: float,
        c2: float,
        bounds: np.ndarray,
        iteration: int,
        max_iterations: int,
) -> np.ndarray:
    pop_size, dim = velocities.shape
    r1 = np.random.rand(pop_size, dim)
    r2 = np.random.rand(pop_size, dim)
    cognitive = c1 * r1 * (pbest_positions - positions)
    social = c2 * r2 * (gbest_position - positions)
    return w * velocities + cognitive + social


def invalid_velocity(
        velocities: np.ndarray,
        positions: np.ndarray,
        pbest_positions: np.ndarray,
        pbest_fitness: np.ndarray,
        gbest_position: np.ndarray,
        gbest_fitness: float,
        w: float,
        c1: float,
        c2: float,
        bounds: np.ndarray,
        iteration: int,
        max_iterations: int,
) -> np.ndarray:
    return np.zeros(velocities.shape[1], dtype=float)


def test_pso_velocity_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "pso_velocity_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["pop_size"] == 30
    assert metadata["max_iterations"] == 200
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {10}


def test_pso_velocity_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["pop_size"] == 50
    assert metadata["max_iterations"] == 500
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_pso_velocity_evaluates_standard_update_quickly():
    evaluator = PSOVelocityEvaluation(
        split="train",
        pop_size=5,
        max_iterations=3,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", standard_pso_update)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_pso_velocity_rejects_invalid_shape():
    evaluator = PSOVelocityEvaluation(
        split="train",
        pop_size=5,
        max_iterations=3,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_velocity) is None
