from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.aco_pheromone.dataset import load_split_instances
from llm4ad.task.optimization.aco_pheromone.evaluation import ACOPheromoneEvaluation


def ant_system_update(
        pheromone: np.ndarray,
        ant_tours: list,
        tour_costs: np.ndarray,
        best_tour: np.ndarray,
        best_cost: float,
        rho: float,
        iteration: int,
        max_iterations: int,
) -> np.ndarray:
    n = pheromone.shape[0]
    pheromone = (1.0 - rho) * pheromone
    for tour, cost in zip(ant_tours, tour_costs):
        delta = 1.0 / cost
        for i in range(n):
            u, v = int(tour[i]), int(tour[(i + 1) % n])
            pheromone[u, v] += delta
            pheromone[v, u] += delta
    return pheromone


def invalid_pheromone(
        pheromone: np.ndarray,
        ant_tours: list,
        tour_costs: np.ndarray,
        best_tour: np.ndarray,
        best_cost: float,
        rho: float,
        iteration: int,
        max_iterations: int,
) -> np.ndarray:
    return np.zeros(pheromone.shape[0], dtype=float)


def test_aco_pheromone_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "aco_pheromone_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 3
    assert metadata["n_cities"] == 20
    assert metadata["n_ants"] == 20
    assert metadata["iter_max"] == 100
    assert metadata["n_runs"] == 3
    assert len(instances) == 3
    assert instances[0]["coordinates"].shape == (20, 2)
    assert instances[0]["distances"].shape == (20, 20)


def test_aco_pheromone_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 5
    assert metadata["n_cities"] == 50
    assert metadata["n_ants"] == 25
    assert metadata["iter_max"] == 200
    assert metadata["n_runs"] == 10
    assert len(instances) == 5
    assert instances[0]["coordinates"].shape == (50, 2)
    assert instances[0]["distances"].shape == (50, 50)


def test_aco_pheromone_evaluates_ant_system_quickly():
    evaluator = ACOPheromoneEvaluation(
        split="train",
        n_ants=3,
        iter_max=2,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", ant_system_update)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_aco_pheromone_rejects_invalid_shape():
    evaluator = ACOPheromoneEvaluation(
        split="train",
        n_ants=3,
        iter_max=2,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_pheromone) is None
