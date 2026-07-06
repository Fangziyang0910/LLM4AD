from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.cvrp_aco.dataset import load_split_instances
from llm4ad.task.optimization.cvrp_aco.evaluation import CVRPACOEvaluation


def inverse_distance(
        distance_matrix: np.ndarray,
        coordinates: np.ndarray,
        demands: np.ndarray,
        capacity: int,
) -> np.ndarray:
    return 1.0 / (distance_matrix + 1e-9)


def test_cvrp_aco_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "cvrp_aco_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 10
    assert metadata["problem_size"] == 50
    assert metadata["capacity"] == 50
    assert instances.shape == (10, 51, 3)


def test_cvrp_aco_evaluates_inverse_distance_heuristic_quickly():
    evaluator = CVRPACOEvaluation(split="train", n_ants=4, n_iterations=1)

    score = evaluator.evaluate_program("_", inverse_distance)

    assert isinstance(score, float)
    assert score < 0
