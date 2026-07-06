from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.tsp_rnr.dataset import load_split_instances
from llm4ad.task.optimization.tsp_rnr.evaluation import TSPRnrEvaluation


def first_nodes(current_tour: np.ndarray, distance_matrix: np.ndarray, n_destroy: int) -> np.ndarray:
    return current_tour[:n_destroy]


def invalid_nodes(current_tour: np.ndarray, distance_matrix: np.ndarray, n_destroy: int) -> np.ndarray:
    return np.array([-1] * n_destroy, dtype=int)


def test_tsp_rnr_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "tsp_rnr_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["n_nodes"] == 50
    assert metadata["n_destroy"] == 10
    assert metadata["iter_max"] == 100
    assert metadata["time_max"] == 5.0
    assert metadata["seed"] == 2024
    assert len(instances) == 5
    assert instances[0]["coordinates"].shape == (50, 2)
    assert instances[0]["distance_matrix"].shape == (50, 50)


def test_tsp_rnr_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 16
    assert metadata["n_nodes"] == 50
    assert metadata["iter_max"] == 200
    assert metadata["time_max"] == 10.0
    assert metadata["seed"] == 2025
    assert len(instances) == 16
    assert instances[0]["coordinates"].shape == (50, 2)


def test_tsp_rnr_evaluates_destroy_operator_quickly():
    evaluator = TSPRnrEvaluation(split="train", iter_max=2, time_max=0.1)

    score = evaluator.evaluate_program("_", first_nodes)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score < 0.0


def test_tsp_rnr_invalid_destroy_operator_keeps_initial_solution():
    evaluator = TSPRnrEvaluation(split="train", iter_max=2, time_max=0.1)

    score = evaluator.evaluate_program("_", invalid_nodes)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score < 0.0
