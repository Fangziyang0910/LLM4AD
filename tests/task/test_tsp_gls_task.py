from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.main.tsp_gls.dataset import load_split_instances
from llm4ad.task.optimization.main.tsp_gls.evaluation import TSPGLSEvaluation


def identity_update(
        edge_distance: np.ndarray,
        local_opt_tour: np.ndarray,
        edge_n_used: np.ndarray,
) -> np.ndarray:
    return edge_distance.copy()


def invalid_update(
        edge_distance: np.ndarray,
        local_opt_tour: np.ndarray,
        edge_n_used: np.ndarray,
) -> np.ndarray:
    return np.ones((edge_distance.shape[0] + 1, edge_distance.shape[1]), dtype=float)


def test_tsp_gls_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "tsp_gls_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 3
    assert metadata["problem_size"] == 100
    assert metadata["time_limit"] == 10.0
    assert metadata["ite_max"] == 1000
    assert len(instances) == 3
    assert instances[0]["coordinates"].shape == (100, 2)
    assert instances[0]["distance_matrix"].shape == (100, 100)
    assert instances[0]["optimal_tour"].shape == (100,)
    assert instances[0]["optimal_cost"] > 0


def test_tsp_gls_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 16
    assert metadata["problem_size"] == 20
    assert metadata["time_limit"] == 10.0
    assert metadata["ite_max"] == 1000
    assert len(instances) == 16
    assert instances[0]["coordinates"].shape == (20, 2)
    assert instances[0]["distance_matrix"].shape == (20, 20)


def test_tsp_gls_evaluates_identity_update_quickly():
    evaluator = TSPGLSEvaluation(split="train", time_limit=0.0, ite_max=0)

    score = evaluator.evaluate_program("_", identity_update)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 0.0


def test_tsp_gls_penalizes_invalid_update():
    evaluator = TSPGLSEvaluation(split="test_full", time_limit=0.01, ite_max=1)

    score = evaluator.evaluate_program("_", invalid_update)

    assert isinstance(score, float)
    assert score < -1e6
