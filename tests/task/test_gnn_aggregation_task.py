from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.gnn_aggregation.dataset import load_split_instances
from llm4ad.task.optimization.gnn_aggregation.evaluation import GNNAggregationEvaluation


def mean_aggregate(node_features: np.ndarray, adj_matrix: np.ndarray, iteration: int) -> np.ndarray:
    degree = adj_matrix.sum(axis=1, keepdims=True)
    degree = np.where(degree == 0, 1.0, degree)
    return adj_matrix @ node_features / degree


def invalid_shape(node_features: np.ndarray, adj_matrix: np.ndarray, iteration: int) -> np.ndarray:
    return np.zeros((node_features.shape[0] + 1, node_features.shape[1]), dtype=float)


def invalid_nonfinite(node_features: np.ndarray, adj_matrix: np.ndarray, iteration: int) -> np.ndarray:
    return np.full_like(node_features, np.nan, dtype=float)


def test_gnn_aggregation_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "gnn_aggregation_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 20
    assert metadata["n_nodes"] == 30
    assert metadata["n_feat"] == 4
    assert metadata["n_layers"] == 3
    assert metadata["seed"] == 2024
    assert len(instances) == 20
    assert instances[0]["adj_matrix"].shape == (30, 30)
    assert instances[0]["node_features"].shape == (30, 4)
    assert instances[0]["labels"].shape == (30,)


def test_gnn_aggregation_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 64
    assert metadata["n_nodes"] == 30
    assert metadata["n_feat"] == 4
    assert metadata["n_layers"] == 3
    assert metadata["seed"] == 2025
    assert len(instances) == 64
    assert instances[0]["adj_matrix"].shape == (30, 30)


def test_gnn_aggregation_evaluates_mean_aggregation():
    evaluator = GNNAggregationEvaluation(split="train")

    score = evaluator.evaluate_program("_", mean_aggregate)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 0.0


def test_gnn_aggregation_rejects_invalid_outputs():
    evaluator = GNNAggregationEvaluation(split="train")

    assert evaluator.evaluate_program("_", invalid_shape) is None
    assert evaluator.evaluate_program("_", invalid_nonfinite) is None
