from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.tpe_bandwidth.dataset import load_split_instances
from llm4ad.task.optimization.other.tpe_bandwidth.evaluation import TPEBandwidthEvaluation


def default_weights(n: int) -> np.ndarray:
    if n == 0:
        return np.array([])
    if n < 25:
        return np.ones(n)
    return np.concatenate([np.linspace(1.0 / n, 1.0, num=n - 25), np.ones(25)])


def invalid_weights(n: int) -> np.ndarray:
    return np.full(n + 1, -1.0)


def test_tpe_bandwidth_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "tpe_bandwidth_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["n_startup"] == 10
    assert metadata["n_iter"] == 30
    assert metadata["n_runs"] == 3
    assert len(instances) == 5
    assert {instance["name"] for instance in instances} == {
        "sphere",
        "rastrigin",
        "ackley",
        "griewank",
        "narrow",
    }


def test_tpe_bandwidth_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["n_startup"] == 20
    assert metadata["n_iter"] == 60
    assert metadata["n_runs"] == 10
    assert len(instances) == 10


def test_tpe_bandwidth_evaluates_default_weights_quickly():
    evaluator = TPEBandwidthEvaluation(split="train", n_startup=2, n_iter=3, n_runs=1)

    score = evaluator.evaluate_program("_", default_weights)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 0.0


def test_tpe_bandwidth_rejects_invalid_weights():
    evaluator = TPEBandwidthEvaluation(split="train", n_startup=2, n_iter=3, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_weights) is None
