from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.bp_online.dataset import load_split_instances
from llm4ad.task.optimization.bp_online.evaluation import BPOnlineEvaluation


def best_fit_score(item: int, bins: np.ndarray) -> np.ndarray:
    return -bins


def invalid_score(item: int, bins: np.ndarray) -> np.ndarray:
    return np.zeros(len(bins) + 1, dtype=float)


def test_bp_online_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "bp_online_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["capacity"] == 100
    assert metadata["n_items"] == 5000
    assert len(instances) == 5
    assert instances[0]["group_label"] == "5k_c100"
    assert instances[0]["capacity"] == 100
    assert instances[0]["num_items"] == 5000
    assert instances[0]["items"].shape == (5000,)
    assert instances[0]["l1_bound"] > 0


def test_bp_online_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 30
    assert len(metadata["groups"]) == 6
    assert len(instances) == 30
    assert {instance["group_label"] for instance in instances} == {
        "1k_c100",
        "1k_c500",
        "5k_c100",
        "5k_c500",
        "10k_c100",
        "10k_c500",
    }


def test_bp_online_evaluates_best_fit():
    evaluator = BPOnlineEvaluation(split="train")

    score = evaluator.evaluate_program("_", best_fit_score)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 0.0


def test_bp_online_rejects_invalid_scores():
    evaluator = BPOnlineEvaluation(split="train")

    assert evaluator.evaluate_program("_", invalid_score) is None
