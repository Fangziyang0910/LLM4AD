from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.nsga2_crowding.dataset import load_split_instances
from llm4ad.task.optimization.other.nsga2_crowding.evaluation import NSGA2CrowdingEvaluation


def standard_crowding(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    distance = np.zeros(n)
    for obj in range(m):
        order = np.argsort(F[:, obj])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = F[order[-1], obj] - F[order[0], obj]
        if span < 1e-10:
            continue
        for k in range(1, n - 1):
            distance[order[k]] += (F[order[k + 1], obj] - F[order[k - 1], obj]) / span
    return distance


def invalid_shape(F: np.ndarray) -> np.ndarray:
    return np.zeros(len(F) + 1, dtype=float)


def test_nsga2_crowding_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "nsga2_crowding_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 2
    assert metadata["pop_size"] == 100
    assert metadata["n_gen"] == 100
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == ["ZDT1", "ZDT2"]
    assert {instance["n_var"] for instance in instances} == {30}


def test_nsga2_crowding_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 3
    assert metadata["pop_size"] == 100
    assert metadata["n_gen"] == 200
    assert metadata["n_runs"] == 10
    assert [instance["name"] for instance in instances] == ["ZDT1", "ZDT2", "ZDT3"]


def test_nsga2_crowding_evaluates_standard_crowding_quickly():
    evaluator = NSGA2CrowdingEvaluation(split="train", pop_size=10, n_gen=2, n_runs=1)

    score = evaluator.evaluate_program("_", standard_crowding)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score >= 0.0


def test_nsga2_crowding_rejects_invalid_shape():
    evaluator = NSGA2CrowdingEvaluation(split="train", pop_size=10, n_gen=2, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_shape) is None
