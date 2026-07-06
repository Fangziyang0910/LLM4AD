from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.bpp_offline_aco.dataset import load_split_instances
from llm4ad.task.optimization.bpp_offline_aco.evaluation import BPPOfflineACOEvaluation


def demand_similarity(demand: np.ndarray, capacity: int) -> np.ndarray:
    return np.tile(demand / demand.max(), (demand.shape[0], 1))


def test_bpp_offline_aco_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "bpp_offline_aco_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["n_items"] == 500
    assert metadata["capacity"] == 150
    assert instances.shape == (5, 500)


def test_bpp_offline_aco_evaluates_seed_heuristic_quickly():
    evaluator = BPPOfflineACOEvaluation(split="train", n_ants=2, n_iterations=1)

    score = evaluator.evaluate_program("_", demand_similarity)

    assert isinstance(score, float)
    assert score < 0
