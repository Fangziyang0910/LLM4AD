from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.mkp_aco.dataset import load_split_instances
from llm4ad.task.optimization.mkp_aco.evaluation import MKPACOEvaluation


def prize_weight_ratio(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return prize / (np.sum(weight, axis=1) + 1e-9)


def test_mkp_aco_loads_fixed_train_split():
    dataset, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "mkp_aco_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["n_items"] == 100
    assert metadata["n_dimensions"] == 5
    assert dataset["prizes"].shape == (5, 100)
    assert dataset["weights"].shape == (5, 100, 5)


def test_mkp_aco_evaluates_seed_heuristic_quickly():
    evaluator = MKPACOEvaluation(split="train", n_ants=2, n_iterations=1)

    score = evaluator.evaluate_program("_", prize_weight_ratio)

    assert isinstance(score, float)
    assert score > 0
