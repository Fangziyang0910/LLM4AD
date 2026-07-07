from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.main.circle_packing.dataset import load_split_instances
from llm4ad.task.optimization.main.circle_packing.evaluation import CirclePackingEvaluation


def grid_packing():
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 5)
    centers = np.array([[x, y] for y in ys for x in xs], dtype=float)[:26]
    radii = np.full(26, 0.06, dtype=float)
    return centers, radii


def overlapping_packing():
    centers, radii = grid_packing()
    centers[1] = centers[0]
    return centers, radii


def test_circle_packing_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "circle_packing_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 1
    assert instances[0]["n_circles"] == 26
    assert instances[0]["square_size"] == 1.0


def test_circle_packing_evaluates_valid_packing():
    evaluator = CirclePackingEvaluation(split="train")

    score = evaluator.evaluate_program("_", grid_packing)

    assert score == np.sum(np.full(26, 0.06))


def test_circle_packing_rejects_invalid_overlap():
    evaluator = CirclePackingEvaluation(split="train")

    assert evaluator.evaluate_program("_", overlapping_packing) is None
