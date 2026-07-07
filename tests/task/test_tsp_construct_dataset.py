from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.tsp_construct.dataset import (
    DATA_DIR,
    DEFAULT_DATASET_ID,
    load_manifest,
    load_split_instances,
)
from llm4ad.task.optimization.main.tsp_construct.evaluation import TSPEvaluation


def test_default_tsp_evaluation_uses_fixed_train_split():
    evaluator = TSPEvaluation()

    assert evaluator.dataset_metadata["dataset_id"] == DEFAULT_DATASET_ID
    assert evaluator.dataset_metadata["split"] == "train"
    assert evaluator.n_instance == 64
    assert evaluator.problem_size == 50
    assert len(evaluator._datasets) == 64


def test_tsp_dataset_splits_are_persisted_with_expected_shapes():
    manifest = load_manifest()
    assert set(manifest["splits"]) == {"train", "test_id", "test_ood_100", "test_ood_200"}

    expected_shapes = {
        "train": (64, 50, 2),
        "test_id": (250, 50, 2),
        "test_ood_100": (250, 100, 2),
        "test_ood_200": (250, 200, 2),
    }
    for split, shape in expected_shapes.items():
        split_info = manifest["splits"][split]
        with np.load(DATA_DIR / split_info["filename"]) as data:
            coordinates = data["coordinates"]
        instances, metadata = load_split_instances(split)

        assert coordinates.shape == shape
        assert len(instances) == shape[0]
        assert metadata["n_instances"] == shape[0]
        assert metadata["problem_size"] == shape[1]


def test_tsp_evaluation_can_select_test_split():
    evaluator = TSPEvaluation(split="test_ood_100")

    assert evaluator.dataset_metadata["split"] == "test_ood_100"
    assert evaluator.dataset_metadata["role"] == "test"
    assert evaluator.n_instance == 250
    assert evaluator.problem_size == 100


def test_tsp_evaluation_rejects_legacy_generated_mode():
    with pytest.raises(TypeError):
        TSPEvaluation(n_instance=16, problem_size=50)
