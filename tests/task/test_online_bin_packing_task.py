from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.online_bin_packing.dataset import load_split_instances
from llm4ad.task.optimization.main.online_bin_packing.evaluation import OBPEvaluation

PRIORITY_PROGRAM = """
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -bins
"""


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -bins


def test_online_bin_packing_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "online_bin_packing_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 4
    assert len(instances) == 4


def test_online_bin_packing_evaluates_best_fit():
    evaluator = OBPEvaluation(split="train")

    score = evaluator.evaluate_program("_", priority)

    assert isinstance(score, float)
    assert score < 0


def test_online_bin_packing_process_parallel_matches_sequential():
    sequential = OBPEvaluation(split="train")
    parallel = OBPEvaluation(split="train", eval_workers=2, eval_backend="process")

    expected = sequential.evaluate_program(PRIORITY_PROGRAM, None)
    actual = parallel.evaluate_program(PRIORITY_PROGRAM, None)

    assert actual == pytest.approx(expected)
