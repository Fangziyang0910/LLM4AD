from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.op_aco.dataset import load_split_instances
from llm4ad.task.optimization.main.op_aco.evaluation import OPACOEvaluation

PRIZE_OVER_DISTANCE_PROGRAM = """
import numpy as np

def prize_over_distance(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return prize[np.newaxis, :] / (distance + 1e-9)
"""


def prize_over_distance(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    return prize[np.newaxis, :] / (distance + 1e-9)


def test_op_aco_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "op_aco_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["problem_size"] == 50
    assert instances.shape == (5, 50, 2)


def test_op_aco_evaluates_seed_heuristic_quickly():
    evaluator = OPACOEvaluation(split="train", n_ants=4, n_iterations=1)

    score = evaluator.evaluate_program("_", prize_over_distance)

    assert isinstance(score, float)
    assert score > 0


def test_op_aco_process_parallel_matches_sequential():
    sequential = OPACOEvaluation(split="train", n_ants=4, n_iterations=1)
    parallel = OPACOEvaluation(
        split="train",
        n_ants=4,
        n_iterations=1,
        eval_workers=2,
        eval_backend="process",
    )

    expected = sequential.evaluate_program(PRIZE_OVER_DISTANCE_PROGRAM, None)
    actual = parallel.evaluate_program(PRIZE_OVER_DISTANCE_PROGRAM, None)

    assert actual == pytest.approx(expected)
