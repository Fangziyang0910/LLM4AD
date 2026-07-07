from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.online_bin_packing_2O.evaluation import OBP_2O_Evaluation

PRIORITY_PROGRAM = """
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -bins
"""


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -bins


def test_online_bin_packing_2o_evaluates_best_fit():
    evaluator = OBP_2O_Evaluation(split="train")

    score = evaluator.evaluate_program("_", priority)

    assert isinstance(score, np.ndarray)
    assert score.shape == (2,)
    assert score[0] < 0
    assert np.isfinite(score).all()


def test_online_bin_packing_2o_process_parallel_matches_first_objective():
    sequential = OBP_2O_Evaluation(split="train")
    parallel = OBP_2O_Evaluation(split="train", eval_workers=2, eval_backend="process")

    expected = sequential.evaluate_program(PRIORITY_PROGRAM, None)
    actual = parallel.evaluate_program(PRIORITY_PROGRAM, None)

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] <= 0
    assert np.isfinite(actual).all()
