from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation
from llm4ad.task.optimization.main.tsp_gls_2O.get_instance import TSPInstance

IDENTITY_UPDATE_PROGRAM = """
import numpy as np

def identity_update(
        edge_distance: np.ndarray,
        local_opt_tour: np.ndarray,
        edge_n_used: np.ndarray,
) -> np.ndarray:
    return edge_distance.copy()
"""


def identity_update(
        edge_distance: np.ndarray,
        local_opt_tour: np.ndarray,
        edge_n_used: np.ndarray,
) -> np.ndarray:
    return edge_distance.copy()


def _small_evaluator(**kwargs) -> TSP_GLS_2O_Evaluation:
    evaluator = TSP_GLS_2O_Evaluation(
        split="train",
        perturbation_moves_val=1,
        iter_limit_val=0,
        **kwargs,
    )
    rng = np.random.default_rng(123)
    evaluator._datasets = [
        TSPInstance(rng.random((8, 2))),
        TSPInstance(rng.random((8, 2))),
    ]
    evaluator.n_instance = 2
    evaluator.problem_size = 8
    return evaluator


def test_tsp_gls_2o_evaluates_identity_update():
    evaluator = _small_evaluator()

    score = evaluator.evaluate_program("_", identity_update)

    assert isinstance(score, np.ndarray)
    assert score.shape == (2,)
    assert score[0] < 0
    assert np.isfinite(score).all()


def test_tsp_gls_2o_process_parallel_matches_first_objective():
    sequential = _small_evaluator()
    parallel = _small_evaluator(eval_workers=2, eval_backend="process")

    expected = sequential.evaluate_program(IDENTITY_UPDATE_PROGRAM, None)
    actual = parallel.evaluate_program(IDENTITY_UPDATE_PROGRAM, None)

    assert actual[0] == pytest.approx(expected[0])
    assert actual[1] <= 0
    assert np.isfinite(actual).all()
