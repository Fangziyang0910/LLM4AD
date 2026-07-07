from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.bo_acquisition.dataset import load_split_instances
from llm4ad.task.optimization.main.bo_acquisition.evaluation import BOAcquisitionEvaluation

LCB_PROGRAM = """
import numpy as np

def lcb(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    return -mu + 2.0 * sigma
"""


def lcb(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    return -mu + 2.0 * sigma


def test_bo_acquisition_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "bo_acquisition_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 2
    assert [instance["name"] for instance in instances] == ["Branin", "Hartmann3"]


def test_bo_acquisition_evaluates_lcb_quickly():
    evaluator = BOAcquisitionEvaluation(
        split="train",
        n_init=3,
        n_iter=1,
        n_candidates=16,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", lcb)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_bo_acquisition_process_parallel_matches_sequential():
    kwargs = dict(
        split="train",
        n_init=3,
        n_iter=1,
        n_candidates=16,
        n_runs=1,
    )
    sequential = BOAcquisitionEvaluation(**kwargs)
    parallel = BOAcquisitionEvaluation(**kwargs, eval_workers=2, eval_backend="process")

    expected = sequential.evaluate_program(LCB_PROGRAM, None)
    actual = parallel.evaluate_program(LCB_PROGRAM, None)

    assert actual == pytest.approx(expected)
