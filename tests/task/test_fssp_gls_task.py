from __future__ import annotations

import numpy as np
import pytest

from llm4ad.task.optimization.main.fssp_gls.dataset import load_split_instances
from llm4ad.task.optimization.main.fssp_gls.evaluation import FSSPGLSEvaluation

SIMPLE_PERTURBATION_PROGRAM = """
import numpy as np

def simple_perturbation(current_sequence: list, time_matrix: np.ndarray, m: int, n: int):
    job_loads = np.sum(time_matrix, axis=1)
    jobs = np.argsort(job_loads)[-min(3, n):].tolist()
    return time_matrix.copy(), jobs
"""


def simple_perturbation(current_sequence: list, time_matrix: np.ndarray, m: int, n: int):
    job_loads = np.sum(time_matrix, axis=1)
    jobs = np.argsort(job_loads)[-min(3, n):].tolist()
    return time_matrix.copy(), jobs


def test_fssp_gls_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "fssp_gls_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 3
    assert len(instances) == 3
    assert instances[0]["n_jobs"] == 50
    assert np.asarray(instances[0]["processing_times"]).shape[0] == 50


def test_fssp_gls_evaluates_simple_heuristic_quickly():
    evaluator = FSSPGLSEvaluation(split="train", time_max=0.001, iter_max=1)

    score = evaluator.evaluate_program("_", simple_perturbation)

    assert isinstance(score, float)
    assert score < 0


def test_fssp_gls_process_parallel_matches_sequential():
    sequential = FSSPGLSEvaluation(split="train", time_max=0.0, iter_max=0)
    parallel = FSSPGLSEvaluation(
        split="train",
        time_max=0.0,
        iter_max=0,
        eval_workers=2,
        eval_backend="process",
    )

    expected = sequential.evaluate_program(SIMPLE_PERTURBATION_PROGRAM, None)
    actual = parallel.evaluate_program(SIMPLE_PERTURBATION_PROGRAM, None)

    assert actual == pytest.approx(expected)
