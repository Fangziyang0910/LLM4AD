from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.cmaes_cov_update.dataset import load_split_instances
from llm4ad.task.optimization.other.cmaes_cov_update.evaluation import CMAESCovUpdateEvaluation


def rank_one_rank_mu(
        C: np.ndarray,
        p_c: np.ndarray,
        weights: np.ndarray,
        y_k: np.ndarray,
        c1: float,
        cmu: float,
        cc: float,
        hsig: float,
        n: int,
) -> np.ndarray:
    rank1 = c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * C)
    rankmu = cmu * np.sum(
        [weights[i] * np.outer(y_k[i], y_k[i]) for i in range(len(weights))],
        axis=0,
    )
    return (1.0 - c1 - cmu) * C + rank1 + rankmu


def invalid_covariance(
        C: np.ndarray,
        p_c: np.ndarray,
        weights: np.ndarray,
        y_k: np.ndarray,
        c1: float,
        cmu: float,
        cc: float,
        hsig: float,
        n: int,
) -> np.ndarray:
    return np.zeros((n,), dtype=float)


def test_cmaes_cov_update_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "cmaes_cov_update_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["max_evals"] == 2000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {10}


def test_cmaes_cov_update_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["max_evals"] == 10000
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_cmaes_cov_update_evaluates_baseline_quickly():
    evaluator = CMAESCovUpdateEvaluation(
        split="train",
        max_evals=20,
        n_runs=1,
    )

    score = evaluator.evaluate_program("_", rank_one_rank_mu)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_cmaes_cov_update_rejects_invalid_shape():
    evaluator = CMAESCovUpdateEvaluation(
        split="train",
        max_evals=20,
        n_runs=1,
    )

    assert evaluator.evaluate_program("_", invalid_covariance) is None
