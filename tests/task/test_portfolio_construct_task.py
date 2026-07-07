from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.portfolio_construct.dataset import load_split_instances
from llm4ad.task.optimization.other.portfolio_construct.evaluation import PortfolioConstructEvaluation


def individual_sharpe(
        asset_returns: np.ndarray,
        selected_indices: np.ndarray,
        candidate_indices: np.ndarray,
) -> np.ndarray:
    return np.array([
        asset_returns[i].mean() / (asset_returns[i].std() + 1e-8)
        for i in candidate_indices
    ])


def invalid_shape(
        asset_returns: np.ndarray,
        selected_indices: np.ndarray,
        candidate_indices: np.ndarray,
) -> np.ndarray:
    return np.zeros(len(candidate_indices) + 1, dtype=float)


def invalid_nonfinite(
        asset_returns: np.ndarray,
        selected_indices: np.ndarray,
        candidate_indices: np.ndarray,
) -> np.ndarray:
    return np.full(len(candidate_indices), np.nan, dtype=float)


def test_portfolio_construct_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "portfolio_construct_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["n_assets"] == 20
    assert metadata["n_select"] == 5
    assert metadata["n_periods"] == 252
    assert len(instances) == 5
    assert instances[0]["asset_returns"].shape == (20, 252)


def test_portfolio_construct_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 16
    assert metadata["n_assets"] == 20
    assert metadata["n_select"] == 5
    assert metadata["n_periods"] == 252
    assert len(instances) == 16
    assert instances[0]["asset_returns"].shape == (20, 252)


def test_portfolio_construct_evaluates_individual_sharpe():
    evaluator = PortfolioConstructEvaluation(split="train")

    score = evaluator.evaluate_program("_", individual_sharpe)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_portfolio_construct_rejects_invalid_scores():
    evaluator = PortfolioConstructEvaluation(split="train")

    assert evaluator.evaluate_program("_", invalid_shape) is None
    assert evaluator.evaluate_program("_", invalid_nonfinite) is None
