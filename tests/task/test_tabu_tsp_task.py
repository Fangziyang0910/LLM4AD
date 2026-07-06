from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.tabu_tsp.dataset import load_split_instances
from llm4ad.task.optimization.tabu_tsp.evaluation import TabuTSPEvaluation


def best_non_tabu(
        delta_costs: np.ndarray,
        is_tabu_mask: np.ndarray,
        best_cost: float,
        current_cost: float,
        tabu_ages: np.ndarray,
        iteration: int,
        max_iterations: int,
) -> np.ndarray:
    scores = np.full(len(delta_costs), -np.inf)
    non_tabu = ~is_tabu_mask
    scores[non_tabu] = -delta_costs[non_tabu]
    aspiration = is_tabu_mask & (current_cost + delta_costs < best_cost)
    scores[aspiration] = -delta_costs[aspiration] + 1e6
    return scores


def forbid_all(
        delta_costs: np.ndarray,
        is_tabu_mask: np.ndarray,
        best_cost: float,
        current_cost: float,
        tabu_ages: np.ndarray,
        iteration: int,
        max_iterations: int,
) -> np.ndarray:
    return np.full(len(delta_costs), -np.inf)


def test_tabu_tsp_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "tabu_tsp_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["n_iter"] == 200
    assert metadata["n_runs"] == 3
    assert len(instances) == 5
    assert instances[0]["n_nodes"] == 20
    assert instances[0]["coordinates"].shape == (20, 2)
    assert instances[0]["distances"].shape == (20, 20)


def test_tabu_tsp_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 20
    assert metadata["n_iter"] == 500
    assert metadata["n_runs"] == 10
    assert len(instances) == 20
    assert {instance["n_nodes"] for instance in instances} == {20, 30}
    assert [group["seed"] for group in metadata["groups"]] == [100, 200]


def test_tabu_tsp_evaluates_baseline_quickly():
    evaluator = TabuTSPEvaluation(split="train", n_iter=5, n_runs=1)

    score = evaluator.evaluate_program("_", best_non_tabu)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score < 0.0


def test_tabu_tsp_handles_all_forbidden_moves_like_source():
    evaluator = TabuTSPEvaluation(split="train", n_iter=5, n_runs=1)

    score = evaluator.evaluate_program("_", forbid_all)

    assert isinstance(score, float)
    assert np.isfinite(score)
