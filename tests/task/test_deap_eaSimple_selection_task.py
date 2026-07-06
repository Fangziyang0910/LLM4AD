from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.deap_eaSimple_selection.dataset import load_split_instances
from llm4ad.task.optimization.deap_eaSimple_selection.evaluation import EASimpleSelectionEvaluation


def tournament_select(fitnesses: np.ndarray, k: int, tournament_size: int) -> np.ndarray:
    selected = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(len(fitnesses), tournament_size, replace=False)
        selected[i] = candidates[np.argmin(fitnesses[candidates])]
    return selected


def invalid_select(fitnesses: np.ndarray, k: int, tournament_size: int) -> np.ndarray:
    return np.full(k, len(fitnesses), dtype=int)


def test_deap_ea_simple_selection_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "deap_eaSimple_selection_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["pop_size"] == 50
    assert metadata["n_gen"] == 100
    assert metadata["n_runs"] == 3
    assert len(instances) == 5
    assert {instance["name"] for instance in instances} == {
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    }
    assert all(instance["dim"] == 10 for instance in instances)


def test_deap_ea_simple_selection_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["pop_size"] == 100
    assert metadata["n_gen"] == 200
    assert metadata["n_runs"] == 10
    assert len(instances) == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_deap_ea_simple_selection_evaluates_tournament_quickly():
    evaluator = EASimpleSelectionEvaluation(split="train", pop_size=8, n_gen=2, n_runs=1)

    score = evaluator.evaluate_program("_", tournament_select)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score <= 0.0


def test_deap_ea_simple_selection_rejects_invalid_indices():
    evaluator = EASimpleSelectionEvaluation(split="train", pop_size=8, n_gen=2, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_select) is None
