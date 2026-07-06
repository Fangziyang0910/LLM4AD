from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.evo_dynamic.dataset import load_split_instances
from llm4ad.task.optimization.evo_dynamic.evaluation import EvoDynamicEvaluation


def hypermutation(
        population: np.ndarray,
        fitness: np.ndarray,
        best_position: np.ndarray,
        bounds: np.ndarray,
) -> np.ndarray:
    sigma = (bounds[1] - bounds[0]).mean() * 0.1
    new_population = population + np.random.normal(0.0, sigma, population.shape)
    return np.clip(new_population, bounds[0], bounds[1])


def invalid_shape(
        population: np.ndarray,
        fitness: np.ndarray,
        best_position: np.ndarray,
        bounds: np.ndarray,
) -> np.ndarray:
    return population[0]


def test_evo_dynamic_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "evo_dynamic_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["pop_size"] == 30
    assert metadata["k_iter"] == 30
    assert metadata["run_seed_mode"] == "global"
    assert metadata["run_seed"] == 42
    assert len(instances) == 5
    assert instances[0]["n_dims"] == 10
    assert len(instances[0]["trajectory"]) == 10
    assert instances[0]["trajectory"][0].shape == (10,)


def test_evo_dynamic_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 64
    assert metadata["pop_size"] == 30
    assert metadata["k_iter"] == 50
    assert metadata["run_seed_mode"] == "local_id"
    assert len(instances) == 64
    assert {instance["n_dims"] for instance in instances} == {10, 20}
    assert {instance["n_changes"] for instance in instances} == {15}


def test_evo_dynamic_evaluates_hypermutation_quickly():
    evaluator = EvoDynamicEvaluation(split="train", pop_size=8, k_iter=2)

    score = evaluator.evaluate_program("_", hypermutation)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score < 0.0


def test_evo_dynamic_rejects_invalid_population_shape():
    evaluator = EvoDynamicEvaluation(split="train", pop_size=8, k_iter=2)

    assert evaluator.evaluate_program("_", invalid_shape) is None
