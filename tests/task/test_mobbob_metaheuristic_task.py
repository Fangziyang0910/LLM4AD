from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.mobbob_metaheuristic.dataset import load_split_instances
from llm4ad.task.optimization.mobbob_metaheuristic.evaluation import MoBBOBMetaheuristicEvaluation


def random_front(func, dim: int, bounds: np.ndarray, budget: int, n_obj: int) -> np.ndarray:
    lower, upper = bounds[0], bounds[1]
    xs, fs = [], []
    for _ in range(max(1, budget)):
        x = lower + (upper - lower) * np.random.rand(dim)
        xs.append(x)
        fs.append(func(x))
    X = np.array(xs)
    F = np.array(fs)
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated_by[i] = False
        if dominated_by.any():
            keep[i] = False
    return X[keep]


class RandomFrontClass:
    def __init__(self, func, dim, bounds, budget, n_obj):
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.budget = budget
        self.n_obj = n_obj

    def solve(self):
        return random_front(self.func, self.dim, self.bounds, self.budget, self.n_obj)


def invalid_front(func, dim: int, bounds: np.ndarray, budget: int, n_obj: int) -> np.ndarray:
    return np.empty((0, dim))


def test_mobbob_metaheuristic_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "mobbob_metaheuristic_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 4
    assert metadata["dim"] == 10
    assert metadata["budget"] == 10000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == ["zdt1", "zdt2", "zdt3", "zdt4"]
    assert instances[-1]["bounds"][0].shape == (10,)


def test_mobbob_metaheuristic_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 4
    assert metadata["budget"] == 5000
    assert metadata["n_runs"] == 5
    assert [instance["name"] for instance in instances] == ["zdt1", "zdt2", "zdt3", "zdt4"]


def test_mobbob_metaheuristic_evaluates_function_solver_quickly():
    evaluator = MoBBOBMetaheuristicEvaluation(split="train", budget=10, n_runs=1)

    score = evaluator.evaluate_program("_", random_front)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score >= 0.0


def test_mobbob_metaheuristic_accepts_source_style_class():
    evaluator = MoBBOBMetaheuristicEvaluation(split="train", budget=10, n_runs=1)

    score = evaluator.evaluate_program("_", RandomFrontClass)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_mobbob_metaheuristic_rejects_empty_front():
    evaluator = MoBBOBMetaheuristicEvaluation(split="train", budget=10, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_front) is None
