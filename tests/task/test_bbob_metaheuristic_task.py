from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.bbob_metaheuristic.dataset import load_split_instances
from llm4ad.task.optimization.bbob_metaheuristic.evaluation import BBOBMetaheuristicEvaluation


def random_search(func, dim: int, bounds: np.ndarray, budget: int) -> np.ndarray:
    lower, upper = bounds[0], bounds[1]
    best_x = lower + (upper - lower) * np.random.rand(dim)
    best_f = func(best_x)
    for _ in range(max(0, budget - 1)):
        x = lower + (upper - lower) * np.random.rand(dim)
        fx = func(x)
        if fx < best_f:
            best_x = x
            best_f = fx
    return best_x


class BaselineES:
    def __init__(self, func, dim, bounds, budget):
        self.func = func
        self.dim = dim
        self.lower = bounds[0].copy()
        self.upper = bounds[1].copy()
        self.budget = budget

    def solve(self):
        x_parent = self.lower + (self.upper - self.lower) * np.random.rand(self.dim)
        f_parent = self.func(x_parent)
        sigma = float((self.upper - self.lower).mean()) / 4.0
        for _ in range(max(0, self.budget - 1)):
            x = np.clip(x_parent + np.random.randn(self.dim) * sigma, self.lower, self.upper)
            fx = self.func(x)
            if fx < f_parent:
                x_parent = x
                f_parent = fx
        return x_parent


def invalid_solution(func, dim: int, bounds: np.ndarray, budget: int) -> np.ndarray:
    return np.zeros(dim + 1, dtype=float)


def test_bbob_metaheuristic_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "bbob_metaheuristic_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 5
    assert metadata["budget"] == 1000
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == [
        "sphere",
        "rastrigin",
        "ackley",
        "rosenbrock",
        "griewank",
    ]
    assert {instance["dim"] for instance in instances} == {10}


def test_bbob_metaheuristic_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 10
    assert metadata["budget"] == 10000
    assert metadata["n_runs"] == 10
    assert {instance["dim"] for instance in instances} == {10, 20}


def test_bbob_metaheuristic_evaluates_function_solver_quickly():
    evaluator = BBOBMetaheuristicEvaluation(split="train", budget=10, n_runs=1)

    score = evaluator.evaluate_program("_", random_search)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_bbob_metaheuristic_accepts_source_style_class():
    evaluator = BBOBMetaheuristicEvaluation(split="train", budget=10, n_runs=1)

    score = evaluator.evaluate_program("_", BaselineES)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_bbob_metaheuristic_rejects_invalid_solution_shape():
    evaluator = BBOBMetaheuristicEvaluation(split="train", budget=10, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_solution) is None
