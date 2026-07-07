from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.other.nsga2_pymoo.dataset import load_split_instances
from llm4ad.task.optimization.other.nsga2_pymoo.evaluation import NSGA2PymooEvaluation


def sbx(x1: np.ndarray, x2: np.ndarray) -> tuple:
    eta = 15.0
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if np.random.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-10:
            u = np.random.random()
            beta = (
                (2 * u) ** (1.0 / (eta + 1))
                if u <= 0.5
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
            )
            c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * abs(x2[i] - x1[i]))
            c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * abs(x2[i] - x1[i]))
    return c1, c2


def invalid_shape(x1: np.ndarray, x2: np.ndarray) -> tuple:
    return x1[:-1], x2[:-1]


def test_nsga2_pymoo_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "nsga2_pymoo_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 2
    assert metadata["pop_size"] == 100
    assert metadata["n_gen"] == 100
    assert metadata["n_runs"] == 3
    assert [instance["name"] for instance in instances] == ["zdt1", "zdt2"]


def test_nsga2_pymoo_loads_posthoc_test_split():
    instances, metadata = load_split_instances("test_full")

    assert metadata["role"] == "test"
    assert metadata["n_instances"] == 3
    assert metadata["pop_size"] == 100
    assert metadata["n_gen"] == 200
    assert metadata["n_runs"] == 10
    assert [instance["name"] for instance in instances] == ["zdt1", "zdt2", "zdt3"]


def test_nsga2_pymoo_evaluates_sbx_quickly():
    evaluator = NSGA2PymooEvaluation(split="train", pop_size=10, n_gen=2, n_runs=1)

    score = evaluator.evaluate_program("_", sbx)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score >= 0.0


def test_nsga2_pymoo_rejects_invalid_shape():
    evaluator = NSGA2PymooEvaluation(split="train", pop_size=10, n_gen=2, n_runs=1)

    assert evaluator.evaluate_program("_", invalid_shape) is None
