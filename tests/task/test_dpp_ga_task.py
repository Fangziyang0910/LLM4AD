from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.main.dpp_ga.dataset import load_split_instances
from llm4ad.task.optimization.main.dpp_ga.evaluation import DPPGAEvaluation, seed_crossover

SEED_CROSSOVER_PROGRAM = """
import numpy as np

def seed_crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    n_parents, n_decap = parents.shape
    left_halves = parents[:, :n_decap // 2]
    right_halves = parents[:, n_decap // 2:]
    parent_pairs = np.stack([
        np.random.choice(range(n_parents), 2, replace=False)
        for _ in range(n_pop)
    ])
    return np.concatenate([
        left_halves[parent_pairs[:, 0]],
        right_halves[parent_pairs[:, 1]],
    ], axis=1)
"""


def bad_crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    return np.zeros((n_pop, 1), dtype=int)


def test_dpp_ga_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "dpp_ga_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 3
    assert metadata["parameters"]["n_decap"] == 20
    assert len(instances) == 3
    assert {"probe", "keepout", "keepout_num"} <= instances[0].keys()


def test_dpp_ga_evaluates_seed_crossover_quickly():
    evaluator = DPPGAEvaluation(
        split="train",
        n_pop=4,
        n_iter=1,
        elite_rate=0.25,
        max_instances=1,
    )

    score = evaluator.evaluate_program("_", seed_crossover)

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_dpp_ga_rejects_bad_crossover_shape():
    evaluator = DPPGAEvaluation(
        split="train",
        n_pop=4,
        n_iter=1,
        elite_rate=0.25,
        max_instances=1,
    )

    assert evaluator.evaluate_program("_", bad_crossover) is None


def test_dpp_ga_process_parallel_smoke():
    evaluator = DPPGAEvaluation(
        split="train",
        n_pop=4,
        n_iter=1,
        elite_rate=0.25,
        max_instances=2,
        eval_workers=2,
        eval_backend="process",
    )

    score = evaluator.evaluate_program(SEED_CROSSOVER_PROGRAM, None)

    assert isinstance(score, float)
    assert np.isfinite(score)
