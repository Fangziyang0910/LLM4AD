from __future__ import annotations

import numpy as np

from llm4ad.task.optimization.dpp_ga.dataset import load_split_instances
from llm4ad.task.optimization.dpp_ga.evaluation import DPPGAEvaluation, seed_crossover


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
