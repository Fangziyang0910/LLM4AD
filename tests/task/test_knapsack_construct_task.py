from __future__ import annotations

import pytest

from llm4ad.task.optimization.main.knapsack_construct.dataset import load_split_instances
from llm4ad.task.optimization.main.knapsack_construct.evaluation import KnapsackEvaluation

SELECT_NEXT_ITEM_PROGRAM = """
def select_next_item(remaining_capacity: int, remaining_items: list):
    best_item = None
    best_ratio = -1.0
    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            ratio = value / weight
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item
    return best_item
"""


def select_next_item(remaining_capacity: int, remaining_items: list):
    best_item = None
    best_ratio = -1.0
    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            ratio = value / weight
            if ratio > best_ratio:
                best_ratio = ratio
                best_item = item
    return best_item


def test_knapsack_construct_loads_fixed_train_split():
    instances, metadata = load_split_instances("train")

    assert metadata["dataset_id"] == "knapsack_construct_v1"
    assert metadata["split"] == "train"
    assert metadata["n_instances"] == 64
    assert len(instances) == 64


def test_knapsack_construct_evaluates_seed_heuristic():
    evaluator = KnapsackEvaluation(split="train")

    score = evaluator.evaluate_program("_", select_next_item)

    assert isinstance(score, float)
    assert score < 0


def test_knapsack_construct_process_parallel_matches_sequential():
    sequential = KnapsackEvaluation(split="train")
    parallel = KnapsackEvaluation(split="train", eval_workers=4, eval_backend="process")

    expected = sequential.evaluate_program(SELECT_NEXT_ITEM_PROGRAM, None)
    actual = parallel.evaluate_program(SELECT_NEXT_ITEM_PROGRAM, None)

    assert actual == pytest.approx(expected)
