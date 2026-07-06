from __future__ import annotations

import numpy as np

from llm4ad.task.machine_learning.game_2048.evaluation import (
    Action,
    Game2048Evaluation,
    play_2048,
)


def left_policy(board: np.ndarray) -> int:
    return 2


def enum_policy(board: np.ndarray) -> Action:
    return Action.LEFT


def bad_policy(board: np.ndarray) -> int:
    return 99


def test_game_2048_fixed_seed_is_deterministic():
    first = play_2048(left_policy, seed=42, max_steps=64)
    second = play_2048(left_policy, seed=42, max_steps=64)

    assert first == second
    assert first[1] <= 64


def test_game_2048_evaluates_integer_policy():
    evaluator = Game2048Evaluation(max_steps=64)

    score = evaluator.evaluate_program("_", left_policy)

    assert isinstance(score, float)
    assert np.isfinite(score)
    assert evaluator.dataset_metadata["seed"] == 42


def test_game_2048_accepts_action_enum_policy():
    evaluator = Game2048Evaluation(max_steps=64)

    score = evaluator.evaluate_program("_", enum_policy)

    assert isinstance(score, float)


def test_game_2048_rejects_invalid_action():
    evaluator = Game2048Evaluation(max_steps=64)

    assert evaluator.evaluate_program("_", bad_policy) is None
