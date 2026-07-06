from __future__ import annotations

import random
from enum import Enum
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.machine_learning.game_2048.template import (
    task_description,
    template_program,
)

__all__ = ["Action", "Game2048Evaluation", "play_2048"]


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


def _to_action(action: Any) -> Action:
    if isinstance(action, Action):
        return action
    if isinstance(action, np.integer):
        action = int(action)
    if isinstance(action, int):
        return Action(action)
    raise ValueError(f"Unknown 2048 action: {action!r}")


class Env2048:
    """Minimal deterministic 2048 environment used by the ShinkaEvolve example."""

    def __init__(self, max_steps: int = 2000):
        self.max_steps = int(max_steps)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.current_step = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.board = np.zeros((4, 4), dtype=np.int32)
        self.score = 0
        self.game_over = False
        self.current_step = 0
        self._add_random_tile()
        self._add_random_tile()
        return self.board.copy()

    def step(self, action: Action) -> tuple[np.ndarray, int, bool]:
        if self.game_over:
            return self.board.copy(), 0, True

        self.current_step += 1
        moved = self._execute_move(action)
        reached_2048 = bool(np.any(self.board == 11))
        reward = 1 if reached_2048 else 0

        if moved:
            self._add_random_tile()

        max_steps_exceeded = self.current_step >= self.max_steps
        self.game_over = self._is_game_over() or reached_2048 or max_steps_exceeded
        return self.board.copy(), reward, self.game_over

    @property
    def max_value_reached(self) -> int:
        if np.all(self.board == 0):
            return 0
        return int(2 ** np.max(self.board))

    def _execute_move(self, action: Action) -> bool:
        before = self.board.copy()
        if action == Action.LEFT:
            self._move_left()
        elif action == Action.RIGHT:
            self._move_right()
        elif action == Action.UP:
            self._move_up()
        elif action == Action.DOWN:
            self._move_down()
        return not np.array_equal(before, self.board)

    def _move_left(self) -> None:
        for i in range(4):
            self.board[i] = self._merge_line(self.board[i])

    def _move_right(self) -> None:
        for i in range(4):
            self.board[i] = self._merge_line(self.board[i][::-1])[::-1]

    def _move_up(self) -> None:
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T

    def _move_down(self) -> None:
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T

    def _merge_line(self, line: np.ndarray) -> np.ndarray:
        non_zero = line[line != 0]
        merged = []
        i = 0
        while i < len(non_zero):
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = int(non_zero[i]) + 1
                merged.append(merged_value)
                self.score += 2 ** merged_value
                i += 2
            else:
                merged.append(int(non_zero[i]))
                i += 1

        result = np.zeros(4, dtype=np.int32)
        result[:len(merged)] = merged
        return result

    def _add_random_tile(self) -> None:
        empty_positions = np.argwhere(self.board == 0)
        if len(empty_positions) == 0:
            return
        pos = empty_positions[np.random.randint(len(empty_positions))]
        value = 1 if np.random.random() < 0.9 else 2
        self.board[pos[0], pos[1]] = value

    def _is_game_over(self) -> bool:
        if np.any(self.board == 0):
            return False
        for i in range(4):
            for j in range(3):
                if self.board[i, j] == self.board[i, j + 1]:
                    return False
        for i in range(3):
            for j in range(4):
                if self.board[i, j] == self.board[i + 1, j]:
                    return False
        return True


def play_2048(
        get_best_move: Callable[[np.ndarray], Any],
        *,
        seed: int = 42,
        max_steps: int = 2000,
) -> tuple[int, int, bool, bool]:
    env = Env2048(max_steps=max_steps)
    board = env.reset(seed=seed)
    actions = 0
    done = False
    reward = 0

    while not done and actions < max_steps:
        action = _to_action(get_best_move(board.copy()))
        board, reward, done = env.step(action)
        actions += 1
        if reward > 0:
            break

    reached_2048 = bool(reward)
    reached_max_steps = actions >= max_steps and not reached_2048
    return env.max_value_reached, actions, reached_2048, reached_max_steps


def evaluate(
        get_best_move: Callable[[np.ndarray], Any],
        *,
        seed: int = 42,
        max_steps: int = 2000,
) -> float | None:
    try:
        max_value_reached, num_actions, _, _ = play_2048(
            get_best_move,
            seed=seed,
            max_steps=max_steps,
        )
        return float(max_value_reached / 512 - num_actions * 0.002)
    except Exception:
        return None


class Game2048Evaluation(Evaluation):
    """Evaluator for ShinkaEvolve's 2048 policy-design example."""

    def __init__(self, timeout_seconds=20, seed: int = 42, max_steps: int = 2000, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.dataset_metadata = {
            "dataset_id": "game_2048_fixed_seed_v1",
            "task": "game_2048",
            "split": "fixed",
            "n_instances": 1,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "source": "reference_code/ShinkaEvolve/examples/game_2048",
        }

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return evaluate(callable_func, seed=self.seed, max_steps=self.max_steps)
