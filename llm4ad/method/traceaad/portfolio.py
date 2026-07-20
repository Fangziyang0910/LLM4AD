"""根据改法的历史收益和尝试次数选择下一种改法。"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .operators import Operator, OperatorContext


@dataclass
class OperatorStats:
    attempts: int = 0
    total_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.attempts if self.attempts else 0.0


@dataclass(frozen=True)
class PortfolioWeights:
    ucb_c: float = 0.5


@dataclass(frozen=True)
class SelectionDecision:
    operator: Operator
    scores: dict[str, float]
    eligible: tuple[str, ...]


def signed_utility(delta_norm: float) -> float:
    return math.tanh(delta_norm)


def aggregate_batch_utility(utilities: list[float]) -> float:
    if not utilities:
        return -1.0
    return sum(utilities) / len(utilities)


class OperatorPortfolio:
    def __init__(
        self,
        operators: tuple[Operator, ...],
        weights: PortfolioWeights,
    ) -> None:
        self.operators = operators
        self.weights = weights
        self.stats = {op.name: OperatorStats() for op in operators}

    def choose(self, ctx: OperatorContext) -> SelectionDecision:
        candidates = [op for op in self.operators if op.trigger(ctx)]
        if not candidates:
            candidates = [self.operators[0]]
        total_attempts = sum(stats.attempts for stats in self.stats.values())
        scores: dict[str, float] = {}
        for op in candidates:
            stats = self.stats[op.name]
            if stats.attempts == 0:
                score = float("inf")
            else:
                exploration = self.weights.ucb_c * math.sqrt(
                    math.log(total_attempts + 1) / stats.attempts
                )
                score = stats.mean_reward + exploration
            scores[op.name] = score
        selected = max(
            candidates,
            key=lambda op: (scores[op.name], -self.operators.index(op)),
        )
        return SelectionDecision(
            operator=selected,
            scores=scores,
            eligible=tuple(op.name for op in candidates),
        )

    def record(self, op: Operator, reward: float) -> None:
        stats = self.stats[op.name]
        stats.attempts += 1
        stats.total_reward += max(-1.0, min(1.0, reward))

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "attempts": stats.attempts,
                "mean_reward": stats.mean_reward,
            }
            for name, stats in self.stats.items()
        }


__all__ = [
    "OperatorPortfolio",
    "OperatorStats",
    "PortfolioWeights",
    "SelectionDecision",
    "signed_utility",
    "aggregate_batch_utility",
]
