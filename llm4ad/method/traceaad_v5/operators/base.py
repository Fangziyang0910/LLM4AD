"""TraceAAD v5 semantic-operator protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod


def classify_outcome(delta: float | None, positive_threshold: float = 1e-6) -> str:
    if delta is None:
        return "unknown"
    if delta > positive_threshold:
        return "improve"
    if delta < -positive_threshold:
        return "regress"
    return "plateau"


class Operator(ABC):
    name: str = ""

    @abstractmethod
    def build_constraint(self) -> str:
        """注入 action prompt 的算子特定约束文本。"""


__all__ = [
    "Operator",
    "classify_outcome",
]
