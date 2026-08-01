"""TraceAAD v4 单父代语义算子与有向适应度变化。"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import OperatorName


@dataclass(frozen=True, slots=True)
class Operator:
    name: str
    constraint: str


DEFAULT_OPERATORS: tuple[Operator, ...] = (
    Operator(
        name=OperatorName.IDEATE,
        constraint=(
            "Propose one genuinely new algorithmic idea grounded in the full history. "
            "Use later regressions and plateaus as tested boundaries."
        ),
    ),
    Operator(
        name=OperatorName.REFINE,
        constraint=(
            "Preserve one valuable idea already present in the history and make one focused "
            "mechanism or parameter refinement."
        ),
    ),
)


def directed_delta(
    parent_fitness: float | None, child_fitness: float | None, maximize: bool
) -> float | None:
    if parent_fitness is None or child_fitness is None:
        return None
    return (
        child_fitness - parent_fitness if maximize else parent_fitness - child_fitness
    )


def classify_outcome(delta: float | None, positive_threshold: float = 1e-6) -> str:
    if delta is None:
        return "unknown"
    if delta > positive_threshold:
        return "improve"
    if delta < -positive_threshold:
        return "regress"
    return "plateau"


__all__ = [
    "Operator",
    "DEFAULT_OPERATORS",
    "directed_delta",
    "classify_outcome",
]
