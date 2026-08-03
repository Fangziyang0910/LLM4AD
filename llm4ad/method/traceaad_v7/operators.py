"""Semantic operators and uniform V7 operator sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import OperatorName


def classify_outcome(delta: float | None, positive_threshold: float = 0.0) -> str:
    if delta is None:
        return "unknown"
    if delta > positive_threshold:
        return "improve"
    if delta < -positive_threshold:
        return "regress"
    return "plateau"


@dataclass(frozen=True, slots=True)
class Operator:
    name: OperatorName
    prompt_constraint: str


TRACE_IDEATE = Operator(
    OperatorName.IDEATE,
    (
        "For each action, propose a genuinely new algorithmic idea grounded in the "
        "retained trajectory history. Treat later regressions and plateaus as tested "
        "boundaries while changing the primary program along a new direction."
    ),
)
TRACE_REFINE = Operator(
    OperatorName.REFINE,
    (
        "For each action, make one focused, evidence-grounded refinement to a "
        "mechanism that has shown value or to a weakness exposed by the history."
    ),
)
TRACE_SYNTHESIZE = Operator(
    OperatorName.SYNTHESIZE,
    (
        "For each action, identify a supported principle in both the primary and "
        "reference trajectories, then make the two principles interact functionally "
        "in the primary program. Do not concatenate or copy whole implementations."
    ),
)
TRACE_TRANSFER = Operator(
    OperatorName.TRANSFER,
    (
        "For each action, keep the primary program's core structure and adapt exactly "
        "one supported idea from the reference trajectory to the primary task logic "
        "and tested history."
    ),
)


DEFAULT_OPERATORS: tuple[Operator, ...] = (
    TRACE_IDEATE,
    TRACE_REFINE,
    TRACE_SYNTHESIZE,
    TRACE_TRANSFER,
)

DUAL_OPERATORS = frozenset({OperatorName.SYNTHESIZE, OperatorName.TRANSFER})
SINGLE_OPERATORS = frozenset({OperatorName.IDEATE, OperatorName.REFINE})


@dataclass(frozen=True, slots=True)
class OperatorDecision:
    operator: Operator
    use_dual: bool
    reason: str


def select_operator(
    *,
    operators: tuple[Operator, ...],
    allow_dual: bool,
    rng: random.Random,
) -> OperatorDecision:
    """Sample uniformly from the operators available in the current state."""
    available = tuple(
        operator
        for operator in operators
        if allow_dual or operator.name in SINGLE_OPERATORS
    )
    if not available:
        raise ValueError("at least one available TraceAAD operator is required")
    operator = available[rng.randrange(len(available))]
    return OperatorDecision(
        operator=operator,
        use_dual=operator.name in DUAL_OPERATORS,
        reason="all_active_routes" if allow_dual else "single_active_route",
    )


def is_dual_operator(name: OperatorName | str) -> bool:
    return OperatorName(name) in DUAL_OPERATORS


__all__ = [
    "DEFAULT_OPERATORS",
    "DUAL_OPERATORS",
    "Operator",
    "OperatorDecision",
    "SINGLE_OPERATORS",
    "TRACE_IDEATE",
    "TRACE_REFINE",
    "TRACE_SYNTHESIZE",
    "TRACE_TRANSFER",
    "classify_outcome",
    "is_dual_operator",
    "select_operator",
]
