"""Semantic operators and uniform V6 operator sampling."""

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


class Operator:
    name: OperatorName
    prompt_constraint: str


class TraceIdeateOp(Operator):
    name = OperatorName.IDEATE
    prompt_constraint = (
        "Consider a new algorithmic direction grounded in the task and history."
    )


class TraceRefineOp(Operator):
    name = OperatorName.REFINE
    prompt_constraint = "Consider an improvement to the current program grounded in the observed history."


class TraceSynthesizeOp(Operator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = "Consider combining useful ideas shown by the displayed routes."


class TraceTransferOp(Operator):
    name = OperatorName.TRANSFER
    prompt_constraint = "Consider adapting a useful idea from the reference route."


DEFAULT_OPERATORS: tuple[type[Operator], ...] = (
    TraceIdeateOp,
    TraceRefineOp,
    TraceSynthesizeOp,
    TraceTransferOp,
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
    "TraceIdeateOp",
    "TraceRefineOp",
    "TraceSynthesizeOp",
    "TraceTransferOp",
    "classify_outcome",
    "is_dual_operator",
    "select_operator",
]
