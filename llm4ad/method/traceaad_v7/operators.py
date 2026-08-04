"""Semantic operators and uniform V7 operator sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import OperatorName, ProgramNode


def classify_outcome(delta: float | None, positive_threshold: float = 0.0) -> str:
    if delta is None:
        return "unknown"
    if delta > positive_threshold:
        return "improve"
    if delta < -positive_threshold:
        return "regress"
    return "plateau"


def classify_program_outcome(
    parent: ProgramNode | None,
    child: ProgramNode,
    *,
    maximize: bool,
    positive_threshold: float = 0.0,
) -> str:
    """Classify the same comparator used for program selection."""
    del positive_threshold  # selection is exact; the threshold is for trend only
    if parent is None:
        return "strict_fitness"
    delta = (
        child.fitness - parent.fitness
        if maximize
        else parent.fitness - child.fitness
    )
    if delta is None:
        return "unknown"
    if delta > 0.0:
        return "strict_fitness"
    if delta < 0.0:
        return "regress"
    if child.fitness == parent.fitness and child.program_loc < parent.program_loc:
        return "tie_shorter"
    return "plateau"


@dataclass(frozen=True, slots=True)
class Operator:
    name: OperatorName
    prompt_constraint: str


TRACE_IDEATE = Operator(
    OperatorName.IDEATE,
    (
        "For each action, propose one concrete algorithmic direction not already tried "
        "in the retained history. A regression or plateau rules out repeating that "
        "specific implementation; it does not rule out the whole underlying idea."
    ),
)
TRACE_REFINE = Operator(
    OperatorName.REFINE,
    (
        "For each action, change exactly one existing mechanism. Tie it to a mechanism "
        "that advanced the route or to one concrete weakness exposed by the history; "
        "state the code-level change rather than a general goal."
    ),
)
TRACE_SYNTHESIZE = Operator(
    OperatorName.SYNTHESIZE,
    (
        "For each action, identify one result-supported principle from the primary "
        "trajectory and one from the distinct reference trajectory, then make them "
        "interact at one concrete program interface. Do not concatenate or copy "
        "whole implementations."
    ),
)
TRACE_TRANSFER = Operator(
    OperatorName.TRANSFER,
    (
        "For each action, keep the primary program's core structure and adapt exactly "
        "one result-supported principle that is absent from the primary program, "
        "using the distinct reference trajectory as its source."
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
    "classify_program_outcome",
    "is_dual_operator",
    "select_operator",
]
