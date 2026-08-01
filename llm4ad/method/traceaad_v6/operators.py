"""Semantic operators and V6 operator scheduling."""

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
        "For each action, propose one genuinely new algorithmic direction grounded "
        "in the task structure and outside the retained history. Use later failures "
        "and plateaus only as tested boundaries while changing the primary program."
    )


class TraceRefineOp(Operator):
    name = OperatorName.REFINE
    prompt_constraint = (
        "For each action, make one focused, evidence-grounded refinement to a "
        "mechanism that has shown value or to a weakness exposed by the history. "
        "You may add, replace, merge, or delete code."
    )


REFINE_TRIM_CONSTRAINT = (
    "Nearby attempts increased LOC without route progress. Prefer deleting, "
    "replacing, or merging unsupported mechanisms rather than adding more code. "
    "Make one focused, evidence-grounded refinement."
)


class TraceSynthesizeOp(Operator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = (
        "For each action, take one supported principle from each of the primary and "
        "reference trajectories, then make the two principles interact functionally "
        "in the primary program. Do not concatenate or copy whole implementations."
    )


class TraceTransferOp(Operator):
    name = OperatorName.TRANSFER
    prompt_constraint = (
        "For each action, keep the primary program's core structure and adapt exactly "
        "one supported idea from the reference trajectory to the primary trajectory's "
        "task logic and tested history."
    )


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


def recent_route_progress(
    *,
    edge_route_improvements: tuple[bool, ...],
    window: int = 3,
) -> bool:
    if not edge_route_improvements:
        return False
    return any(edge_route_improvements[-window:])


def select_operator(
    *,
    operators: tuple[Operator, ...],
    mature: bool,
    has_qualified_reference: bool,
    anchor_role: str,
    recent_progress: bool,
    prefer_trim_refine: bool,
    rng: random.Random,
    dual_probability: float = 0.25,
) -> OperatorDecision:
    by_name = {operator.name: operator for operator in operators}
    use_dual = (
        mature
        and has_qualified_reference
        and OperatorName.TRANSFER in by_name
        and OperatorName.SYNTHESIZE in by_name
        and rng.random() < dual_probability
    )
    if use_dual:
        name = rng.choice((OperatorName.TRANSFER, OperatorName.SYNTHESIZE))
        return OperatorDecision(
            operator=by_name[name],
            use_dual=True,
            reason="mature_with_reference",
        )

    use_endpoint_progress = anchor_role.startswith("endpoint") and recent_progress
    if use_endpoint_progress:
        name = rng.choices(
            (OperatorName.REFINE, OperatorName.IDEATE),
            weights=(2, 1),
            k=1,
        )[0]
        reason = "endpoint_with_progress"
    else:
        name = rng.choices(
            (OperatorName.REFINE, OperatorName.IDEATE),
            weights=(1, 2),
            k=1,
        )[0]
        reason = (
            "compact_best_restart"
            if anchor_role == "compact_best"
            else "no_recent_progress"
        )
    operator = by_name[name]
    if name == OperatorName.REFINE and prefer_trim_refine:
        operator = _RefineTrimOp()
        reason = f"{reason}_trim"
    return OperatorDecision(operator=operator, use_dual=False, reason=reason)


class _RefineTrimOp(Operator):
    name = OperatorName.REFINE
    prompt_constraint = REFINE_TRIM_CONSTRAINT


def is_dual_operator(name: OperatorName | str) -> bool:
    return OperatorName(name) in DUAL_OPERATORS


__all__ = [
    "DEFAULT_OPERATORS",
    "DUAL_OPERATORS",
    "Operator",
    "OperatorDecision",
    "REFINE_TRIM_CONSTRAINT",
    "SINGLE_OPERATORS",
    "TraceIdeateOp",
    "TraceRefineOp",
    "TraceSynthesizeOp",
    "TraceTransferOp",
    "classify_outcome",
    "is_dual_operator",
    "recent_route_progress",
    "select_operator",
]
