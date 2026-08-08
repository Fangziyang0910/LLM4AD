"""The four trajectory-semantic operators used by TraceAAD V9.1."""

from __future__ import annotations

from .schema import OperatorName


def classify_outcome(delta: float, positive_threshold: float = 1e-6) -> str:
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
        "Propose a genuinely new algorithmic direction from the formation history. "
        "Use the previously tested direct branches as boundaries and do not repeat them."
    )


class TraceRefineOp(Operator):
    name = OperatorName.REFINE
    prompt_constraint = (
        "Make one focused correction to a mechanism that showed value or to a weakness "
        "exposed by the formation history and direct child attempts."
    )


class TraceSynthesizeOp(Operator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = (
        "Identify one supported principle in each branch and make the two principles "
        "interact functionally in the current program. Do not concatenate implementations."
    )


class TraceTransferOp(Operator):
    name = OperatorName.TRANSFER
    prompt_constraint = (
        "Keep the current program's core structure and adapt exactly one supported idea "
        "from the reference root branch to the current branch's tested history."
    )


DEFAULT_OPERATORS: tuple[type[Operator], ...] = (
    TraceIdeateOp,
    TraceRefineOp,
    TraceSynthesizeOp,
    TraceTransferOp,
)
DUAL_OPERATORS = frozenset({OperatorName.SYNTHESIZE, OperatorName.TRANSFER})


__all__ = [
    "DEFAULT_OPERATORS",
    "DUAL_OPERATORS",
    "Operator",
    "TraceIdeateOp",
    "TraceRefineOp",
    "TraceSynthesizeOp",
    "TraceTransferOp",
    "classify_outcome",
]
