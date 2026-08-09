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
        "Make one focused correction to a tested mechanism. Use its recorded result to "
        "preserve what worked and repair one concrete weakness."
    )


class TraceSynthesizeOp(Operator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = (
        "Compare the two complete histories and combine one change from each whose observed "
        "effects are compatible. Make them interact; do not concatenate implementations."
    )


class TraceTransferOp(Operator):
    name = OperatorName.TRANSFER
    prompt_constraint = (
        "Keep the current program's core structure and adapt exactly one previously tested "
        "change from the reference trajectory to the current trajectory's evidence."
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
