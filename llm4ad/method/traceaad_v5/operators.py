"""The four semantic operators used by TraceAAD V5."""

from __future__ import annotations

from .schema import OperatorName


def classify_outcome(delta: float | None, positive_threshold: float = 1e-6) -> str:
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
        "For each action, propose a genuinely new algorithmic idea grounded in the "
        "retained trajectory history. Use later regressions and plateaus as tested "
        "boundaries while changing the primary program along a new direction."
    )


class TraceRefineOp(Operator):
    name = OperatorName.REFINE
    prompt_constraint = (
        "For each action, make one focused, evidence-grounded refinement to a "
        "mechanism that has shown value or to a weakness exposed by the history."
    )


class TraceSynthesizeOp(Operator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = (
        "For each action, identify a supported principle in both the primary and "
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


__all__ = [
    "DEFAULT_OPERATORS",
    "Operator",
    "TraceIdeateOp",
    "TraceRefineOp",
    "TraceSynthesizeOp",
    "TraceTransferOp",
    "classify_outcome",
]
