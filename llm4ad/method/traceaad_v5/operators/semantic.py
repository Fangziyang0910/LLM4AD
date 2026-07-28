"""Four independent semantic operators used by TraceAAD v5."""

from __future__ import annotations

from ..schema import OperatorName
from .base import Operator


class SemanticOperator(Operator):
    def build_constraint(self) -> str:
        return self.prompt_constraint


class TraceIdeateOp(SemanticOperator):
    name = OperatorName.IDEATE
    prompt_constraint = (
        "For each action, propose a genuinely new algorithmic idea grounded in the "
        "retained trajectory history. Use later regressions and plateaus as tested "
        "boundaries while changing the primary program along a new direction."
    )


class TraceRefineOp(SemanticOperator):
    name = OperatorName.REFINE
    prompt_constraint = (
        "For each action, make one focused, evidence-grounded refinement to a "
        "mechanism that has shown value or to a weakness exposed by the history."
    )


class TraceSynthesizeOp(SemanticOperator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = (
        "For each action, identify a supported principle in both the primary and "
        "reference trajectories, then make the two principles interact functionally "
        "in the primary program. Do not concatenate or copy whole implementations."
    )


class TraceTransferOp(SemanticOperator):
    name = OperatorName.TRANSFER
    prompt_constraint = (
        "For each action, keep the primary program's core structure and adapt exactly "
        "one supported idea from the reference trajectory to the primary trajectory's "
        "task logic and tested history."
    )


DEFAULT_SEMANTIC_OPERATORS: tuple[type[SemanticOperator], ...] = (
    TraceIdeateOp,
    TraceRefineOp,
    TraceSynthesizeOp,
    TraceTransferOp,
)


__all__ = [
    "SemanticOperator",
    "TraceIdeateOp",
    "TraceRefineOp",
    "TraceSynthesizeOp",
    "TraceTransferOp",
    "DEFAULT_SEMANTIC_OPERATORS",
]
