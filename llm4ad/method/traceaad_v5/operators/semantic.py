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
        "Propose one genuinely new algorithmic idea grounded in the full history. "
        "Use later regressions and plateaus as tested boundaries."
    )


class TraceRefineOp(SemanticOperator):
    name = OperatorName.REFINE
    prompt_constraint = (
        "Make one evidence-grounded refinement. You may deepen, repair, replace, "
        "delete, merge, or simplify existing logic; do not default to adding branches."
    )


class TraceSynthesizeOp(SemanticOperator):
    name = OperatorName.SYNTHESIZE
    prompt_constraint = (
        "Extract one supported principle from each trajectory and make them interact "
        "functionally in the primary program. Do not concatenate two implementations."
    )


class TraceTransferOp(SemanticOperator):
    name = OperatorName.TRANSFER
    prompt_constraint = (
        "Keep the primary program's core structure and adapt exactly one supported "
        "idea from the reference trajectory."
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
