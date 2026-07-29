"""TraceAAD v4 的语义算子协议。"""

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
        "Preserve one valuable idea already present in the history and make one focused "
        "mechanism or parameter refinement."
    )


DEFAULT_SEMANTIC_OPERATORS: tuple[type[SemanticOperator], ...] = (
    TraceIdeateOp,
    TraceRefineOp,
)


__all__ = [
    "SemanticOperator",
    "TraceIdeateOp",
    "TraceRefineOp",
    "DEFAULT_SEMANTIC_OPERATORS",
]
