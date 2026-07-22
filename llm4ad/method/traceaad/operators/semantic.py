"""TraceAAD v4 的语义算子协议。"""
from __future__ import annotations

from ..schema import NodeId, OperatorName, Trajectory
from .base import Operator, OperatorContext


class SemanticOperator(Operator):
    requires_second: bool = False
    prompt_name: str = ""

    def trigger(self, ctx: OperatorContext) -> bool:
        return not self.requires_second or len(ctx.memory.active()) >= 2

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        return ctx.selected.endpoint_id, "endpoint_or_best"

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return self.prompt_constraint

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        assert base_node_id is not None
        return ctx.memory.branch_from(
            trajectory_id=ctx.selected.id,
            base_node_id=base_node_id,
            child_id=child_id,
            edge_id=edge_id,
        )


class TraceIdeateOp(SemanticOperator):
    name = OperatorName.IDEATE
    prompt_name = "trace_ideate"
    prompt_constraint = (
        "Propose one genuinely new algorithmic idea grounded in the full history. "
        "Use later regressions and plateaus as tested boundaries."
    )


class TraceRefineOp(SemanticOperator):
    name = OperatorName.REFINE
    prompt_name = "trace_refine"
    prompt_constraint = (
        "Preserve one valuable idea already present in the history and make one focused "
        "mechanism or parameter refinement."
    )


class TraceSynthesizeOp(SemanticOperator):
    name = OperatorName.SYNTHESIZE
    prompt_name = "trace_synthesize"
    requires_second = True
    prompt_constraint = (
        "Extract useful principles from both complete histories and define an explicit "
        "operational relationship that synthesizes them."
    )


class TraceTransferOp(SemanticOperator):
    name = OperatorName.TRANSFER
    prompt_name = "trace_transfer"
    requires_second = True
    prompt_constraint = (
        "Transfer exactly one evidence-backed idea from the second complete history into "
        "the primary algorithm while preserving its main structure."
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
