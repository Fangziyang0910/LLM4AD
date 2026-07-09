"""Distill / Simplify —— generalization + complexity control（design §4.4）。"""
from __future__ import annotations

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp, trajectory_step_outcomes


class DistillSimplifyOp(_ExtendFromEndpointOp):
    name = OperatorName.SIMPLIFY
    role = "simplify"

    def trigger(self, ctx: OperatorContext) -> bool:
        node = ctx.graph.get_node(ctx.selected.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return False
        outcomes = trajectory_step_outcomes(ctx.graph, ctx.selected, ctx.maximize, ctx.positive_threshold)
        saturated = bool(outcomes) and outcomes[-1][3] == "plateau"
        # 收紧：仅在高复杂度 且（饱和或稳健）时简化，避免在低质 trajectory 上无谓简化
        return node.complexity > 30 and (saturated or node.robustness >= 0.5)

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "Simplify: remove redundant or low-contribution code components while keeping the core "
            "mechanisms that produced improvements. Reduce complexity WITHOUT lowering fitness. Preserve "
            "the algorithmic idea, trim boilerplate and dead branches."
        )
