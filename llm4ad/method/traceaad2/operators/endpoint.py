"""Endpoint Refine —— exploitation（design §4.1）。"""
from __future__ import annotations

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp, trajectory_step_outcomes


class EndpointRefineOp(_ExtendFromEndpointOp):
    name = OperatorName.ENDPOINT
    role = "exploit"

    def trigger(self, ctx: OperatorContext) -> bool:
        node = ctx.graph.get_node(ctx.selected.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return False
        outcomes = trajectory_step_outcomes(ctx.graph, ctx.selected, ctx.maximize, ctx.positive_threshold)
        if not outcomes:
            return True  # length-1 轨迹，默认可继续
        # 最近步是 improve/plateau 可继续 exploit；regress 让位给 backtrack
        return outcomes[-1][3] != "regress"

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "Continue refining the current best direction. Propose ONE targeted modification that "
            "strengthens the mechanism which recently improved fitness. Avoid directions that regressed."
        )
