"""Endpoint Refine —— exploitation（design §4.1）。"""
from __future__ import annotations

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp


class EndpointRefineOp(_ExtendFromEndpointOp):
    name = OperatorName.ENDPOINT
    role = "exploit"

    def trigger(self, ctx: OperatorContext) -> bool:
        # 图中节点均已评估合法；可行性上始终可 refine。
        return True

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "Continue refining the current best direction. Propose ONE targeted modification that "
            "strengthens the mechanism which recently improved fitness. Use the elite curriculum as "
            "observed evidence, but do not copy a whole historical trace. Avoid directions that regressed."
        )
