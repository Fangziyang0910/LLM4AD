"""从当前轨迹终点继续改进。"""
from __future__ import annotations

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp


class EndpointRefineOp(_ExtendFromEndpointOp):
    name = OperatorName.ENDPOINT

    def trigger(self, ctx: OperatorContext) -> bool:
        # 图中节点均已评估合法；可行性上始终可 refine。
        return True

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "Continue refining the current best direction. Propose ONE targeted modification that "
            "strengthens the mechanism which recently improved fitness. Use the recorded trajectory "
            "as evidence and avoid directions that regressed."
        )
