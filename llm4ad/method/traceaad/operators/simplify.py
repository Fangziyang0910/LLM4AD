"""Simplify —— complexity control（design §6.4）。

降低代码复杂度而不降低适应度。可行性：活跃池能做相对复杂度比较，且当前终点相对偏复杂。
"""
from __future__ import annotations

import math
import statistics

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp


class SimplifyOp(_ExtendFromEndpointOp):
    name = OperatorName.SIMPLIFY
    role = "simplify"

    def trigger(self, ctx: OperatorContext) -> bool:
        node = ctx.graph.get_node(ctx.selected.endpoint_id)
        endpoint_complexities = {
            t.endpoint_id: ctx.graph.get_node(t.endpoint_id).complexity
            for t in ctx.memory.active()
            if ctx.graph.get_node(t.endpoint_id).fitness is not None
        }
        complexities = sorted(endpoint_complexities.values())
        if len(complexities) < 2 or node.complexity <= 0:
            return False
        upper_quartile = complexities[math.ceil(0.75 * len(complexities)) - 1]
        return (
            node.complexity >= upper_quartile
            and node.complexity > statistics.median(complexities)
        )

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "Simplify the program: remove low-contribution code while preserving the core ideas that "
            "drive fitness. Do not decrease fitness; prefer a shorter, clearer implementation."
        )
