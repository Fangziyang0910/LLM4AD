"""Distill / Simplify —— generalization + complexity control（design §4.4）。"""
from __future__ import annotations

import math
import statistics

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp, trajectory_step_outcomes


class DistillSimplifyOp(_ExtendFromEndpointOp):
    name = OperatorName.SIMPLIFY
    role = "simplify"
    stagnation_threshold = 5

    def trigger(self, ctx: OperatorContext) -> bool:
        node = ctx.graph.get_node(ctx.selected.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return False
        endpoint_complexities = {
            t.endpoint_id: ctx.graph.get_node(t.endpoint_id).complexity
            for t in ctx.memory.active()
            if ctx.graph.get_node(t.endpoint_id).is_valid
            and ctx.graph.get_node(t.endpoint_id).fitness is not None
        }
        complexities = sorted(endpoint_complexities.values())
        if len(complexities) < 2 or node.complexity <= 0:
            return False
        upper_quartile = complexities[math.ceil(0.75 * len(complexities)) - 1]
        relatively_complex = (
            node.complexity >= upper_quartile
            and node.complexity > statistics.median(complexities)
        )
        outcomes = trajectory_step_outcomes(ctx.graph, ctx.selected, ctx.maximize, ctx.positive_threshold)
        saturated = bool(outcomes) and outcomes[-1][3] == "plateau"
        stagnated = ctx.best_stagnation >= self.stagnation_threshold
        return relatively_complex and (saturated or stagnated)

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "Simplify: remove redundant or low-contribution code components while keeping the core "
            "mechanisms that produced improvements. Reduce complexity WITHOUT lowering fitness. Preserve "
            "the algorithmic idea, trim boilerplate and dead branches."
        )
