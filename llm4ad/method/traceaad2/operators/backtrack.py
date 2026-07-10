"""Backtrack Branch —— path correction（design §4.2，优化版：独立选题）。

不再依赖被 selection 选中的 trajectory（它偏好高潜力、近期改进的，最后一步多为 improve，
导致 backtrack 永不触发）。改为主动从 pool 里挑「endpoint 退步/饱和，但内部前缀 branch_score 高」
的 trajectory 来 backtrack——让 path correction 真正运转。
"""
from __future__ import annotations

from ..schema import NodeId, OperatorName, Trajectory
from .base import (
    Operator,
    OperatorContext,
    branch_score,
    select_base_node,
    trajectory_step_outcomes,
)


class BacktrackBranchOp(Operator):
    name = OperatorName.BACKTRACK
    role = "path_correct"

    def _candidates(self, ctx: OperatorContext) -> list[tuple[Trajectory, float]]:
        out: list[tuple[Trajectory, float]] = []
        for t in ctx.memory.active():
            if not t.edge_ids:
                continue
            outcomes = trajectory_step_outcomes(ctx.graph, t, ctx.maximize, ctx.positive_threshold)
            if outcomes and outcomes[-1][3] in ("regress", "plateau"):
                base_id, _ = select_base_node(
                    graph=ctx.graph,
                    trajectory=t,
                    maximize=ctx.maximize,
                    positive_threshold=ctx.positive_threshold,
                )
                if base_id != t.endpoint_id:
                    out.append((t, branch_score(ctx.graph, t, base_id, ctx.maximize)))
        return out

    def select_trajectory(self, ctx: OperatorContext) -> Trajectory | None:
        cands = self._candidates(ctx)
        if not cands:
            return None
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[0][0]

    def trigger(self, ctx: OperatorContext) -> bool:
        return self.select_trajectory(ctx) is not None

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        # 主循环已把 ctx.selected 替换为 backtrack target trajectory
        node_id, reason = select_base_node(
            graph=ctx.graph, trajectory=ctx.selected,
            maximize=ctx.maximize, positive_threshold=ctx.positive_threshold,
        )
        return node_id, reason

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        return (
            "The selected trajectory's endpoint regressed or saturated, but an earlier prefix was strong. "
            "Branch from the selected base node: restart from that high-value prefix and propose a "
            "DIFFERENT modification than the ones that led to the regression or plateau."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        assert base_node_id is not None
        return ctx.memory.branch_from(
            trajectory_id=ctx.selected.id, base_node_id=base_node_id,
            child_id=child_id, edge_id=edge_id,
        )
