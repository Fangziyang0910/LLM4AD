"""Backtrack Branch —— path correction（design §4.2，优化版：独立选题）。

不依赖 UCB 刚选中的轨迹；主动扫描活跃池，找「存在不同于 endpoint 的内部 base」
的多步轨迹。可行性门槛：池中至少有一条长度 ≥ 2 且能选出内部前缀的轨迹。
"""
from __future__ import annotations

from ..schema import NodeId, OperatorName, Trajectory
from .base import (
    Operator,
    OperatorContext,
    branch_score,
    select_base_node,
)


class BacktrackBranchOp(Operator):
    name = OperatorName.BACKTRACK
    role = "path_correct"

    def _candidates(self, ctx: OperatorContext) -> list[tuple[Trajectory, float]]:
        out: list[tuple[Trajectory, float]] = []
        for t in ctx.memory.active():
            if not t.edge_ids:
                continue
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
            "Branch from that high-value prefix and propose a modification DIFFERENT from the one that "
            "caused the regression or plateau. Treat the elite prefix-repair trace as a boundary to repair, "
            "not as a new parent edge."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        assert base_node_id is not None
        return ctx.memory.branch_from(
            trajectory_id=ctx.selected.id, base_node_id=base_node_id,
            child_id=child_id, edge_id=edge_id,
        )
