"""Novelty Jump —— exploration。

始终可作为探索候选；由 portfolio 的历史收益与阶段 bonus 控制调用频率。
生成与当前活跃精英 idea 明显不同的完整新方案，分配到活跃轨迹最少的 island。
"""
from __future__ import annotations

from ..schema import NodeId, OperatorName, Trajectory
from .base import Operator, OperatorContext


class NoveltyJumpOp(Operator):
    name = OperatorName.NOVELTY
    role = "explore"

    def __init__(self, *, max_avoid_ideas: int = 4) -> None:
        self.max_avoid_ideas = max_avoid_ideas

    def trigger(self, ctx: OperatorContext) -> bool:
        return True

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        return None, "fresh_start"

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        avoid = self._avoid_ideas(ctx)
        if avoid:
            listed = "; ".join(f"'{idea}'" for idea in avoid)
            avoid_clause = f" Avoid repeating these existing ideas: {listed}."
        else:
            avoid_clause = ""
        return (
            "Novelty jump: design a NEW complete algorithm that uses a clearly different "
            "algorithmic idea from the current active elites."
            f"{avoid_clause} Build a fresh solution from scratch; do not continue an existing program."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        counts = {i: 0 for i in range(ctx.islands.n_islands)}
        for t in ctx.memory.active():
            counts[t.island_id] = counts.get(t.island_id, 0) + 1
        island_id = min(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return ctx.memory.create_initial(node_id=child_id, island_id=island_id)

    def _avoid_ideas(self, ctx: OperatorContext) -> list[str]:
        scored: list[tuple[float, str]] = []
        for t in ctx.memory.active():
            node = ctx.graph.get_node(t.endpoint_id)
            if not node.idea:
                continue
            score = t.scalar_value if t.scalar_value is not None else (
                node.fitness if node.fitness is not None else float("-inf")
            )
            scored.append((score, node.idea.strip()))
        scored.sort(key=lambda x: x[0], reverse=True)
        seen: set[str] = set()
        out: list[str] = []
        for _, idea in scored:
            if idea in seen:
                continue
            seen.add(idea)
            out.append(idea)
            if len(out) >= self.max_avoid_ideas:
                break
        return out
