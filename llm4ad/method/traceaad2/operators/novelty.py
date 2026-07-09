"""Novelty Jump —— exploration（design §4.5，优化版）。

两处修正：
- trigger 从「unique_ratio<0.4（active 多时永真）」改为「best 连续 N 轮停滞」，避免 novelty 被过度选中。
- 目标族从「最罕见」改为「PatternMemory 高 improve 且当前探索不足」，并避开 anti_pattern——
  让探索朝已证明有效的机制（如 adaptive_exponent）而不是又开一个 row_normalize。
"""
from __future__ import annotations

from ..schema import NodeId, OperatorName, Trajectory
from .base import Operator, OperatorContext

_CANDIDATE_FAMILIES = (
    "local_density", "nn_rank", "row_normalize", "edge_contrast",
    "sparsified_candidate", "adaptive_exponent", "hybrid_distance", "randomization",
)


class NoveltyJumpOp(Operator):
    name = OperatorName.NOVELTY
    role = "explore"

    def trigger(self, ctx: OperatorContext) -> bool:
        return ctx.best_stagnation >= 5

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        return None, "fresh_start"

    def _pick_family(self, ctx: OperatorContext) -> str:
        tag_counts: dict[str, int] = {}
        for t in ctx.memory.active():
            tg = ctx.graph.get_node(t.endpoint_id).mechanism_tag
            tag_counts[tg] = tag_counts.get(tg, 0) + 1
        # 优先 PatternMemory 高 improve 机制，避开 anti_pattern
        scored: list[tuple[float, str]] = []
        for m in ctx.pattern_memory.top_mechanisms(k=8):
            if ctx.pattern_memory.is_anti_pattern(m.mechanism_tag):
                continue
            score = m.generalization_score - 0.02 * tag_counts.get(m.mechanism_tag, 0)
            scored.append((score, m.mechanism_tag))
        if scored:
            scored.sort(reverse=True)
            return scored[0][1]
        # 无蒸馏数据时：候选族里非 anti_pattern 且当前最少用的
        cands = [f for f in _CANDIDATE_FAMILIES if not ctx.pattern_memory.is_anti_pattern(f)] or list(_CANDIDATE_FAMILIES)
        return min(cands, key=lambda f: tag_counts.get(f, 0))

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        target = self._pick_family(ctx)
        ctx.hints["mechanism_tag_hint"] = target
        return (
            f"Novelty jump: best has stagnated. Design a NEW complete algorithm from the '{target}' "
            f"mechanism family (it has shown cross-trajectory promise or is under-explored). Build a "
            f"fresh constructive heuristic from scratch using this family; avoid repeating current "
            f"converged directions."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        tag = ctx.hints.get("mechanism_tag_hint", "other")
        island = ctx.islands.assign(tag)
        return ctx.memory.create_initial(node_id=child_id, island_id=island)
