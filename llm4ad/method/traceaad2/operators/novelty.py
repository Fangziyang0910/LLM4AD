"""Novelty Jump —— exploration（design §4.5，优化版）。

关键修正：
- trigger 从「unique_ratio<0.4（active 多时永真）」改为「best 连续 N 轮停滞」，避免 novelty 被过度选中。
- 目标族只按 novelty fresh-start 自身的后验成功率与尝试数选择，并避开 cooldown/anti-pattern；
  其他算子的成功不会再掩盖某个 family 的 fresh-start 连败。
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

    def __init__(
        self,
        *,
        stagnation_threshold: int = 12,
        trigger_cooldown: int = 8,
        family_failure_limit: int = 2,
        family_cooldown: int = 24,
    ) -> None:
        self.stagnation_threshold = stagnation_threshold
        self.trigger_cooldown = trigger_cooldown
        self.family_failure_limit = family_failure_limit
        self.family_cooldown = family_cooldown
        self._last_trigger_iteration: int | None = None

    def trigger(self, ctx: OperatorContext) -> bool:
        if ctx.best_stagnation < self.stagnation_threshold:
            return False
        if not self._eligible_families(ctx):
            return False
        return (
            self._last_trigger_iteration is None
            or ctx.iteration - self._last_trigger_iteration >= self.trigger_cooldown
        )

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        return None, "fresh_start"

    def _pick_family(self, ctx: OperatorContext) -> str:
        cands = self._eligible_families(ctx)
        if not cands:
            raise RuntimeError("novelty has no eligible mechanism family")

        def score(family: str) -> tuple[float, int, str]:
            attempts = ctx.pattern_memory.mechanism_attempts(
                family, operator=self.name
            )
            successes = ctx.pattern_memory.mechanism_successes(
                family, operator=self.name
            )
            # Beta(1, 1) posterior: untried families remain competitive, while
            # repeated failed fresh starts rotate away regardless of other operators.
            posterior = (successes + 1.0) / (attempts + 2.0)
            return (posterior, -attempts, family)

        return max(cands, key=score)

    def _eligible_families(self, ctx: OperatorContext) -> list[str]:
        return [
            family
            for family in _CANDIDATE_FAMILIES
            if not ctx.pattern_memory.is_anti_pattern(family, operator=self.name)
            and not ctx.pattern_memory.mechanism_in_failure_cooldown(
                family,
                operator=self.name,
                iteration=ctx.iteration,
                failure_limit=self.family_failure_limit,
                cooldown=self.family_cooldown,
            )
        ]

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        target = self._pick_family(ctx)
        ctx.hints["mechanism_tag_hint"] = target
        self._last_trigger_iteration = ctx.iteration
        return (
            f"Novelty jump: best has stagnated. Design a NEW complete algorithm from the '{target}' "
            f"mechanism family (it has favorable fresh-start evidence or is under-explored). Build a "
            f"fresh constructive heuristic from scratch using this family; avoid repeating current "
            f"converged directions."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        tag = ctx.hints.get(
            "observed_mechanism_tag",
            ctx.hints.get("mechanism_tag_hint", "other"),
        )
        island = ctx.islands.assign(tag)
        return ctx.memory.create_initial(node_id=child_id, island_id=island)
