"""Mechanism Crossover —— recombination（design §4.3）。从互补 donor 迁移单个稳定机制。"""
from __future__ import annotations

from ..schema import NodeId, OperatorName, Trajectory
from ..similarity import mechanism_profile, mechanism_similarity
from .base import Operator, OperatorContext


class MechanismCrossoverOp(Operator):
    name = OperatorName.CROSSOVER
    role = "recombine"

    def trigger(self, ctx: OperatorContext) -> bool:
        node = ctx.graph.get_node(ctx.selected.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return False
        return self._select_donor(ctx) is not None

    def _select_donor(self, ctx: OperatorContext) -> Trajectory | None:
        sel_profile = mechanism_profile(ctx.graph, ctx.selected)
        candidates = [t for t in ctx.memory.active() if t.id != ctx.selected.id]
        if not candidates:
            return None
        best: Trajectory | None = None
        best_score = -1.0
        for t in candidates:
            prof = mechanism_profile(ctx.graph, t)
            complementary = 1.0 - mechanism_similarity(sel_profile, prof)
            if complementary < 0.5:  # 必须足够互补
                continue
            value = t.scalar_value if t.scalar_value is not None else 0.0
            score = complementary + 0.3 * value
            if score > best_score:
                best_score, best = score, t
        return best

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        donor = self._select_donor(ctx)
        if donor is None:
            return ctx.selected.endpoint_id, "endpoint"
        donor_node = ctx.graph.get_node(donor.endpoint_id)
        ctx.hints["donor_id"] = donor.id
        ctx.hints["donor_mechanism"] = donor_node.mechanism_tag
        ctx.hints["donor_idea"] = donor_node.idea
        return ctx.selected.endpoint_id, "crossover_base"

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        donor_idea = ctx.hints.get("donor_idea", "an unreported idea")
        donor_mech = ctx.hints.get("donor_mechanism", "a complementary mechanism")
        return (
            f"Recombine: adopt exactly ONE main mechanism from a donor trajectory into the current base "
            f"program. Donor mechanism family: {donor_mech}. Donor idea for reference: {donor_idea}. "
            f"Do NOT replace the whole program — transplant only that single mechanism into the existing "
            f"structure and keep everything else intact."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        assert base_node_id is not None
        return ctx.memory.branch_from(
            trajectory_id=ctx.selected.id, base_node_id=base_node_id,
            child_id=child_id, edge_id=edge_id,
        )
