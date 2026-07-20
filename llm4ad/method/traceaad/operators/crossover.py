"""从另一条互补轨迹引入一个明确的算法思想。"""
from __future__ import annotations

from ..credit import normalize_fitness
from ..schema import NodeId, OperatorName, Trajectory
from ..similarity import (
    code_similarity,
    trajectory_pattern,
    trajectory_pattern_similarity,
)
from .base import Operator, OperatorContext


class MechanismCrossoverOp(Operator):
    name = OperatorName.CROSSOVER

    def trigger(self, ctx: OperatorContext) -> bool:
        return any(
            trajectory.id != ctx.selected.id
            for trajectory in ctx.memory.unique_active()
        )

    def _select_donor(self, ctx: OperatorContext) -> Trajectory | None:
        candidates = [t for t in ctx.memory.active() if t.id != ctx.selected.id]
        if not candidates:
            return None
        fitnesses = [
            node.fitness
            for t in ctx.memory.active()
            if (node := ctx.graph.get_node(t.endpoint_id)).fitness is not None
        ]
        fmin = min(fitnesses) if fitnesses else None
        fmax = max(fitnesses) if fitnesses else None
        sel_code = ctx.graph.get_node(ctx.selected.endpoint_id).code
        sel_pattern = trajectory_pattern(ctx.graph, ctx.selected)
        best: Trajectory | None = None
        best_score = float("-inf")
        for t in candidates:
            endpoint = ctx.graph.get_node(t.endpoint_id)
            if endpoint.fitness is None:
                continue
            sim_code = code_similarity(sel_code, endpoint.code)
            sim_traj = trajectory_pattern_similarity(
                sel_pattern, trajectory_pattern(ctx.graph, t)
            )
            w_code, w_trajectory = ctx.similarity_weights
            total = w_code + w_trajectory
            sim = (
                (w_code * sim_code + w_trajectory * sim_traj) / total
                if total > 0
                else 0.0
            )
            complementary = 1.0 - sim
            quality = (
                t.value.quality
                if t.value is not None
                else normalize_fitness(endpoint.fitness, fmin, fmax, ctx.maximize)
            )
            score = complementary + 0.3 * quality
            if score > best_score:
                best_score, best = score, t
        return best

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        donor = self._select_donor(ctx)
        if donor is None:
            return ctx.selected.endpoint_id, "endpoint"
        donor_node = ctx.graph.get_node(donor.endpoint_id)
        ctx.donor_idea = donor_node.idea
        return ctx.selected.endpoint_id, "crossover_base"

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        donor_idea = ctx.donor_idea or "an unreported idea"
        return (
            "Recombine: transplant exactly ONE clear algorithmic idea from a donor trajectory "
            f"into the current base program. Donor idea for reference: {donor_idea}. "
            "Do NOT replace the whole program — keep the existing structure and change only "
            "that single idea."
        )

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        assert base_node_id is not None
        return ctx.memory.branch_from(
            trajectory_id=ctx.selected.id, base_node_id=base_node_id,
            child_id=child_id, edge_id=edge_id,
        )
