"""算子选择前的轻量预览信号：实际 target/base + 有界 context bonus。"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from .operators import (
    BacktrackBranchOp,
    MechanismCrossoverOp,
    Operator,
    OperatorContext,
    NoveltyJumpOp,
    SimplifyOp,
    branch_score,
    trajectory_step_outcomes,
)
from .schema import OperatorName


@dataclass(frozen=True)
class OperatorPreview:
    operator_name: str
    eligible: bool
    target_trajectory_id: int | None
    base_node_id: int | None
    base_reason: str | None
    context_bonus: float
    details: dict = field(default_factory=dict)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _last_outcome(ctx: OperatorContext) -> str | None:
    outcomes = trajectory_step_outcomes(
        ctx.graph, ctx.selected, ctx.maximize, ctx.positive_threshold
    )
    if not outcomes:
        return None
    return outcomes[-1][3]


def _improve_streak(ctx: OperatorContext) -> int:
    outcomes = trajectory_step_outcomes(
        ctx.graph, ctx.selected, ctx.maximize, ctx.positive_threshold
    )
    streak = 0
    for entry in reversed(outcomes):
        if entry[3] != "improve":
            break
        streak += 1
    return streak


def _complexity_percentile(ctx: OperatorContext) -> float | None:
    node = ctx.graph.get_node(ctx.selected.endpoint_id)
    complexities = sorted(
        ctx.graph.get_node(t.endpoint_id).complexity
        for t in ctx.memory.active()
        if ctx.graph.get_node(t.endpoint_id).fitness is not None
    )
    if len(complexities) < 2 or node.complexity <= 0:
        return None
    upper = complexities[math.ceil(0.75 * len(complexities)) - 1]
    median = statistics.median(complexities)
    return 1.0 if node.complexity >= upper and node.complexity > median else 0.0


def build_operator_previews(
    *,
    ctx: OperatorContext,
    operators: tuple[Operator, ...],
    context_bound: float = 0.2,
    stagnation_scale: int = 12,
    recent_novelty_downside: float = 0.0,
) -> dict[str, OperatorPreview]:
    """为每个算子预计算可行性、实际 target/base 与有界 context bonus。"""
    bound = max(0.0, context_bound)
    previews: dict[str, OperatorPreview] = {}
    stag = min(1.0, max(0.0, ctx.best_stagnation) / max(stagnation_scale, 1))

    for op in operators:
        eligible = bool(op.trigger(ctx))
        if not eligible:
            previews[op.name] = OperatorPreview(
                operator_name=op.name,
                eligible=False,
                target_trajectory_id=None,
                base_node_id=None,
                base_reason=None,
                context_bonus=0.0,
                details={"reason": "ineligible"},
            )
            continue

        if isinstance(op, BacktrackBranchOp) or op.name == OperatorName.BACKTRACK:
            target = op.select_trajectory(ctx)
            if target is None:
                previews[op.name] = OperatorPreview(
                    operator_name=op.name,
                    eligible=False,
                    target_trajectory_id=None,
                    base_node_id=None,
                    base_reason=None,
                    context_bonus=0.0,
                    details={"reason": "no_internal_prefix"},
                )
                continue
            local = OperatorContext(
                graph=ctx.graph,
                memory=ctx.memory,
                experience_memory=ctx.experience_memory,
                islands=ctx.islands,
                selected=target,
                maximize=ctx.maximize,
                positive_threshold=ctx.positive_threshold,
                iteration=ctx.iteration,
                best_stagnation=ctx.best_stagnation,
                hints=dict(ctx.hints),
            )
            base_id, base_reason = op.select_base(local)
            outcomes = trajectory_step_outcomes(
                local.graph, local.selected, local.maximize, local.positive_threshold
            )
            last = outcomes[-1][3] if outcomes else None
            bonus = 0.0
            if last == "regress":
                bonus += 0.15
            elif last == "plateau":
                bonus += 0.05
            depth_gap = 0
            branch_quality = branch_score(
                local.graph, target, base_id, local.maximize
            ) if base_id is not None else None
            if base_id is not None and base_id in target.node_ids:
                depth_gap = len(target.node_ids) - 1 - target.node_ids.index(base_id)
                bonus += min(0.05, 0.01 * depth_gap)
            if branch_quality is not None:
                bonus += 0.02 * math.tanh(branch_quality)
            previews[op.name] = OperatorPreview(
                operator_name=op.name,
                eligible=True,
                target_trajectory_id=target.id,
                base_node_id=base_id,
                base_reason=base_reason,
                context_bonus=_clip(bonus, bound),
                details={
                    "last_outcome": last,
                    "depth_gap": depth_gap,
                    "branch_score": branch_quality,
                    "base_reason": base_reason,
                },
            )
            continue

        if isinstance(op, NoveltyJumpOp) or op.name == OperatorName.NOVELTY:
            bonus = 0.0
            active = ctx.memory.active()
            diversity_values = [
                trajectory.value.diversity
                for trajectory in active
                if trajectory.value is not None
            ]
            pool_diversity = (
                statistics.mean(diversity_values) if diversity_values else 0.5
            )
            if stag > 0.0 and recent_novelty_downside < 0.35:
                bonus += 0.08 * stag
            if pool_diversity < 0.35:
                bonus += 0.04 * (1.0 - pool_diversity / 0.35)
            previews[op.name] = OperatorPreview(
                operator_name=op.name,
                eligible=True,
                target_trajectory_id=ctx.selected.id,
                base_node_id=None,
                base_reason="fresh_start",
                context_bonus=_clip(bonus, bound),
                details={
                    "stagnation_frac": stag,
                    "recent_novelty_downside": recent_novelty_downside,
                    "active_pool_diversity": pool_diversity,
                },
            )
            continue

        base_id, base_reason = op.select_base(ctx)
        last = _last_outcome(ctx)
        bonus = 0.0
        details: dict = {"last_outcome": last}

        if op.name == OperatorName.ENDPOINT:
            streak = _improve_streak(ctx)
            details["improve_streak"] = streak
            if last == "improve":
                bonus += 0.12 + min(0.08, 0.02 * streak)
            elif last == "plateau":
                bonus -= 0.02
            elif last == "regress":
                bonus -= 0.08
        elif isinstance(op, MechanismCrossoverOp) or op.name == OperatorName.CROSSOVER:
            donor = op._select_donor(ctx)  # noqa: SLF001 — preview peeks ranking
            if donor is not None:
                endpoint = ctx.graph.get_node(donor.endpoint_id)
                details["donor_trajectory_id"] = donor.id
                details["donor_idea"] = endpoint.idea
                quality = donor.value.quality if donor.value is not None else 0.5
                bonus += 0.05 + 0.05 * max(0.0, min(1.0, quality))
                if last == "plateau":
                    bonus += 0.04 * stag
            else:
                bonus -= 0.05
        elif isinstance(op, SimplifyOp) or op.name == OperatorName.SIMPLIFY:
            rel = _complexity_percentile(ctx)
            details["relative_complex"] = rel
            if rel is not None and rel >= 1.0:
                bonus += 0.12
            else:
                bonus -= 0.05
        else:
            if last == "improve":
                bonus += 0.05

        previews[op.name] = OperatorPreview(
            operator_name=op.name,
            eligible=True,
            target_trajectory_id=ctx.selected.id,
            base_node_id=base_id,
            base_reason=base_reason,
            context_bonus=_clip(bonus, bound),
            details=details,
        )
    return previews
