"""Stable trajectory-level budget allocation for TraceAAD V9.1."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schema import ProgramNode
from .tree import SearchTree


@dataclass(frozen=True, slots=True)
class TrajectorySelection:
    selected_node_id: int
    quality_rank: int
    quality_pool_ids: tuple[int, ...]
    mode: str
    route_advance_rate: float | None
    wilson_upper: float
    recent_advance_rate: float | None


def wilson_upper_bound(successes: int, trials: int, z: float = 1.0) -> float:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if z < 0:
        raise ValueError("z must be non-negative")
    if trials == 0:
        return 1.0
    rate = successes / trials
    z2 = z * z
    centre = rate + z2 / (2 * trials)
    radius = z * math.sqrt(rate * (1 - rate) / trials + z2 / (4 * trials * trials))
    return (centre + radius) / (1 + z2 / trials)


def trajectory_quality_pool(
    tree: SearchTree, *, pool_size: int
) -> tuple[ProgramNode, ...]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    ranked = sorted(
        tree.nodes(),
        key=lambda node: (-node.directed_fitness, node.program_loc, node.id),
    )
    return tuple(ranked[:pool_size])


def select_trajectory(
    tree: SearchTree,
    *,
    pool_size: int,
    confidence_z: float,
) -> TrajectorySelection:
    pool = trajectory_quality_pool(tree, pool_size=pool_size)
    if not pool:
        raise ValueError("cannot select from an empty trajectory store")
    unverified = [node for node in pool if node.verification_count == 0]
    if unverified:
        selected = unverified[0]
        mode = "basic_validation"
    else:
        selected = max(
            pool,
            key=lambda node: (
                wilson_upper_bound(
                    node.route_advance_count,
                    node.verification_count,
                    confidence_z,
                ),
                sum(node.recent_advances) / len(node.recent_advances),
                node.directed_fitness,
                -node.program_loc,
                -node.id,
            ),
        )
        mode = "trajectory_productivity"
    trials = selected.verification_count
    recent = selected.recent_advances
    return TrajectorySelection(
        selected_node_id=selected.id,
        quality_rank=next(
            index for index, node in enumerate(pool) if node.id == selected.id
        ),
        quality_pool_ids=tuple(node.id for node in pool),
        mode=mode,
        route_advance_rate=(
            None if trials == 0 else selected.route_advance_count / trials
        ),
        wilson_upper=wilson_upper_bound(
            selected.route_advance_count, trials, confidence_z
        ),
        recent_advance_rate=(None if not recent else sum(recent) / len(recent)),
    )


def reference_candidates(
    tree: SearchTree, main_node_id: int, *, pool_size: int
) -> tuple[ProgramNode, ...]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    main = tree.get_node(main_node_id)
    candidates = [
        node
        for node in tree.nodes()
        if not tree.same_lineage(main_node_id, node.id)
        and node.code_hash != main.code_hash
    ]
    candidates.sort(
        key=lambda node: (-node.directed_fitness, node.program_loc, node.id)
    )
    return tuple(candidates[:pool_size])


def sample_reference(
    tree: SearchTree,
    main_node_id: int,
    *,
    pool_size: int,
    rng: random.Random,
) -> ProgramNode | None:
    candidates = reference_candidates(tree, main_node_id, pool_size=pool_size)
    return None if not candidates else rng.choice(candidates)


__all__ = [
    "TrajectorySelection",
    "reference_candidates",
    "sample_reference",
    "select_trajectory",
    "trajectory_quality_pool",
    "wilson_upper_bound",
]
