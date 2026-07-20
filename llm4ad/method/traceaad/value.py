"""轨迹价值与选择。

轨迹只按三项信息判断：当前程序质量、沿途改进潜力、与其它路线的差异。
访问较少的轨迹通过 UCB 获得探索机会。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .credit import compute_path_value, normalize_fitness
from .derivation_graph import DerivationGraph
from .schema import Trajectory, ValueVec
from .similarity import max_similarity_to_active
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True)
class ValueWeights:
    w_quality: float = 0.55
    w_potential: float = 0.25
    w_diversity: float = 0.20
    w_sim_code: float = 0.7
    w_sim_trajectory: float = 0.3
    discount: float = 0.8
    positive_threshold: float = 1e-6
    ucb_c: float = 0.4


def active_fitness_bounds(
    *,
    trajectories: tuple[Trajectory, ...],
    graph: DerivationGraph,
) -> tuple[float | None, float | None]:
    values = {
        graph.get_node(trajectory.endpoint_id).fitness
        for trajectory in trajectories
        if graph.get_node(trajectory.endpoint_id).fitness is not None
    }
    if not values:
        return None, None
    return min(values), max(values)


def compute_value_vec(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    active_others: tuple[Trajectory, ...],
    fmin: float | None,
    fmax: float | None,
    maximize: bool,
    w: ValueWeights,
) -> ValueVec:
    endpoint = graph.get_node(trajectory.endpoint_id)
    if endpoint.fitness is None:
        return ValueVec()
    quality = normalize_fitness(endpoint.fitness, fmin, fmax, maximize)
    potential = compute_path_value(
        trajectory=trajectory,
        graph=graph,
        fmin=fmin,
        fmax=fmax,
        maximize=maximize,
        discount=w.discount,
        positive_threshold=w.positive_threshold,
        w_positive_ratio=0.25,
        w_downside=0.5,
    )
    similarity = max_similarity_to_active(
        graph=graph,
        candidate=trajectory,
        others=active_others,
        weights=(w.w_sim_code, w.w_sim_trajectory),
    )
    return ValueVec(
        quality=quality,
        potential=potential,
        diversity=1.0 - similarity,
    )


def scalarize(value: ValueVec, w: ValueWeights) -> float:
    return (
        w.w_quality * value.quality
        + w.w_potential * value.potential
        + w.w_diversity * value.diversity
    )


def ucb_bonus(*, visit_count: int, total_visits: int, c: float) -> float:
    return max(0.0, c) * math.sqrt(
        math.log(total_visits + 2) / (visit_count + 1)
    )


def score_active_trajectories(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
) -> tuple[Trajectory, ...]:
    actives = memory.unique_active()
    fmin, fmax = active_fitness_bounds(trajectories=actives, graph=graph)
    total_visits = memory.total_visits()
    scored: list[Trajectory] = []
    for trajectory in actives:
        others = tuple(other for other in actives if other.id != trajectory.id)
        value = compute_value_vec(
            trajectory=trajectory,
            graph=graph,
            active_others=others,
            fmin=fmin,
            fmax=fmax,
            maximize=maximize,
            w=w,
        )
        score = scalarize(value, w) + ucb_bonus(
            visit_count=trajectory.visit_count,
            total_visits=total_visits,
            c=w.ucb_c,
        )
        scored.append(memory.set_value(trajectory.id, value, score))
    return tuple(
        sorted(
            scored,
            key=lambda trajectory: (
                -(trajectory.scalar_value or 0.0),
                trajectory.id,
            ),
        )
    )


def select_trajectory(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
) -> Trajectory:
    scored = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=maximize,
        w=w,
    )
    if not scored:
        raise ValueError("no active trajectories available for sampling")
    return scored[0]


__all__ = [
    "ValueWeights",
    "active_fitness_bounds",
    "compute_value_vec",
    "scalarize",
    "score_active_trajectories",
    "select_trajectory",
]
