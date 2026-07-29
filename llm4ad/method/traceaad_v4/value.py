"""TraceAAD v4 的轨迹质量、趋势和 softmax 选择。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .derivation_graph import DerivationGraph
from .schema import Trajectory, ValueVec
from .similarity import trajectory_similarity
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True)
class ValueWeights:
    """Weights for search value and the separate parent-expansion UCB bonus."""

    w_quality: float = 0.6
    w_trend: float = 0.4
    discount: float = 0.8
    positive_threshold: float = 1e-6
    # UCB is an expansion-scheduling bonus, not part of route quality.
    ucb_c: float = 0.25


def _directed_fitness(fitness: float, maximize: bool) -> float:
    return fitness if maximize else -fitness


def active_fitness_bounds(
    *,
    trajectories: tuple[Trajectory, ...],
    graph: DerivationGraph,
) -> tuple[float | None, float | None]:
    values = [
        graph.get_node(t.endpoint_id).fitness
        for t in trajectories
        if graph.get_node(t.endpoint_id).fitness is not None
    ]
    if not values:
        return None, None
    return min(values), max(values)


def _path_trend(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    discount: float,
    positive_threshold: float,
    maximize: bool,
) -> float:
    if not trajectory.edge_ids:
        return 0.5
    signals: list[float] = []
    for edge_id in trajectory.edge_ids:
        edge = graph.get_edge(edge_id)
        delta = edge.delta
        if delta is None:
            parent = graph.get_node(edge.parent_id)
            child = graph.get_node(edge.child_id)
            if parent.fitness is None or child.fitness is None:
                signal = 0.0
            else:
                raw = (
                    (child.fitness - parent.fitness)
                    if maximize
                    else (parent.fitness - child.fitness)
                )
                signal = (
                    1.0
                    if raw > positive_threshold
                    else -1.0
                    if raw < -positive_threshold
                    else 0.0
                )
        else:
            # Edges store the already directed delta in v4.
            signal = (
                1.0
                if delta > positive_threshold
                else -1.0
                if delta < -positive_threshold
                else 0.0
            )
        signals.append(signal)
    denominator = sum(discount**index for index in range(len(signals)))
    weighted = sum(
        (discount ** (len(signals) - 1 - index)) * signal
        for index, signal in enumerate(signals)
    )
    return (weighted / denominator + 1.0) / 2.0 if denominator else 0.5


def _trajectory_best_fitness(
    trajectory: Trajectory,
    graph: DerivationGraph,
    maximize: bool,
) -> float | None:
    values = [
        graph.get_node(node_id).fitness
        for node_id in trajectory.node_ids
        if graph.get_node(node_id).fitness is not None
    ]
    if not values:
        return None
    return max(values) if maximize else min(values)


def _normalize_fitness(
    fitness: float | None,
    *,
    fmin: float | None,
    fmax: float | None,
    maximize: bool,
) -> float:
    if fitness is None or fmin is None or fmax is None or fmax == fmin:
        return 0.5
    directed = _directed_fitness(fitness, maximize)
    directed_bounds = (
        _directed_fitness(fmin, maximize),
        _directed_fitness(fmax, maximize),
    )
    low = min(directed_bounds)
    high = max(directed_bounds)
    return (directed - low) / (high - low) if high > low else 0.5


def compute_value_vec(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    fmin: float | None,
    fmax: float | None,
    maximize: bool,
    w: ValueWeights,
) -> ValueVec:
    """Compute route quality and recent trend.

    Route quality gives 70% weight to the current endpoint and 30% credit to the
    best program observed anywhere in the retained trajectory. The population
    scorer replaces both absolute values with percentile ranks.
    """
    endpoint = graph.get_node(trajectory.endpoint_id)
    if endpoint.fitness is None:
        return ValueVec()
    endpoint_quality = _normalize_fitness(
        endpoint.fitness,
        fmin=fmin,
        fmax=fmax,
        maximize=maximize,
    )
    best_quality = _normalize_fitness(
        _trajectory_best_fitness(trajectory, graph, maximize),
        fmin=fmin,
        fmax=fmax,
        maximize=maximize,
    )
    return ValueVec(
        quality=0.7 * endpoint_quality + 0.3 * best_quality,
        trend=_path_trend(
            trajectory=trajectory,
            graph=graph,
            discount=w.discount,
            positive_threshold=w.positive_threshold,
            maximize=maximize,
        ),
    )


def scalarize(value: ValueVec, w: ValueWeights) -> float:
    return w.w_quality * value.quality + w.w_trend * value.trend


def _percentile_fitness(
    fitness: float | None,
    values: list[float],
    maximize: bool,
) -> float:
    if fitness is None or len(values) <= 1:
        return 0.5 if fitness is not None else 0.0
    directed_values = [_directed_fitness(value, maximize) for value in values]
    directed = _directed_fitness(fitness, maximize)
    less = sum(value < directed for value in directed_values)
    equal = sum(value == directed for value in directed_values)
    rank = less + (equal + 1) / 2.0
    return (rank - 1.0) / (len(values) - 1.0)


def _percentile_quality(
    trajectory: Trajectory,
    trajectories: tuple[Trajectory, ...],
    graph: DerivationGraph,
    maximize: bool,
) -> float:
    endpoint_values = [
        graph.get_node(t.endpoint_id).fitness
        for t in trajectories
        if graph.get_node(t.endpoint_id).fitness is not None
    ]
    endpoint = graph.get_node(trajectory.endpoint_id).fitness
    best_values = [
        best
        for candidate in trajectories
        if (best := _trajectory_best_fitness(candidate, graph, maximize)) is not None
    ]
    endpoint_percentile = _percentile_fitness(
        endpoint,
        endpoint_values,
        maximize,
    )
    best_percentile = _percentile_fitness(
        _trajectory_best_fitness(trajectory, graph, maximize),
        best_values,
        maximize,
    )
    return 0.7 * endpoint_percentile + 0.3 * best_percentile


def score_active_trajectories(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
    trajectories: tuple[Trajectory, ...] | None = None,
) -> tuple[Trajectory, ...]:
    candidates = tuple(memory.active() if trajectories is None else trajectories)
    if not candidates:
        return ()
    fmin, fmax = active_fitness_bounds(trajectories=candidates, graph=graph)
    scored: list[Trajectory] = []
    for trajectory in candidates:
        value = compute_value_vec(
            trajectory=trajectory,
            graph=graph,
            fmin=fmin,
            fmax=fmax,
            maximize=maximize,
            w=w,
        )
        value = ValueVec(
            quality=_percentile_quality(trajectory, candidates, graph, maximize),
            trend=value.trend,
        )
        scored.append(memory.set_value(trajectory.id, value, scalarize(value, w)))
    return tuple(sorted(scored, key=lambda t: (-(t.scalar_value or 0.0), t.id)))


def select_diverse_trajectories(
    *,
    candidates: tuple[Trajectory, ...],
    graph: DerivationGraph,
    count: int,
    reference: tuple[Trajectory, ...] = (),
) -> tuple[Trajectory, ...]:
    """Select a small quality-aware max-min route-diversity reserve.

    This is deliberately a survivor reserve rather than a hard novelty gate: all
    candidates remain eligible for the ordinary softmax survivor draw.
    """
    if count <= 0 or not candidates:
        return ()
    remaining = list(candidates)
    first = max(remaining, key=lambda t: (t.scalar_value or 0.0, -t.id))
    selected = [first]
    remaining.remove(first)
    while remaining and len(selected) < count:

        def reserve_score(candidate: Trajectory) -> tuple[float, float, int]:
            compared = (*reference, *selected)
            similarity = max(
                (
                    trajectory_similarity(graph=graph, left=candidate, right=chosen)
                    for chosen in compared
                ),
                default=0.0,
            )
            novelty = 1.0 - similarity
            scalar = float(candidate.scalar_value or 0.0)
            return (0.85 * novelty + 0.15 * scalar, scalar, -candidate.id)

        choice = max(remaining, key=reserve_score)
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


def _weighted_sample_without_replacement(
    items: list[Trajectory], weights: list[float], count: int
) -> list[Trajectory]:
    selected: list[Trajectory] = []
    remaining = list(zip(items, weights))
    for _ in range(min(count, len(remaining))):
        total = sum(weight for _, weight in remaining)
        if total <= 0.0 or not math.isfinite(total):
            index = random.randrange(len(remaining))
        else:
            needle = random.random() * total
            index = 0
            for index, (_, weight) in enumerate(remaining):
                needle -= weight
                if needle <= 0:
                    break
        selected.append(remaining.pop(index)[0])
    return selected


def _softmax_scores(scores: list[float], temperature: float) -> list[float]:
    if not scores:
        return []
    temperature = max(float(temperature), 1e-8)
    maximum = max(scores)
    return [math.exp((score - maximum) / temperature) for score in scores]


def sample_trajectory(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
    temperature: float = 0.2,
) -> Trajectory:
    scored = list(
        score_active_trajectories(memory=memory, graph=graph, maximize=maximize, w=w)
    )
    if not scored:
        raise ValueError("no eligible active trajectories available for sampling")
    total_visits = sum(max(0, trajectory.visit_count) for trajectory in scored)
    adjusted_scores = [
        float(trajectory.scalar_value or 0.0)
        + w.ucb_c
        * math.sqrt(math.log1p(total_visits) / (1.0 + max(0, trajectory.visit_count)))
        for trajectory in scored
    ]
    return _weighted_sample_without_replacement(
        scored, _softmax_scores(adjusted_scores, temperature), 1
    )[0]


__all__ = [
    "ValueWeights",
    "active_fitness_bounds",
    "compute_value_vec",
    "scalarize",
    "score_active_trajectories",
    "select_diverse_trajectories",
    "sample_trajectory",
]
