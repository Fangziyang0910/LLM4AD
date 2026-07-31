"""Fitness-first trajectory scheduling for TraceAAD v5."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .derivation_graph import DerivationGraph
from .schema import ProgramNode, Trajectory, ValueVec
from .similarity import trajectory_similarity
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True, slots=True)
class ValueWeights:
    endpoint_quality: float = 0.7
    best_quality: float = 0.3
    search_quality: float = 0.8
    search_trend: float = 0.2
    ucb_c: float = 0.25
    discount: float = 0.8
    positive_threshold: float = 1e-12

    def __post_init__(self) -> None:
        if self.endpoint_quality < 0 or self.best_quality < 0:
            raise ValueError("quality weights must be non-negative")
        if self.endpoint_quality + self.best_quality <= 0:
            raise ValueError("at least one quality weight must be positive")
        if self.search_quality < 0 or self.search_trend < 0:
            raise ValueError("search-value weights must be non-negative")
        if self.search_quality + self.search_trend <= 0:
            raise ValueError("at least one search-value weight must be positive")
        if self.ucb_c < 0:
            raise ValueError("ucb_c must be non-negative")


def directed_fitness(fitness: float, maximize: bool) -> float:
    return fitness if maximize else -fitness


def directed_delta(
    parent_fitness: float | None, child_fitness: float | None, maximize: bool
) -> float | None:
    if parent_fitness is None or child_fitness is None:
        return None
    return (
        child_fitness - parent_fitness if maximize else parent_fitness - child_fitness
    )


def program_quality_key(node: ProgramNode, maximize: bool) -> tuple[float, int]:
    if node.fitness is None:
        return (-math.inf, -node.program_loc)
    return (directed_fitness(float(node.fitness), maximize), -node.program_loc)


def is_program_better(
    candidate: ProgramNode,
    incumbent: ProgramNode | None,
    maximize: bool,
) -> bool:
    if candidate.fitness is None:
        return False
    return incumbent is None or program_quality_key(
        candidate, maximize
    ) > program_quality_key(incumbent, maximize)


def compact_best_node(
    trajectory: Trajectory,
    graph: DerivationGraph,
    maximize: bool,
) -> ProgramNode:
    best = graph.get_node(trajectory.node_ids[0])
    for node_id in trajectory.node_ids[1:]:
        candidate = graph.get_node(node_id)
        if is_program_better(candidate, best, maximize):
            best = candidate
    return best


def _tie_aware_percentile(
    key: tuple[float, int],
    keys: list[tuple[float, int]],
) -> float:
    if len(keys) <= 1:
        return 0.5
    less = sum(value < key for value in keys)
    equal = sum(value == key for value in keys)
    average_rank = less + (equal + 1) / 2.0
    return (average_rank - 1.0) / (len(keys) - 1.0)


def _analysis_trend(
    trajectory: Trajectory,
    graph: DerivationGraph,
    *,
    discount: float,
    threshold: float,
) -> float:
    signals: list[float] = []
    for edge_id in trajectory.edge_ids:
        delta = graph.get_edge(edge_id).delta_parent
        if delta is None or abs(delta) <= threshold:
            signals.append(0.0)
        else:
            signals.append(1.0 if delta > 0 else -1.0)
    if not signals:
        return 0.5
    denominator = sum(discount**index for index in range(len(signals)))
    weighted = sum(
        (discount ** (len(signals) - 1 - index)) * signal
        for index, signal in enumerate(signals)
    )
    return (weighted / denominator + 1.0) / 2.0 if denominator else 0.5


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
    endpoint_keys = [
        program_quality_key(graph.get_node(route.endpoint_id), maximize)
        for route in candidates
    ]
    best_nodes = [compact_best_node(route, graph, maximize) for route in candidates]
    best_keys = [program_quality_key(node, maximize) for node in best_nodes]
    scored: list[Trajectory] = []
    for route, best_node in zip(candidates, best_nodes):
        endpoint_percentile = _tie_aware_percentile(
            program_quality_key(graph.get_node(route.endpoint_id), maximize),
            endpoint_keys,
        )
        best_percentile = _tie_aware_percentile(
            program_quality_key(best_node, maximize),
            best_keys,
        )
        denominator = w.endpoint_quality + w.best_quality
        quality = (
            w.endpoint_quality * endpoint_percentile + w.best_quality * best_percentile
        ) / denominator
        value = ValueVec(
            quality=quality,
            trend=_analysis_trend(
                route,
                graph,
                discount=w.discount,
                threshold=w.positive_threshold,
            ),
        )
        search_denominator = w.search_quality + w.search_trend
        search_value = (
            w.search_quality * value.quality + w.search_trend * value.trend
        ) / search_denominator
        scored.append(memory.set_value(route.id, value, search_value))
    return tuple(
        sorted(
            scored,
            key=lambda route: (-(route.scalar_value or 0.0), route.id),
        )
    )


def _softmax_weights(scores: list[float], temperature: float) -> list[float]:
    maximum = max(scores)
    return [
        math.exp((score - maximum) / max(float(temperature), 1e-8)) for score in scores
    ]


def _probabilities(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def weighted_choice(
    items: list[Trajectory],
    weights: list[float],
    rng: random.Random,
) -> Trajectory:
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return items[rng.randrange(len(items))]
    needle = rng.random() * total
    for item, weight in zip(items, weights):
        needle -= weight
        if needle <= 0:
            return item
    return items[-1]


def trajectory_sampling_distribution(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
    temperature: float,
) -> tuple[tuple[Trajectory, float, float], ...]:
    scored = list(
        score_active_trajectories(
            memory=memory,
            graph=graph,
            maximize=maximize,
            w=w,
        )
    )
    if not scored:
        return ()
    total_visits = sum(max(0, route.visit_count) for route in scored)
    adjusted = [
        float(route.scalar_value or 0.0)
        + w.ucb_c
        * math.sqrt(math.log1p(total_visits) / (1.0 + max(0, route.visit_count)))
        for route in scored
    ]
    probabilities = _probabilities(_softmax_weights(adjusted, temperature))
    return tuple(zip(scored, adjusted, probabilities))


def reference_sampling_distribution(
    *,
    candidates: tuple[Trajectory, ...],
    temperature: float,
) -> tuple[tuple[Trajectory, float], ...]:
    if not candidates:
        return ()
    scores = [
        0.0 if route.value is None else float(route.value.quality)
        for route in candidates
    ]
    probabilities = _probabilities(_softmax_weights(scores, temperature))
    return tuple(zip(candidates, probabilities))


def select_diverse_trajectories(
    *,
    candidates: tuple[Trajectory, ...],
    graph: DerivationGraph,
    count: int,
    reference: tuple[Trajectory, ...] = (),
) -> tuple[Trajectory, ...]:
    if count <= 0 or not candidates:
        return ()
    remaining = list(candidates)
    selected: list[Trajectory] = []
    while remaining and len(selected) < count:

        def score(route: Trajectory) -> tuple[float, float, int]:
            compared = (*reference, *selected)
            similarity = max(
                (
                    trajectory_similarity(graph=graph, left=route, right=chosen)
                    for chosen in compared
                ),
                default=0.0,
            )
            return (
                0.85 * (1.0 - similarity) + 0.15 * float(route.scalar_value or 0.0),
                float(route.scalar_value or 0.0),
                -route.id,
            )

        choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


__all__ = [
    "ValueWeights",
    "compact_best_node",
    "directed_delta",
    "directed_fitness",
    "is_program_better",
    "program_quality_key",
    "reference_sampling_distribution",
    "score_active_trajectories",
    "select_diverse_trajectories",
    "trajectory_sampling_distribution",
    "weighted_choice",
]
