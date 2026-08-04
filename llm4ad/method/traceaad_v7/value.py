"""V5-compatible route scoring with V7 budget-aware exploration."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .derivation_graph import DerivationGraph
from .schema import ProgramNode, Trajectory, ValueVec
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True, slots=True)
class ValueWeights:
    """Weights for route quality, trend credit, and budget-aware exploration."""

    endpoint_quality: float = 0.7
    best_quality: float = 0.3
    search_quality: float = 0.8
    search_trend: float = 0.2
    ucb_c: float = 0.25
    discount: float = 0.8
    positive_threshold: float = 1e-12

    def __post_init__(self) -> None:
        values = (
            self.endpoint_quality,
            self.best_quality,
            self.search_quality,
            self.search_trend,
            self.ucb_c,
            self.discount,
            self.positive_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("value weights must be finite")
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
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1]")
        if self.positive_threshold < 0:
            raise ValueError("positive_threshold must be non-negative")


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
    """Order programs by fitness, then by non-empty LOC for exact ties."""
    return (directed_fitness(float(node.fitness), maximize), -node.program_loc)


def is_program_better(
    candidate: ProgramNode,
    incumbent: ProgramNode | None,
    maximize: bool,
) -> bool:
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


def tie_aware_percentile(
    key: tuple[float, int],
    keys: list[tuple[float, int]],
) -> float:
    if len(keys) <= 1:
        return 0.5
    less = sum(value < key for value in keys)
    equal = sum(value == key for value in keys)
    average_rank = less + (equal + 1) / 2.0
    return (average_rank - 1.0) / (len(keys) - 1.0)


def score_active_trajectories(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
    trajectories: tuple[Trajectory, ...] | None = None,
) -> tuple[Trajectory, ...]:
    """Compute ``Q`` and ``P`` from fitness and recent route outcomes."""
    candidates = tuple(memory.active() if trajectories is None else trajectories)
    if not candidates:
        return ()
    endpoint_keys = [
        program_quality_key(graph.get_node(route.endpoint_id), maximize)
        for route in candidates
    ]
    best_nodes = [compact_best_node(route, graph, maximize) for route in candidates]
    best_keys = [program_quality_key(node, maximize) for node in best_nodes]
    quality_denominator = w.endpoint_quality + w.best_quality
    scored: list[Trajectory] = []
    for route, best_node in zip(candidates, best_nodes):
        endpoint_percentile = tie_aware_percentile(
            program_quality_key(graph.get_node(route.endpoint_id), maximize),
            endpoint_keys,
        )
        best_percentile = tie_aware_percentile(
            program_quality_key(best_node, maximize),
            best_keys,
        )
        quality = (
            w.endpoint_quality * endpoint_percentile + w.best_quality * best_percentile
        ) / quality_denominator
        signals: list[float] = []
        for edge_id in route.edge_ids:
            # Trend follows the same comparator as route/global selection,
            # including an exact-fitness shorter-program improvement.
            edge = graph.get_edge(edge_id)
            reason = edge.route_best_update_reason
            if reason in {"strict_fitness", "tie_shorter"}:
                signals.append(1.0)
            elif reason == "regress":
                signals.append(-1.0)
            else:
                delta = edge.delta_route_best
                if delta is None or abs(delta) <= w.positive_threshold:
                    signals.append(0.0)
                else:
                    signals.append(1.0 if delta > 0 else -1.0)
        if signals:
            trend_denominator = sum(w.discount**index for index in range(len(signals)))
            weighted = sum(
                w.discount ** (len(signals) - 1 - index) * signal
                for index, signal in enumerate(signals)
            )
            trend = (weighted / trend_denominator + 1.0) / 2.0
        else:
            trend = 0.5
        value = ValueVec(quality=quality, trend=trend)
        scalar = (w.search_quality * quality + w.search_trend * trend) / (
            w.search_quality + w.search_trend
        )
        scored.append(memory.set_value(route.id, value, scalar))
    return tuple(
        sorted(
            scored,
            key=lambda route: (
                -(route.scalar_value if route.scalar_value is not None else 0.0),
                route.id,
            ),
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


def weighted_choice(items: list, weights: list[float], rng: random.Random):
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
    remaining_budget_ratio: float,
    selection_count: int,
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
    remaining_ratio = min(1.0, max(0.0, float(remaining_budget_ratio)))
    log_term = math.log1p(max(0, int(selection_count)))
    adjusted = [
        float(route.scalar_value if route.scalar_value is not None else 0.0)
        + w.ucb_c
        * remaining_ratio
        * math.sqrt(log_term / (1.0 + max(0, route.visit_count)))
        for route in scored
    ]
    probabilities = _probabilities(_softmax_weights(adjusted, temperature))
    return tuple(zip(scored, adjusted, probabilities))


def reference_sampling_distribution(
    *,
    primary: Trajectory,
    active: tuple[Trajectory, ...],
    temperature: float,
    graph: DerivationGraph | None = None,
    primary_node_id: int | None = None,
) -> tuple[tuple[Trajectory, float, float], ...]:
    """Sample a distinct active route according to its Q value."""
    primary_node = (
        primary.compact_best_id if primary_node_id is None else primary_node_id
    )
    primary_hash = None if graph is None else graph.get_node(primary_node).code_hash
    candidates = [
        route
        for route in active
        if route.id != primary.id
        and (
            graph is None
            or graph.get_node(route.compact_best_id).code_hash != primary_hash
        )
    ]
    if not candidates:
        return ()
    scores = [
        float(route.value.quality if route.value else 0.0) for route in candidates
    ]
    probabilities = _probabilities(_softmax_weights(scores, temperature))
    return tuple(
        (route, score, probability)
        for route, score, probability in zip(candidates, scores, probabilities)
    )


def search_value_survivor_sample(
    candidates: list[Trajectory],
    count: int,
    *,
    temperature: float,
    rng: random.Random,
) -> tuple[Trajectory, ...]:
    """Sample survivors without replacement from the configured Q/P softmax."""
    remaining = list(candidates)
    selected: list[Trajectory] = []
    while remaining and len(selected) < count:
        scores = [
            float(route.scalar_value if route.scalar_value is not None else 0.0)
            for route in remaining
        ]
        maximum = max(scores)
        weights = [
            math.exp((score - maximum) / max(float(temperature), 1e-8))
            for score in scores
        ]
        choice = weighted_choice(remaining, weights, rng)
        remaining.remove(choice)
        selected.append(choice)
    return tuple(selected)


__all__ = [
    "ValueWeights",
    "compact_best_node",
    "directed_delta",
    "directed_fitness",
    "is_program_better",
    "program_quality_key",
    "search_value_survivor_sample",
    "reference_sampling_distribution",
    "score_active_trajectories",
    "tie_aware_percentile",
    "trajectory_sampling_distribution",
    "weighted_choice",
]
