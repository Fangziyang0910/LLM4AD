"""Quality, route credit, and parent selection for TraceAAD V6."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .derivation_graph import DerivationGraph
from .schema import ProgramNode, Trajectory, ValueVec
from .similarity import route_difference
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True, slots=True)
class ValueWeights:
    endpoint_quality: float = 0.7
    best_quality: float = 0.3
    search_quality: float = 0.8
    search_credit: float = 0.2
    ucb_c: float = 0.25
    discount: float = 0.8
    positive_threshold: float = 0.0
    mature_quantile: float = 0.70
    mature_min_edges: int = 2

    def __post_init__(self) -> None:
        if self.endpoint_quality < 0 or self.best_quality < 0:
            raise ValueError("quality weights must be non-negative")
        if self.endpoint_quality + self.best_quality <= 0:
            raise ValueError("at least one quality weight must be positive")
        if self.search_quality < 0 or self.search_credit < 0:
            raise ValueError("search-value weights must be non-negative")
        if self.search_quality + self.search_credit <= 0:
            raise ValueError("at least one search-value weight must be positive")
        if self.ucb_c < 0:
            raise ValueError("ucb_c must be non-negative")
        if not 0.0 <= self.mature_quantile <= 1.0:
            raise ValueError("mature_quantile must be in [0, 1]")
        if self.mature_min_edges < 0:
            raise ValueError("mature_min_edges must be non-negative")


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


def meaningful_improvement(
    delta: float | None, threshold: float
) -> bool:
    return delta is not None and delta > threshold


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


def edge_credit(
    *,
    child: ProgramNode,
    batch_keys: list[tuple[float, int]],
    route_improved: bool,
    maximize: bool,
) -> float:
    if not route_improved or child.fitness is None:
        return 0.0
    return tie_aware_percentile(program_quality_key(child, maximize), batch_keys)


def route_credit(
    trajectory: Trajectory,
    graph: DerivationGraph,
    *,
    discount: float,
) -> float:
    if not trajectory.edge_ids:
        return 0.0
    credits = [
        float(graph.get_edge(edge_id).edge_credit) for edge_id in trajectory.edge_ids
    ]
    weights = [discount ** (len(credits) - 1 - index) for index in range(len(credits))]
    denominator = sum(weights)
    if denominator <= 0:
        return 0.0
    return sum(weight * credit for weight, credit in zip(weights, credits)) / denominator


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
        endpoint_percentile = tie_aware_percentile(
            program_quality_key(graph.get_node(route.endpoint_id), maximize),
            endpoint_keys,
        )
        best_percentile = tie_aware_percentile(
            program_quality_key(best_node, maximize),
            best_keys,
        )
        denominator = w.endpoint_quality + w.best_quality
        quality = (
            w.endpoint_quality * endpoint_percentile + w.best_quality * best_percentile
        ) / denominator
        credit = route_credit(route, graph, discount=w.discount)
        value = ValueVec(quality=quality, credit=credit)
        search_denominator = w.search_quality + w.search_credit
        search_value = (
            w.search_quality * value.quality + w.search_credit * value.credit
        ) / search_denominator
        scored.append(memory.set_value(route.id, value, search_value))
    return tuple(
        sorted(
            scored,
            key=lambda route: (-(route.scalar_value or 0.0), route.id),
        )
    )


def is_mature_trajectory(
    route: Trajectory,
    *,
    active: tuple[Trajectory, ...],
    graph: DerivationGraph,
    w: ValueWeights,
) -> bool:
    if route.value is None:
        return False
    if len(route.edge_ids) < w.mature_min_edges:
        return False
    qualities = sorted(
        (
            0.0 if candidate.value is None else float(candidate.value.quality)
            for candidate in active
        ),
        reverse=True,
    )
    if not qualities:
        return False
    # Top 30% with boundary ties retained.
    top_fraction = 1.0 - w.mature_quantile
    keep_count = max(1, math.ceil(top_fraction * len(qualities)))
    threshold = qualities[min(keep_count, len(qualities)) - 1]
    if route.value.quality + 1e-12 < threshold:
        return False
    return any(
        meaningful_improvement(
            graph.get_edge(edge_id).delta_route_best, w.positive_threshold
        )
        for edge_id in route.edge_ids
    )


def mature_active_trajectories(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    w: ValueWeights,
) -> tuple[Trajectory, ...]:
    scored = score_active_trajectories(
        memory=memory, graph=graph, maximize=maximize, w=w
    )
    return tuple(
        route
        for route in scored
        if is_mature_trajectory(route, active=scored, graph=graph, w=w)
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
    items: list,
    weights: list[float],
    rng: random.Random,
):
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
    batch_count: int,
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
    adjusted = [
        float(route.scalar_value or 0.0)
        + w.ucb_c
        * math.sqrt(math.log1p(max(0, batch_count)) / (1.0 + max(0, route.visit_count)))
        for route in scored
    ]
    probabilities = _probabilities(_softmax_weights(adjusted, temperature))
    return tuple(zip(scored, adjusted, probabilities))


def qualified_reference_candidates(
    *,
    primary: Trajectory,
    anchor_id: int,
    active: tuple[Trajectory, ...],
    graph: DerivationGraph,
    w: ValueWeights,
) -> tuple[tuple[Trajectory, float], ...]:
    """Return mature, distinct references with D > 0 and D >= median."""
    primary_best = graph.get_node(primary.compact_best_id)
    anchor = graph.get_node(anchor_id)
    scored: list[tuple[Trajectory, float]] = []
    for route in active:
        if route.id == primary.id:
            continue
        if not is_mature_trajectory(route, active=active, graph=graph, w=w):
            continue
        ref_best = graph.get_node(route.compact_best_id)
        if ref_best.code_hash in {primary_best.code_hash, anchor.code_hash}:
            continue
        difference = route_difference(graph=graph, left=primary, right=route)
        if difference <= 0:
            continue
        scored.append((route, difference))
    if not scored:
        return ()
    differences = sorted(difference for _, difference in scored)
    median = differences[len(differences) // 2]
    return tuple(
        (route, difference)
        for route, difference in scored
        if difference + 1e-12 >= median
    )


def reference_sampling_distribution(
    *,
    primary: Trajectory,
    candidates: tuple[tuple[Trajectory, float], ...],
    temperature: float,
) -> tuple[tuple[Trajectory, float, float], ...]:
    if not candidates:
        return ()
    scores = []
    for route, difference in candidates:
        quality = 0.0 if route.value is None else float(route.value.quality)
        scores.append(quality * difference)
    probabilities = _probabilities(_softmax_weights(scores, temperature))
    return tuple(
        (route, score, probability)
        for (route, _), score, probability in zip(candidates, scores, probabilities)
    )


def select_diverse_trajectories(
    *,
    candidates: tuple[Trajectory, ...],
    graph: DerivationGraph,
    count: int,
    reference: tuple[Trajectory, ...] = (),
) -> tuple[Trajectory, ...]:
    """Max-min route difference among mature candidates."""
    if count <= 0 or not candidates:
        return ()
    remaining = list(candidates)
    selected: list[Trajectory] = []
    while remaining and len(selected) < count:

        def score(route: Trajectory) -> tuple[float, float, int]:
            compared = (*reference, *selected)
            if not compared:
                return (
                    float(route.scalar_value or 0.0),
                    float(route.scalar_value or 0.0),
                    -route.id,
                )
            min_diff = min(
                route_difference(graph=graph, left=route, right=chosen)
                for chosen in compared
            )
            return (
                min_diff,
                float(route.scalar_value or 0.0),
                -route.id,
            )

        choice = max(remaining, key=score)
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


def deduplicate_by_endpoint_hash(
    *,
    routes: tuple[Trajectory, ...],
    graph: DerivationGraph,
    best_trajectory_id: int | None,
) -> tuple[Trajectory, ...]:
    groups: dict[str, list[Trajectory]] = {}
    for route in routes:
        code_hash = graph.get_node(route.endpoint_id).code_hash
        groups.setdefault(code_hash, []).append(route)
    kept: list[Trajectory] = []
    for group in groups.values():
        if best_trajectory_id is not None and any(
            route.id == best_trajectory_id for route in group
        ):
            kept.append(next(route for route in group if route.id == best_trajectory_id))
            continue
        kept.append(
            max(
                group,
                key=lambda route: (
                    float(route.scalar_value or 0.0),
                    -route.id,
                ),
            )
        )
    return tuple(sorted(kept, key=lambda route: (-(route.scalar_value or 0.0), route.id)))


__all__ = [
    "ValueWeights",
    "compact_best_node",
    "deduplicate_by_endpoint_hash",
    "directed_delta",
    "directed_fitness",
    "edge_credit",
    "is_mature_trajectory",
    "is_program_better",
    "mature_active_trajectories",
    "meaningful_improvement",
    "program_quality_key",
    "qualified_reference_candidates",
    "reference_sampling_distribution",
    "route_credit",
    "score_active_trajectories",
    "select_diverse_trajectories",
    "tie_aware_percentile",
    "trajectory_sampling_distribution",
    "weighted_choice",
]
