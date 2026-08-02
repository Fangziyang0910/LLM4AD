"""Program ordering and quality-based route selection for TraceAAD V6."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .derivation_graph import DerivationGraph
from .schema import ProgramNode, Trajectory, ValueVec
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True, slots=True)
class ValueWeights:
    """Weights for route quality and UCB parent selection.

    Route value is deliberately just the quality percentile ``Q``.  Program
    length is a tie-break only when replacing the global/compact-best program;
    it is not part of route scoring.
    """

    endpoint_quality: float = 0.7
    best_quality: float = 0.3
    ucb_c: float = 0.25
    positive_threshold: float = 0.0

    def __post_init__(self) -> None:
        if self.endpoint_quality < 0 or self.best_quality < 0:
            raise ValueError("quality weights must be non-negative")
        if self.endpoint_quality + self.best_quality <= 0:
            raise ValueError("at least one quality weight must be positive")
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
    """Order programs by fitness, then by non-empty LOC for exact ties."""
    if node.fitness is None:
        return (-math.inf, -node.program_loc)
    return (directed_fitness(float(node.fitness), maximize), -node.program_loc)


def fitness_percentile_key(node: ProgramNode, maximize: bool) -> tuple[float, int]:
    """Order route candidates by evaluator fitness only.

    The second component keeps the tuple shape used by the tie-aware percentile
    helper while ensuring equal-fitness programs receive the same rank.
    """
    if node.fitness is None:
        return (-math.inf, 0)
    return (directed_fitness(float(node.fitness), maximize), 0)


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
    """Compute Q from endpoint and compact-best fitness percentiles."""
    candidates = tuple(memory.active() if trajectories is None else trajectories)
    if not candidates:
        return ()
    endpoint_keys = [
        fitness_percentile_key(graph.get_node(route.endpoint_id), maximize)
        for route in candidates
    ]
    best_nodes = [compact_best_node(route, graph, maximize) for route in candidates]
    best_keys = [fitness_percentile_key(node, maximize) for node in best_nodes]
    denominator = w.endpoint_quality + w.best_quality
    scored: list[Trajectory] = []
    for route, best_node in zip(candidates, best_nodes):
        endpoint_percentile = tie_aware_percentile(
            fitness_percentile_key(graph.get_node(route.endpoint_id), maximize),
            endpoint_keys,
        )
        best_percentile = tie_aware_percentile(
            fitness_percentile_key(best_node, maximize),
            best_keys,
        )
        quality = (
            w.endpoint_quality * endpoint_percentile + w.best_quality * best_percentile
        ) / denominator
        scored.append(memory.set_value(route.id, ValueVec(quality=quality)))
    return tuple(
        sorted(
            scored,
            key=lambda route: (
                -(route.value.quality if route.value else 0.0),
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
    log_term = math.log1p(max(0, batch_count))
    adjusted = [
        float(route.value.quality if route.value else 0.0)
        + w.ucb_c * math.sqrt(log_term / (1.0 + max(0, route.visit_count)))
        for route in scored
    ]
    probabilities = _probabilities(_softmax_weights(adjusted, temperature))
    return tuple(zip(scored, adjusted, probabilities))


def reference_sampling_distribution(
    *,
    primary: Trajectory,
    active: tuple[Trajectory, ...],
    temperature: float,
) -> tuple[tuple[Trajectory, float, float], ...]:
    """Sample any other active route according to its Q value."""
    candidates = [route for route in active if route.id != primary.id]
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


def quality_survivor_sample(
    candidates: list[Trajectory],
    count: int,
    *,
    temperature: float,
    rng: random.Random,
) -> tuple[Trajectory, ...]:
    """Sample survivors without replacement from the Q-softmax distribution."""
    remaining = list(candidates)
    selected: list[Trajectory] = []
    while remaining and len(selected) < count:
        scores = [
            float(route.value.quality if route.value else 0.0) for route in remaining
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
    "fitness_percentile_key",
    "is_program_better",
    "program_quality_key",
    "quality_survivor_sample",
    "reference_sampling_distribution",
    "score_active_trajectories",
    "tie_aware_percentile",
    "trajectory_sampling_distribution",
    "weighted_choice",
]
