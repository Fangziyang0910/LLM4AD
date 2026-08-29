"""Opportunity allocation and evidence-driven action selection for V9.20."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .landscape import RegionStats
from .schema import (
    ACTION_TEMPERATURE,
    COVERAGE_MIX,
    ESS_FRACTION,
    MIN_ESS_TARGET,
    Action,
    Algorithm,
)
from .tree import Tree

LN_BETA_LOW = -30.0
LN_BETA_HIGH = 30.0
BETA_ITERATIONS = 80


def continuation_value(algorithm: Algorithm) -> float:
    """Smoothed probability that a direct edit improves its parent."""
    return (1.0 + algorithm.improvements) / (2.0 + algorithm.opportunities)


def uncertainty_value(algorithm: Algorithm) -> float:
    """One-step uncertainty bonus; new nodes receive the largest bonus."""
    return 1.0 / math.sqrt(1.0 + algorithm.opportunities)


def target_ess(pool_size: int) -> float:
    return min(float(pool_size), max(ESS_FRACTION * pool_size, MIN_ESS_TARGET))


def boltzmann_probabilities(beta: float, scores: list[float]) -> list[float]:
    peak = max(scores)
    weights = [math.exp(beta * (score - peak)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def effective_sample_size(probabilities: list[float]) -> float:
    return 1.0 / sum(probability * probability for probability in probabilities)


def solve_beta(scores: list[float], target: float) -> float:
    if len(scores) <= 1 or max(scores) == min(scores):
        return 0.0
    lo, hi = LN_BETA_LOW, LN_BETA_HIGH
    for _ in range(BETA_ITERATIONS):
        mid = 0.5 * (lo + hi)
        probabilities = boltzmann_probabilities(math.exp(mid), scores)
        if effective_sample_size(probabilities) > target:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def _uniform_or_normalized(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    total = sum(max(0.0, value) for value in values.values())
    if total <= 0.0:
        mass = 1.0 / len(values)
        return {key: mass for key in values}
    return {key: max(0.0, value) / total for key, value in values.items()}


@dataclass(frozen=True, slots=True)
class AllocationStats:
    quality: dict[int, float]
    continuation: dict[int, float]
    coverage: dict[int, float]
    probabilities: dict[int, float]
    beta: float
    ess: float


def allocation_stats(tree: Tree, stats: RegionStats) -> AllocationStats:
    algorithms = tree.valid_algorithms()
    quality = dict(stats.q)
    continuation = {
        algorithm.id: continuation_value(algorithm) for algorithm in algorithms
    }
    raw_coverage = {
        algorithm.id: 1.0 / (1.0 + stats.raw_coverage[algorithm.id])
        for algorithm in algorithms
    }
    coverage = _uniform_or_normalized(raw_coverage)
    promise = {
        node_id: 0.5 * (quality[node_id] + continuation[node_id])
        for node_id in quality
    }
    ordered = list(promise)
    beta = solve_beta(list(promise.values()), target_ess(len(ordered)))
    q_probabilities = boltzmann_probabilities(beta, list(promise.values()))
    probabilities = {
        node_id: (1.0 - COVERAGE_MIX) * q_probability
        + COVERAGE_MIX * coverage[node_id]
        for node_id, q_probability in zip(ordered, q_probabilities)
    }
    return AllocationStats(
        quality=quality,
        continuation=continuation,
        coverage=coverage,
        probabilities=probabilities,
        beta=beta,
        ess=effective_sample_size(list(probabilities.values())),
    )


@dataclass(frozen=True, slots=True)
class ParentDecision:
    parent: Algorithm
    beta: float
    ess: float
    marker: float
    snapshot: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: Action
    utilities: dict[str, float]
    probabilities: dict[str, float]
    draw: float


def reference_utility(
    *,
    parent_id: int,
    reference_id: int | None,
    quality: dict[int, float],
    landscape,
) -> float | None:
    """Return a normalized value for a behavior-different reference.

    The value is deliberately computed only after reference retrieval.  It is
    an action utility, not another parent score: quality and distance are
    each ranked among the possible references and then averaged.
    """
    if reference_id is None:
        return None
    candidates = [node_id for node_id in landscape.node_ids if node_id != parent_id]
    if reference_id not in candidates:
        raise ValueError("reference is not a valid alternative to the parent")
    q_rank = _midrank_percentile({node_id: quality[node_id] for node_id in candidates})
    d_rank = _midrank_percentile(
        {node_id: landscape.distance(parent_id, node_id) for node_id in candidates}
    )
    return 0.5 * q_rank[reference_id] + 0.5 * d_rank[reference_id]


def _midrank_percentile(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    result: dict[int, float] = {}
    index = 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][1] == ordered[index][1]:
            stop += 1
        rank = (index + stop) / 2.0
        value = 0.5 if len(ordered) == 1 else rank / (len(ordered) - 1)
        for position in range(index, stop + 1):
            result[ordered[position][0]] = value
        index = stop + 1
    return result


def sample_parent(
    *,
    tree: Tree,
    stats: RegionStats,
    rng: random.Random,
) -> ParentDecision:
    allocation = allocation_stats(tree, stats)
    ordered = list(allocation.probabilities)
    marker = rng.random()
    cumulative = 0.0
    chosen = ordered[-1]
    for node_id in ordered:
        cumulative += allocation.probabilities[node_id]
        if marker <= cumulative:
            chosen = node_id
            break
    snapshot = tuple(
        {
            "id": node_id,
            "Q": allocation.quality[node_id],
            "C": allocation.continuation[node_id],
            "B": allocation.coverage[node_id],
            "U": stats.underdevelopment[node_id],
            "opportunities": tree.get_algorithm(node_id).opportunities,
            "pi": allocation.probabilities[node_id],
        }
        for node_id in ordered
    )
    return ParentDecision(
        parent=tree.get_algorithm(chosen),
        beta=allocation.beta,
        ess=allocation.ess,
        marker=marker,
        snapshot=snapshot,
    )


def _softmax(utilities: dict[str, float]) -> dict[str, float]:
    available = {key: value for key, value in utilities.items() if math.isfinite(value)}
    if not available:
        raise ValueError("at least one finite action utility is required")
    peak = max(available.values())
    weights = {
        key: math.exp((value - peak) / ACTION_TEMPERATURE)
        for key, value in available.items()
    }
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def decide_action(
    *,
    algorithm: Algorithm,
    reference_value: float | None,
    rng: random.Random,
) -> ActionDecision:
    utilities = {
        Action.DEVELOP.value: continuation_value(algorithm),
        Action.EXPLORE.value: uncertainty_value(algorithm),
    }
    if reference_value is not None:
        utilities[Action.CROSSOVER.value] = float(reference_value)
    probabilities = _softmax(utilities)
    draw = rng.random()
    cumulative = 0.0
    chosen = next(iter(probabilities))
    for action, probability in probabilities.items():
        cumulative += probability
        if draw <= cumulative:
            chosen = action
            break
    return ActionDecision(
        action=Action(chosen),
        utilities=utilities,
        probabilities=probabilities,
        draw=draw,
    )


__all__ = [
    "ActionDecision",
    "AllocationStats",
    "ParentDecision",
    "allocation_stats",
    "boltzmann_probabilities",
    "continuation_value",
    "decide_action",
    "effective_sample_size",
    "sample_parent",
    "reference_utility",
    "solve_beta",
    "target_ess",
    "uncertainty_value",
]
