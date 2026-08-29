"""Parent allocation and action selection for TraceAAD V9.19."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .landscape import RegionStats
from .schema import (
    ESS_FRACTION,
    CROSSOVER_PROBABILITY,
    EXPLORE_MAX,
    EXPLORE_MIN,
    EXPLORE_SLOPE,
    MIN_ESS_TARGET,
    TRAJECTORY_WINDOW,
    W_PROMISE,
    W_TRAJECTORY,
    W_UNDERDEVELOPMENT,
    Action,
    Algorithm,
)
from .tree import Tree

LN_BETA_LOW = -30.0
LN_BETA_HIGH = 30.0
BETA_ITERATIONS = 80


def edge_value(*, improved: bool, novelty: float) -> float:
    """v_i = y_i + 0.5 (1 - y_i) ν_i."""
    return 1.0 if improved else 0.5 * novelty


def _edge_value_of(tree: Tree, node_id: int) -> float:
    node = tree.get_algorithm(node_id)
    parent = tree.get_algorithm(node.parent_id)
    return edge_value(
        improved=tree.quality(node) > tree.quality(parent),
        novelty=0.0 if node.novelty is None else node.novelty,
    )


def t_response(tree: Tree, node_id: int, *, window: int = TRAJECTORY_WINDOW) -> float:
    """T(a) = (1 + sum v_i) / (2 + h) over the last h formation edges."""
    path = tree.formation_path(node_id)
    h = min(window, len(path))
    if h == 0:
        return 0.5
    total = sum(_edge_value_of(tree, edge_node) for edge_node in path[-h:])
    return (1.0 + total) / (2.0 + h)


def node_score(stats: RegionStats, trajectory: dict[int, float]) -> dict[int, float]:
    """S(a) = 0.75 P + 0.10 U + 0.15 T."""
    return {
        node_id: (
            W_PROMISE * stats.promise[node_id]
            + W_UNDERDEVELOPMENT * stats.underdevelopment[node_id]
            + W_TRAJECTORY * trajectory[node_id]
        )
        for node_id in stats.q
    }


def boltzmann_probabilities(beta: float, scores: list[float]) -> list[float]:
    peak = max(scores)
    weights = [math.exp(beta * (score - peak)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def effective_sample_size(probabilities: list[float]) -> float:
    return 1.0 / sum(p * p for p in probabilities)


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


def target_ess(pool_size: int) -> float:
    return min(float(pool_size), max(ESS_FRACTION * pool_size, MIN_ESS_TARGET))


def explore_probability(t_value: float) -> float:
    """p_E = clip(0.60 - 0.60 T, 0.10, 0.60). Neutral T=0.5 gives 0.30."""
    return min(EXPLORE_MAX, max(EXPLORE_MIN, EXPLORE_SLOPE - EXPLORE_SLOPE * t_value))


@dataclass(frozen=True, slots=True)
class ParentDecision:
    parent: Algorithm
    beta: float
    ess: float
    decision_index: int
    marker: float
    snapshot: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: Action
    t_response: float
    p_explore: float
    p_crossover: float
    p_develop: float
    draw: float


def decide_action(
    *, t_value: float, rng: random.Random, allow_crossover: bool = True
) -> ActionDecision:
    p_explore = explore_probability(t_value)
    # Reserve a stable share for behavior-conditioned transfer.  The
    # remaining mass is development, so T still controls the explore/develop
    # trade-off without crowding crossover out when a node is stale.
    p_crossover = (
        min(CROSSOVER_PROBABILITY, max(0.0, 1.0 - p_explore))
        if allow_crossover
        else 0.0
    )
    p_develop = max(0.0, 1.0 - p_explore - p_crossover)
    draw = rng.random()
    if draw < p_explore:
        action = Action.EXPLORE
    elif draw < p_explore + p_crossover:
        action = Action.CROSSOVER
    else:
        action = Action.DEVELOP
    return ActionDecision(
        action=action,
        t_response=t_value,
        p_explore=p_explore,
        p_crossover=p_crossover,
        p_develop=p_develop,
        draw=draw,
    )


def trajectory_response(algorithm: Algorithm) -> float:
    """Combine formation response with the node's latest direct outcomes.

    The original V9.19 stored ``T`` only at node creation.  A node that was
    subsequently selected many times therefore kept receiving a high score
    after repeated failures.  Laplace smoothing keeps new nodes neutral and
    lets direct opportunity outcomes lower or raise the effective response.
    """
    attempts = algorithm.successful_opportunities + algorithm.failed_opportunities
    if attempts == 0:
        return algorithm.t_response
    direct = (1.0 + algorithm.successful_opportunities) / (2.0 + attempts)
    return 0.5 * algorithm.t_response + 0.5 * direct


def sample_parent(
    *,
    tree: Tree,
    stats: RegionStats,
    trajectory: dict[int, float],
    rng: random.Random,
    decision_index: int,
) -> ParentDecision:
    scores = node_score(stats, trajectory)
    ordered = list(scores)
    values = [scores[node_id] for node_id in ordered]
    beta = solve_beta(values, target_ess(len(ordered)))
    probabilities = boltzmann_probabilities(beta, values)
    marker = rng.random()
    cumulative = 0.0
    chosen_position = len(ordered) - 1
    for position, probability in enumerate(probabilities):
        cumulative += probability
        if marker <= cumulative:
            chosen_position = position
            break
    snapshot = tuple(
        {
            "id": node_id,
            "q": stats.q[node_id],
            "P": stats.promise[node_id],
            "U": stats.underdevelopment[node_id],
            "B": stats.raw_coverage[node_id],
            "c_t": tree.get_algorithm(node_id).opportunities,
            "T": trajectory[node_id],
            "S": scores[node_id],
        }
        for node_id in ordered
    )
    return ParentDecision(
        parent=tree.get_algorithm(ordered[chosen_position]),
        beta=beta,
        ess=effective_sample_size(probabilities),
        decision_index=decision_index,
        marker=marker,
        snapshot=snapshot,
    )


__all__ = [
    "ActionDecision",
    "ParentDecision",
    "boltzmann_probabilities",
    "decide_action",
    "edge_value",
    "effective_sample_size",
    "explore_probability",
    "node_score",
    "sample_parent",
    "solve_beta",
    "t_response",
    "target_ess",
    "trajectory_response",
]
