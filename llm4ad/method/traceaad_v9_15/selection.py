"""Developmental-state budget allocation for TraceAAD V9.15.

One budget unit answers, in order: which operator intent (stagnation-driven
p_E), which parent algorithm (score q + B + C_traj sampled through a
Boltzmann distribution whose inverse temperature is solved for a target
effective sample size).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schema import (
    BASE_EXPLORE_PROBABILITY,
    BONUS_CAP_SCALE,
    ESS_FRACTION,
    EXPLORE_PROBABILITY_MAX,
    EXPLORE_PROBABILITY_MIN,
    MIN_ESS_TARGET,
    STAGNATION_GAIN,
    STAGNATION_WINDOW,
    TRAJECTORY_WINDOW,
    Algorithm,
    Intent,
)
from .tree import Tree

LN_BETA_LOW = -30.0
LN_BETA_HIGH = 30.0
BETA_ITERATIONS = 80


def explore_probability(n_stag: int) -> float:
    """Smooth stagnation response p_E = clip(p_0 + alpha * r, p_min, p_max)."""
    r_stag = n_stag / (STAGNATION_WINDOW + n_stag)
    value = BASE_EXPLORE_PROBABILITY + STAGNATION_GAIN * r_stag
    return min(max(value, EXPLORE_PROBABILITY_MIN), EXPLORE_PROBABILITY_MAX)


def protection_bonus(
    tree: Tree, algorithm: Algorithm, *, intent: Intent, scale: float | None
) -> float:
    """B(a, o): bounded, fast-decaying survival grace for Explore-born nodes."""
    if intent is not Intent.REFINE or algorithm.created_by != Intent.EXPLORE.value:
        return 0.0
    if scale is None:
        return 0.0
    gap = -tree.formation_gain(algorithm)
    if gap <= 0:
        return 0.0
    return min(gap, BONUS_CAP_SCALE * scale) / (algorithm.refine_count + 1)


def formation_gains(tree: Tree, algorithm: Algorithm) -> tuple[float, ...]:
    """Delta-q over the last min(k, depth - 1) formation steps of the lineage."""
    chain = tree.ancestor_ids(algorithm.id)
    window = min(TRAJECTORY_WINDOW, len(chain) - 2)
    if window <= 0:
        return ()
    lineage = chain[-(window + 1) :]
    return tuple(
        tree.quality(tree.get_algorithm(lineage[i + 1]))
        - tree.quality(tree.get_algorithm(lineage[i]))
        for i in range(window)
    )


def trajectory_bonus(
    tree: Tree, algorithm: Algorithm, *, scale: float | None
) -> float:
    """C_traj(a) = f_succ * mean positive gain * headroom / (headroom + s_t)."""
    gains = formation_gains(tree, algorithm)
    if not gains:
        return 0.0
    f_succ = sum(1 for gain in gains if gain > 0) / len(gains)
    mean_positive = sum(max(0.0, gain) for gain in gains) / len(gains)
    headroom = max(tree.best_quality() - tree.quality(algorithm), 0.0)
    if headroom <= 0 or scale is None:
        return 0.0
    return f_succ * mean_positive * headroom / (headroom + scale)


def selection_score(
    tree: Tree,
    algorithm: Algorithm,
    *,
    intent: Intent,
    scale: float | None,
) -> float:
    return (
        tree.quality(algorithm)
        + protection_bonus(tree, algorithm, intent=intent, scale=scale)
        + trajectory_bonus(tree, algorithm, scale=scale)
    )


def boltzmann_probabilities(beta: float, scores: list[float]) -> list[float]:
    peak = max(scores)
    weights = [math.exp(beta * (score - peak)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def effective_sample_size(probabilities: list[float]) -> float:
    return 1.0 / sum(p * p for p in probabilities)


def solve_beta(scores: list[float], target_ess: float) -> float:
    """Bisection over ln(beta); ESS decreases monotonically as beta grows."""
    lo, hi = LN_BETA_LOW, LN_BETA_HIGH
    for _ in range(BETA_ITERATIONS):
        mid = 0.5 * (lo + hi)
        probabilities = boltzmann_probabilities(math.exp(mid), scores)
        if effective_sample_size(probabilities) > target_ess:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def target_ess(pool_size: int) -> float:
    return max(ESS_FRACTION * pool_size, MIN_ESS_TARGET)


@dataclass(frozen=True, slots=True)
class Decision:
    intent: Intent
    parent: Algorithm
    p_explore: float
    beta: float
    ess: float
    n_valid: int
    parent_q: float
    parent_bonus: float
    parent_ctraj: float


def decide(tree: Tree, *, n_stag: int, seed: int | None, n_eval: int) -> Decision:
    """Draw the operator intent, then sample the parent anchor."""
    if not tree.valid_algorithms():
        raise ValueError("cannot allocate budget without an algorithm")

    p_explore = explore_probability(n_stag)
    rng = random.Random(f"{seed}:{n_eval}")
    intent = Intent.EXPLORE if rng.random() < p_explore else Intent.REFINE

    algorithms = tree.valid_algorithms()
    scale = tree.positive_gain_scale()
    scores = [
        selection_score(tree, algorithm, intent=intent, scale=scale)
        for algorithm in algorithms
    ]
    beta = solve_beta(scores, target_ess(len(algorithms)))
    probabilities = boltzmann_probabilities(beta, scores)

    marker = rng.random()
    index = 0
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if marker <= cumulative:
            break
    parent = algorithms[index]

    return Decision(
        intent=intent,
        parent=parent,
        p_explore=p_explore,
        beta=beta,
        ess=effective_sample_size(probabilities),
        n_valid=len(algorithms),
        parent_q=tree.quality(parent),
        parent_bonus=protection_bonus(
            tree, parent, intent=intent, scale=scale
        ),
        parent_ctraj=trajectory_bonus(tree, parent, scale=scale),
    )


__all__ = [
    "Decision",
    "boltzmann_probabilities",
    "decide",
    "effective_sample_size",
    "explore_probability",
    "formation_gains",
    "protection_bonus",
    "selection_score",
    "solve_beta",
    "target_ess",
    "trajectory_bonus",
]
