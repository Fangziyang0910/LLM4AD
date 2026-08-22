"""Fixed-intent and quality-only parent allocation for TraceAAD V9.16."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schema import (
    ESS_FRACTION,
    EXPLORE_PROBABILITY,
    LANDING_PROBABILITY,
    MIN_ESS_TARGET,
    Algorithm,
    Intent,
)
from .tree import Tree

LN_BETA_LOW = -30.0
LN_BETA_HIGH = 30.0
BETA_ITERATIONS = 80


def selection_score(tree: Tree, algorithm: Algorithm) -> float:
    """The ordinary parent score is the current true quality q(a)."""
    return tree.quality(algorithm)


def boltzmann_probabilities(beta: float, scores: list[float]) -> list[float]:
    peak = max(scores)
    weights = [math.exp(beta * (score - peak)) for score in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def effective_sample_size(probabilities: list[float]) -> float:
    return 1.0 / sum(p * p for p in probabilities)


def solve_beta(scores: list[float], target_ess: float) -> float:
    """Solve for the inverse temperature whose ESS reaches the target."""
    if len(scores) <= 1 or max(scores) == min(scores):
        return 0.0
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
    return min(float(pool_size), max(ESS_FRACTION * pool_size, MIN_ESS_TARGET))


def landing_ticket(*, seed: int | None, entry_id: int) -> bool:
    """Draw the one deterministic ticket associated with an Explore entry."""
    return random.Random(f"{seed}:landing:{entry_id}").random() < LANDING_PROBABILITY


@dataclass(frozen=True, slots=True)
class Decision:
    intent: Intent
    parent: Algorithm
    p_explore: float
    beta: float
    ess: float
    n_valid: int
    parent_q: float


def decide(
    tree: Tree,
    *,
    seed: int | None,
    n_eval: int,
    decision_index: int | None = None,
) -> Decision:
    """Draw one fixed-mixture intent, then sample a q-ranked parent.

    The derived ordinary-decision stream is independent of landing tickets.
    ``decision_index`` is persisted by the caller so checkpoint recovery does
    not change the random stream when a slot is retried.
    """
    algorithms = tree.valid_algorithms()
    if not algorithms:
        raise ValueError("cannot allocate budget without an algorithm")
    index = n_eval if decision_index is None else decision_index
    rng = random.Random(f"{seed}:ordinary:{index}")
    intent = (
        Intent.EXPLORE if rng.random() < EXPLORE_PROBABILITY else Intent.REFINE
    )
    scores = [selection_score(tree, algorithm) for algorithm in algorithms]
    beta = solve_beta(scores, target_ess(len(scores)))
    probabilities = boltzmann_probabilities(beta, scores)
    marker = rng.random()
    cumulative = 0.0
    chosen = len(algorithms) - 1
    for i, probability in enumerate(probabilities):
        cumulative += probability
        if marker <= cumulative:
            chosen = i
            break
    parent = algorithms[chosen]
    return Decision(
        intent=intent,
        parent=parent,
        p_explore=EXPLORE_PROBABILITY,
        beta=beta,
        ess=effective_sample_size(probabilities),
        n_valid=len(algorithms),
        parent_q=tree.quality(parent),
    )


__all__ = [
    "Decision",
    "boltzmann_probabilities",
    "decide",
    "effective_sample_size",
    "landing_ticket",
    "selection_score",
    "solve_beta",
    "target_ess",
]
