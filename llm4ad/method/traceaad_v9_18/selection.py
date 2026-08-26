"""Atomic anchor allocation for TraceAAD V9.18-R0."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schema import (
    ESS_FRACTION,
    EXPLORE_PROBABILITY,
    MIN_ESS_TARGET,
    OPPORTUNITY_LAMBDA,
    OPPORTUNITY_TAU,
    Algorithm,
    Intent,
)
from .tree import Tree

LN_BETA_LOW = -30.0
LN_BETA_HIGH = 30.0
BETA_ITERATIONS = 80


def opportunity_value(algorithm: Algorithm, *, tau: float = OPPORTUNITY_TAU) -> float:
    """Return the bounded, decaying entry opportunity used by R0."""
    if not algorithm.is_explore_entry:
        return 0.0
    if tau <= 0:
        raise ValueError("opportunity tau must be positive")
    return math.exp(-algorithm.n_after / tau)


def robust_root_scale(tree: Tree) -> float:
    """Freeze a quality-unit scale from exactly the valid initialization roots."""
    roots = tree.root_algorithms()
    if not roots:
        return 0.0
    values = sorted(tree.quality(root) for root in roots)
    median = values[len(values) // 2]
    deviations = sorted(abs(value - median) for value in values)
    return deviations[len(deviations) // 2]


def selection_score(
    tree: Tree,
    algorithm: Algorithm,
    *,
    sigma_q: float = 0.0,
    allocation_mode: str = "q",
    opportunity_lambda: float = OPPORTUNITY_LAMBDA,
    opportunity_tau: float = OPPORTUNITY_TAU,
) -> float:
    """Compute the pre-decision score without mutating the tree."""
    score = tree.quality(algorithm)
    if allocation_mode == "q":
        return score
    if allocation_mode != "opportunity":
        raise ValueError(f"unknown allocation mode: {allocation_mode}")
    if opportunity_lambda < 0:
        raise ValueError("opportunity lambda must be non-negative")
    return score + opportunity_lambda * sigma_q * opportunity_value(
        algorithm, tau=opportunity_tau
    )


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


@dataclass(frozen=True, slots=True)
class Decision:
    intent: Intent
    parent: Algorithm
    p_explore: float
    beta: float
    ess: float
    n_valid: int
    parent_q: float
    sigma_q: float
    allocation_mode: str
    selected_score: float
    opportunity: float
    decision_index: int
    operator_draw: float
    selection_scores: tuple[tuple[int, float], ...]
    # (algorithm id, q, opportunity, n_after before this decision, S)
    selection_snapshot: tuple[tuple[int, float, float, int, float], ...]


def decide(
    tree: Tree,
    *,
    seed: int | None,
    n_eval: int,
    sigma_q: float = 0.0,
    allocation_mode: str = "q",
    decision_index: int | None = None,
    opportunity_lambda: float = OPPORTUNITY_LAMBDA,
    opportunity_tau: float = OPPORTUNITY_TAU,
) -> Decision:
    """Draw one fixed-mixture intent, then sample one scored anchor."""
    algorithms = tree.valid_algorithms()
    if not algorithms:
        raise ValueError("cannot allocate budget without an algorithm")
    index = n_eval if decision_index is None else decision_index
    rng = random.Random(f"{seed}:ordinary:{index}")
    operator_draw = rng.random()
    intent = (
        Intent.EXPLORE if operator_draw < EXPLORE_PROBABILITY else Intent.REFINE
    )
    scores = [
        selection_score(
            tree,
            algorithm,
            sigma_q=sigma_q,
            allocation_mode=allocation_mode,
            opportunity_lambda=opportunity_lambda,
            opportunity_tau=opportunity_tau,
        )
        for algorithm in algorithms
    ]
    beta = solve_beta(scores, target_ess(len(scores)))
    probabilities = boltzmann_probabilities(beta, scores)
    selection_snapshot = tuple(
        (
            item.id,
            tree.quality(item),
            opportunity_value(item, tau=opportunity_tau),
            item.n_after,
            score,
        )
        for item, score in zip(algorithms, scores)
    )
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
        sigma_q=sigma_q,
        allocation_mode=allocation_mode,
        selected_score=scores[chosen],
        opportunity=opportunity_value(parent, tau=opportunity_tau),
        decision_index=index,
        operator_draw=operator_draw,
        selection_scores=tuple((item.id, score) for item, score in zip(algorithms, scores)),
        selection_snapshot=selection_snapshot,
    )


__all__ = [
    "Decision",
    "boltzmann_probabilities",
    "decide",
    "effective_sample_size",
    "opportunity_value",
    "robust_root_scale",
    "selection_score",
    "solve_beta",
    "target_ess",
]
