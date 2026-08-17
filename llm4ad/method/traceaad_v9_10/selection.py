"""Discounted Thompson joint anchor-intent allocation for TraceAAD V9.10.

The joint arm z = (anchor, intent) is the only online investment unit. Each
arm keeps a discounted Beta pseudo-posterior over the short-delay success of
actions started on the arm's parent chain; per response all arms are sampled
once and normalized into the joint allocation omega.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from scipy import special

from .forest import Forest
from .schema import (
    CHILD_WINDOW,
    EXPLORE_PRIOR,
    PARENT_CHAIN_HALF_LIFE,
    PARENT_CHAIN_WINDOW,
    PRIOR_STRENGTH,
    RECENCY_HALF_LIFE,
    REFINE_PRIOR,
    Action,
    ActionStatus,
    Intent,
)


@dataclass(slots=True)
class ArmScore:
    anchor_id: int
    program_id: int
    intent: Intent
    a_post: float
    b_post: float
    posterior_mean: float
    n_settled: int
    evidence_mass: float
    n_selected: int
    theta: float
    omega: float = 0.0
    mu: float = 0.0
    pi: float = 0.0


@dataclass(frozen=True, slots=True)
class Choice:
    intent: Intent
    anchor_id: int
    order: int
    arms: tuple[ArmScore, ...]
    action_stats: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        selected = next(
            item
            for item in self.arms
            if item.anchor_id == self.anchor_id and item.intent is self.intent
        )
        return {
            "intent": self.intent.value,
            "selected_anchor_id": self.anchor_id,
            "decision_order": self.order,
            "selected_arm": asdict(selected),
            "diagnostics": {
                **allocation_diagnostics(self.arms, self.anchor_id, self.intent),
                **self.action_stats,
            },
        }


@dataclass(frozen=True, slots=True)
class Posterior:
    intent: Intent
    a_post: float
    b_post: float
    n_settled: int
    evidence_mass: float


def prior_counts(intent: Intent) -> tuple[float, float]:
    """Intent prior (alpha_0, beta_0) shared by every new joint arm."""
    prior = REFINE_PRIOR if intent is Intent.REFINE else EXPLORE_PRIOR
    return PRIOR_STRENGTH * prior, PRIOR_STRENGTH * (1.0 - prior)


def parent_chain_window(forest: Forest, anchor_id: int) -> tuple[int, ...]:
    """The anchor plus its most recent PARENT_CHAIN_WINDOW ancestors, nearest first."""
    chain = forest.ancestor_ids(anchor_id)
    return tuple(reversed(chain[-(PARENT_CHAIN_WINDOW + 1) :]))


def start_quality(forest: Forest, action: Action) -> float:
    anchor = forest.get_anchor(action.anchor_id)
    return forest.get_program(anchor.program_id).q


def settle_pending_actions(forest: Forest, *, now_order: int) -> tuple[Action, ...]:
    """Settle pending actions against their current short-delay observation window.

    Settlement is one-way: an action settles at most once, either as a success
    when some observed descendant within CHILD_WINDOW generations beats the
    start anchor's quality, or as a failure once the window depth is reached
    without such a descendant. Childless actions (invalid, no-op, ancestral
    return, repeated connection) settle as immediate failures.
    """
    settled: list[Action] = []
    for action in forest.actions():
        if action.status is not ActionStatus.PENDING:
            continue
        if action.child_id is None:
            action.observed_depth = 0
            action.settle(0, now_order)
            settled.append(action)
            continue
        best_q, depth = forest.window_stats(action.child_id, max_depth=CHILD_WINDOW)
        action.observed_depth = depth
        action.window_best_q = best_q
        if best_q > start_quality(forest, action):
            action.settle(1, now_order)
            settled.append(action)
        elif depth >= CHILD_WINDOW:
            action.settle(0, now_order)
            settled.append(action)
    return tuple(settled)


def discounted_posterior(
    forest: Forest, anchor_id: int, intent: Intent, *, now_order: int
) -> Posterior:
    """Beta pseudo-posterior of one joint arm at the upcoming response order.

    Settled actions started on the anchor's parent-chain window with the same
    intent enter as weighted pseudo-counts: parent-chain distance decays with
    PARENT_CHAIN_HALF_LIFE and response age with RECENCY_HALF_LIFE.
    """
    alpha, beta = prior_counts(intent)
    n_settled = 0
    evidence_mass = 0.0
    for distance, chain_anchor_id in enumerate(parent_chain_window(forest, anchor_id)):
        distance_weight = 2.0 ** (-distance / PARENT_CHAIN_HALF_LIFE)
        for action in forest.actions_for_arm(chain_anchor_id, intent):
            if action.status is not ActionStatus.SETTLED:
                continue
            recency = 2.0 ** (-(now_order - action.order) / RECENCY_HALF_LIFE)
            weight = distance_weight * recency
            evidence_mass += weight
            n_settled += 1
            if action.result == 1:
                alpha += weight
            else:
                beta += weight
    return Posterior(
        intent=intent,
        a_post=alpha,
        b_post=beta,
        n_settled=n_settled,
        evidence_mass=evidence_mass,
    )


def unit_interval(seed: int | None, order: int, salt: str) -> float:
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(f"v9.10:{salt}:{token}:{order}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def beta_quantile(draw: float, alpha: float, beta: float) -> float:
    value = float(special.betaincinv(alpha, beta, draw))
    if not math.isfinite(value):
        return alpha / (alpha + beta)
    return min(max(value, 0.0), 1.0)


def sample_index(weights: Sequence[float], draw: float) -> int:
    if not weights:
        raise ValueError("cannot sample from an empty distribution")
    total = sum(weights)
    threshold = draw * total
    cumulative = 0.0
    last = len(weights) - 1
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold < cumulative or index == last:
            return index
    return last


def score_arms(
    forest: Forest, *, now_order: int, seed: int | None
) -> tuple[ArmScore, ...]:
    """Sample one Beta draw per joint arm and normalize into the joint allocation."""
    arms: list[ArmScore] = []
    for anchor in forest.anchors():
        for intent in Intent:
            posterior = discounted_posterior(forest, anchor.id, intent, now_order=now_order)
            draw = unit_interval(seed, now_order, f"theta:{anchor.id}:{intent.value}")
            arms.append(
                ArmScore(
                    anchor_id=anchor.id,
                    program_id=anchor.program_id,
                    intent=intent,
                    a_post=posterior.a_post,
                    b_post=posterior.b_post,
                    posterior_mean=posterior.a_post / (posterior.a_post + posterior.b_post),
                    n_settled=posterior.n_settled,
                    evidence_mass=posterior.evidence_mass,
                    n_selected=anchor.count(intent),
                    theta=beta_quantile(draw, posterior.a_post, posterior.b_post),
                )
            )
    total = math.fsum(item.theta for item in arms)
    if not math.isfinite(total) or total <= 0.0:
        even = 1.0 / len(arms)
        for item in arms:
            item.omega = even
    else:
        for item in arms:
            item.omega = item.theta / total
    marginals: dict[int, float] = {}
    for item in arms:
        marginals[item.anchor_id] = marginals.get(item.anchor_id, 0.0) + item.omega
    for item in arms:
        item.mu = marginals[item.anchor_id]
        item.pi = 1.0 if item.mu <= 0.0 else item.omega / item.mu
    return tuple(arms)


def action_statistics(forest: Forest) -> dict[str, int]:
    actions = forest.actions()
    pending = success = failure = 0
    for item in actions:
        if item.status is ActionStatus.PENDING:
            pending += 1
        elif item.result == 1:
            success += 1
        else:
            failure += 1
    return {
        "n_actions": len(actions),
        "n_pending": pending,
        "n_settled_success": success,
        "n_settled_failure": failure,
    }


def allocation_diagnostics(
    arms: Sequence[ArmScore], selected_anchor_id: int, selected_intent: Intent
) -> dict[str, Any]:
    if not arms:
        return {}
    ordered = sorted(
        arms, key=lambda item: (-item.omega, item.anchor_id, item.intent.value)
    )
    arm_entropy = -sum(
        item.omega * math.log(item.omega) for item in arms if item.omega > 0.0
    )
    marginals: dict[int, float] = {}
    for item in arms:
        marginals[item.anchor_id] = marginals.get(item.anchor_id, 0.0) + item.omega
    anchor_entropy = -sum(
        value * math.log(value) for value in marginals.values() if value > 0.0
    )
    multiplicity: dict[int, int] = {}
    for item in arms:
        multiplicity[item.program_id] = multiplicity.get(item.program_id, 0) + 1
    selected = next(
        item
        for item in arms
        if item.anchor_id == selected_anchor_id and item.intent is selected_intent
    )

    def share(count: int) -> float:
        return sum(item.omega for item in ordered[:count])

    return {
        "n_anchors": len(marginals),
        "n_arms": len(arms),
        "arm_entropy": arm_entropy,
        "effective_arms": math.exp(arm_entropy),
        "anchor_marginal_entropy": anchor_entropy,
        "effective_anchors": math.exp(anchor_entropy),
        "refine_share": sum(
            item.omega for item in arms if item.intent is Intent.REFINE
        ),
        "explore_share": sum(
            item.omega for item in arms if item.intent is Intent.EXPLORE
        ),
        "top5_omega": share(5),
        "top10_omega": share(10),
        "top20_omega": share(20),
        "max_program_multiplicity": max(multiplicity.values()),
        "repeated_program_omega": sum(
            item.omega for item in arms if multiplicity[item.program_id] > 1
        ),
        "selected_theta": selected.theta,
        "selected_omega": selected.omega,
        "selected_posterior_mean": selected.posterior_mean,
        "selected_n_settled": selected.n_settled,
        "selected_evidence_mass": selected.evidence_mass,
        "selected_n_selected": selected.n_selected,
    }


def select(forest: Forest, *, seed: int | None, order: int) -> Choice:
    """Jointly select one anchor-intent arm for the response at the given order."""
    arms = score_arms(forest, now_order=order, seed=seed)
    if not arms:
        raise ValueError("cannot allocate budget without an anchor")
    ordered = sorted(arms, key=lambda item: (item.anchor_id, item.intent.value))
    draw = unit_interval(seed, order, "arm")
    chosen = ordered[sample_index([item.omega for item in ordered], draw)]
    return Choice(
        intent=chosen.intent,
        anchor_id=chosen.anchor_id,
        order=order,
        arms=arms,
        action_stats=action_statistics(forest),
    )


__all__ = [
    "ArmScore",
    "Choice",
    "Posterior",
    "action_statistics",
    "allocation_diagnostics",
    "beta_quantile",
    "discounted_posterior",
    "parent_chain_window",
    "prior_counts",
    "sample_index",
    "score_arms",
    "select",
    "settle_pending_actions",
    "start_quality",
    "unit_interval",
]
