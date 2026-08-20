"""Route-then-anchor allocation with progress-conditioned operator choice."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .forest import Forest
from .schema import EXPLORE_PROBABILITY_MAX, EXPLORE_PROBABILITY_MIN, Intent, PROGRESS_WINDOW


@dataclass(frozen=True, slots=True)
class RouteScore:
    route_id: int
    q: float
    n: int
    optimism: float
    score: float


@dataclass(frozen=True, slots=True)
class AnchorScore:
    anchor_id: int
    q: float
    n: int
    optimism: float
    score: float


@dataclass(frozen=True, slots=True)
class Choice:
    intent: Intent
    anchor_id: int
    route_id: int
    routes: tuple[RouteScore, ...]
    anchors: tuple[AnchorScore, ...]
    refine_failure_evidence: float
    progress_observations: int
    explore_probability: float


def score_routes(forest: Forest, s: float) -> tuple[RouteScore, ...]:
    best: dict[int, float] = {}
    spent: dict[int, int] = {}
    for anchor in forest.anchors():
        q = forest.get_program(anchor.program_id).q
        best[anchor.root_id] = max(best.get(anchor.root_id, -float("inf")), q)
        spent[anchor.root_id] = spent.get(anchor.root_id, 0) + anchor.n

    result: list[RouteScore] = []
    for root in forest.root_ids:
        q = best[root]
        n = spent[root]
        opt = s / math.sqrt(n + 1)
        result.append(RouteScore(root, q, n, opt, q + opt))
    return tuple(result)


def score_anchors(forest: Forest, s: float, selected_route: int) -> tuple[AnchorScore, ...]:
    scored: list[AnchorScore] = []
    for a in forest.anchors():
        if a.root_id == selected_route:
            q = forest.get_program(a.program_id).q
            opt = s / math.sqrt(a.n + 1)
            scored.append(AnchorScore(a.id, q, a.n, opt, q + opt))
    return tuple(scored)


def _refinement_segment_anchor_ids(forest: Forest, anchor_id: int) -> set[int]:
    path = forest.ancestor_ids(anchor_id)
    segment_start = 0
    for index, path_anchor_id in enumerate(path[1:], start=1):
        attempt_id = forest.get_anchor(path_anchor_id).attempt_id
        if (
            attempt_id is not None
            and forest.get_attempt(attempt_id).intent == Intent.EXPLORE.value
        ):
            segment_start = index
    return set(path[segment_start:])


def refine_failure_evidence(
    forest: Forest,
    anchor_id: int,
    *,
    window: int = PROGRESS_WINDOW,
) -> tuple[float, int]:
    if window <= 0:
        raise ValueError("progress window must be positive")
    segment = _refinement_segment_anchor_ids(forest, anchor_id)
    attempts = sorted(
        (
            attempt
            for attempt in forest.attempts()
            if attempt.anchor_id in segment
            and attempt.stage == "search"
            and attempt.intent == Intent.REFINE.value
        ),
        key=lambda a: a.order,
    )[-window:]
    if not attempts:
        return 0.0, 0
    successes = sum(a.child_id is not None and a.dq is not None and a.dq > 0 for a in attempts)
    return (len(attempts) - successes) / window, len(attempts)


def explore_probability(
    refine_failure_evidence: float,
    *,
    minimum: float = EXPLORE_PROBABILITY_MIN,
    maximum: float = EXPLORE_PROBABILITY_MAX,
) -> float:
    if not 0.0 <= refine_failure_evidence <= 1.0:
        raise ValueError("Refine failure evidence must be in [0, 1]")
    if not 0.0 <= minimum <= maximum <= 1.0:
        raise ValueError("invalid Explore probability bounds")
    return minimum + (maximum - minimum) * refine_failure_evidence


def unit_interval(seed: int | None, iteration: int, salt: str) -> float:
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(f"v9.12:{salt}:{token}:{iteration}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def select(
    forest: Forest,
    s: float,
    *,
    seed: int | None,
    iteration: int,
    progress_window: int = PROGRESS_WINDOW,
) -> Choice:
    routes = score_routes(forest, s)
    if not routes:
        raise ValueError("cannot allocate budget without an anchor")
    route = max(
        routes,
        key=lambda x: (
            x.score,
            -x.n,
            -forest.get_anchor(x.route_id).order,
            -x.route_id,
        ),
    )
    anchors = score_anchors(forest, s, route.route_id)
    anchor = max(
        anchors,
        key=lambda x: (
            x.score,
            -x.n,
            -forest.get_anchor(x.anchor_id).order,
            -x.anchor_id,
        ),
    )
    failure_evidence, observations = refine_failure_evidence(
        forest,
        anchor.anchor_id,
        window=progress_window,
    )
    probability = explore_probability(failure_evidence)
    intent = (
        Intent.EXPLORE
        if unit_interval(seed, iteration, "operator") < probability
        else Intent.REFINE
    )
    return Choice(
        intent,
        anchor.anchor_id,
        route.route_id,
        routes,
        anchors,
        failure_evidence,
        observations,
        probability,
    )


def allocation_diagnostics(choice: Choice) -> dict[str, Any]:
    selected = next(item for item in choice.anchors if item.anchor_id == choice.anchor_id)
    return {
        "selected_anchor_id": choice.anchor_id,
        "selected_route_id": choice.route_id,
        "selected_q": selected.q,
        "refine_failure_evidence": choice.refine_failure_evidence,
        "progress_observations": choice.progress_observations,
        "explore_probability": choice.explore_probability,
        "intent": choice.intent.value,
    }


__all__ = [
    "AnchorScore",
    "Choice",
    "RouteScore",
    "allocation_diagnostics",
    "explore_probability",
    "refine_failure_evidence",
    "score_anchors",
    "score_routes",
    "select",
    "unit_interval",
]
