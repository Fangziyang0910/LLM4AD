"""Route-then-anchor allocation: q + s / sqrt(n + 1) at both levels."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .forest import Forest


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
    anchor_id: int
    route_id: int
    routes: tuple[RouteScore, ...]
    anchors: tuple[AnchorScore, ...]


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
    for anchor in forest.anchors():
        if anchor.root_id == selected_route:
            q = forest.get_program(anchor.program_id).q
            opt = s / math.sqrt(anchor.n + 1)
            scored.append(AnchorScore(anchor.id, q, anchor.n, opt, q + opt))
    return tuple(scored)


def select(forest: Forest, s: float) -> Choice:
    routes = score_routes(forest, s)
    if not routes:
        raise ValueError("cannot allocate budget without an anchor")

    chosen_route = max(
        routes,
        key=lambda r: (r.score, -r.n, -forest.get_anchor(r.route_id).order, -r.route_id),
    )
    anchors = score_anchors(forest, s, chosen_route.route_id)
    chosen = max(
        anchors,
        key=lambda a: (a.score, -a.n, -forest.get_anchor(a.anchor_id).order, -a.anchor_id),
    )
    return Choice(
        anchor_id=chosen.anchor_id,
        route_id=chosen_route.route_id,
        routes=routes,
        anchors=anchors,
    )


__all__ = [
    "AnchorScore",
    "Choice",
    "RouteScore",
    "score_anchors",
    "score_routes",
    "select",
]
