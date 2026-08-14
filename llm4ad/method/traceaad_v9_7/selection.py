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
        if anchor.root_id not in best or q > best[anchor.root_id]:
            best[anchor.root_id] = q
        spent[anchor.root_id] = spent.get(anchor.root_id, 0) + anchor.n
    return tuple(
        RouteScore(
            route_id=root,
            q=best[root],
            n=spent[root],
            optimism=s / math.sqrt(spent[root] + 1),
            score=best[root] + s / math.sqrt(spent[root] + 1),
        )
        for root in forest.root_ids
    )


def score_anchors(forest: Forest, s: float, selected_route: int) -> tuple[AnchorScore, ...]:
    scored: list[AnchorScore] = []
    for anchor in forest.anchors():
        if anchor.root_id != selected_route:
            continue
        q = forest.get_program(anchor.program_id).q
        optimism = s / math.sqrt(anchor.n + 1)
        scored.append(
            AnchorScore(
                anchor_id=anchor.id,
                q=q,
                n=anchor.n,
                optimism=optimism,
                score=q + optimism,
            )
        )
    return tuple(scored)


def select(forest: Forest, s: float) -> Choice:
    routes = score_routes(forest, s)
    if not routes:
        raise ValueError("cannot allocate budget without an anchor")

    def route_key(item: RouteScore) -> tuple[float, int, int, int]:
        root = forest.get_anchor(item.route_id)
        return (item.score, -item.n, -root.order, -root.id)

    chosen_route = max(routes, key=route_key)
    anchors = score_anchors(forest, s, chosen_route.route_id)

    def anchor_key(item: AnchorScore) -> tuple[float, int, int, int]:
        anchor = forest.get_anchor(item.anchor_id)
        return (item.score, -item.n, -anchor.order, -anchor.id)

    chosen = max(anchors, key=anchor_key)
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
