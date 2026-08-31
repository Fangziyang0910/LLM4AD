"""Route-then-anchor allocation: q + s / sqrt(n + 1) at both levels."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .forest import Forest


@dataclass(frozen=True, slots=True)
class Score:
    """One route or anchor allocation score: id is a root or anchor id."""

    id: int
    q: float
    n: int
    optimism: float
    score: float


@dataclass(frozen=True, slots=True)
class Choice:
    anchor_id: int
    route_id: int
    routes: tuple[Score, ...]
    anchors: tuple[Score, ...]


def score_routes(forest: Forest, s: float) -> tuple[Score, ...]:
    best: dict[int, float] = {}
    spent: dict[int, int] = {}
    for anchor in forest.anchors():
        q = forest.get_program(anchor.program_id).q
        best[anchor.root_id] = max(best.get(anchor.root_id, -float("inf")), q)
        spent[anchor.root_id] = spent.get(anchor.root_id, 0) + anchor.n

    result: list[Score] = []
    for root in forest.root_ids:
        q = best[root]
        n = spent[root]
        opt = s / math.sqrt(n + 1)
        result.append(Score(root, q, n, opt, q + opt))
    return tuple(result)


def score_anchors(forest: Forest, s: float, selected_route: int) -> tuple[Score, ...]:
    scored: list[Score] = []
    for anchor in forest.anchors():
        if anchor.root_id == selected_route:
            q = forest.get_program(anchor.program_id).q
            opt = s / math.sqrt(anchor.n + 1)
            scored.append(Score(anchor.id, q, anchor.n, opt, q + opt))
    return tuple(scored)


def select(forest: Forest, s: float) -> Choice:
    routes = score_routes(forest, s)
    if not routes:
        raise ValueError("cannot allocate budget without an anchor")

    def pick(scores: tuple[Score, ...]) -> Score:
        return max(
            scores,
            key=lambda item: (
                item.score,
                -item.n,
                -forest.get_anchor(item.id).order,
                -item.id,
            ),
        )

    chosen_route = pick(routes)
    anchors = score_anchors(forest, s, chosen_route.id)
    chosen = pick(anchors)
    return Choice(
        anchor_id=chosen.id,
        route_id=chosen_route.id,
        routes=routes,
        anchors=anchors,
    )


__all__ = [
    "Choice",
    "Score",
    "score_anchors",
    "score_routes",
    "select",
]
