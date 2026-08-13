"""Route-then-anchor optimistic allocation.

Both levels use the same V9.6 formula ``q + s / sqrt(n + 1)``. The route
level scores each initial-root lineage by its best directed fitness plus an
optimism term that decays with the total generations already spent inside
the route; the anchor level then applies the V9.6 anchor rule within the
selected route. Anchor-level optimism prevents a single state from being
over- or under-visited; route-level optimism prevents the budget from
concentrating long-term on one initial branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .forest import SearchForest


@dataclass(frozen=True, slots=True)
class RouteScore:
    root_state_id: int
    best_directed_fitness: float
    generation_count_n: int
    optimism: float
    score: float


@dataclass(frozen=True, slots=True)
class StateScore:
    state_id: int
    directed_fitness: float
    generation_count_n: int
    optimism: float
    score: float


@dataclass(frozen=True, slots=True)
class AllocationDecision:
    state_id: int
    root_state_id: int
    route_scores: tuple[RouteScore, ...]
    state_scores: tuple[StateScore, ...]


def route_root_id(forest: SearchForest, state_id: int) -> int:
    return forest.ancestor_state_ids(state_id)[0]


def score_routes(forest: SearchForest, optimism_scale: float) -> tuple[RouteScore, ...]:
    best: dict[int, float] = {}
    spent: dict[int, int] = {}
    for state in forest.states():
        root_id = route_root_id(forest, state.state_id)
        quality = forest.get_artifact(state.artifact_id).directed_fitness
        if root_id not in best or quality > best[root_id]:
            best[root_id] = quality
        spent[root_id] = spent.get(root_id, 0) + state.generation_count_n
    scored: list[RouteScore] = []
    for root_id in forest.root_state_ids:
        optimism = optimism_scale / math.sqrt(spent[root_id] + 1)
        scored.append(
            RouteScore(
                root_state_id=root_id,
                best_directed_fitness=best[root_id],
                generation_count_n=spent[root_id],
                optimism=optimism,
                score=best[root_id] + optimism,
            )
        )
    return tuple(scored)


def score_states(
    forest: SearchForest, optimism_scale: float, root_state_id: int
) -> tuple[StateScore, ...]:
    scored: list[StateScore] = []
    for state in forest.states():
        if route_root_id(forest, state.state_id) != root_state_id:
            continue
        artifact = forest.get_artifact(state.artifact_id)
        optimism = optimism_scale / math.sqrt(state.generation_count_n + 1)
        scored.append(
            StateScore(
                state_id=state.state_id,
                directed_fitness=artifact.directed_fitness,
                generation_count_n=state.generation_count_n,
                optimism=optimism,
                score=artifact.directed_fitness + optimism,
            )
        )
    return tuple(scored)


def select_anchor(forest: SearchForest, optimism_scale: float) -> AllocationDecision:
    route_scores = score_routes(forest, optimism_scale)
    if not route_scores:
        raise ValueError("cannot allocate budget without an AnchorState")

    def route_key(item: RouteScore) -> tuple[float, int, int, int]:
        root = forest.get_state(item.root_state_id)
        return (
            item.score,
            -item.generation_count_n,
            -root.creation_order,
            -root.state_id,
        )

    selected_route = max(route_scores, key=route_key)
    state_scores = score_states(
        forest, optimism_scale, selected_route.root_state_id
    )

    def state_key(item: StateScore) -> tuple[float, int, int, int]:
        state = forest.get_state(item.state_id)
        return (
            item.score,
            -item.generation_count_n,
            -state.creation_order,
            -state.state_id,
        )

    selected_state = max(state_scores, key=state_key)
    return AllocationDecision(
        state_id=selected_state.state_id,
        root_state_id=selected_route.root_state_id,
        route_scores=route_scores,
        state_scores=state_scores,
    )


__all__ = [
    "AllocationDecision",
    "RouteScore",
    "StateScore",
    "route_root_id",
    "score_routes",
    "score_states",
    "select_anchor",
]
