"""Quality-guided optimistic allocation over every valid AnchorState."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .forest import SearchForest


@dataclass(frozen=True, slots=True)
class StateScore:
    state_id: int
    directed_fitness: float
    generation_count_n: int
    optimism: float
    score: float


def score_states(forest: SearchForest, optimism_scale: float) -> tuple[StateScore, ...]:
    scored: list[StateScore] = []
    for state in forest.states():
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


def select_anchor(
    forest: SearchForest, optimism_scale: float
) -> tuple[int, tuple[StateScore, ...]]:
    scored = score_states(forest, optimism_scale)
    if not scored:
        raise ValueError("cannot allocate budget without an AnchorState")

    def key(item: StateScore) -> tuple[float, int, int, int]:
        state = forest.get_state(item.state_id)
        return (
            item.score,
            -item.generation_count_n,
            -state.creation_order,
            -state.state_id,
        )

    selected = max(scored, key=key)
    return selected.state_id, scored


__all__ = ["StateScore", "score_states", "select_anchor"]
