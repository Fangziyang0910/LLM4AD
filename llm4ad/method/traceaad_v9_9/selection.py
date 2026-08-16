"""Anchor-first joint allocation for TraceAAD V9.9."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from itertools import groupby
from typing import Any, Mapping, Sequence

from .forest import Forest
from .schema import (
    EXPLORE_PRIOR,
    LAMBDA_U,
    PATH_HALF_LIFE,
    PROTOCOL_ID,
    RANK_HALF_LIFE,
    REFINE_PRIOR,
    TEMPERATURE,
    Intent,
)


@dataclass(frozen=True, slots=True)
class AnchorScore:
    anchor_id: int
    q: float
    n_refine: int
    n_explore: int
    u_refine: float
    u_explore: float
    c_refine: float
    s_refine: float
    s_explore: float
    w_refine: float
    w_explore: float
    total: float
    mu: float
    pi_refine: float
    pi_explore: float


@dataclass(frozen=True, slots=True)
class Choice:
    intent: Intent
    anchor_id: int
    scores: tuple[AnchorScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "selected_anchor_id": self.anchor_id,
            "anchors": [asdict(item) for item in self.scores],
        }


def midrank_percentiles(qualities: Mapping[int, float]) -> dict[int, float]:
    items = tuple(qualities.items())
    count = len(items)
    if count == 0:
        return {}
    if count == 1:
        program_id, _ = items[0]
        return {program_id: 0.5}
    values = [quality for _, quality in items]
    ranks: dict[int, float] = {}
    for program_id, quality in items:
        lower = sum(1 for other in values if other < quality)
        equal = sum(1 for other in values if other == quality)
        ranks[program_id] = (lower + (equal - 1) / 2) / (count - 1)
    return ranks


def under_exposure(count: int) -> float:
    return 1.0 / math.sqrt(count + 1)


def distance_grace(
    forest: Forest,
    anchor_id: int,
    ranks: Mapping[int, float],
    *,
    half_life: float = PATH_HALF_LIFE,
) -> float:
    anchor = forest.get_anchor(anchor_id)
    if anchor.parent_id is None:
        return 0.0
    current_rank = ranks[anchor.program_id]
    best = 0.0
    distance = 0
    current = anchor.parent_id
    while current is not None:
        distance += 1
        ancestor = forest.get_anchor(current)
        gap = max(ranks[ancestor.program_id] - current_rank, 0.0)
        decayed = (2.0 ** (-distance / half_life)) * gap
        if decayed > best:
            best = decayed
        current = ancestor.parent_id
    return best


def geometric_rank_weights(
    totals: Sequence[tuple[int, float]],
    *,
    half_life: float = RANK_HALF_LIFE,
) -> dict[int, float]:
    if not totals:
        return {}
    ordered = sorted(totals, key=lambda item: (-item[1], item[0]))
    weights: dict[int, float] = {}
    position = 1
    for _, group in groupby(ordered, key=lambda item: item[1]):
        members = list(group)
        shared = (
            sum(2.0 ** (-(index - 1) / half_life) for index in range(position, position + len(members)))
            / len(members)
        )
        for anchor_id, _ in members:
            weights[anchor_id] = shared
        position += len(members)
    total = sum(weights.values())
    return {anchor_id: weight / total for anchor_id, weight in weights.items()}


def score_anchors(forest: Forest) -> tuple[AnchorScore, ...]:
    live_ids = forest.live_program_ids()
    ranks = midrank_percentiles(
        {program_id: forest.get_program(program_id).q for program_id in live_ids}
    )
    scored: list[AnchorScore] = []
    for anchor in forest.anchors():
        quality = ranks[anchor.program_id]
        u_refine = under_exposure(anchor.n_refine)
        u_explore = under_exposure(anchor.n_explore)
        c_refine = distance_grace(forest, anchor.id, ranks) / math.sqrt(anchor.n_refine + 1)
        s_refine = quality + LAMBDA_U * u_refine + c_refine
        s_explore = quality + LAMBDA_U * u_explore
        w_refine = REFINE_PRIOR * math.exp(s_refine / TEMPERATURE)
        w_explore = EXPLORE_PRIOR * math.exp(s_explore / TEMPERATURE)
        total = w_refine + w_explore
        scored.append(
            AnchorScore(
                anchor_id=anchor.id,
                q=quality,
                n_refine=anchor.n_refine,
                n_explore=anchor.n_explore,
                u_refine=u_refine,
                u_explore=u_explore,
                c_refine=c_refine,
                s_refine=s_refine,
                s_explore=s_explore,
                w_refine=w_refine,
                w_explore=w_explore,
                total=total,
                mu=0.0,
                pi_refine=w_refine / total,
                pi_explore=w_explore / total,
            )
        )
    mu = geometric_rank_weights(tuple((item.anchor_id, item.total) for item in scored))
    return tuple(
        AnchorScore(
            **{
                **asdict(item),
                "mu": mu[item.anchor_id],
            }
        )
        for item in scored
    )


def unit_interval(seed: int | None, iteration: int, salt: str) -> float:
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}:{salt}:{token}:{iteration}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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


def select(forest: Forest, *, seed: int | None, iteration: int) -> Choice:
    scores = score_anchors(forest)
    if not scores:
        raise ValueError("cannot allocate budget without an anchor")
    ordered = sorted(scores, key=lambda item: item.anchor_id)
    chosen = ordered[sample_index(tuple(item.mu for item in ordered), unit_interval(seed, iteration, "anchor"))]
    intent = (
        Intent.REFINE
        if unit_interval(seed, iteration, "operator") < chosen.pi_refine
        else Intent.EXPLORE
    )
    return Choice(intent=intent, anchor_id=chosen.anchor_id, scores=scores)


__all__ = [
    "AnchorScore",
    "Choice",
    "distance_grace",
    "geometric_rank_weights",
    "midrank_percentiles",
    "score_anchors",
    "select",
    "under_exposure",
    "unit_interval",
]
