"""Operator-conditional hypothesis and anchor allocation for TraceAAD V9.8."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .forest import Forest
from .schema import AllocationPolicy, Intent


@dataclass(frozen=True, slots=True)
class HypothesisScore:
    hypothesis_id: int
    q: float
    n: int
    u: float
    c: float
    m: float
    score: float


@dataclass(frozen=True, slots=True)
class RouteScore:
    route_id: int
    q: float
    n: int
    u: float
    score: float


@dataclass(frozen=True, slots=True)
class AnchorScore:
    anchor_id: int
    q: float
    n: int
    u: float
    score: float


@dataclass(frozen=True, slots=True)
class Choice:
    intent: Intent
    hypothesis_id: int
    anchor_id: int
    policy: AllocationPolicy
    hypotheses: tuple[HypothesisScore, ...]
    anchors: tuple[AnchorScore, ...]
    routes: tuple[RouteScore, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "policy": self.policy.value,
            "selected_hypothesis_id": self.hypothesis_id,
            "selected_anchor_id": self.anchor_id,
            "hypotheses": [asdict(item) for item in self.hypotheses],
            "anchors": [asdict(item) for item in self.anchors],
            "routes": [asdict(item) for item in self.routes],
        }


def hypothesis_scores(
    forest: Forest,
    *,
    s0: float,
    intent: Intent,
    policy: AllocationPolicy,
) -> tuple[HypothesisScore, ...]:
    scored: list[HypothesisScore] = []
    for hypothesis in forest.hypotheses():
        q = forest.hypothesis_frontier(hypothesis.id)
        n = hypothesis.count(intent)
        u = s0 / math.sqrt(n + 1)
        c = 0.0
        m = 0.0
        if intent is Intent.REFINE:
            if hypothesis.q_base is not None:
                c = max(hypothesis.q_base - q, 0.0) / math.sqrt(n + 1)
            m = (q - hypothesis.q0) / max(1, hypothesis.n_refine)

        if policy is AllocationPolicy.HYPOTHESIS_UNIFORM:
            score = -float(n)
            used_u = used_c = used_m = 0.0
        else:
            used_u = u
            used_c = (
                c
                if policy in {AllocationPolicy.FULL, AllocationPolicy.Q_U_C}
                else 0.0
            )
            used_m = m if policy is AllocationPolicy.FULL else 0.0
            score = q + used_u + used_c + used_m
        scored.append(
            HypothesisScore(
                hypothesis_id=hypothesis.id,
                q=q,
                n=n,
                u=used_u,
                c=used_c,
                m=used_m,
                score=score,
            )
        )
    return tuple(scored)


def route_scores(
    forest: Forest, *, s0: float, intent: Intent
) -> tuple[RouteScore, ...]:
    scored: list[RouteScore] = []
    for root_id in forest.root_ids:
        anchors = forest.anchors_in_route(root_id)
        q = max(forest.get_program(anchor.program_id).q for anchor in anchors)
        n = sum(anchor.count(intent) for anchor in anchors)
        u = s0 / math.sqrt(n + 1)
        scored.append(RouteScore(route_id=root_id, q=q, n=n, u=u, score=q + u))
    return tuple(scored)


def anchor_scores(
    forest: Forest,
    *,
    s0: float,
    intent: Intent,
    hypothesis_id: int | None = None,
    root_id: int | None = None,
) -> tuple[AnchorScore, ...]:
    if (hypothesis_id is None) == (root_id is None):
        raise ValueError("exactly one allocation boundary must be supplied")
    scored: list[AnchorScore] = []
    for anchor in forest.anchors():
        if hypothesis_id is not None and anchor.hypothesis_id != hypothesis_id:
            continue
        if root_id is not None and anchor.root_id != root_id:
            continue
        q = forest.get_program(anchor.program_id).q
        n = anchor.count(intent)
        u = s0 / math.sqrt(n + 1)
        scored.append(AnchorScore(anchor_id=anchor.id, q=q, n=n, u=u, score=q + u))
    return tuple(scored)


def select(
    forest: Forest,
    *,
    s0: float,
    intent: Intent,
    policy: AllocationPolicy = AllocationPolicy.FULL,
) -> Choice:
    hypotheses = hypothesis_scores(forest, s0=s0, intent=intent, policy=policy)
    if not hypotheses:
        raise ValueError("cannot allocate budget without a hypothesis")

    routes: tuple[RouteScore, ...] = ()
    if policy is AllocationPolicy.ROUTE_Q_U:
        routes = route_scores(forest, s0=s0, intent=intent)

        def route_key(item: RouteScore) -> tuple[float, int, int, int]:
            root = forest.get_anchor(item.route_id)
            return item.score, -item.n, -root.order, -root.id

        selected_route = max(routes, key=route_key)
        anchors = anchor_scores(
            forest, s0=s0, intent=intent, root_id=selected_route.route_id
        )
    else:

        def hypothesis_key(
            item: HypothesisScore,
        ) -> tuple[float, int, float, int, int]:
            hypothesis = forest.get_hypothesis(item.hypothesis_id)
            return item.score, -item.n, item.q, -hypothesis.order, -hypothesis.id

        selected_hypothesis = max(hypotheses, key=hypothesis_key)
        anchors = anchor_scores(
            forest,
            s0=s0,
            intent=intent,
            hypothesis_id=selected_hypothesis.hypothesis_id,
        )

    def anchor_key(item: AnchorScore) -> tuple[float, int, int, int, int]:
        anchor = forest.get_anchor(item.anchor_id)
        program = forest.get_program(anchor.program_id)
        return item.score, -item.n, -program.length, -anchor.order, -anchor.id

    chosen_anchor = max(anchors, key=anchor_key)
    chosen_hypothesis_id = forest.get_anchor(chosen_anchor.anchor_id).hypothesis_id
    return Choice(
        intent=intent,
        hypothesis_id=chosen_hypothesis_id,
        anchor_id=chosen_anchor.anchor_id,
        policy=policy,
        hypotheses=hypotheses,
        anchors=anchors,
        routes=routes,
    )


__all__ = [
    "AnchorScore",
    "Choice",
    "HypothesisScore",
    "RouteScore",
    "anchor_scores",
    "hypothesis_scores",
    "route_scores",
    "select",
]
