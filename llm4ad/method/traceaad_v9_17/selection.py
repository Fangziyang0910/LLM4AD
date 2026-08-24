"""Hypothesis competition and within-hypothesis allocation for V9.17."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .schema import Algorithm, Hypothesis
from .tree import Tree


def rank_hypotheses(hypotheses: Iterable[Hypothesis]) -> list[Hypothesis]:
    return sorted(hypotheses, key=lambda item: (-item.best_quality, item.id))


def competition_line(hypotheses: Iterable[Hypothesis]) -> float | None:
    items = tuple(hypotheses)
    return min((item.best_quality for item in items), default=None)


def refine_score(tree: Tree, algorithm: Algorithm, scale: float) -> float:
    return tree.quality(algorithm) + scale / math.sqrt(algorithm.refine_count + 1)


def select_refine_parent(
    tree: Tree, hypothesis_id: int, *, scale: float
) -> Algorithm:
    algorithms = tree.hypothesis_algorithms(hypothesis_id)
    if not algorithms:
        raise RuntimeError(f"hypothesis {hypothesis_id} has no valid algorithms")
    return max(
        algorithms,
        key=lambda item: (
            refine_score(tree, item, scale),
            -item.refine_count,
            -item.id,
        ),
    )


__all__ = [
    "competition_line",
    "rank_hypotheses",
    "refine_score",
    "select_refine_parent",
]
