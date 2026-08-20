"""Quality-and-count budget allocation."""

from __future__ import annotations

import math

from .schema import Algorithm
from .tree import Tree


def selection_score(tree: Tree, algorithm: Algorithm) -> float:
    return tree.quality(algorithm) + 1.0 / math.sqrt(algorithm.count + 1)


def select(tree: Tree) -> Algorithm:
    algorithms = tree.valid_algorithms()
    if not algorithms:
        raise ValueError("cannot allocate budget without an algorithm")
    return max(
        algorithms,
        key=lambda algorithm: (
            selection_score(tree, algorithm),
            -algorithm.count,
            -algorithm.id,
        ),
    )


__all__ = ["select", "selection_score"]
