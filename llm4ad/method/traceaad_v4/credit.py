"""计算单步有向适应度变化。"""

from __future__ import annotations


def directed_delta(
    parent_fitness: float | None, child_fitness: float | None, maximize: bool
) -> float | None:
    if parent_fitness is None or child_fitness is None:
        return None
    return (
        child_fitness - parent_fitness if maximize else parent_fitness - child_fitness
    )
