"""Anchor-level budget selection for TraceAAD V9.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .schema import ProgramNode
from .tree import FactGraph

QUALITY_EXPLORATION_INTERVAL: Final[int] = 5


@dataclass(frozen=True, slots=True)
class AnchorSelection:
    selected_node_id: int
    quality_rank: int
    quality_pool_ids: tuple[int, ...]
    mode: str
    budget_value: float


def anchor_rank_key(node: ProgramNode) -> tuple[float, float, int, int]:
    return (
        -node.budget_value,
        -node.directed_fitness,
        node.program_loc,
        node.creation_order,
    )


def quality_pool(
    graph: FactGraph,
    *,
    eligible_node_ids: set[int] | None = None,
    pool_size: int = 10,
) -> tuple[ProgramNode, ...]:
    nodes = (
        graph.nodes()
        if eligible_node_ids is None
        else tuple(graph.get_node(node_id) for node_id in eligible_node_ids)
    )
    return tuple(sorted(nodes, key=anchor_rank_key)[:pool_size])


def select_anchor(
    graph: FactGraph,
    *,
    eligible_node_ids: set[int] | None = None,
    pool_size: int = 10,
    decision_order: int = 0,
) -> AnchorSelection:
    pool = quality_pool(graph, eligible_node_ids=eligible_node_ids, pool_size=pool_size)
    if not pool:
        raise ValueError("cannot select from an empty anchor store")
    untested = [node for node in pool if node.budget_event_count == 0]
    if len(pool) > 1 and (decision_order + 1) % QUALITY_EXPLORATION_INTERVAL == 0:
        selected = min(
            pool,
            key=lambda node: (
                node.budget_event_count,
                node.creation_order,
                anchor_rank_key(node),
            ),
        )
        mode = "quality_local_exploration"
    elif untested:
        selected = untested[0]
        mode = "basic_validation"
    else:
        selected = pool[0]
        mode = "anchor_productivity"
    return AnchorSelection(
        selected_node_id=selected.id,
        quality_rank=pool.index(selected),
        quality_pool_ids=tuple(node.id for node in pool),
        mode=mode,
        budget_value=selected.budget_value,
    )


__all__ = [
    "QUALITY_EXPLORATION_INTERVAL",
    "AnchorSelection",
    "anchor_rank_key",
    "quality_pool",
    "select_anchor",
]
