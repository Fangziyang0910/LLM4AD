"""Independent single-parent derivation forest for TraceAAD v6."""

from __future__ import annotations

from .complexity import code_hash, nonempty_loc
from .schema import (
    EdgeId,
    ImprovementEdge,
    NodeId,
    OperatorName,
    ProgramNode,
    TrajectoryId,
)


class DerivationGraph:
    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0
        self._nodes: dict[NodeId, ProgramNode] = {}
        self._edges: dict[EdgeId, ImprovementEdge] = {}
        self._incoming_edge_by_child: dict[NodeId, EdgeId] = {}

    def add_node(self, *, code: str, idea: str, fitness: float | None) -> ProgramNode:
        node = ProgramNode(
            id=self._next_node_id,
            code=code,
            idea=idea,
            fitness=fitness,
            program_loc=nonempty_loc(code),
            code_hash=code_hash(code),
        )
        self._nodes[node.id] = node
        self._next_node_id += 1
        return node

    def get_node(self, node_id: NodeId) -> ProgramNode:
        return self._nodes[node_id]

    def nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(self._nodes.values())

    def add_edge(
        self,
        *,
        parent_id: NodeId,
        child_id: NodeId,
        action: str,
        operator: OperatorName,
        anchor_role: str,
        primary_trajectory_id: TrajectoryId,
        reference_trajectory_id: TrajectoryId | None = None,
        reference_program_id: NodeId | None = None,
        delta_parent: float | None = None,
        delta_route_best: float | None = None,
        delta_global_best: float | None = None,
        delta_loc: int = 0,
        code_change_ratio: float = 0.0,
        outcome: str = "unknown",
        edge_credit: float = 0.0,
        iteration: int | None = None,
        new_global_best: bool = False,
        global_best_update_reason: str | None = None,
    ) -> ImprovementEdge:
        if parent_id not in self._nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        if child_id not in self._nodes:
            raise KeyError(f"unknown child node: {child_id}")
        if parent_id == child_id:
            raise ValueError("an improvement edge cannot point to the same node")
        if child_id in self._incoming_edge_by_child:
            raise ValueError(f"child node already has a parent: {child_id}")
        edge = ImprovementEdge(
            id=self._next_edge_id,
            parent_id=parent_id,
            child_id=child_id,
            operator=operator,
            action=action,
            anchor_role=anchor_role,
            primary_trajectory_id=primary_trajectory_id,
            reference_trajectory_id=reference_trajectory_id,
            reference_program_id=reference_program_id,
            delta_parent=delta_parent,
            delta_route_best=delta_route_best,
            delta_global_best=delta_global_best,
            delta_loc=delta_loc,
            code_change_ratio=code_change_ratio,
            outcome=outcome,
            edge_credit=float(edge_credit),
            iteration=iteration,
            new_global_best=new_global_best,
            global_best_update_reason=global_best_update_reason,
        )
        self._edges[edge.id] = edge
        self._incoming_edge_by_child[child_id] = edge.id
        self._next_edge_id += 1
        return edge

    def get_edge(self, edge_id: EdgeId) -> ImprovementEdge:
        return self._edges[edge_id]

    def edges(self) -> tuple[ImprovementEdge, ...]:
        return tuple(self._edges.values())


__all__ = ["DerivationGraph"]
