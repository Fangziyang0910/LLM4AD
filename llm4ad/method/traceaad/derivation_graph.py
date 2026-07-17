"""DerivationGraph —— Program Memory（结果库，ground truth）。

单父 DAG：每个 child 恰好一条入边。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from .schema import EdgeId, ImprovementEdge, NodeId, ProgramNode


class DerivationGraph:
    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0
        self._nodes: dict[NodeId, ProgramNode] = {}
        self._edges: dict[EdgeId, ImprovementEdge] = {}
        self._incoming_edge_by_child: dict[NodeId, EdgeId] = {}
        self._outgoing_edges_by_parent: dict[NodeId, list[EdgeId]] = defaultdict(list)

    def add_node(self, *, code: str, idea: str, fitness: float | None, **fields) -> ProgramNode:
        metrics = fields.get("complexity_metrics") or {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        node = ProgramNode(
            id=self._next_node_id,
            code=code,
            idea=idea,
            fitness=fitness,
            complexity=float(fields.get("complexity", 0.0) or 0.0),
            runtime=float(fields.get("runtime", 0.0) or 0.0),
            complexity_metrics=dict(metrics),
        )
        self._nodes[node.id] = node
        self._next_node_id += 1
        return node

    def get_node(self, node_id: NodeId) -> ProgramNode:
        return self._nodes[node_id]

    def nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(self._nodes.values())

    def add_edge(self, *, parent_id: NodeId, child_id: NodeId, action: str, **fields) -> ImprovementEdge:
        if parent_id not in self._nodes:
            raise KeyError(f"unknown parent node: {parent_id}")
        if child_id not in self._nodes:
            raise KeyError(f"unknown child node: {child_id}")
        if parent_id == child_id:
            raise ValueError("an improvement edge cannot point to the same node")
        if child_id in self._incoming_edge_by_child:
            raise ValueError(f"child node already has a parent edge: {child_id}")

        edge = ImprovementEdge(
            id=self._next_edge_id,
            parent_id=parent_id,
            child_id=child_id,
            action=action,
            operator=fields.get("operator", "unknown"),
            delta=fields.get("delta"),
            outcome=fields.get("outcome", "unknown"),
            iteration=fields.get("iteration"),
        )
        self._edges[edge.id] = edge
        self._incoming_edge_by_child[child_id] = edge.id
        self._outgoing_edges_by_parent[parent_id].append(edge.id)
        self._next_edge_id += 1
        return edge

    def get_edge(self, edge_id: EdgeId) -> ImprovementEdge:
        return self._edges[edge_id]

    def edges(self) -> tuple[ImprovementEdge, ...]:
        return tuple(self._edges.values())

    def fitness_range(self) -> tuple[float | None, float | None]:
        values = [n.fitness for n in self._nodes.values() if n.fitness is not None]
        if not values:
            return None, None
        return min(values), max(values)
