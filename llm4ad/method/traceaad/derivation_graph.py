from __future__ import annotations

from collections import defaultdict

from .schema import EdgeId, ImprovementEdge, NodeId, ProgramNode


class DerivationGraph:
    def __init__(self) -> None:
        self._next_node_id = 0
        self._next_edge_id = 0
        self._nodes: dict[NodeId, ProgramNode] = {}
        self._edges: dict[EdgeId, ImprovementEdge] = {}
        self._incoming_edge_by_child: dict[NodeId, EdgeId] = {}
        self._outgoing_edges_by_parent: dict[NodeId, list[EdgeId]] = defaultdict(list)

    def add_node(
            self,
            *,
            code: str,
            idea: str,
            fitness: float | None,
            is_valid: bool,
            iteration: int | None = None,
            sample_order: int | None = None,
    ) -> ProgramNode:
        node = ProgramNode(
            id=self._next_node_id,
            code=code,
            idea=idea,
            fitness=fitness,
            is_valid=is_valid,
            iteration=iteration,
            sample_order=sample_order,
        )
        self._nodes[node.id] = node
        self._next_node_id += 1
        return node

    def add_edge(
            self,
            *,
            parent_id: NodeId,
            child_id: NodeId,
            action: str,
            iteration: int | None = None,
    ) -> ImprovementEdge:
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
            iteration=iteration,
        )
        self._edges[edge.id] = edge
        self._incoming_edge_by_child[child_id] = edge.id
        self._outgoing_edges_by_parent[parent_id].append(edge.id)
        self._next_edge_id += 1
        return edge

    def get_node(self, node_id: NodeId) -> ProgramNode:
        return self._nodes[node_id]

    def get_edge(self, edge_id: EdgeId) -> ImprovementEdge:
        return self._edges[edge_id]

    def nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(self._nodes.values())

    def edges(self) -> tuple[ImprovementEdge, ...]:
        return tuple(self._edges.values())

    def outgoing_edges(self, node_id: NodeId) -> tuple[ImprovementEdge, ...]:
        if node_id not in self._nodes:
            raise KeyError(f"unknown node: {node_id}")
        return tuple(self._edges[edge_id] for edge_id in self._outgoing_edges_by_parent[node_id])
