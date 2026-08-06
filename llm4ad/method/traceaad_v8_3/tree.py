"""Single-parent complete tree used by TraceAAD V8.3."""

from __future__ import annotations

import math

from .schema import AlgorithmRecord, OperatorName, TreeEdge, TreeNode, VirtualRoot


class SearchTree:
    def __init__(self) -> None:
        self.root = VirtualRoot()
        self._nodes: dict[int, TreeNode] = {}
        self._edges: dict[tuple[int, int], TreeEdge] = {}
        self._next_node_id = 0

    def nodes(self) -> tuple[TreeNode, ...]:
        return tuple(self._nodes.values())

    def edges(self) -> tuple[TreeEdge, ...]:
        return tuple(self._edges.values())

    def get_node(self, node_id: int) -> TreeNode:
        return self._nodes[node_id]

    def get_edge(self, parent_id: int, child_id: int) -> TreeEdge:
        return self._edges[(parent_id, child_id)]

    def children(self, node_id: int) -> tuple[TreeNode, ...]:
        """Return valid direct children in creation order."""
        ids = self.root.child_ids if node_id == self.root.id else self.get_node(node_id).child_ids
        return tuple(self.get_node(child_id) for child_id in ids)

    def ancestors(self, node_id: int) -> tuple[TreeNode, ...]:
        """Return program ancestors from the root branch to ``node_id``."""
        return tuple(self.get_node(item) for item in self.ancestor_node_ids(node_id))

    def subtree_value(self, node_id: int, *, maximize: bool = True) -> float:
        """Return the best original fitness in a node's single-parent subtree."""
        node = self.get_node(node_id)
        values = [node.algorithm.fitness]
        values.extend(self.subtree_value(child_id, maximize=maximize) for child_id in node.child_ids)
        return max(values) if maximize else min(values)

    def add_initial(self, algorithm: AlgorithmRecord, creation_order: int | None = None) -> TreeNode:
        if creation_order is None:
            creation_order = len(self._nodes)
        node = self._new_node(algorithm, self.root.id, 1, creation_order)
        self._nodes[node.id] = node
        self.root.child_ids.append(node.id)
        self.root.visit_count += 1
        return node

    def add_child(
        self,
        parent_id: int,
        algorithm: AlgorithmRecord,
        operator: OperatorName,
        reference_node_id: int | None,
        creation_order: int | None = None,
    ) -> tuple[TreeNode, TreeEdge]:
        parent = self.get_node(parent_id)
        if creation_order is None:
            creation_order = len(self._nodes)
        node = self._new_node(algorithm, parent_id, parent.depth + 1, creation_order)
        edge = TreeEdge(parent_id, node.id, operator, reference_node_id)
        self._nodes[node.id] = node
        self._edges[(parent_id, node.id)] = edge
        parent.child_ids.append(node.id)
        return node, edge

    def _new_node(
        self,
        algorithm: AlgorithmRecord,
        parent_id: int,
        depth: int,
        creation_order: int,
    ) -> TreeNode:
        if not math.isfinite(algorithm.fitness):
            raise ValueError("algorithm fitness must be finite")
        node = TreeNode(
            id=self._next_node_id,
            algorithm=algorithm,
            parent_id=parent_id,
            depth=depth,
            creation_order=creation_order,
        )
        self._next_node_id += 1
        return node

    def ancestor_node_ids(self, node_id: int) -> tuple[int, ...]:
        result: list[int] = []
        current = node_id
        while current != self.root.id:
            result.append(current)
            current = self.get_node(current).parent_id
        return tuple(reversed(result))

    def ancestor_edge_ids(self, node_id: int) -> tuple[tuple[int, int], ...]:
        ids = self.ancestor_node_ids(node_id)
        return tuple((self.get_node(child).parent_id, child) for child in ids[1:])

    def selected_path(self, node_id: int) -> tuple[int, ...]:
        return (self.root.id, *self.ancestor_node_ids(node_id))

    def record_selection(self, node_id: int) -> tuple[int, ...]:
        path = self.selected_path(node_id)
        self.root.visit_count += 1
        for current in path[1:]:
            self.get_node(current).visit_count += 1
        self.get_node(node_id).expansion_attempts += 1
        return path


__all__ = ["SearchTree"]
