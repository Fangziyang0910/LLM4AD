"""Complete single-parent search tree for TraceAAD V8."""

from __future__ import annotations

import math
from collections.abc import Iterable

from .complexity import code_change_ratio, code_hash, nonempty_loc
from .schema import ImprovementEdge, OperatorName, ProgramNode, VirtualRoot


def is_node_better(candidate: ProgramNode, incumbent: ProgramNode | None) -> bool:
    if incumbent is None:
        return True
    if candidate.directed_fitness != incumbent.directed_fitness:
        return candidate.directed_fitness > incumbent.directed_fitness
    if candidate.program_loc != incumbent.program_loc:
        return candidate.program_loc < incumbent.program_loc
    return candidate.id < incumbent.id


class SearchTree:
    """Owns the complete structural tree and optimistic subtree backup."""

    def __init__(self) -> None:
        self.root = VirtualRoot()
        self._nodes: dict[int, ProgramNode] = {}
        self._edges: dict[int, ImprovementEdge] = {}
        self._next_node_id = 0
        self._next_edge_id = 0

    def nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(self._nodes.values())

    def edges(self) -> tuple[ImprovementEdge, ...]:
        return tuple(self._edges.values())

    def get_node(self, node_id: int) -> ProgramNode:
        return self._nodes[node_id]

    def get_edge(self, edge_id: int) -> ImprovementEdge:
        return self._edges[edge_id]

    def add_initial(
        self,
        *,
        code: str,
        idea: str,
        fitness: float,
        maximize: bool,
        creation_order: int,
    ) -> ProgramNode:
        node = self._new_node(
            code=code,
            idea=idea,
            fitness=fitness,
            maximize=maximize,
            parent_id=self.root.id,
            incoming_edge_id=None,
            depth=1,
            creation_order=creation_order,
            batch_id=None,
            operator="init",
        )
        self.root.child_ids.append(node.id)
        self.root.visit_count += 1
        self._backup_root()
        return node

    def add_child(
        self,
        *,
        parent_id: int,
        code: str,
        idea: str,
        fitness: float,
        maximize: bool,
        creation_order: int,
        operator: OperatorName,
        action: str,
        reference_node_id: int | None,
        reference_root_branch_id: int | None,
        global_best_directed_fitness: float | None,
        new_global_best: bool,
        global_best_update_reason: str | None,
        iteration: int,
        batch_id: int,
        sibling_seq: int,
        sample_order: int,
        positive_threshold: float = 1e-6,
    ) -> tuple[ProgramNode, ImprovementEdge, list[dict[str, int | float | None]]]:
        parent = self.get_node(parent_id)
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        child = self._new_node(
            code=code,
            idea=idea,
            fitness=fitness,
            maximize=maximize,
            parent_id=parent_id,
            incoming_edge_id=edge_id,
            depth=parent.depth + 1,
            creation_order=creation_order,
            batch_id=batch_id,
            operator=str(operator),
        )
        delta_parent = child.directed_fitness - parent.directed_fitness
        delta_global = (
            None
            if global_best_directed_fitness is None
            else child.directed_fitness - global_best_directed_fitness
        )
        if delta_parent > positive_threshold:
            outcome = "improve"
        elif delta_parent < -positive_threshold:
            outcome = "regress"
        else:
            outcome = "plateau"
        edge = ImprovementEdge(
            id=edge_id,
            parent_id=parent_id,
            child_id=child.id,
            operator=operator,
            action=action,
            implemented_idea=idea,
            reference_node_id=reference_node_id,
            reference_root_branch_id=reference_root_branch_id,
            delta_parent=delta_parent,
            delta_global_best=delta_global,
            outcome=outcome,
            delta_loc=child.program_loc - parent.program_loc,
            code_change_ratio=code_change_ratio(parent.code, child.code),
            new_global_best=new_global_best,
            global_best_update_reason=global_best_update_reason,
            iteration=iteration,
            batch_id=batch_id,
            sibling_seq=sibling_seq,
            sample_order=sample_order,
        )
        self._edges[edge.id] = edge
        parent.child_ids.append(child.id)
        backup_changes = self.backup_from(parent.id)
        return child, edge, backup_changes

    def _new_node(
        self,
        *,
        code: str,
        idea: str,
        fitness: float,
        maximize: bool,
        parent_id: int,
        incoming_edge_id: int | None,
        depth: int,
        creation_order: int,
        batch_id: int | None,
        operator: str,
    ) -> ProgramNode:
        if not math.isfinite(fitness):
            raise ValueError("program fitness must be finite")
        node_id = self._next_node_id
        self._next_node_id += 1
        directed = fitness if maximize else -fitness
        node = ProgramNode(
            id=node_id,
            code=code,
            idea=idea,
            fitness=fitness,
            directed_fitness=directed,
            program_loc=nonempty_loc(code),
            code_hash=code_hash(code),
            parent_id=parent_id,
            incoming_edge_id=incoming_edge_id,
            child_ids=[],
            depth=depth,
            visit_count=1,
            subtree_value=directed,
            subtree_best_node_id=node_id,
            creation_order=creation_order,
            batch_id=batch_id,
            operator=operator,
        )
        self._nodes[node_id] = node
        return node

    def ancestor_node_ids(self, node_id: int) -> tuple[int, ...]:
        path: list[int] = []
        current = node_id
        while current != self.root.id:
            path.append(current)
            current = self.get_node(current).parent_id
        return tuple(reversed(path))

    def ancestor_edge_ids(self, node_id: int) -> tuple[int, ...]:
        return tuple(
            self.get_node(item).incoming_edge_id
            for item in self.ancestor_node_ids(node_id)[1:]
            if self.get_node(item).incoming_edge_id is not None
        )

    def selected_path(self, node_id: int) -> tuple[int, ...]:
        return (self.root.id, *self.ancestor_node_ids(node_id))

    def root_branch_id(self, node_id: int) -> int:
        return self.ancestor_node_ids(node_id)[0]

    def subtree_best(self, node_id: int) -> ProgramNode:
        return self.get_node(self.get_node(node_id).subtree_best_node_id)

    def record_batch_visit(self, selected_node_id: int) -> tuple[int, ...]:
        path = self.selected_path(selected_node_id)
        self.root.visit_count += 1
        for node_id in path[1:]:
            self.get_node(node_id).visit_count += 1
        return path

    def backup_from(self, node_id: int) -> list[dict[str, int | float | None]]:
        changes: list[dict[str, int | float | None]] = []
        root_before_value = self.root.subtree_value
        root_before_best = self.root.subtree_best_node_id
        current = node_id
        while current != self.root.id:
            node = self.get_node(current)
            before_value = node.subtree_value
            before_best = node.subtree_best_node_id
            candidates = [node, *(self.subtree_best(child) for child in node.child_ids)]
            best = candidates[0]
            for candidate in candidates[1:]:
                if is_node_better(candidate, best):
                    best = candidate
            node.subtree_value = best.directed_fitness
            node.subtree_best_node_id = best.id
            if before_value != node.subtree_value or before_best != best.id:
                changes.append(
                    {
                        "node_id": node.id,
                        "before_value": before_value,
                        "after_value": node.subtree_value,
                        "before_best_node_id": before_best,
                        "after_best_node_id": best.id,
                    }
                )
            current = node.parent_id
        self._backup_root()
        if (
            root_before_value != self.root.subtree_value
            or root_before_best != self.root.subtree_best_node_id
        ):
            changes.append(
                {
                    "node_id": self.root.id,
                    "before_value": root_before_value,
                    "after_value": self.root.subtree_value,
                    "before_best_node_id": root_before_best,
                    "after_best_node_id": self.root.subtree_best_node_id,
                }
            )
        return changes

    def _backup_root(self) -> None:
        if not self.root.child_ids:
            self.root.subtree_value = None
            self.root.subtree_best_node_id = None
            return
        best = self.subtree_best(self.root.child_ids[0])
        for branch_id in self.root.child_ids[1:]:
            candidate = self.subtree_best(branch_id)
            if is_node_better(candidate, best):
                best = candidate
        self.root.subtree_value = best.directed_fitness
        self.root.subtree_best_node_id = best.id

    def descendants(self, node_id: int) -> Iterable[ProgramNode]:
        pending = list(reversed(self.get_node(node_id).child_ids))
        while pending:
            child_id = pending.pop()
            child = self.get_node(child_id)
            yield child
            pending.extend(reversed(child.child_ids))


__all__ = ["SearchTree", "is_node_better"]
