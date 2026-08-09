"""Trajectory store for TraceAAD V9.1.

The tree preserves lineage and auditability.  It does not propagate value or
visits: verification evidence belongs to the trajectory that consumed budget.
"""

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

    def best_node(self) -> ProgramNode | None:
        best: ProgramNode | None = None
        for node in self._nodes.values():
            if is_node_better(node, best):
                best = node
        return best

    def add_initial(
        self,
        *,
        code: str,
        idea: str,
        fitness: float,
        maximize: bool,
        creation_order: int,
        bootstrap_reference_node_ids: tuple[int, ...] = (),
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
            bootstrap_reference_node_ids=bootstrap_reference_node_ids,
            trajectory_best_value=None,
            trajectory_best_node_id=None,
        )
        self.root.child_ids.append(node.id)
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
        reference_node_id: int | None,
        global_best_directed_fitness: float | None,
        new_global_best: bool,
        global_best_update_reason: str | None,
        iteration: int,
        batch_id: int,
        sibling_seq: int,
        sample_order: int,
        positive_threshold: float = 1e-6,
    ) -> tuple[ProgramNode, ImprovementEdge]:
        parent = self.get_node(parent_id)
        edge_id = self._next_edge_id
        self._next_edge_id += 1
        directed = fitness if maximize else -fitness
        advances_route = directed > parent.trajectory_best_value + positive_threshold
        if advances_route:
            trajectory_best_value = directed
            trajectory_best_node_id: int | None = self._next_node_id
        else:
            trajectory_best_value = parent.trajectory_best_value
            trajectory_best_node_id = parent.trajectory_best_node_id
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
            bootstrap_reference_node_ids=(),
            trajectory_best_value=trajectory_best_value,
            trajectory_best_node_id=trajectory_best_node_id,
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
            implemented_idea=idea,
            reference_node_id=reference_node_id,
            delta_parent=delta_parent,
            delta_global_best=delta_global,
            trajectory_best_before=parent.trajectory_best_value,
            advances_parent_trajectory=advances_route,
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
        return child, edge

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
        bootstrap_reference_node_ids: tuple[int, ...],
        trajectory_best_value: float | None,
        trajectory_best_node_id: int | None,
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
            creation_order=creation_order,
            batch_id=batch_id,
            operator=operator,
            bootstrap_reference_node_ids=list(bootstrap_reference_node_ids),
            trajectory_best_value=(
                directed if trajectory_best_value is None else trajectory_best_value
            ),
            trajectory_best_node_id=(
                node_id if trajectory_best_node_id is None else trajectory_best_node_id
            ),
        )
        self._nodes[node_id] = node
        return node

    def record_verification(
        self,
        node_id: int,
        *,
        valid_candidate_count: int,
        route_advanced: bool,
        global_advanced: bool,
        batch_id: int,
        recent_window: int,
    ) -> None:
        """Attach one budget event only to the selected trajectory."""
        if valid_candidate_count < 0:
            raise ValueError("valid_candidate_count must be non-negative")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        node = self.get_node(node_id)
        node.verification_count += 1
        node.valid_candidate_count += valid_candidate_count
        node.route_advance_count += int(route_advanced)
        node.global_advance_count += int(global_advanced)
        node.recent_advances.append(bool(route_advanced))
        del node.recent_advances[:-recent_window]
        node.last_verification_batch_id = batch_id

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

    def is_ancestor(self, possible_ancestor_id: int, node_id: int) -> bool:
        return possible_ancestor_id in self.ancestor_node_ids(node_id)[:-1]

    def same_lineage(self, first_id: int, second_id: int) -> bool:
        return (
            first_id == second_id
            or self.is_ancestor(first_id, second_id)
            or self.is_ancestor(second_id, first_id)
        )

    def descendants(self, node_id: int) -> Iterable[ProgramNode]:
        pending = list(reversed(self.get_node(node_id).child_ids))
        while pending:
            child_id = pending.pop()
            child = self.get_node(child_id)
            yield child
            pending.extend(reversed(child.child_ids))


__all__ = ["SearchTree", "is_node_better"]
