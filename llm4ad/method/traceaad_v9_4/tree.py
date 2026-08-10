"""Fact graph and anchor-local credit for TraceAAD V9.4."""

from __future__ import annotations

import math
from typing import Final

from .complexity import code_change_ratio, code_hash, nonempty_loc
from .schema import (
    EventStatus,
    GenerationEvent,
    ProgramNode,
    TrajectoryCreditUpdate,
    VirtualRoot,
)

TRAJECTORY_CREDIT_DISCOUNT: Final[float] = 0.5
TRAJECTORY_CREDIT_DEPTH: Final[int] = 3


def is_node_better(candidate: ProgramNode, incumbent: ProgramNode | None) -> bool:
    if incumbent is None:
        return True
    return (
        candidate.directed_fitness,
        -candidate.program_loc,
        -candidate.creation_order,
    ) > (
        incumbent.directed_fitness,
        -incumbent.program_loc,
        -incumbent.creation_order,
    )


class FactGraph:
    """Store real lineage and bounded, distance-decayed trajectory credit."""

    def __init__(self) -> None:
        self.root = VirtualRoot()
        self._nodes: dict[int, ProgramNode] = {}
        self._events: dict[int, GenerationEvent] = {}
        self._next_node_id = 0
        self._next_event_id = 0

    def nodes(self) -> tuple[ProgramNode, ...]:
        return tuple(self._nodes.values())

    def events(self) -> tuple[GenerationEvent, ...]:
        return tuple(self._events.values())

    def get_node(self, node_id: int) -> ProgramNode:
        return self._nodes[node_id]

    def get_event(self, event_id: int) -> GenerationEvent:
        return self._events[event_id]

    def best_node(self) -> ProgramNode | None:
        best: ProgramNode | None = None
        for node in self._nodes.values():
            if is_node_better(node, best):
                best = node
        return best

    def add_root(
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
            incoming_event_id=None,
            depth=1,
            creation_order=creation_order,
        )
        self.root.child_ids.append(node.id)
        return node

    def add_valid_event(
        self,
        *,
        anchor_id: int,
        code: str,
        idea: str,
        fitness: float,
        maximize: bool,
        stage: str,
        iteration: int | None,
        budget_order: int,
        new_global_best: bool,
        global_best_update_reason: str | None,
    ) -> tuple[ProgramNode, GenerationEvent]:
        anchor = self.get_node(anchor_id)
        event_id = self._next_event_id
        self._next_event_id += 1
        child = self._new_node(
            code=code,
            idea=idea,
            fitness=fitness,
            maximize=maximize,
            parent_id=anchor.id,
            incoming_event_id=event_id,
            depth=anchor.depth + 1,
            creation_order=budget_order,
        )
        delta = child.directed_fitness - anchor.directed_fitness
        credit_updates = self._trajectory_credit_updates(
            anchor_id=anchor.id,
            result_directed_fitness=child.directed_fitness,
        )
        event = GenerationEvent(
            id=event_id,
            anchor_id=anchor.id,
            child_id=child.id,
            idea=idea,
            status=EventStatus.VALID,
            failure_kind=None,
            error_type=None,
            error_message=None,
            result_fitness=fitness,
            anchor_credit=credit_updates[0].credit,
            credit_updates=credit_updates,
            outcome="improve" if delta > 0 else "regress" if delta < 0 else "plateau",
            delta_parent=delta,
            delta_loc=child.program_loc - anchor.program_loc,
            code_change_ratio=code_change_ratio(anchor.code, child.code),
            new_global_best=new_global_best,
            strict_breakthrough=global_best_update_reason == "strict_fitness",
            global_best_update_reason=global_best_update_reason,
            stage=stage,
            iteration=iteration,
            budget_order=budget_order,
        )
        self._events[event.id] = event
        anchor.child_ids.append(child.id)
        self._record_direct_attempt(anchor, budget_order)
        self._record_trajectory_updates(credit_updates)
        return child, event

    def add_invalid_event(
        self,
        *,
        anchor_id: int,
        idea: str,
        code: str | None = None,
        failure_kind: str,
        error_type: str | None,
        error_message: str | None,
        stage: str,
        iteration: int | None,
        budget_order: int,
    ) -> GenerationEvent:
        anchor = self.get_node(anchor_id)
        event_id = self._next_event_id
        self._next_event_id += 1
        credit_updates = self._trajectory_credit_updates(
            anchor_id=anchor.id,
            result_directed_fitness=None,
        )
        event = GenerationEvent(
            id=event_id,
            anchor_id=anchor.id,
            child_id=None,
            idea=idea,
            status=EventStatus.INVALID,
            failure_kind=failure_kind,
            error_type=error_type,
            error_message=error_message,
            result_fitness=None,
            anchor_credit=0.0,
            credit_updates=credit_updates,
            outcome="invalid",
            delta_parent=None,
            delta_loc=None if code is None else nonempty_loc(code) - anchor.program_loc,
            code_change_ratio=(
                None if code is None else code_change_ratio(anchor.code, code)
            ),
            new_global_best=False,
            strict_breakthrough=False,
            global_best_update_reason=None,
            stage=stage,
            iteration=iteration,
            budget_order=budget_order,
        )
        self._events[event.id] = event
        self._record_direct_attempt(anchor, budget_order)
        self._record_trajectory_updates(credit_updates)
        return event

    def _new_node(
        self,
        *,
        code: str,
        idea: str,
        fitness: float,
        maximize: bool,
        parent_id: int,
        incoming_event_id: int | None,
        depth: int,
        creation_order: int,
    ) -> ProgramNode:
        if not math.isfinite(fitness):
            raise ValueError("program fitness must be finite")
        node_id = self._next_node_id
        self._next_node_id += 1
        node = ProgramNode(
            id=node_id,
            code=code,
            idea=idea,
            fitness=fitness,
            directed_fitness=fitness if maximize else -fitness,
            program_loc=nonempty_loc(code),
            code_hash=code_hash(code),
            parent_id=parent_id,
            incoming_event_id=incoming_event_id,
            child_ids=[],
            depth=depth,
            creation_order=creation_order,
        )
        self._nodes[node.id] = node
        return node

    @staticmethod
    def _record_direct_attempt(anchor: ProgramNode, budget_order: int) -> None:
        anchor.budget_event_count += 1
        anchor.last_budget_order = budget_order

    def _trajectory_credit_updates(
        self,
        *,
        anchor_id: int,
        result_directed_fitness: float | None,
    ) -> tuple[TrajectoryCreditUpdate, ...]:
        updates: list[TrajectoryCreditUpdate] = []
        current_id = anchor_id
        distance = 1
        while current_id != self.root.id and distance <= TRAJECTORY_CREDIT_DEPTH:
            node = self.get_node(current_id)
            improvement = (
                0.0
                if result_directed_fitness is None
                else max(0.0, result_directed_fitness - node.directed_fitness)
            )
            updates.append(
                TrajectoryCreditUpdate(
                    node_id=node.id,
                    distance=distance,
                    credit=(TRAJECTORY_CREDIT_DISCOUNT**distance) * improvement,
                )
            )
            current_id = node.parent_id
            distance += 1
        return tuple(updates)

    def _record_trajectory_updates(
        self, updates: tuple[TrajectoryCreditUpdate, ...]
    ) -> None:
        for update in updates:
            node = self.get_node(update.node_id)
            node.trajectory_event_count += 1
            node.trajectory_credit_sum += update.credit

    def ancestor_node_ids(self, node_id: int) -> tuple[int, ...]:
        path: list[int] = []
        current = node_id
        while current != self.root.id:
            path.append(current)
            current = self.get_node(current).parent_id
        return tuple(reversed(path))

    def formation_event_ids(self, node_id: int) -> tuple[int, ...]:
        return tuple(
            self.get_node(path_node_id).incoming_event_id
            for path_node_id in self.ancestor_node_ids(node_id)[1:]
            if self.get_node(path_node_id).incoming_event_id is not None
        )

    def downstream_events(
        self, anchor_id: int, *, max_depth: int
    ) -> tuple[tuple[GenerationEvent, int, int | None], ...]:
        anchor = self.get_node(anchor_id)
        selected: list[tuple[GenerationEvent, int, int | None]] = []
        for event in self._events.values():
            event_anchor = self.get_node(event.anchor_id)
            path = self.ancestor_node_ids(event_anchor.id)
            if anchor_id not in path:
                continue
            relative_depth = event_anchor.depth - anchor.depth + 1
            if relative_depth > max_depth:
                continue
            branch_id = self._branch_id(anchor_id, event)
            selected.append((event, relative_depth, branch_id))
        selected.sort(key=lambda item: item[0].budget_order)
        return tuple(selected)

    def _branch_id(self, anchor_id: int, event: GenerationEvent) -> int | None:
        if event.anchor_id == anchor_id:
            return event.child_id
        path = self.ancestor_node_ids(event.anchor_id)
        position = path.index(anchor_id)
        return path[position + 1]


__all__ = [
    "TRAJECTORY_CREDIT_DEPTH",
    "TRAJECTORY_CREDIT_DISCOUNT",
    "FactGraph",
    "is_node_better",
]
