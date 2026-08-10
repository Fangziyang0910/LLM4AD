"""Fact graph and short-rollout credit for TraceAAD V9.3."""

from __future__ import annotations

import math

from .complexity import code_change_ratio, code_hash, nonempty_loc
from .schema import EventStatus, GenerationEvent, ProgramNode, VirtualRoot


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
    """Store real lineage and events; it performs no tree-value backup."""

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
        rollout_id: int,
        rollout_step: int,
        rollout_start_anchor_id: int,
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
        event = GenerationEvent(
            id=event_id,
            anchor_id=anchor.id,
            child_id=child.id,
            idea=idea,
            status=EventStatus.VALID,
            failure_kind=None,
            result_fitness=fitness,
            credit_value=child.directed_fitness,
            outcome="improve" if delta > 0 else "regress" if delta < 0 else "plateau",
            delta_parent=delta,
            delta_loc=child.program_loc - anchor.program_loc,
            code_change_ratio=code_change_ratio(anchor.code, child.code),
            new_global_best=new_global_best,
            global_best_update_reason=global_best_update_reason,
            stage=stage,
            iteration=iteration,
            budget_order=budget_order,
            rollout_id=rollout_id,
            rollout_step=rollout_step,
            rollout_start_anchor_id=rollout_start_anchor_id,
        )
        self._events[event.id] = event
        anchor.child_ids.append(child.id)
        return child, event

    def add_invalid_event(
        self,
        *,
        anchor_id: int,
        idea: str,
        code: str | None = None,
        failure_kind: str,
        stage: str,
        iteration: int | None,
        budget_order: int,
        rollout_id: int,
        rollout_step: int,
        rollout_start_anchor_id: int,
    ) -> GenerationEvent:
        anchor = self.get_node(anchor_id)
        event_id = self._next_event_id
        self._next_event_id += 1
        event = GenerationEvent(
            id=event_id,
            anchor_id=anchor.id,
            child_id=None,
            idea=idea,
            status=EventStatus.INVALID,
            failure_kind=failure_kind,
            result_fitness=None,
            credit_value=anchor.directed_fitness,
            outcome="invalid",
            delta_parent=None,
            delta_loc=None if code is None else nonempty_loc(code) - anchor.program_loc,
            code_change_ratio=(
                None if code is None else code_change_ratio(anchor.code, code)
            ),
            new_global_best=False,
            global_best_update_reason=None,
            stage=stage,
            iteration=iteration,
            budget_order=budget_order,
            rollout_id=rollout_id,
            rollout_step=rollout_step,
            rollout_start_anchor_id=rollout_start_anchor_id,
        )
        self._events[event.id] = event
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

    def record_rollout_outcome(
        self, anchor_id: int, credit_value: float, budget_order: int
    ) -> None:
        anchor = self.get_node(anchor_id)
        anchor.budget_event_count += 1
        anchor.outcome_value_sum += credit_value
        anchor.last_budget_order = budget_order

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


__all__ = ["FactGraph", "is_node_better"]
