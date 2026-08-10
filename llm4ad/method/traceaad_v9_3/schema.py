"""Search facts and short-rollout credit for TraceAAD V9.3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

PROTOCOL_ID: Final[str] = "traceaad-v9.3-short-rollout-two-stage-comment-free"
GENERATION_OPERATOR: Final[str] = "trajectory_rollout_step"


class EventStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(slots=True)
class VirtualRoot:
    id: int = -1
    child_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ProgramNode:
    """One executable anchor and its anchor-level production evidence."""

    id: int
    code: str
    idea: str
    fitness: float
    directed_fitness: float
    program_loc: int
    code_hash: str
    parent_id: int
    incoming_event_id: int | None
    child_ids: list[int]
    depth: int
    creation_order: int
    budget_event_count: int = 0
    outcome_value_sum: float = 0.0
    last_budget_order: int | None = None

    @property
    def budget_value(self) -> float:
        return (self.directed_fitness + self.outcome_value_sum) / (
            1 + self.budget_event_count
        )


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """One evaluated step inside a short trajectory rollout."""

    id: int
    anchor_id: int
    child_id: int | None
    idea: str
    status: EventStatus
    failure_kind: str | None
    result_fitness: float | None
    credit_value: float
    outcome: str
    delta_parent: float | None
    delta_loc: int | None
    code_change_ratio: float | None
    new_global_best: bool
    global_best_update_reason: str | None
    stage: str
    iteration: int | None
    budget_order: int
    rollout_id: int
    rollout_step: int
    rollout_start_anchor_id: int


__all__ = [
    "EventStatus",
    "GENERATION_OPERATOR",
    "GenerationEvent",
    "PROTOCOL_ID",
    "ProgramNode",
    "VirtualRoot",
]
