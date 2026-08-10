"""Search facts and anchor credit for TraceAAD V9.4."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

PROTOCOL_ID: Final[str] = (
    "traceaad-v9.4-single-step-joint-generation-decayed-trajectory-credit-"
    "quality-local-coverage-verified-failure-memory"
)
GENERATION_OPERATOR: Final[str] = "trajectory_step"


class EventStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(slots=True)
class VirtualRoot:
    id: int = -1
    child_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ProgramNode:
    """One executable anchor and its bounded trajectory-production evidence."""

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
    trajectory_event_count: int = 0
    trajectory_credit_sum: float = 0.0
    last_budget_order: int | None = None

    @property
    def budget_value(self) -> float:
        if self.trajectory_event_count == 0:
            return self.directed_fitness
        return self.directed_fitness + (
            self.trajectory_credit_sum / self.trajectory_event_count
        )


@dataclass(frozen=True, slots=True)
class TrajectoryCreditUpdate:
    """One event's discounted contribution to one visible ancestor."""

    node_id: int
    distance: int
    credit: float


@dataclass(frozen=True, slots=True)
class FailureObservation:
    """Verified parser or evaluator feedback retained across local branches."""

    failure_kind: str
    error_type: str | None
    error_message: str | None
    budget_order: int


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """One complete budget event performed from an executable anchor."""

    id: int
    anchor_id: int
    child_id: int | None
    idea: str
    status: EventStatus
    failure_kind: str | None
    error_type: str | None
    error_message: str | None
    result_fitness: float | None
    anchor_credit: float
    credit_updates: tuple[TrajectoryCreditUpdate, ...]
    outcome: str
    delta_parent: float | None
    delta_loc: int | None
    code_change_ratio: float | None
    new_global_best: bool
    strict_breakthrough: bool
    global_best_update_reason: str | None
    stage: str
    iteration: int | None
    budget_order: int


__all__ = [
    "EventStatus",
    "FailureObservation",
    "GENERATION_OPERATOR",
    "GenerationEvent",
    "PROTOCOL_ID",
    "ProgramNode",
    "TrajectoryCreditUpdate",
    "VirtualRoot",
]
