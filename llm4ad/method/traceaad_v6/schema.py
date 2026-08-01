"""TraceAAD v6 search-state schema."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

NodeId: TypeAlias = int
EdgeId: TypeAlias = int
TrajectoryId: TypeAlias = int


class TrajectoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class OperatorName(StrEnum):
    IDEATE = "trace_ideate"
    REFINE = "trace_refine"
    SYNTHESIZE = "trace_synthesize"
    TRANSFER = "trace_transfer"


@dataclass(frozen=True, slots=True)
class ProgramNode:
    id: NodeId
    code: str
    idea: str
    fitness: float | None
    program_loc: int
    code_hash: str


@dataclass(frozen=True, slots=True)
class ImprovementEdge:
    id: EdgeId
    parent_id: NodeId
    child_id: NodeId
    operator: OperatorName
    action: str
    anchor_role: str
    primary_trajectory_id: TrajectoryId
    reference_trajectory_id: TrajectoryId | None = None
    reference_program_id: NodeId | None = None
    delta_parent: float | None = None
    delta_route_best: float | None = None
    delta_global_best: float | None = None
    delta_loc: int = 0
    code_change_ratio: float = 0.0
    outcome: str = "unknown"
    edge_credit: float = 0.0
    iteration: int | None = None
    new_global_best: bool = False
    global_best_update_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ValueVec:
    """Route quality and retained-path credit for continued development."""

    quality: float = 0.0
    credit: float = 0.0


@dataclass(frozen=True, slots=True)
class Trajectory:
    id: TrajectoryId
    node_ids: tuple[NodeId, ...]
    edge_ids: tuple[EdgeId, ...]
    endpoint_id: NodeId
    compact_best_id: NodeId
    visit_count: int = 0
    status: TrajectoryStatus = TrajectoryStatus.ACTIVE
    value: ValueVec | None = None
    scalar_value: float | None = None


@dataclass(frozen=True, slots=True)
class AnchorAttempt:
    """A tested attempt from an anchor, including failed non-program candidates."""

    anchor_node_id: NodeId
    primary_trajectory_id: TrajectoryId
    operator: str
    action: str
    iteration: int | None
    status: str
    idea: str = ""
    fitness: float | None = None
    delta_parent: float | None = None
    delta_route_best: float | None = None
    delta_global_best: float | None = None
    outcome: str = "unknown"
    edge_id: EdgeId | None = None
    child_id: NodeId | None = None
    program_loc: int | None = None
    failure_kind: str | None = None
    new_route_best: bool = False
    new_global_best: bool = False


__all__ = [
    "AnchorAttempt",
    "EdgeId",
    "ImprovementEdge",
    "NodeId",
    "OperatorName",
    "ProgramNode",
    "Trajectory",
    "TrajectoryId",
    "TrajectoryStatus",
    "ValueVec",
]
