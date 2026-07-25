"""TraceAAD v5 search-state schema."""

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


class ExperienceKind(StrEnum):
    EFFECTIVE = "effective"
    PITFALL = "pitfall"
    EXPLORE = "explore"


@dataclass(frozen=True, slots=True)
class EvalResult:
    fitness: float | None


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
    root_lineage_id: int
    reference_trajectory_id: TrajectoryId | None = None
    reference_program_id: NodeId | None = None
    delta_parent: float | None = None
    delta_route_best: float | None = None
    delta_global_best: float | None = None
    delta_loc: int = 0
    code_change_ratio: float = 0.0
    outcome: str = "unknown"
    iteration: int | None = None
    new_global_best: bool = False
    global_best_update_reason: str | None = None

    @property
    def delta(self) -> float | None:
        return self.delta_parent


@dataclass(frozen=True, slots=True)
class ValueVec:
    """Search quality; trend is retained only as an analysis field."""

    quality: float = 0.0
    trend: float = 0.5


@dataclass(frozen=True, slots=True)
class Trajectory:
    id: TrajectoryId
    root_lineage_id: int
    node_ids: tuple[NodeId, ...]
    edge_ids: tuple[EdgeId, ...]
    endpoint_id: NodeId
    compact_best_id: NodeId
    visit_count: int = 0
    reference_use_count: int = 0
    status: TrajectoryStatus = TrajectoryStatus.ACTIVE
    value: ValueVec | None = None
    scalar_value: float | None = None


@dataclass(frozen=True, slots=True)
class GlobalExperienceEntry:
    kind: ExperienceKind
    statement: str
    condition: str
    evidence_edge_ids: tuple[EdgeId, ...]


__all__ = [
    "EdgeId",
    "EvalResult",
    "ExperienceKind",
    "GlobalExperienceEntry",
    "ImprovementEdge",
    "NodeId",
    "OperatorName",
    "ProgramNode",
    "Trajectory",
    "TrajectoryId",
    "TrajectoryStatus",
    "ValueVec",
]
