"""TraceAAD 的程序、修改、轨迹和种群状态。"""

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


@dataclass(frozen=True, slots=True)
class EvalResult:
    """单次程序评估结果。现有平台 task 返回标量 fitness。"""

    fitness: float | None


@dataclass(frozen=True, slots=True)
class ProgramNode:
    id: NodeId
    code: str
    idea: str
    fitness: float | None


@dataclass(frozen=True, slots=True)
class ImprovementEdge:
    id: EdgeId
    parent_id: NodeId
    child_id: NodeId
    action: str
    operator: str = "unknown"
    delta: float | None = None
    outcome: str = "unknown"
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class ValueVec:
    """轨迹的搜索质量和近期走势。"""

    quality: float = 0.0
    trend: float = 0.0


@dataclass(frozen=True, slots=True)
class Trajectory:
    id: TrajectoryId
    node_ids: tuple[NodeId, ...]
    edge_ids: tuple[EdgeId, ...]
    endpoint_id: NodeId
    visit_count: int = 0
    status: TrajectoryStatus = TrajectoryStatus.ACTIVE
    value: ValueVec | None = None
    scalar_value: float | None = None
