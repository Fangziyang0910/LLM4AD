"""TraceAAD 的程序、修改、轨迹和跨轨迹经验。"""
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
    ENDPOINT = "endpoint_refine"
    BACKTRACK = "backtrack_branch"
    CROSSOVER = "mechanism_crossover"
    NOVELTY = "novelty_jump"


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
    """轨迹的终点质量、沿途改进潜力和路线差异。"""

    quality: float = 0.0
    potential: float = 0.0
    diversity: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return self.quality, self.potential, self.diversity


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

    @property
    def path_key(self) -> tuple[tuple[NodeId, ...], tuple[EdgeId, ...]]:
        return self.node_ids, self.edge_ids


@dataclass(frozen=True, slots=True)
class ExperienceExample:
    """一条边级 action 经验（成功或失败）。"""
    edge_id: EdgeId
    operator: str
    action: str
    delta: float
    outcome: str
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class ExperienceBatch:
    """ExperienceMemory.examples 的有界返回。"""

    positives: tuple[ExperienceExample, ...] = ()
    negatives: tuple[ExperienceExample, ...] = ()
