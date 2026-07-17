"""TraceAAD 领域数据结构。

对应 TraceAAD 机制设计的三层记忆对象：
- ProgramNode：代码、思想、适应度、复杂度、runtime
- ImprovementEdge：动作、算子、有向 Δ、outcome
- Trajectory：有界路径 + ValueVec
- ExperienceExample / ExperienceBatch：边级成功/失败 action 经验查询结果
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, TypeAlias

NodeId: TypeAlias = int
EdgeId: TypeAlias = int
TrajectoryId: TypeAlias = int
IslandId: TypeAlias = int


class TrajectoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class OperatorName(StrEnum):
    ENDPOINT = "endpoint_refine"
    BACKTRACK = "backtrack_branch"
    CROSSOVER = "mechanism_crossover"
    SIMPLIFY = "simplify"
    NOVELTY = "novelty_jump"


def _empty_metrics() -> Mapping[str, float]:
    return {}


@dataclass(frozen=True, slots=True)
class EvalResult:
    """单次程序评估结果。现有平台 task 返回标量 fitness。"""
    fitness: float | None
    complexity: float = 0.0
    runtime: float = 0.0
    complexity_metrics: Mapping[str, float] = field(default_factory=_empty_metrics)


@dataclass(frozen=True, slots=True)
class ProgramNode:
    id: NodeId
    code: str
    idea: str
    fitness: float | None
    complexity: float = 0.0
    runtime: float = 0.0
    complexity_metrics: Mapping[str, float] = field(default_factory=_empty_metrics)


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
    """多维 trajectory 价值（采样用 scalarize，survival 用 non-dominated）。

    compactness / speed 均为「越大越好」的池相对效率维（由 raw complexity/runtime 翻转）。
    """
    quality: float = 0.0
    potential: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    compactness: float = 0.0
    speed: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.quality,
            self.potential,
            self.diversity,
            self.novelty,
            self.compactness,
            self.speed,
        )


@dataclass(frozen=True, slots=True)
class Trajectory:
    id: TrajectoryId
    node_ids: tuple[NodeId, ...]
    edge_ids: tuple[EdgeId, ...]
    endpoint_id: NodeId
    base_id: NodeId
    island_id: IslandId = 0
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
