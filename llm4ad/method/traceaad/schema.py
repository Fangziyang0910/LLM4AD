"""TraceAAD 领域数据结构。

对应 TraceAAD 机制设计的三层记忆对象：
- ProgramNode：代码、思想、适应度、复杂度、机制标签
- ImprovementEdge：动作、算子、机制、有向 Δ、outcome
- Trajectory：有界路径 + ValueVec
- Pattern：PatternMemory 中的机制 / 教训 / 反模式
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

NodeId: TypeAlias = int
EdgeId: TypeAlias = int
TrajectoryId: TypeAlias = int
IslandId: TypeAlias = int
PatternId: TypeAlias = int


class TrajectoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class OperatorName(StrEnum):
    ENDPOINT = "endpoint_refine"
    BACKTRACK = "backtrack_branch"
    CROSSOVER = "mechanism_crossover"
    SIMPLIFY = "distill_simplify"
    NOVELTY = "novelty_jump"


@dataclass(frozen=True, slots=True)
class EvalResult:
    """单次程序评估结果。现有平台 task 返回标量 fitness。"""
    fitness: float | None
    complexity: int = 0


@dataclass(frozen=True, slots=True)
class ProgramNode:
    id: NodeId
    code: str
    idea: str
    fitness: float | None
    is_valid: bool
    complexity: int = 0
    mechanism_tag: str = "other"


@dataclass(frozen=True, slots=True)
class ImprovementEdge:
    id: EdgeId
    parent_id: NodeId
    child_id: NodeId
    action: str
    operator: str = "unknown"
    mechanism_tag: str = "other"
    delta: float | None = None
    outcome: str = "unknown"
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class ValueVec:
    """多维 trajectory 价值（采样用 scalarize，survival 用 non-dominated）。"""
    quality: float = 0.0
    potential: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.quality, self.potential, self.diversity, self.novelty)


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
class Pattern:
    """Pattern Memory 的一条蒸馏知识（机制 / 教训 / 反模式）。"""
    id: PatternId
    kind: str
    text: str
    mechanism_tag: str
    support_ids: tuple[int, ...] = field(default_factory=tuple)
    improve_rate: float = 0.0  # 唯一图边上的改进率，用于排序/淘汰
    confidence: float = 0.0
