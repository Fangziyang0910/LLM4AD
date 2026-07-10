"""TraceAAD2 领域数据结构。

对应 traceaad-fusion-design.md §3 三层记忆的数据对象。相对 v1 的 schema：
- ProgramNode 扩展多目标/泛化字段（runtime/complexity/robustness/mechanism_tag/confidence）
- ImprovementEdge 扩展过程信息（operator/mechanism_tag/delta/outcome），支撑 stepwise credit
- Trajectory 扩展 base_id/island_id/value，并显式携带 ValueVec
- 新增 EvalResult / ValueVec / StepInfo / Pattern
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


class StepOutcome(StrEnum):
    IMPROVE = "improve"
    NEUTRAL = "neutral"
    REGRESS = "regress"
    PLATEAU = "plateau"
    UNKNOWN = "unknown"


class OperatorName(StrEnum):
    INIT = "init"
    ENDPOINT = "endpoint_refine"
    BACKTRACK = "backtrack_branch"
    CROSSOVER = "mechanism_crossover"
    SIMPLIFY = "distill_simplify"
    SCALE_TRANSFER = "scale_transfer"
    NOVELTY = "novelty_jump"


class PatternKind(StrEnum):
    MECHANISM = "mechanism"
    LESSON = "lesson"
    ANTI_PATTERN = "anti_pattern"


@dataclass(frozen=True, slots=True)
class EvalResult:
    """单次程序评估的富结果。

    fitness_vector 默认 None：现有平台 task 只返回标量；支持 held-out/per-instance
    结果的 evaluator 可同时返回 vector 与 robustness，供真实跨实例信用使用。
    """
    fitness: float | None
    runtime: float = 0.0
    complexity: int = 0
    robustness: float = 0.0
    confidence: float = 1.0
    fitness_vector: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class ProgramNode:
    id: NodeId
    code: str
    idea: str
    fitness: float | None
    is_valid: bool
    runtime: float = 0.0
    complexity: int = 0
    robustness: float = 0.0
    fitness_vector: tuple[float, ...] | None = None
    mechanism_tag: str = "other"
    confidence: float = 1.0
    iteration: int | None = None
    sample_order: int | None = None


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
    generalization_signal: float = 0.0
    iteration: int | None = None


@dataclass(frozen=True, slots=True)
class ValueVec:
    """多维 trajectory 价值（不塌缩成单标量；采样用 scalarize，survival 用 non-dominated）。"""
    quality: float = 0.0
    potential: float = 0.0
    diversity: float = 0.0
    novelty: float = 0.0
    generalization: float = 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.quality, self.potential, self.diversity, self.novelty, self.generalization)


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
    def length(self) -> int:
        return len(self.node_ids)

    @property
    def path_key(self) -> tuple[tuple[NodeId, ...], tuple[EdgeId, ...]]:
        return self.node_ids, self.edge_ids


@dataclass(frozen=True, slots=True)
class StepInfo:
    """一条 trajectory 上某一步的展开视图（从 graph edge 派生，供 value/credit 使用）。"""
    edge_id: EdgeId
    parent_id: NodeId
    child_id: NodeId
    operator: str
    mechanism_tag: str
    delta: float | None
    outcome: str
    generalization_signal: float


@dataclass(frozen=True, slots=True)
class Pattern:
    """Pattern Memory 的一条蒸馏知识（机制 / 教训 / 反模式）。"""
    id: PatternId
    kind: str
    text: str
    mechanism_tag: str
    support_ids: tuple[int, ...] = field(default_factory=tuple)
    generalization_score: float = 0.0
    confidence: float = 0.0
    updated_iter: int = 0
