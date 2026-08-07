"""Core state for the independent TraceAAD V8.3 implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

PROTOCOL_ID = "traceaad-v8.3"


class OperatorName(StrEnum):
    REFINE = "trace_refine"
    TUNE = "trace_tune"
    SIMPLIFY = "trace_simplify"
    INNOVATE = "trace_innovate"
    CROSSOVER = "trace_crossover"


@dataclass(frozen=True, slots=True)
class AlgorithmRecord:
    design_idea: str
    code: str
    description: str
    fitness: float
    evaluation_time: float


@dataclass(slots=True)
class TreeNode:
    id: int
    algorithm: AlgorithmRecord
    parent_id: int
    child_ids: list[int] = field(default_factory=list)
    depth: int = 1
    visit_count: int = 1
    expansion_attempts: int = 0
    credit_sum: float = 0.0
    credit_count: int = 0
    creation_order: int = 0


@dataclass(slots=True)
class TreeEdge:
    parent_id: int
    child_id: int
    operator: OperatorName
    reference_node_id: int | None
    # Rank-normalized direct qualities captured when this edge is created.
    # They keep the expansion arm's historical credit stationary as later
    # candidates change the global rank reference set.
    parent_quality: float | None = None
    child_quality: float | None = None


@dataclass(slots=True)
class VirtualRoot:
    id: int = -1
    child_ids: list[int] = field(default_factory=list)
    visit_count: int = 0


@dataclass(frozen=True, slots=True)
class SelectionStep:
    decision_node_id: int
    option: str
    target_node_id: int | None
    score: float
    quality: float
    exploration: float
    probability: float
    prior: float
    credit_mean: float
    credit_count: int


@dataclass(frozen=True, slots=True)
class SelectionResult:
    node_id: int
    path: tuple[int, ...]
    steps: tuple[SelectionStep, ...]


__all__ = [
    "AlgorithmRecord",
    "OperatorName",
    "PROTOCOL_ID",
    "SelectionResult",
    "SelectionStep",
    "TreeEdge",
    "TreeNode",
    "VirtualRoot",
]
