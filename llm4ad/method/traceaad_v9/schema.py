"""TraceAAD V9 tree-search state schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, TypeAlias

PROTOCOL_ID: Final[str] = "traceaad-v9-core"

NodeId: TypeAlias = int
EdgeId: TypeAlias = int


class OperatorName(StrEnum):
    IDEATE = "trace_ideate"
    REFINE = "trace_refine"
    SYNTHESIZE = "trace_synthesize"
    TRANSFER = "trace_transfer"


@dataclass(slots=True)
class VirtualRoot:
    """Structural root. It deliberately has no program or fitness fields."""

    id: int = -1
    child_ids: list[NodeId] = field(default_factory=list)
    visit_count: int = 0
    subtree_value: float | None = None
    subtree_best_node_id: NodeId | None = None


@dataclass(slots=True)
class ProgramNode:
    id: NodeId
    code: str
    idea: str
    fitness: float
    directed_fitness: float
    program_loc: int
    code_hash: str
    parent_id: int
    incoming_edge_id: EdgeId | None
    child_ids: list[NodeId]
    depth: int
    visit_count: int
    expansion_count: int
    subtree_value: float
    subtree_best_node_id: NodeId
    creation_order: int
    batch_id: int | None
    operator: str


@dataclass(frozen=True, slots=True)
class ImprovementEdge:
    id: EdgeId
    parent_id: NodeId
    child_id: NodeId
    operator: OperatorName
    implemented_idea: str
    reference_node_id: NodeId | None
    reference_root_branch_id: NodeId | None
    delta_parent: float
    delta_global_best: float | None
    outcome: str
    delta_loc: int
    code_change_ratio: float
    new_global_best: bool
    global_best_update_reason: str | None
    iteration: int
    batch_id: int
    sibling_seq: int
    sample_order: int


@dataclass(frozen=True, slots=True)
class SelectionStep:
    decision_node_id: int
    option: str
    target_node_id: NodeId | None
    quality: float
    raw_value: float | None
    option_visits: int
    score: float


__all__ = [
    "EdgeId",
    "ImprovementEdge",
    "NodeId",
    "OperatorName",
    "PROTOCOL_ID",
    "ProgramNode",
    "SelectionStep",
    "VirtualRoot",
]
