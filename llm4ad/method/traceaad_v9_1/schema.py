"""TraceAAD V9.1 trajectory-centred search state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, TypeAlias

PROTOCOL_ID: Final[str] = "traceaad-v9.1-trajectory-centered"

NodeId: TypeAlias = int
EdgeId: TypeAlias = int


class OperatorName(StrEnum):
    IDEATE = "trace_ideate"
    REFINE = "trace_refine"
    SYNTHESIZE = "trace_synthesize"
    TRANSFER = "trace_transfer"


@dataclass(slots=True)
class VirtualRoot:
    """Structural container only; it never receives search credit."""

    id: int = -1
    child_ids: list[NodeId] = field(default_factory=list)


@dataclass(slots=True)
class ProgramNode:
    """One executable endpoint together with its unique formation trajectory."""

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
    creation_order: int
    batch_id: int | None
    operator: str
    bootstrap_reference_node_ids: list[NodeId]
    trajectory_best_value: float
    trajectory_best_node_id: NodeId
    verification_count: int = 0
    valid_candidate_count: int = 0
    route_advance_count: int = 0
    global_advance_count: int = 0
    recent_advances: list[bool] = field(default_factory=list)
    last_verification_batch_id: int | None = None


@dataclass(frozen=True, slots=True)
class ImprovementEdge:
    id: EdgeId
    parent_id: NodeId
    child_id: NodeId
    operator: OperatorName
    implemented_idea: str
    reference_node_id: NodeId | None
    delta_parent: float
    delta_global_best: float | None
    trajectory_best_before: float
    advances_parent_trajectory: bool
    outcome: str
    delta_loc: int
    code_change_ratio: float
    new_global_best: bool
    global_best_update_reason: str | None
    iteration: int
    batch_id: int
    sibling_seq: int
    sample_order: int


__all__ = [
    "EdgeId",
    "ImprovementEdge",
    "NodeId",
    "OperatorName",
    "PROTOCOL_ID",
    "ProgramNode",
    "VirtualRoot",
]
