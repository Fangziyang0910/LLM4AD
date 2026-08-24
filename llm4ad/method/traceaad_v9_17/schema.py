"""State and fixed controls for TraceAAD V9.17."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
ACTIVE_CAPACITY: Final[int] = 8
BLOCK_HORIZON: Final[int] = 3
MAX_HISTORY_EVENTS: Final[int] = 8


class Intent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


class HypothesisStatus(StrEnum):
    MATURING = "maturing"
    ACTIVE = "active"
    RESERVE = "reserve"


class Phase(StrEnum):
    ROOTS = "roots"
    INITIAL_MATURATION = "initial_maturation"
    DEVELOPMENT = "development"
    DISCOVERY = "discovery"
    MATURATION = "maturation"
    TERMINAL = "terminal"


class BlockKind(StrEnum):
    INITIAL_MATURATION = "initial_maturation"
    DEVELOPMENT = "development"
    MATURATION = "maturation"
    TERMINAL = "terminal"


@dataclass(slots=True)
class Algorithm:
    id: int
    code: str | None
    fitness: float | None
    parent_id: int | None
    hypothesis_id: int | None = None
    idea: str | None = None
    diff: str = ""
    added: int = 0
    removed: int = 0
    result: str | None = None
    created_by: str | None = None
    count: int = 0
    refine_count: int = 0
    explore_count: int = 0


@dataclass(slots=True)
class Hypothesis:
    id: int
    origin_node_id: int
    source_hypothesis_id: int | None
    status: HypothesisStatus
    frontier_node_id: int
    best_quality: float
    primary_slots: int = 1
    last_block_gain: float | None = None


@dataclass(slots=True)
class BlockState:
    id: int
    hypothesis_id: int
    kind: BlockKind
    q_before: float
    target_steps: int
    completed_steps: int = 0
    valid_results: int = 0
    selected_parent_ids: list[int] = field(default_factory=list)


@dataclass(slots=True)
class GenerationState:
    prompt: str
    parent_id: int
    intent: str | None
    mode: str
    hypothesis_id: int | None
    block_id: int | None = None
    block_step: int | None = None
    attempt: int = 1
    primary_charged: bool = False
    failed_code: str = ""
    failure_feedback: str = ""


@dataclass(slots=True)
class Pending:
    response: str
    attempt: int
    attempt_kind: str


__all__ = [
    "ACTIVE_CAPACITY",
    "BLOCK_HORIZON",
    "INITIAL_ROOT_COUNT",
    "MAX_HISTORY_EVENTS",
    "Algorithm",
    "BlockKind",
    "BlockState",
    "GenerationState",
    "Hypothesis",
    "HypothesisStatus",
    "Intent",
    "Pending",
    "Phase",
]
