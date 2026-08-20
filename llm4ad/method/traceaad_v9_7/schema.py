"""Protocol constants and search facts for TraceAAD V9.7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

REFINE_PROBABILITY: Final[float] = 0.7
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"


class Intent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


class Outcome(StrEnum):
    IMPROVE = "improve"
    PLATEAU = "plateau"
    REGRESS = "regress"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Program:
    id: int
    code: str
    fitness: float
    q: float
    length: int
    order: int


@dataclass(slots=True)
class Anchor:
    id: int
    program_id: int
    parent_id: int | None
    attempt_id: int | None
    root_id: int
    order: int
    n: int = 0


@dataclass(frozen=True, slots=True)
class Attempt:
    id: int
    anchor_id: int | None
    child_id: int | None
    program_id: int | None
    intent: str | None
    idea: str | None
    diff: str | None
    added: int
    removed: int
    parent_fitness: float | None
    child_fitness: float | None
    dq: float | None
    outcome: Outcome | None
    kind: str
    order: int
    stage: str
    iteration: int | None


@dataclass(slots=True)
class Pending:
    id: int
    anchor_id: int | None
    stage: str
    iteration: int | None
    order: int
    intent: str | None
    response: str


__all__ = [
    "INITIAL_ROOT_COUNT",
    "LOGICAL_MODEL_NAME",
    "MAX_HISTORY_EVENTS",
    "REFINE_PROBABILITY",
    "Anchor",
    "Attempt",
    "Intent",
    "Outcome",
    "Pending",
    "Program",
]
