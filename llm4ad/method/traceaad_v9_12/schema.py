"""Protocol constants and search facts for TraceAAD V9.12."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

PROTOCOL_ID: Final[str] = "traceaad-v9.12-progress-conditioned-operator"
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
PROGRESS_WINDOW: Final[int] = 8
EXPLORE_PROBABILITY_MIN: Final[float] = 0.10
EXPLORE_PROBABILITY_MAX: Final[float] = 0.30
MIN_EXPLORE_REMAINING_EVALS: Final[int] = 2
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"


class Intent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


class Regime(StrEnum):
    DEVELOP = "develop"
    EXPLORE = "explore"
    FOLLOWUP = "followup"


class Outcome(StrEnum):
    IMPROVE = "improve"
    PLATEAU = "plateau"
    REGRESS = "regress"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Program:
    id: int
    code: str
    code_hash: str
    fitness: float
    q: float
    length: int
    order: int
    evaluation_seconds: float | None = None


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
    "EXPLORE_PROBABILITY_MAX",
    "EXPLORE_PROBABILITY_MIN",
    "MIN_EXPLORE_REMAINING_EVALS",
    "PROTOCOL_ID",
    "PROGRESS_WINDOW",
    "Anchor",
    "Attempt",
    "Intent",
    "Outcome",
    "Pending",
    "Program",
    "Regime",
]
