"""Protocol constants and search facts for TraceAAD V9.9."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

PROTOCOL_ID: Final[str] = "traceaad-v9.9-anchor-operator-joint"
SCORE_FORMULA_VERSION: Final[str] = "anchor-joint-q-u-c-v1"
REFINE_PRIOR: Final[float] = 0.7
EXPLORE_PRIOR: Final[float] = 0.3
INITIAL_ROOT_COUNT: Final[int] = 8
ROOT_CANDIDATE_COUNT: Final[int] = 12
MAX_HISTORY_EVENTS: Final[int] = 8
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"
DEFAULT_MAX_RESPONSES: Final[int] = 5000
DEFAULT_MAX_CONSECUTIVE_ERRORS: Final[int] = 50
LAMBDA_U: Final[float] = 0.25
PATH_HALF_LIFE: Final[float] = 4.0
RANK_HALF_LIFE: Final[float] = 5.0
TEMPERATURE: Final[float] = 0.5


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
    n_refine: int = 0
    n_explore: int = 0

    def count(self, intent: Intent | str) -> int:
        return self.n_refine if Intent(intent) is Intent.REFINE else self.n_explore

    def increment(self, intent: Intent | str) -> None:
        if Intent(intent) is Intent.REFINE:
            self.n_refine += 1
        else:
            self.n_explore += 1


@dataclass(frozen=True, slots=True)
class Attempt:
    id: int
    response_id: str
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
    response_id: str
    anchor_id: int | None
    stage: str
    iteration: int | None
    order: int
    intent: str | None
    prompt: str
    generation_seed: int | None
    selection: dict[str, Any] | None
    response: str | None = None


__all__ = [
    "DEFAULT_MAX_CONSECUTIVE_ERRORS",
    "DEFAULT_MAX_RESPONSES",
    "EXPLORE_PRIOR",
    "INITIAL_ROOT_COUNT",
    "LAMBDA_U",
    "LOGICAL_MODEL_NAME",
    "MAX_HISTORY_EVENTS",
    "PATH_HALF_LIFE",
    "PROTOCOL_ID",
    "RANK_HALF_LIFE",
    "REFINE_PRIOR",
    "ROOT_CANDIDATE_COUNT",
    "SCORE_FORMULA_VERSION",
    "TEMPERATURE",
    "Anchor",
    "Attempt",
    "Intent",
    "Outcome",
    "Pending",
    "Program",
]
