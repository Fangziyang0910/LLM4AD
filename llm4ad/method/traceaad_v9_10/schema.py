"""Protocol constants and search facts for TraceAAD V9.10."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

REFINE_PRIOR: Final[float] = 0.7
EXPLORE_PRIOR: Final[float] = 0.3
PRIOR_STRENGTH: Final[float] = 2.0
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"
DEFAULT_MAX_RESPONSES: Final[int] = 5000
DEFAULT_MAX_CONSECUTIVE_ERRORS: Final[int] = 50
RECENCY_HALF_LIFE: Final[float] = 20.0
CHILD_WINDOW: Final[int] = 3
PARENT_CHAIN_WINDOW: Final[int] = 4
PARENT_CHAIN_HALF_LIFE: Final[float] = 2.0


class Intent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


class Outcome(StrEnum):
    IMPROVE = "improve"
    PLATEAU = "plateau"
    REGRESS = "regress"
    INVALID = "invalid"


class ActionStatus(StrEnum):
    PENDING = "pending"
    SETTLED = "settled"


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
    action_id: int | None
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


@dataclass(slots=True)
class Action:
    """One model response from a search anchor plus its short-window result."""

    id: int
    response_id: str
    anchor_id: int
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
    status: ActionStatus = ActionStatus.PENDING
    result: int | None = None
    observed_depth: int = 0
    window_best_q: float | None = None
    settled_order: int | None = None

    def settle(self, result: int, settled_order: int) -> None:
        self.status = ActionStatus.SETTLED
        self.result = result
        self.settled_order = settled_order


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
    "CHILD_WINDOW",
    "DEFAULT_MAX_CONSECUTIVE_ERRORS",
    "DEFAULT_MAX_RESPONSES",
    "EXPLORE_PRIOR",
    "INITIAL_ROOT_COUNT",
    "LOGICAL_MODEL_NAME",
    "MAX_HISTORY_EVENTS",
    "PARENT_CHAIN_HALF_LIFE",
    "PARENT_CHAIN_WINDOW",
    "PRIOR_STRENGTH",
    "RECENCY_HALF_LIFE",
    "REFINE_PRIOR",
    "Action",
    "ActionStatus",
    "Anchor",
    "Intent",
    "Outcome",
    "Pending",
    "Program",
]
