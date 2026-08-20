"""Protocol constants and search facts for TraceAAD V9.13."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

PROTOCOL_ID: Final[str] = "traceaad-v9.13-region-frontier-explore-r3"
# The intent schedule reuses the V9.7 mapping so that a V9.13 control branch
# (treatment PP) draws the identical Refine/Explore sequence as V9.7 from the
# same seed; Stage A branches forked from one prefix stay intent-aligned.
INTENT_SCHEDULE_ID: Final[str] = "traceaad-v9.7-route-refine-explore"
REFINE_PROBABILITY: Final[float] = 0.7
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"
FRONTIER_ACTIVATION_EVALS: Final[int] = 200


class Treatment(StrEnum):
    """Frozen Explore context treatment selected in Stage P."""

    PP = "pp"
    FP = "fp"


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
    treatment: str = Treatment.PP.value


@dataclass(slots=True)
class Pending:
    id: int
    anchor_id: int | None
    stage: str
    iteration: int | None
    order: int
    intent: str | None
    response: str
    treatment: str = Treatment.PP.value


__all__ = [
    "FRONTIER_ACTIVATION_EVALS",
    "INITIAL_ROOT_COUNT",
    "INTENT_SCHEDULE_ID",
    "LOGICAL_MODEL_NAME",
    "MAX_HISTORY_EVENTS",
    "PROTOCOL_ID",
    "REFINE_PROBABILITY",
    "Anchor",
    "Attempt",
    "Intent",
    "Outcome",
    "Pending",
    "Program",
    "Treatment",
]
