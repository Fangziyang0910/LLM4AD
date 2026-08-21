"""Protocol constants and search facts for TraceAAD V9.8."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

REFINE_PROBABILITY: Final[float] = 0.7
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"
DEFAULT_MAX_RESPONSES: Final[int] = 5000


class Intent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


class Outcome(StrEnum):
    IMPROVE = "improve"
    PLATEAU = "plateau"
    REGRESS = "regress"
    INVALID = "invalid"


class AllocationPolicy(StrEnum):
    """Pre-registered allocation arms used by Stage A and the formal method."""

    FULL = "q_u_c_m"
    Q_U_C = "q_u_c"
    Q_U = "q_u"
    HYPOTHESIS_UNIFORM = "hypothesis_uniform"
    ROUTE_Q_U = "route_q_u"


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
    hypothesis_id: int
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
class Hypothesis:
    id: int
    entry_anchor_id: int
    parent_hypothesis_id: int | None
    creation_attempt_id: int | None
    root_id: int
    q0: float
    q_base: float | None
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
    source_hypothesis_id: int | None
    child_id: int | None
    child_hypothesis_id: int | None
    program_id: int | None
    intent: str | None
    idea: str | None
    diff: str | None
    added: int
    removed: int
    parent_fitness: float | None
    child_fitness: float | None
    dq: float | None
    frontier_before: float | None
    frontier_after: float | None
    realized_gain: float | None
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
    hypothesis_id: int | None
    stage: str
    iteration: int | None
    order: int
    intent: str | None
    prompt: str
    generation_seed: int | None
    selection: dict[str, Any] | None
    response: str | None = None


__all__ = [
    "DEFAULT_MAX_RESPONSES",
    "INITIAL_ROOT_COUNT",
    "LOGICAL_MODEL_NAME",
    "MAX_HISTORY_EVENTS",
    "REFINE_PROBABILITY",
    "AllocationPolicy",
    "Anchor",
    "Attempt",
    "Hypothesis",
    "Intent",
    "Outcome",
    "Pending",
    "Program",
]
