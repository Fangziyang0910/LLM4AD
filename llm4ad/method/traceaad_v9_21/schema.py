"""Small, explicit state objects for TraceAAD V9.21."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
REALIZATIONS_PER_IDEA: Final[int] = 2
BATCH_SIZE: Final[int] = 4
MAX_HISTORY_EVENTS: Final[int] = 8
MAX_REALIZATION_EVENTS: Final[int] = 6
MAX_REPAIRS: Final[int] = 2
RESPONSE_CLIP: Final[float] = 2.0


@dataclass(slots=True)
class ProgramNode:
    id: int
    code: str
    fitness: float
    parent_id: int | None
    hypothesis_id: int | None
    idea: str | None
    role: str
    slot: int


@dataclass(slots=True)
class Realization:
    id: int
    hypothesis_id: int
    idea: str
    parent_id: int | None
    slot: int
    outcome: str
    fitness: float | None
    response: float
    node_id: int | None = None
    error: str | None = None
    attempt: int = 1


@dataclass(slots=True)
class Hypothesis:
    id: int
    entry_idea: str
    source_node_id: int
    scaffold_node_id: int
    working_node_id: int | None
    parent_hypothesis_id: int | None
    donor_node_id: int | None
    created_batch: int
    responses: list[float] = field(default_factory=list)
    realization_ids: list[int] = field(default_factory=list)
    last_batch: int = 0

    @property
    def trials(self) -> int:
        return len(self.responses)

    @property
    def response_mean(self) -> float:
        if not self.responses:
            return 0.0
        return sum(self.responses) / len(self.responses)


@dataclass(slots=True)
class Attempt:
    slot: int
    batch: int
    hypothesis_id: int | None
    proposal: str
    idea: str | None
    parent_id: int | None
    node_id: int | None
    outcome: str
    fitness: float | None
    response: float | None
    error: str | None


@dataclass(slots=True)
class Pending:
    prompt: str
    response: str
    parent_id: int | None
    hypothesis_id: int | None
    proposal: str
    idea: str
    base_code: str
    batch: int
    slot: int
    base_scaffold_fitness: float
    base_parent_fitness: float
    attempt: int = 1


__all__ = [
    "Attempt",
    "BATCH_SIZE",
    "Hypothesis",
    "INITIAL_ROOT_COUNT",
    "MAX_HISTORY_EVENTS",
    "MAX_REALIZATION_EVENTS",
    "MAX_REPAIRS",
    "Pending",
    "ProgramNode",
    "REALIZATIONS_PER_IDEA",
    "Realization",
    "RESPONSE_CLIP",
]
