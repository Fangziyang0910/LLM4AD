"""State and explicit controls for TraceAAD V9.22."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
IDEAS_PER_BATCH: Final[int] = 2
REALIZATIONS_PER_IDEA: Final[int] = 2
BATCH_SIZE: Final[int] = IDEAS_PER_BATCH * REALIZATIONS_PER_IDEA
MAX_HISTORY_EVENTS: Final[int] = 8
MAX_REALIZATION_EVENTS: Final[int] = 6
MAX_REPAIRS: Final[int] = 2

# Responses are percentile differences, so they already have a stable
# dimensionless range.  Failure records use the lower bound explicitly.
RESPONSE_CLIP: Final[float] = 1.0


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
    proposal: str
    outcome: str
    fitness: float | None
    response: float
    response_working: float
    response_scaffold: float
    improved_working: bool
    improved_scaffold: bool
    usable: bool
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
    response_working_values: list[float] = field(default_factory=list)
    response_scaffold_values: list[float] = field(default_factory=list)
    realization_ids: list[int] = field(default_factory=list)
    usable_trials: int = 0
    working_improvements: int = 0
    scaffold_improvements: int = 0
    action_trials: dict[str, int] = field(default_factory=dict)
    action_usable_trials: dict[str, int] = field(default_factory=dict)
    action_improvements: dict[str, int] = field(default_factory=dict)
    last_batch: int = 0

    @property
    def trials(self) -> int:
        return len(self.response_working_values)

    @property
    def response_working_mean(self) -> float:
        if not self.response_working_values:
            return 0.0
        return sum(self.response_working_values) / len(self.response_working_values)

    @property
    def response_scaffold_mean(self) -> float:
        if not self.response_scaffold_values:
            return 0.0
        return sum(self.response_scaffold_values) / len(self.response_scaffold_values)

    @property
    def response_mean(self) -> float:
        """Compatibility view for compact ledgers and older reports."""
        return self.response_working_mean


@dataclass(slots=True)
class Attempt:
    slot: int
    batch: int
    hypothesis_id: int | None
    source_hypothesis_id: int | None
    proposal: str
    idea: str | None
    parent_id: int | None
    node_id: int | None
    outcome: str
    fitness: float | None
    response: float | None
    response_working: float | None
    response_scaffold: float | None
    error: str | None


@dataclass(slots=True)
class Pending:
    prompt: str
    response: str
    parent_id: int | None
    hypothesis_id: int | None
    source_hypothesis_id: int | None
    proposal: str
    idea: str
    base_code: str
    batch: int
    slot: int
    base_scaffold_fitness: float
    base_parent_fitness: float
    base_working_fitness: float
    quality_reference: list[float]
    attempt: int = 1


__all__ = [
    "Attempt",
    "BATCH_SIZE",
    "Hypothesis",
    "IDEAS_PER_BATCH",
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
