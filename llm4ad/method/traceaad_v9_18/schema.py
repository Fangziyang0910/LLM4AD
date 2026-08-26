"""State and fixed controls for TraceAAD V9.18-R0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8

# Keep the operator mixture fixed so allocation is the only online intervention.
REFINE_PROBABILITY: Final[float] = 0.70
EXPLORE_PROBABILITY: Final[float] = 0.30

# Quality allocation with the retained ESS-controlled Boltzmann rule.
ESS_FRACTION: Final[float] = 0.10
MIN_ESS_TARGET: Final[float] = 2.0

# R0 opportunity prior. This is deliberately small and applies only to valid
# Explore entries; it is not a continuation-value estimate.
OPPORTUNITY_LAMBDA: Final[float] = 0.10
OPPORTUNITY_TAU: Final[float] = 2.0
GLOBAL_FACTS_WINDOW: Final[int] = 32


class Intent(StrEnum):
    REFINE = "refine"
    EXPLORE = "explore"


@dataclass(slots=True)
class Algorithm:
    id: int
    code: str | None
    fitness: float | None
    parent_id: int | None
    count: int = 0
    idea: str | None = None
    refine_count: int = 0
    explore_count: int = 0
    created_by: str | None = None
    entry_id: int | None = None
    created_slot: int = 0
    n_after: int = 0
    is_explore_entry: bool = False


@dataclass(slots=True)
class Pending:
    parent_id: int
    intent: str | None
    response: str
    mode: str = "ordinary"
    entry_id: int | None = None
    decision_index: int | None = None
    request_seed: int | None = None
    prompt_hash: str | None = None
    prompt_chars: int | None = None
    facts_hash: str | None = None
    facts_omitted: bool = False


__all__ = [
    "ESS_FRACTION",
    "EXPLORE_PROBABILITY",
    "INITIAL_ROOT_COUNT",
    "GLOBAL_FACTS_WINDOW",
    "MAX_HISTORY_EVENTS",
    "MIN_ESS_TARGET",
    "OPPORTUNITY_LAMBDA",
    "OPPORTUNITY_TAU",
    "REFINE_PROBABILITY",
    "Algorithm",
    "Intent",
    "Pending",
]
