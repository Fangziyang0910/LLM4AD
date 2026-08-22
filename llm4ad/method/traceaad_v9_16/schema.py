"""State and fixed controls for TraceAAD V9.16."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8

# Keep the operator mixture fixed so landing is the only online intervention.
REFINE_PROBABILITY: Final[float] = 0.70
EXPLORE_PROBABILITY: Final[float] = 0.30

LANDING_RATIO: Final[float] = 0.10
LANDING_PROBABILITY: Final[float] = 0.125
LANDING_HORIZON: Final[int] = 3

# Quality-only parent allocation with the retained ESS-controlled Boltzmann rule.
ESS_FRACTION: Final[float] = 0.10
MIN_ESS_TARGET: Final[float] = 2.0


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


@dataclass(slots=True)
class Pending:
    parent_id: int
    intent: str | None
    response: str
    mode: str = "ordinary"
    entry_id: int | None = None
    landing_id: int | None = None
    landing_step: int | None = None


@dataclass(slots=True)
class LandingState:
    id: int
    entry_id: int
    origin_id: int
    origin_parent_id: int
    latest_valid_id: int
    completed_steps: int = 0
    valid_steps: int = 0
    strict_improvements: int = 0
    start_eval: int = 0
    final_gain: float | None = None
    max_gain: float = 0.0


__all__ = [
    "ESS_FRACTION",
    "EXPLORE_PROBABILITY",
    "INITIAL_ROOT_COUNT",
    "LANDING_HORIZON",
    "LANDING_PROBABILITY",
    "LANDING_RATIO",
    "MAX_HISTORY_EVENTS",
    "MIN_ESS_TARGET",
    "REFINE_PROBABILITY",
    "Algorithm",
    "Intent",
    "LandingState",
    "Pending",
]
