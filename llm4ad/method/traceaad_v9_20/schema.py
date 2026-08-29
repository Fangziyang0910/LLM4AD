"""State and frozen controls for TraceAAD V9.20.

V9.20 keeps the online state deliberately small.  A node stores its measured
quality, behavior profile, formation path, and the direct opportunity ledger
needed to estimate whether another edit is worth purchasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 6
TRAJECTORY_WINDOW: Final[int] = 4
NEIGHBORHOOD_FRACTION: Final[float] = 0.05
MIN_NEIGHBORS: Final[int] = 2

# The allocation policy is an explicit mixture: most budget follows quality
# and observed continuation value, while a fixed minority covers under-tested
# behavior neighborhoods.
COVERAGE_MIX: Final[float] = 0.20
ESS_FRACTION: Final[float] = 0.10
MIN_ESS_TARGET: Final[float] = 2.0

# Action utilities are sampled with this temperature.  It is a probability
# calibration constant, not another quality signal.
ACTION_TEMPERATURE: Final[float] = 0.35
CROSSOVER_REFERENCE_MIX: Final[float] = 0.50
EXPLORE_MIN: Final[float] = 0.15
EXPLORE_MAX: Final[float] = 0.55
EXPLORE_NEUTRAL: Final[float] = 0.35

MAX_REPAIRS: Final[int] = 2

NEAR_KNOWN: Final[str] = "near-known"
INTERMEDIATE: Final[str] = "intermediate"
FAR_FROM_ARCHIVE: Final[str] = "far-from-archive"
NOVELTY_LOW: Final[float] = 1.0 / 3.0
NOVELTY_HIGH: Final[float] = 2.0 / 3.0


class Action(StrEnum):
    DEVELOP = "develop"
    EXPLORE = "explore"
    CROSSOVER = "crossover"


@dataclass(slots=True)
class Algorithm:
    id: int
    code: str | None
    fitness: float | None
    parent_id: int | None
    idea: str | None = None
    action: str | None = None
    created_slot: int = 0
    novelty: float | None = None
    behavior_tag: str | None = None
    opportunities: int = 0
    improvements: int = 0
    failures: int = 0
    last_outcome: str | None = None


@dataclass(slots=True)
class Pending:
    parent_id: int
    action: str | None
    response: str
    exact_prompt: str
    mode: str = "ordinary"
    request_seed: int | None = None
    attempt: int = 1
    decision_index: int | None = None
    quality_value: float | None = None
    continuation_value: float | None = None
    coverage_value: float | None = None
    allocation_probability: float | None = None
    action_probabilities: dict[str, float] | None = None
    reference_id: int | None = None
    reference_value: float | None = None


@dataclass(slots=True)
class Attempt:
    slot: int
    parent_id: int
    action: str | None
    outcome: str
    reference_id: int | None = None


__all__ = [
    "ACTION_TEMPERATURE",
    "Action",
    "Algorithm",
    "Attempt",
    "COVERAGE_MIX",
    "CROSSOVER_REFERENCE_MIX",
    "ESS_FRACTION",
    "EXPLORE_MAX",
    "EXPLORE_MIN",
    "EXPLORE_NEUTRAL",
    "FAR_FROM_ARCHIVE",
    "INITIAL_ROOT_COUNT",
    "INTERMEDIATE",
    "MAX_HISTORY_EVENTS",
    "MAX_REPAIRS",
    "MIN_NEIGHBORS",
    "NEIGHBORHOOD_FRACTION",
    "MIN_ESS_TARGET",
    "NEAR_KNOWN",
    "NOVELTY_HIGH",
    "NOVELTY_LOW",
    "Pending",
    "TRAJECTORY_WINDOW",
]
