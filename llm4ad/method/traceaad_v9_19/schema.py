"""State and frozen controls for TraceAAD V9.19."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
TRAJECTORY_WINDOW: Final[int] = 4

NEIGHBORHOOD_FRACTION: Final[float] = 0.05
MIN_NEIGHBORS: Final[int] = 2

W_PROMISE: Final[float] = 0.75
W_UNDERDEVELOPMENT: Final[float] = 0.10
W_TRAJECTORY: Final[float] = 0.15

ESS_FRACTION: Final[float] = 0.10
MIN_ESS_TARGET: Final[float] = 2.0

EXPLORE_SLOPE: Final[float] = 0.60
EXPLORE_MIN: Final[float] = 0.10
EXPLORE_MAX: Final[float] = 0.60
EXPLORE_NEUTRAL: Final[float] = 0.30
CROSSOVER_PROBABILITY: Final[float] = 0.25

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
    t_response: float = 0.5
    novelty: float | None = None
    behavior_tag: str | None = None
    opportunities: int = 0
    successful_opportunities: int = 0
    failed_opportunities: int = 0


@dataclass(slots=True)
class Pending:
    parent_id: int
    action: str | None
    response: str
    exact_prompt: str
    mode: str = "ordinary"
    request_seed: int | None = None
    decision_index: int | None = None
    promise: float | None = None
    underdevelopment: float | None = None
    t_response: float | None = None
    p_explore: float | None = None
    beta: float | None = None
    ess: float | None = None
    pool_size: int | None = None
    neighborhood_size: int | None = None
    reference_id: int | None = None


@dataclass(slots=True)
class Attempt:
    slot: int
    parent_id: int
    action: str | None
    outcome: str


__all__ = [
    "ESS_FRACTION",
    "EXPLORE_MAX",
    "EXPLORE_MIN",
    "EXPLORE_NEUTRAL",
    "CROSSOVER_PROBABILITY",
    "EXPLORE_SLOPE",
    "FAR_FROM_ARCHIVE",
    "INITIAL_ROOT_COUNT",
    "INTERMEDIATE",
    "MAX_HISTORY_EVENTS",
    "MAX_REPAIRS",
    "MIN_ESS_TARGET",
    "MIN_NEIGHBORS",
    "NEAR_KNOWN",
    "NEIGHBORHOOD_FRACTION",
    "NOVELTY_HIGH",
    "NOVELTY_LOW",
    "TRAJECTORY_WINDOW",
    "W_PROMISE",
    "W_TRAJECTORY",
    "W_UNDERDEVELOPMENT",
    "Action",
    "Algorithm",
    "Attempt",
    "Pending",
]
