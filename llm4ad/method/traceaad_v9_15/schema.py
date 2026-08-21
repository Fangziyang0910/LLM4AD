"""Core state used by TraceAAD V9.15."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8

# Operator schedule: p_E = clip(p_0 + alpha * n_stag / (W_stag + n_stag), p_min, p_max).
BASE_EXPLORE_PROBABILITY: Final[float] = 0.20
STAGNATION_WINDOW: Final[float] = 50.0
STAGNATION_GAIN: Final[float] = 0.30
EXPLORE_PROBABILITY_MIN: Final[float] = 0.15
EXPLORE_PROBABILITY_MAX: Final[float] = 0.50

# Protected exploration bonus: gamma(n_R) * min(parent gap, delta_t) with
# gamma(n) = 1 / (n + 1) and delta_t = BONUS_CAP_SCALE * s_t.
BONUS_CAP_SCALE: Final[float] = 2.0

# Trajectory continuation window: the last min(k, depth - 1) formation steps
# between evaluated ancestors; depth-1 because the virtual root has no fitness.
TRAJECTORY_WINDOW: Final[int] = 6

# ESS control: target ESS = ESS_FRACTION * |pool|, floored so the target stays
# feasible while the pool holds fewer than ten algorithms (ESS >= 1 always).
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


@dataclass(slots=True)
class Pending:
    parent_id: int
    intent: str | None
    response: str


__all__ = [
    "BASE_EXPLORE_PROBABILITY",
    "BONUS_CAP_SCALE",
    "ESS_FRACTION",
    "EXPLORE_PROBABILITY_MAX",
    "EXPLORE_PROBABILITY_MIN",
    "INITIAL_ROOT_COUNT",
    "MAX_HISTORY_EVENTS",
    "MIN_ESS_TARGET",
    "STAGNATION_GAIN",
    "STAGNATION_WINDOW",
    "TRAJECTORY_WINDOW",
    "Algorithm",
    "Intent",
    "Pending",
]
