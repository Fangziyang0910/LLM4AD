"""Core state used by TraceAAD V9.14."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

REFINE_PROBABILITY: Final[float] = 0.7
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8


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


@dataclass(slots=True)
class Pending:
    parent_id: int
    intent: str | None
    response: str


__all__ = [
    "INITIAL_ROOT_COUNT",
    "MAX_HISTORY_EVENTS",
    "REFINE_PROBABILITY",
    "Algorithm",
    "Intent",
    "Pending",
]
