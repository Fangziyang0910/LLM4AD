"""TraceAAD V10: trajectory-aware joint design opportunity allocation."""

from .artifacts import RunArtifacts
from .schema import (
    COMPETITIVE_SET_SIZE,
    FORMATION_WINDOW,
    G_HORIZONS,
    INITIAL_ROOT_COUNT,
    MAX_REPAIRS,
    RESTART_CARDS,
    SCREEN_SIZE,
    REFERENCE_COUNT,
)
from .traceaad import TraceAADV10

__all__ = [
    "COMPETITIVE_SET_SIZE",
    "FORMATION_WINDOW",
    "G_HORIZONS",
    "INITIAL_ROOT_COUNT",
    "MAX_REPAIRS",
    "REFERENCE_COUNT",
    "RESTART_CARDS",
    "SCREEN_SIZE",
    "RunArtifacts",
    "TraceAADV10",
]
