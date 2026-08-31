"""State and protocol constants for TraceAAD V10.

V10 implements Trajectory-aware Joint Design Opportunity Allocation: each
primary slot first builds a bounded opportunity set over screened archive
states, asks a valuation critic for the plausibly optimal subset, allocates
the slot by lexicographic coverage, and only then conditions generation on
the selected opportunity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

# Protocol constants (V10 design, section 9).
SCREEN_SIZE: Final[int] = 8  # K_s: starts entering the relative judgement
REFERENCE_COUNT: Final[int] = 2  # K_d: transfer references per start
COMPETITIVE_SET_SIZE: Final[int] = 4  # K_c: opportunities kept by the critic
FORMATION_WINDOW: Final[int] = 8  # H_tau: formation path steps kept per node
G_HORIZONS: Final[tuple[int, ...]] = (1, 2, 4)  # H_G
INITIAL_ROOT_COUNT: Final[int] = 8  # N_root: valid roots before the loop
RESTART_CARDS: Final[int] = 3  # N_card: verified improvement cards for Restart
MAX_REPAIRS: Final[int] = 2  # bounded execution repairs before giving up

DEVELOP: Final[str] = "develop"
PIVOT: Final[str] = "pivot"
TRANSFER: Final[str] = "transfer"
RESTART: Final[str] = "restart"
SEMANTIC_REPAIR: Final[str] = "semantic_repair"

OPERATORS: Final[tuple[str, ...]] = (
    DEVELOP,
    PIVOT,
    TRANSFER,
    RESTART,
    SEMANTIC_REPAIR,
)
OPENING_OPERATORS: Final[frozenset[str]] = frozenset({PIVOT, TRANSFER, RESTART})

VALID_OUTCOMES: Final[frozenset[str]] = frozenset({"improve", "plateau", "regress"})


@dataclass(slots=True)
class ProgramNode:
    """One valid, evaluated program in the archive."""

    id: int
    code: str
    fitness: float
    parent_id: int | None
    thread_id: int
    idea: str | None
    slot: int  # primary slot at which the node entered the archive


@dataclass(slots=True)
class Thread:
    """One experiment thread Gamma: a direction's claim and budget use.

    ``best_history[h - 1]`` is the best valid quality inside the thread after
    ``h`` primary slots, so ``G_h = best_history[h - 1] - q_origin`` whenever
    ``q_origin`` is defined (initialization threads have none).
    """

    id: int
    origin_action: str  # "init" | "pivot" | "transfer" | "restart"
    origin_idea: str
    origin_slot: int
    created_node_id: int
    q_origin: float | None = None
    opportunities_used: int = 0
    best_history: list[float] = field(default_factory=list)

    @property
    def best_fitness(self) -> float | None:
        return self.best_history[-1] if self.best_history else None

    def g_value(self, horizon: int) -> float | None:
        """Return G_h when the thread both defines and has observed it."""
        if self.q_origin is None or self.opportunities_used < horizon:
            return None
        return self.best_history[horizon - 1] - self.q_origin


@dataclass(slots=True)
class AttemptRecord:
    """One ledger entry: a primary allocation from a start (or a restart)."""

    slot: int
    round_index: int
    operator: str
    idea: str
    outcome: str
    start_id: int | None
    start_fitness: float | None
    child_id: int | None
    child_fitness: float | None
    thread_of_start: int | None
    created_thread: int | None
    reference_id: int | None = None
    q_origin: float | None = None  # frozen restart reference quality
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Opportunity:
    """One design experiment a = (s, o, r) offered to the critic."""

    opportunity_id: str
    operator: str
    start_id: int | None  # None only for Restart
    reference_id: int | None = None


@dataclass(frozen=True, slots=True)
class CompetitiveEntry:
    """One critic-selected opportunity with its valuation metadata."""

    opportunity: Opportunity
    rank: int
    reason: str
    evidence_refs: tuple[str, ...]
    payoff_horizon: str | None
    semantic_mismatch: str | None


@dataclass(frozen=True, slots=True)
class CriticResult:
    """Parsed critic output, or the conservative fallback."""

    entries: tuple[CompetitiveEntry, ...]
    not_applicable: tuple[tuple[str, str], ...]
    invalid: bool  # True when the conservative fallback was used


@dataclass(slots=True)
class Pending:
    """A generated candidate awaiting formal settlement."""

    prompt: str
    response: str
    operator: str
    idea: str
    slot: int
    round_index: int
    start_id: int | None
    reference_id: int | None
    base_code: str  # start code (empty for Restart), shown during repairs
    start_fitness: float | None
    q_origin: float | None  # Restart: global best frozen at allocation
    semantic_mismatch: str | None
    opportunity_id: str
    critic_rank: int | None
    attempt: int = 1


__all__ = [
    "AttemptRecord",
    "COMPETITIVE_SET_SIZE",
    "CompetitiveEntry",
    "CriticResult",
    "DEVELOP",
    "FORMATION_WINDOW",
    "G_HORIZONS",
    "INITIAL_ROOT_COUNT",
    "MAX_REPAIRS",
    "OPENING_OPERATORS",
    "OPERATORS",
    "Opportunity",
    "Pending",
    "PIVOT",
    "ProgramNode",
    "REFERENCE_COUNT",
    "RESTART",
    "RESTART_CARDS",
    "SCREEN_SIZE",
    "SEMANTIC_REPAIR",
    "Thread",
    "TRANSFER",
    "VALID_OUTCOMES",
]
