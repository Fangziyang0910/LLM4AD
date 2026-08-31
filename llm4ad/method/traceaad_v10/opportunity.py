"""Opportunity construction: screening, references, coverage, observations.

Implements the V10 allocation-side views of the archive: the bounded
screening index that decides which states the critic sees, the cross-thread
transfer references, the per-operator observation facts, and the
lexicographic coverage used to allocate inside the competitive set.
"""

from __future__ import annotations

import random
from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from .schema import (
    DEVELOP,
    G_HORIZONS,
    OPENING_OPERATORS,
    OPERATORS,
    PIVOT,
    REFERENCE_COUNT,
    RESTART,
    SCREEN_SIZE,
    SEMANTIC_REPAIR,
    TRANSFER,
    VALID_OUTCOMES,
    AttemptRecord,
    CompetitiveEntry,
    Opportunity,
    ProgramNode,
    Thread,
)


def mid_rank(fitness: float, values: Sequence[float]) -> float:
    """Standardized mid-rank over the archive; best 1, worst 0, flat 0.5."""
    if len(values) <= 1:
        return 0.5
    return _mid_rank_sorted(fitness, sorted(values))


def _mid_rank_sorted(fitness: float, ordered: Sequence[float]) -> float:
    if len(ordered) <= 1:
        return 0.5
    less = bisect_left(ordered, fitness)
    equal = bisect_right(ordered, fitness) - less
    rank = less + (equal - 1) / 2.0
    return rank / (len(ordered) - 1)


def start_use_counts(attempts: Sequence[AttemptRecord]) -> dict[int, int]:
    """n_t(s): times each node was used as a start, failures included."""
    counts: dict[int, int] = {}
    for attempt in attempts:
        if attempt.start_id is not None:
            counts[attempt.start_id] = counts.get(attempt.start_id, 0) + 1
    return counts


def screening_index(rank: float, uses: int) -> float:
    """R_t(s) = Q_t(s) + 1 / sqrt(1 + n_t(s)); both terms in [0, 1]."""
    return rank + 1.0 / ((1 + uses) ** 0.5)


def screen_shortlist(
    nodes: dict[int, ProgramNode],
    attempts: Sequence[AttemptRecord],
    best_node_id: int | None,
    *,
    size: int = SCREEN_SIZE,
    rng: random.Random,
) -> list[int]:
    """Return the K_s start ids by screening index, incumbent guaranteed."""
    if not nodes:
        return []
    ordered = sorted(node.fitness for node in nodes.values())
    uses = start_use_counts(attempts)
    scored = [
        (screening_index(_mid_rank_sorted(node.fitness, ordered), uses.get(node.id, 0)), node.id)
        for node in nodes.values()
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    shortlist = _top_k(scored, size, rng)
    if best_node_id is not None and best_node_id in nodes and best_node_id not in shortlist:
        members = _scored_of(scored, shortlist)
        lowest = min(item[0] for item in members)
        replaceable = [item for item in members if item[0] == lowest]
        shortlist.remove(rng.choice(replaceable)[1])
        shortlist.append(best_node_id)
    shortlist.sort()
    return shortlist


def _scored_of(
    scored: list[tuple[float, int]], ids: Sequence[int]
) -> list[tuple[float, int]]:
    wanted = set(ids)
    return [item for item in scored if item[1] in wanted]


def _top_k(
    scored: list[tuple[float, int]], size: int, rng: random.Random
) -> list[int]:
    """Take the top-k indices, breaking boundary ties with seeded chance."""
    if len(scored) <= size:
        return [item[1] for item in scored]
    boundary = scored[size - 1][0]
    guaranteed = [item for item in scored[: size - 1] if item[0] > boundary]
    tie_group = [item for item in scored if item[0] == boundary]
    deficit = size - len(guaranteed)
    if len(tie_group) < deficit:  # float-adjacent scores at the boundary
        remaining = [item for item in scored if item not in guaranteed]
        remaining.sort(key=lambda item: (-item[0], item[1]))
        return [item[1] for item in (guaranteed + remaining[:deficit])]
    chosen = guaranteed + rng.sample(tie_group, deficit)
    return [item[1] for item in chosen]


def ancestors(nodes: dict[int, ProgramNode], node_id: int) -> set[int]:
    seen: set[int] = set()
    current = nodes[node_id].parent_id
    while current is not None and current in nodes and current not in seen:
        seen.add(current)
        current = nodes[current].parent_id
    return seen


def descendants(nodes: dict[int, ProgramNode], node_id: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for node in nodes.values():
        if node.parent_id is not None:
            children.setdefault(node.parent_id, []).append(node.id)
    seen: set[int] = set()
    stack = list(children.get(node_id, []))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(children.get(current, []))
    return seen


def reference_candidates(
    nodes: dict[int, ProgramNode], start: ProgramNode
) -> list[int]:
    """Other-thread, non-ancestor, non-descendant nodes, best quality first."""
    excluded = ancestors(nodes, start.id) | descendants(nodes, start.id) | {start.id}
    candidates = [
        node
        for node in nodes.values()
        if node.id not in excluded and node.thread_id != start.thread_id
    ]
    candidates.sort(key=lambda node: (-node.fitness, node.id))
    return [node.id for node in candidates]


def build_opportunities(
    nodes: dict[int, ProgramNode],
    shortlist: Sequence[int],
    *,
    reference_count: int = REFERENCE_COUNT,
) -> tuple[list[Opportunity], dict[int, list[int]]]:
    """Build Omega_t: per-start Develop/Pivot/SemanticRepair + Transfers, plus
    exactly one global Restart."""
    opportunities: list[Opportunity] = []
    references: dict[int, list[int]] = {}
    counter = 0

    def _add(operator: str, start_id: int | None, reference_id: int | None) -> None:
        nonlocal counter
        counter += 1
        opportunities.append(
            Opportunity(
                opportunity_id=f"O{counter}",
                operator=operator,
                start_id=start_id,
                reference_id=reference_id,
            )
        )

    for start_id in shortlist:
        start = nodes[start_id]
        refs = reference_candidates(nodes, start)[:reference_count]
        references[start_id] = refs
        _add(DEVELOP, start_id, None)
        _add(PIVOT, start_id, None)
        for reference_id in refs:
            _add(TRANSFER, start_id, reference_id)
        _add(SEMANTIC_REPAIR, start_id, None)
    _add(RESTART, None, None)
    return opportunities, references


def coverage_tuple(
    attempt: Opportunity,
    attempts: Sequence[AttemptRecord],
    nodes: dict[int, ProgramNode],
) -> tuple[int, int, int]:
    """C(a) = (n(s, o), n(Gamma(s), o), n_G(o)); Restart thread coverage 0.

    Every restart shares the empty start state, so n(s, Restart) counts past
    restarts; only the thread element is zeroed because the empty start owns
    no thread.
    """
    start_operator = 0
    thread_operator = 0
    global_operator = 0
    for record in attempts:
        if record.operator != attempt.operator:
            continue
        global_operator += 1
        if attempt.start_id is None:
            if record.start_id is None:
                start_operator += 1
        elif record.start_id == attempt.start_id:
            start_operator += 1
        if (
            attempt.start_id is not None
            and record.thread_of_start == nodes[attempt.start_id].thread_id
        ):
            thread_operator += 1
    return start_operator, thread_operator, global_operator


def pair_counts(attempts: Sequence[AttemptRecord]) -> dict[tuple[int, int], int]:
    """n_pair(s, r): transfer allocations from s with reference r."""
    counts: dict[tuple[int, int], int] = {}
    for record in attempts:
        if record.operator == TRANSFER and record.start_id is not None:
            key = (record.start_id, int(record.reference_id))
            counts[key] = counts.get(key, 0) + 1
    return counts


def select_by_coverage(
    entries: Sequence[CompetitiveEntry],
    attempts: Sequence[AttemptRecord],
    nodes: dict[int, ProgramNode],
    rng: random.Random,
) -> CompetitiveEntry:
    """Lexicographic minimum of coverage, then smaller critic rank, then rng."""
    if not entries:
        raise ValueError("cannot allocate an empty competitive set")
    keyed = [
        (
            coverage_tuple(entry.opportunity, attempts, nodes),
            entry.rank,
            index,
            entry,
        )
        for index, entry in enumerate(entries)
    ]
    keyed.sort(key=lambda item: (item[0], item[1], item[2]))
    best_key = (keyed[0][0], keyed[0][1])
    tied = [item[3] for item in keyed if (item[0], item[1]) == best_key]
    return tied[0] if len(tied) == 1 else rng.choice(tied)


def operator_observations(
    attempts: Sequence[AttemptRecord], threads: dict[int, Thread]
) -> dict[str, dict[str, float | int | None]]:
    """Per-operator observation facts shown to the critic.

    Opening operators additionally report P(G_h > 0) over their threads and
    how often dropped threads recovered above their origin quality.  These
    are observed facts only; they never enter the allocation formula.
    """
    observations: dict[str, dict[str, float | int | None]] = {}
    for operator in OPERATORS:
        records = [record for record in attempts if record.operator == operator]
        trials = len(records)
        valid = [record for record in records if record.outcome in VALID_OUTCOMES]
        improvements = [record for record in valid if record.outcome == "improve"]
        deltas = [
            record.child_fitness - baseline
            for record in valid
            if record.child_fitness is not None
            and (baseline := record.start_fitness if record.start_fitness is not None else record.q_origin) is not None
        ]
        payload: dict[str, float | int | None] = {
            "trials": trials,
            "valid_rate": (len(valid) / trials) if trials else None,
            "one_step_improvement_rate": (len(improvements) / trials) if trials else None,
            "mean_one_step_change": (sum(deltas) / len(deltas)) if deltas else None,
        }
        if operator in OPENING_OPERATORS:
            payload.update(_opening_thread_stats(operator, threads))
        observations[operator] = payload
    return observations


def _opening_thread_stats(operator: str, threads: dict[int, Thread]) -> dict[str, float | int | None]:
    owned = [thread for thread in threads.values() if thread.origin_action == operator]
    stats: dict[str, float | int | None] = {}
    for horizon in G_HORIZONS:
        observed = [
            thread.g_value(horizon)
            for thread in owned
            if thread.g_value(horizon) is not None
        ]
        stats[f"P_G{horizon}_positive"] = (
            (sum(1 for value in observed if value > 0) / len(observed))
            if observed
            else None
        )
    dropped = [
        thread
        for thread in owned
        if thread.q_origin is not None
        and thread.opportunities_used >= 1
        and (thread.best_history[0] - thread.q_origin) <= 0
    ]
    recovered = []
    for thread in dropped:
        for used, best in enumerate(thread.best_history, start=1):
            if best > thread.q_origin:
                recovered.append(used)
                break
    stats["dropped_threads"] = len(dropped)
    stats["recovered_threads"] = len(recovered)
    stats["mean_recovery_slots"] = (sum(recovered) / len(recovered)) if recovered else None
    return stats


def g_summary(thread: Thread) -> dict[str, float | int | None]:
    """G digest for one thread; missing observations stay None, never zero."""
    summary: dict[str, float | int | None] = {
        "origin_action": thread.origin_action,
        "origin_idea": thread.origin_idea,
        "q_origin": thread.q_origin,
        "opportunities_used": thread.opportunities_used,
        "best_fitness": thread.best_fitness,
    }
    for horizon in G_HORIZONS:
        summary[f"G{horizon}"] = thread.g_value(horizon)
    return summary


__all__ = [
    "build_opportunities",
    "coverage_tuple",
    "descendants",
    "ancestors",
    "g_summary",
    "mid_rank",
    "operator_observations",
    "pair_counts",
    "reference_candidates",
    "screen_shortlist",
    "screening_index",
    "select_by_coverage",
    "start_use_counts",
]
