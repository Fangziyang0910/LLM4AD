"""Parent improvement path as the default generation history.

Selection rule: show the most recent formation steps on the unique parent
chain from the route root to the current anchor, at most eight events.
Direct attempts from the current state remain search facts but are never
selected into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .forest import SearchForest
from .schema import AttemptRecord, DirectOutcome

MAX_HISTORY_EVENTS = 8
CHANGE_EXAMPLES_PER_SIDE = 2
CHANGE_LINE_MAX_CHARS = 520
IDEA_LINE_MAX_CHARS = 300


@dataclass(frozen=True, slots=True)
class HistorySelection:
    event_ids: tuple[int, ...]
    formation_event_ids: tuple[int, ...]
    formation_pool_ids: tuple[int, ...]


def select_history(
    forest: SearchForest, anchor_state_id: int, *, max_events: int = MAX_HISTORY_EVENTS
) -> HistorySelection:
    formation_pool = forest.formation_attempt_ids(anchor_state_id)
    selected_formation = formation_pool[-max_events:]
    return HistorySelection(
        event_ids=selected_formation,
        formation_event_ids=selected_formation,
        formation_pool_ids=formation_pool,
    )


def drop_oldest_event(selection: HistorySelection) -> HistorySelection:
    dropped = selection.event_ids[0]
    return replace(
        selection,
        event_ids=selection.event_ids[1:],
        formation_event_ids=tuple(
            item for item in selection.formation_event_ids if item != dropped
        ),
    )


def render_history(forest: SearchForest, selection: HistorySelection) -> str:
    lines = ["[Recent Algorithm Improvement History]"]
    if not selection.event_ids:
        lines.append("No history events are shown for this algorithm.")
        return "\n".join(lines)
    for index, attempt_id in enumerate(selection.event_ids, start=1):
        lines.append("")
        lines.append(f"[History {index}] Formation step")
        lines.extend(_render_event(forest.get_attempt(attempt_id)))
    return "\n".join(lines)


def _render_event(attempt: AttemptRecord) -> list[str]:
    # Formation steps created a child state, so invalid never renders.
    assert attempt.direct_outcome is not None
    assert attempt.direct_outcome is not DirectOutcome.INVALID
    return [
        f"Idea: {_one_line(attempt.declared_idea or 'unavailable', IDEA_LINE_MAX_CHARS)}",
        f"Change: {_compact_change(attempt)}",
        f"Result: {attempt.direct_outcome.value}",
        (
            "Fitness: "
            f"{format_fitness(attempt.parent_fitness)} -> "
            f"{format_fitness(attempt.child_fitness)}"
        ),
    ]


def _compact_change(attempt: AttemptRecord) -> str:
    if not attempt.actual_diff:
        return "no recorded code change"
    statistics = attempt.diff_statistics
    added_count = 0 if statistics is None else statistics.added_lines
    removed_count = 0 if statistics is None else statistics.removed_lines
    removed = _example_lines(_changed_code_lines(attempt.actual_diff, "-"))
    added = _example_lines(_changed_code_lines(attempt.actual_diff, "+"))
    summary = (
        f"+{added_count}/-{removed_count} lines; "
        f"removed: {' | '.join(f'`{line}`' for line in removed) or 'none'}; "
        f"added: {' | '.join(f'`{line}`' for line in added) or 'none'}"
    )
    return _one_line(summary, CHANGE_LINE_MAX_CHARS)


def _changed_code_lines(actual_diff: str, prefix: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for line in actual_diff.splitlines():
        if not line.startswith(prefix) or line.startswith(prefix * 3):
            continue
        code = line[1:].strip()
        if not code or code.startswith(("#", '"""', "'''")):
            continue
        normalized = _one_line(code, 150)
        if normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _example_lines(lines: list[str]) -> list[str]:
    if len(lines) <= CHANGE_EXAMPLES_PER_SIDE:
        return lines
    return [lines[0], lines[-1]]


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "CHANGE_EXAMPLES_PER_SIDE",
    "CHANGE_LINE_MAX_CHARS",
    "MAX_HISTORY_EVENTS",
    "HistorySelection",
    "drop_oldest_event",
    "format_fitness",
    "render_history",
    "select_history",
]
