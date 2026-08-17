"""Formation path rendering for TraceAAD V9.10."""

from __future__ import annotations

from .forest import Forest
from .schema import MAX_HISTORY_EVENTS, Action, Intent, Outcome

CHANGE_EXAMPLES_PER_SIDE = 2
CHANGE_LINE_MAX_CHARS = 520


def parent_path(
    forest: Forest, anchor_id: int, *, max_events: int = MAX_HISTORY_EVENTS
) -> tuple[int, ...]:
    return forest.parent_path_ids(anchor_id)[-max_events:]


def drop_oldest(action_ids: tuple[int, ...]) -> tuple[int, ...]:
    return action_ids[1:]


def render_path(forest: Forest, action_ids: tuple[int, ...]) -> str:
    lines = ["[Recent Algorithm Formation Path]"]
    if not action_ids:
        lines.append("No formation events are shown for this algorithm.")
        return "\n".join(lines)
    for index, action_id in enumerate(action_ids, start=1):
        lines.extend(("", f"[History {index}] Formation step"))
        lines.extend(_render_event(forest.get_action(action_id)))
    return "\n".join(lines)


def _render_event(action: Action) -> list[str]:
    assert action.outcome is not None and action.outcome is not Outcome.INVALID
    intent = Intent(action.intent)
    return [
        f"Operator: {intent.value.capitalize()}",
        f"Idea: {action.idea or 'unavailable'}",
        f"Change: {_compact_change(action)}",
        f"Result: {action.outcome.value}",
        (
            "Fitness: "
            f"{format_fitness(action.parent_fitness)} -> "
            f"{format_fitness(action.child_fitness)}"
        ),
    ]


def _compact_change(action: Action) -> str:
    if not action.diff:
        return "no recorded code change"
    removed = _example_lines(_changed_code_lines(action.diff, "-"))
    added = _example_lines(_changed_code_lines(action.diff, "+"))
    summary = (
        f"+{action.added}/-{action.removed} lines; "
        f"removed: {' | '.join(f'`{line}`' for line in removed) or 'none'}; "
        f"added: {' | '.join(f'`{line}`' for line in added) or 'none'}"
    )
    return one_line(summary, CHANGE_LINE_MAX_CHARS)


def _changed_code_lines(diff: str, prefix: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith(prefix) or line.startswith(prefix * 3):
            continue
        code = line[1:].strip()
        if not code or code.startswith(("#", '"""', "'''")):
            continue
        normalized = one_line(code, 150)
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


def one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "CHANGE_EXAMPLES_PER_SIDE",
    "CHANGE_LINE_MAX_CHARS",
    "drop_oldest",
    "format_fitness",
    "one_line",
    "parent_path",
    "render_path",
]
