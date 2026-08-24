"""Parent improvement history rendered for V9.17 generation."""

from __future__ import annotations

from .schema import Algorithm
from .tree import Tree, VIRTUAL_ROOT_ID

CHANGE_EXAMPLES_PER_SIDE = 2
CHANGE_LINE_MAX_CHARS = 520


def parent_path(tree: Tree, anchor_id: int, *, max_events: int) -> tuple[int, ...]:
    return tuple(
        item for item in tree.ancestor_ids(anchor_id) if item != VIRTUAL_ROOT_ID
    )[-max_events:]


def render_path(tree: Tree, algorithm_ids: tuple[int, ...]) -> str:
    lines = ["[Recent Algorithm Improvement History]"]
    if not algorithm_ids:
        lines.append("No history events are shown for this algorithm.")
        return "\n".join(lines)
    for index, algorithm_id in enumerate(algorithm_ids, start=1):
        item = tree.get_algorithm(algorithm_id)
        lines.extend(
            [
                "",
                f"[History {index}] Formation step",
                f"Idea: {item.idea or 'unavailable'}",
                f"Change: {_compact_change(item)}",
                f"Result: {item.result or 'evaluated'}",
                f"Fitness: {_fitness_transition(tree, item)}",
            ]
        )
    return "\n".join(lines)


def _fitness_transition(tree: Tree, item: Algorithm) -> str:
    if item.parent_id in {None, VIRTUAL_ROOT_ID}:
        return format_fitness(item.fitness)
    parent = tree.get_algorithm(item.parent_id)
    return f"{format_fitness(parent.fitness)} -> {format_fitness(item.fitness)}"


def _compact_change(item: Algorithm) -> str:
    if not item.diff:
        return "initial program" if item.parent_id == VIRTUAL_ROOT_ID else "no recorded code change"
    removed = _example_lines(_changed_code_lines(item.diff, "-"))
    added = _example_lines(_changed_code_lines(item.diff, "+"))
    summary = (
        f"+{item.added}/-{item.removed} lines; "
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
        if not code or code.startswith(("#", '\"\"\"', "'''")):
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


__all__ = ["format_fitness", "one_line", "parent_path", "render_path"]
