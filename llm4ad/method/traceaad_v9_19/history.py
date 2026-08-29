"""Render the formation trajectory shown to the generator."""

from __future__ import annotations

from .schema import MAX_HISTORY_EVENTS, Algorithm
from .tree import Tree


def _path_ids(tree: Tree, algorithm_id: int) -> tuple[int, ...]:
    return tree.formation_path(algorithm_id)[-MAX_HISTORY_EVENTS:]


def render_path(tree: Tree, algorithm_id: int) -> str:
    ids = _path_ids(tree, algorithm_id)
    lines = ["[Recent Algorithm Improvement History]"]
    if not ids:
        lines.append("No history events are shown for this algorithm.")
        return "\n".join(lines)
    for index, node_id in enumerate(ids, start=1):
        child = tree.get_algorithm(node_id)
        parent = tree.get_algorithm(child.parent_id)
        lines.extend(
            [
                "",
                f"[History {index}] Formation step",
                f"Idea: {child.idea or 'unavailable'}",
                f"Result: {_outcome(tree, parent, child)}",
                f"Fitness: {format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}",
                f"Behavior: {child.behavior_tag or 'unavailable'}",
            ]
        )
    return "\n".join(lines)


def formation_path_records(tree: Tree, algorithm_id: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for node_id in _path_ids(tree, algorithm_id):
        child = tree.get_algorithm(node_id)
        parent = tree.get_algorithm(child.parent_id)
        records.append(
            {
                "idea": child.idea,
                "result": _outcome(tree, parent, child),
                "parent_fitness": parent.fitness,
                "child_fitness": child.fitness,
                "behavior": child.behavior_tag,
            }
        )
    return records


def _outcome(tree: Tree, parent: Algorithm, child: Algorithm) -> str:
    change = tree.quality(child) - tree.quality(parent)
    if change > 0:
        return "improve"
    if change < 0:
        return "regress"
    return "plateau"


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "format_fitness",
    "formation_path_records",
    "one_line",
    "render_path",
]
