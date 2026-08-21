"""Extract and render the path that formed an algorithm (unchanged from V9.14)."""

from __future__ import annotations

from .schema import MAX_HISTORY_EVENTS, Algorithm
from .tree import Tree


def parent_path(
    tree: Tree, algorithm_id: int, *, max_events: int = MAX_HISTORY_EVENTS
) -> tuple[int, ...]:
    return tree.ancestor_ids(algorithm_id)[2:][-max_events:]


def render_path(tree: Tree, algorithm_ids: tuple[int, ...]) -> str:
    lines = ["[Recent Algorithm Improvement History]"]
    if not algorithm_ids:
        lines.append("No history events are shown for this algorithm.")
        return "\n".join(lines)

    for index, algorithm_id in enumerate(algorithm_ids, start=1):
        algorithm = tree.get_algorithm(algorithm_id)
        parent = tree.get_algorithm(algorithm.parent_id)
        lines.extend(
            [
                "",
                f"[History {index}] Formation step",
                f"Idea: {algorithm.idea or 'unavailable'}",
                (
                    f"Result: {_outcome(tree, parent, algorithm)} "
                    f"(Fitness: {format_fitness(parent.fitness)} -> "
                    f"{format_fitness(algorithm.fitness)})"
                ),
            ]
        )
    return "\n".join(lines)


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


__all__ = ["format_fitness", "one_line", "parent_path", "render_path"]
