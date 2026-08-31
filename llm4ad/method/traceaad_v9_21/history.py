"""Compact views of real formation and realization evidence."""

from __future__ import annotations

from .schema import MAX_HISTORY_EVENTS, MAX_REALIZATION_EVENTS, Hypothesis, Realization


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.8g}"


def edge_outcome(parent_fitness: float, child_fitness: float) -> str:
    if child_fitness > parent_fitness:
        return "improve"
    if child_fitness < parent_fitness:
        return "regress"
    return "plateau"


def render_formation_path(nodes: dict, node_id: int | None) -> str:
    chain = []
    current = node_id
    while current is not None and current in nodes:
        node = nodes[current]
        if node.parent_id is None or node.parent_id not in nodes:
            break
        parent = nodes[node.parent_id]
        chain.append((parent, node))
        current = parent.id
    chain.reverse()
    chain = chain[-MAX_HISTORY_EVENTS:]
    lines = ["[Real Formation Path]"]
    if not chain:
        lines.append("No earlier formation edge is available.")
        return "\n".join(lines)
    for index, (parent, child) in enumerate(chain, 1):
        lines.extend(
            [
                "",
                f"[Formation {index}] {edge_outcome(parent.fitness, child.fitness)}",
                f"Idea: {child.idea or 'unavailable'}",
                f"Fitness: {format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}",
            ]
        )
    return "\n".join(lines)


def render_ledger(hypothesis: Hypothesis, realizations: list[Realization]) -> str:
    items = [item for item in realizations if item.hypothesis_id == hypothesis.id]
    items = items[-MAX_REALIZATION_EVENTS:]
    lines = [
        f"Entry idea: {hypothesis.entry_idea}",
        f"Paired experiments: {hypothesis.trials}",
        f"Mean response to scaffold: {hypothesis.response_mean:.6g}",
        "Recent realizations:",
    ]
    if not items:
        lines.append("none")
        return "\n".join(lines)
    for item in items:
        fitness = "failure" if item.fitness is None else format_fitness(item.fitness)
        lines.append(
            f"- {item.outcome}, fitness={fitness}, response={item.response:.6g}, idea={item.idea}"
        )
    return "\n".join(lines)


def render_public_card(nodes: dict, node_id: int | None, *, max_code_chars: int = 9000) -> str | None:
    if node_id is None or node_id not in nodes:
        return None
    node = nodes[node_id]
    parent = nodes.get(node.parent_id) if node.parent_id is not None else None
    if parent is None or node.fitness <= parent.fitness:
        return None
    code = node.code
    if len(code) > max_code_chars:
        code = code[:max_code_chars] + "\n# [public card code truncated]"
    return "\n".join(
        [
            "This is one measured strict-improvement edge from another branch.",
            f"Measured fitness: {format_fitness(parent.fitness)} -> {format_fitness(node.fitness)}",
            f"Recorded idea: {node.idea or 'unavailable'}",
            "Use only a compatible mechanism; preserve the current scaffold.",
            "```python",
            code,
            "```",
        ]
    )


__all__ = ["edge_outcome", "format_fitness", "render_formation_path", "render_ledger", "render_public_card"]
