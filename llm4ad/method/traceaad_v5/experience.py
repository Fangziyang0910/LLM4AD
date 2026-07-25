"""Periodic plain-text reflection over recently evaluated code changes."""

from __future__ import annotations

from .derivation_graph import DerivationGraph


def build_reflection_prompt(
    *,
    task_description: str,
    maximize: bool,
    old_experience: str,
    recent_edge_ids: tuple[int, ...],
    graph: DerivationGraph,
) -> str:
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            "Fitness direction: "
            + ("higher is better" if maximize else "lower is better"),
            "",
            "[Previous Global Reflection]",
            old_experience.strip() or "None yet.",
            "",
            "[Recent Code Changes]",
            _render_recent_changes(graph, recent_edge_ids),
            "",
            "[Reflection Request]",
            (
                "Reflect on these recent algorithm changes and their fitness results. "
                "Rewrite the global reflection as at most 5 concise points that can "
                "help decide subsequent modifications."
            ),
            (
                "Focus on useful directions, recurring problems, and promising next "
                "steps. Return only the concise reflection text."
            ),
        ]
    ).strip()


def _render_recent_changes(
    graph: DerivationGraph,
    recent_edge_ids: tuple[int, ...],
) -> str:
    sections: list[str] = []
    for position, edge_id in enumerate(recent_edge_ids, start=1):
        edge = graph.get_edge(edge_id)
        parent = graph.get_node(edge.parent_id)
        child = graph.get_node(edge.child_id)
        sections.extend(
            [
                f"Change {position}:",
                f"[Action] {_one_line(edge.action, 120)}",
                f"[Implementation Idea] {_one_line(child.idea, 120)}",
                (
                    "[Fitness Result] "
                    f"{_score(parent.fitness)} -> {_score(child.fitness)} "
                    f"({edge.outcome})"
                ),
            ]
        )
    return "\n".join(sections)


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


__all__ = ["build_reflection_prompt"]
