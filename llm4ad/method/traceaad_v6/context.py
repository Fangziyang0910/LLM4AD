"""Render bounded TraceAAD V6 formation histories for model context."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ...base import Function
from .derivation_graph import DerivationGraph
from .prompt import fitness_direction_hint, format_fitness
from .schema import ProgramNode, Trajectory


@dataclass(frozen=True, slots=True)
class RenderedHistory:
    text: str
    edge_ids: tuple[int, ...]
    formation_edge_ids: tuple[int, ...]
    tested_after_edge_ids: tuple[int, ...]


def trajectory_history(
    graph: DerivationGraph,
    trajectory: Trajectory,
    *,
    base_node_id: int | None = None,
    max_steps: int = 8,
    positive_threshold: float = 0.0,
) -> RenderedHistory:
    """Render recent path facts, truncating deterministically by time.

    ``positive_threshold`` remains accepted for callers that build histories
    from older configurations, but it does not affect context selection.
    """
    del positive_threshold
    if max_steps <= 0:
        return RenderedHistory(
            text="No earlier change is shown.",
            edge_ids=(),
            formation_edge_ids=(),
            tested_after_edge_ids=(),
        )
    if not trajectory.edge_ids:
        return RenderedHistory(
            text="This is an initial program with no previous changes.",
            edge_ids=(),
            formation_edge_ids=(),
            tested_after_edge_ids=(),
        )
    base_id = trajectory.endpoint_id if base_node_id is None else base_node_id
    if base_id not in trajectory.node_ids:
        raise ValueError(f"base node is outside trajectory: {base_id}")
    base_index = trajectory.node_ids.index(base_id)
    before = trajectory.edge_ids[:base_index]
    after = trajectory.edge_ids[base_index:]
    selected_ids = set(trajectory.edge_ids[-max_steps:])
    selected_before = tuple(edge_id for edge_id in before if edge_id in selected_ids)
    selected_after = tuple(edge_id for edge_id in after if edge_id in selected_ids)
    sections: list[str] = []
    if selected_before:
        sections.append("[How This Program Was Reached]")
        sections.extend(_render_edges(graph, selected_before))
    if selected_after:
        sections.append("[Later Changes From This Program]")
        sections.extend(_render_edges(graph, selected_after))
    if not sections:
        sections.append("No earlier change is shown.")
    selected = (*selected_before, *selected_after)
    return RenderedHistory(
        text="\n".join(sections),
        edge_ids=selected,
        formation_edge_ids=tuple(selected_before),
        tested_after_edge_ids=tuple(selected_after),
    )


def reference_history(
    graph: DerivationGraph,
    trajectory: Trajectory,
    *,
    base_node_id: int,
    max_steps: int = 8,
    positive_threshold: float = 0.0,
) -> RenderedHistory:
    """Use the same factual history renderer for a reference route."""
    return trajectory_history(
        graph,
        trajectory,
        base_node_id=base_node_id,
        max_steps=max_steps,
        positive_threshold=positive_threshold,
    )


def build_action_prompt(
    *,
    base_node: ProgramNode,
    primary_history: RenderedHistory,
    operator_constraint: str,
    task_description: str,
    template_function: Function,
    action_count: int,
    maximize: bool,
    reference_node: ProgramNode | None = None,
    reference_history: RenderedHistory | None = None,
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    action_label = "action" if action_count == 1 else "actions"
    sections = [
        "[Task]",
        task_description.strip(),
        fitness_direction_hint(maximize),
        "",
        "[Current Program History]",
        primary_history.text,
        "",
        "[Current Program]",
        "```python",
        base_node.code.rstrip(),
        "```",
    ]
    if reference_node is not None and reference_history is not None:
        sections.extend(
            [
                "",
                "[Reference Program History]",
                reference_history.text,
                "",
                "[Reference Program]",
                "```python",
                reference_node.code.rstrip(),
                "```",
            ]
        )
    sections.extend(
        [
            "",
            "[Improvement Suggestion]",
            operator_constraint,
            "",
            "[Target Function]",
            str(target).rstrip(),
            "",
            "[Action Guidance]",
            "The histories provide factual context for the next algorithmic change.",
            "Only executable suggestions can be evaluated by the search.",
            (
                f"You may provide up to {action_count} numbered single-line {action_label}; "
                "include only suggestions that are useful for this program."
            ),
            "Keep each suggestion within the target function's existing contract.",
            (
                "When a reference program is shown, a suggestion may explain how an "
                "idea from it could be adapted to the primary program."
            ),
        ]
    )
    return "\n".join(sections).strip()


def _render_edges(graph: DerivationGraph, edge_ids: tuple[int, ...]) -> list[str]:
    lines: list[str] = []
    for position, edge_id in enumerate(edge_ids, start=1):
        edge = graph.get_edge(edge_id)
        parent = graph.get_node(edge.parent_id)
        child = graph.get_node(edge.child_id)
        lines.extend(
            [
                (
                    f"Step {position}: {edge.outcome}; fitness "
                    f"{format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}"
                ),
                f"  Action: {_one_line(edge.action, 300)}",
                f"  Implemented Idea: {_one_line(child.idea, 300)}",
                (
                    f"  Code change: {edge.code_change_ratio:.0%}; "
                    f"LOC {parent.program_loc} -> {child.program_loc}"
                ),
            ]
        )
    return lines


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


__all__ = [
    "RenderedHistory",
    "build_action_prompt",
    "reference_history",
    "trajectory_history",
]
