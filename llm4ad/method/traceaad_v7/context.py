"""Render bounded TraceAAD V7 formation histories for model context."""

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
    carried_edge_ids: tuple[int, ...] = ()


def trajectory_history(
    graph: DerivationGraph,
    trajectory: Trajectory,
    *,
    base_node_id: int | None = None,
    max_steps: int = 8,
) -> RenderedHistory:
    """Render bounded path facts while preserving both sides of an anchor."""
    if max_steps <= 0:
        return RenderedHistory(
            text="No earlier change is shown.",
            edge_ids=(),
            formation_edge_ids=(),
            tested_after_edge_ids=(),
            carried_edge_ids=(),
        )
    if not trajectory.edge_ids:
        return RenderedHistory(
            text="This is an initial program with no previous changes.",
            edge_ids=(),
            formation_edge_ids=(),
            tested_after_edge_ids=(),
            carried_edge_ids=(),
        )
    base_id = trajectory.endpoint_id if base_node_id is None else base_node_id
    if base_id not in trajectory.node_ids:
        raise ValueError(f"base node is outside trajectory: {base_id}")
    base_index = trajectory.node_ids.index(base_id)
    before = trajectory.edge_ids[:base_index]
    after = trajectory.edge_ids[base_index:]
    carried = tuple(
        edge_id for edge_id in trajectory.evidence_edge_ids
        if edge_id not in after
    )
    before_budget = min(len(before), max_steps // 2)
    attempt_budget = max_steps - before_budget
    selected_before = tuple(before[-before_budget:]) if before_budget else ()
    attempt_ids = tuple(dict.fromkeys((*after, *carried)))
    selected_attempts: set[int] = set()
    if attempt_budget:
        # Keep at least the latest structural and carried attempt when both
        # kinds exist, then fill remaining slots by recency.
        for edge_id in (after[-1:] + carried[-1:]):
            if len(selected_attempts) >= attempt_budget:
                break
            selected_attempts.add(edge_id)
        for edge_id in reversed(attempt_ids):
            if len(selected_attempts) >= attempt_budget:
                break
            selected_attempts.add(edge_id)
    selected_after = tuple(edge_id for edge_id in after if edge_id in selected_attempts)
    selected_carried = tuple(
        edge_id for edge_id in carried if edge_id in selected_attempts
    )
    sections: list[str] = []
    position = 1
    if selected_before:
        sections.append("[How This Program Was Reached]")
        sections.extend(
            _render_edges(
                graph,
                selected_before,
                start_position=position,
            )
        )
        position += len(selected_before)
    if selected_after:
        sections.append("[Later Attempts From This Program]")
        sections.extend(
            _render_edges(
                graph,
                selected_after,
                start_position=position,
            )
        )
        position += len(selected_after)
    if selected_carried:
        sections.append("[Carried Route Evidence]")
        sections.extend(
            _render_edges(graph, selected_carried, start_position=position)
        )
    if not sections:
        sections.append("No earlier change is shown.")
    selected = (*selected_before, *selected_after, *selected_carried)
    return RenderedHistory(
        text="\n".join(sections),
        edge_ids=selected,
        formation_edge_ids=tuple(selected_before),
        tested_after_edge_ids=tuple(selected_after),
        carried_edge_ids=tuple(selected_carried),
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
            "Use the history as local evidence: do not repeat an identical tested change.",
            "Each action must be one concrete, code-level modification that can be implemented without further interpretation.",
            "Do not output analysis, comparisons, audit fields, or explanations outside the action strings.",
            (
                f"Put up to {action_count} single-line {action_label} in the "
                "schema-defined actions array; "
                "include only suggestions that are useful for this program."
            ),
            "Keep each suggestion within the target function's existing contract.",
        ]
    )
    return "\n".join(sections).strip()


def _render_edges(
    graph: DerivationGraph,
    edge_ids: tuple[int, ...],
    *,
    start_position: int,
) -> list[str]:
    lines: list[str] = []
    for position, edge_id in enumerate(edge_ids, start=start_position):
        edge = graph.get_edge(edge_id)
        parent = graph.get_node(edge.parent_id)
        child = graph.get_node(edge.child_id)
        lines.extend(
            [
                (
                    f"Step {position}: fitness "
                    f"{format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}; "
                    f"parent result={edge.outcome}; "
                    f"route gain={_format_gain(edge.delta_route_best)}; "
                    f"route best update={edge.route_best_update_reason or 'no'}; "
                    "global breakthrough="
                    f"{'yes' if edge.global_best_update_reason in {'strict_fitness', 'tie_shorter'} else 'no'}"
                ),
                f"  Requested change: {_one_line(edge.action, 360)}",
            ]
        )
    return lines


def _format_gain(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    return f"{delta:+.6g}"


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


__all__ = [
    "RenderedHistory",
    "build_action_prompt",
    "trajectory_history",
]
