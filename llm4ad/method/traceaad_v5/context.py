"""Render bounded TraceAAD v5 trajectories for model context."""

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
) -> RenderedHistory:
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
    if len(trajectory.edge_ids) <= max_steps:
        selected_before = before
        selected_after = after
    elif after:
        formation_limit = max_steps // 2
        tested_limit = max_steps - formation_limit
        selected_before = before[-formation_limit:] if formation_limit else ()
        selected_after = after[:tested_limit]
    else:
        selected_before = before[-max_steps:]
        selected_after = ()
    sections: list[str] = []
    if selected_before:
        sections.append("[How This Program Was Reached]")
        sections.extend(_render_edges(graph, trajectory, selected_before))
    if selected_after:
        sections.append("[Later Attempts From This Program]")
        sections.extend(_render_edges(graph, trajectory, selected_after))
    if not sections:
        sections.append("No earlier change is shown.")
    selected = (*selected_before, *selected_after)
    return RenderedHistory(
        text="\n".join(sections),
        edge_ids=selected,
        formation_edge_ids=tuple(selected_before),
        tested_after_edge_ids=tuple(selected_after),
    )


def _render_edges(
    graph: DerivationGraph,
    trajectory: Trajectory,
    edge_ids: tuple[int, ...],
) -> list[str]:
    lines: list[str] = []
    for position, edge_id in enumerate(edge_ids, start=1):
        edge = graph.get_edge(edge_id)
        parent = graph.get_node(edge.parent_id)
        child = graph.get_node(edge.child_id)
        lines.extend(
            [
                (
                    f"Step {position}: {edge.outcome}; fitness "
                    f"{format_fitness(parent.fitness)} -> "
                    f"{format_fitness(child.fitness)}"
                ),
                f"  Planned: {_one_line(edge.action, 300)}",
                f"  Implemented: {_one_line(child.idea, 300)}",
                (
                    f"  Code change: {edge.code_change_ratio:.0%}; "
                    f"LOC {parent.program_loc} -> {child.program_loc}"
                ),
            ]
        )
    return lines


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
    ]
    sections.extend(
        [
            "",
            "[Current Program History]",
            primary_history.text,
            "",
            "[Current Program]",
            "```python",
            base_node.code.rstrip(),
            "```",
        ]
    )
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
            "[Improvement Direction]",
            operator_constraint,
            "",
            "[Target Function]",
            str(target).rstrip(),
            "",
            "[Action Contract]",
            (
                "Use the improvement direction and the provided trajectory histories "
                "to decide the next algorithmic modification."
            ),
            (
                "Ground each proposal in the task structure and avoid constants "
                "justified only by the observed training size."
            ),
            (
                f"Propose exactly {action_count} concrete, self-contained action lines. "
                "Each action must state what to change and how it differs from relevant "
                "attempts already shown in the histories."
            ),
            (
                "Each change must be implementable inside the target function using "
                "only its arguments and locally computed values. Do not assume hidden "
                "state or change the function signature."
            ),
            (
                "Each action must require a substantive change to executable behavior "
                "or algorithmic structure; do not restate the current program."
            ),
            (
                "If a reference program is shown, each action must state what is "
                "learned from it and how that knowledge is adapted to or made to "
                "interact with the primary program, according to the improvement "
                "direction."
            ),
            (
                f"Return exactly {action_count} numbered single-line {action_label} and "
                "nothing else."
            ),
        ]
    )
    return "\n".join(sections).strip()


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


__all__ = [
    "RenderedHistory",
    "build_action_prompt",
    "trajectory_history",
]
