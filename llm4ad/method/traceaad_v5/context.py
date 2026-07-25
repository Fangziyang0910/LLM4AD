"""Bounded prompt views over full TraceAAD v5 trajectory state."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ...base import Function
from .derivation_graph import DerivationGraph
from .prompt import fitness_direction_hint, format_fitness
from .schema import GlobalExperienceEntry, Trajectory


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
            text="No previous modification exists; this is an initial program.",
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
    if after:
        formation_limit = max_steps // 2
        tested_limit = max_steps - formation_limit
        selected_before = before[-formation_limit:] if formation_limit else ()
        selected_after = after[:tested_limit]
    else:
        selected_before = before[-max_steps:]
        selected_after = ()
    sections: list[str] = []
    if selected_before:
        sections.append("[Formation History]")
        sections.extend(_render_edges(graph, trajectory, selected_before))
    if selected_after:
        sections.append("[Tested After Anchor: non-ancestor evidence]")
        sections.extend(_render_edges(graph, trajectory, selected_after))
    if not sections:
        sections.append("No displayed edge fits the current prompt window.")
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
    for edge_id in edge_ids:
        edge = graph.get_edge(edge_id)
        parent = graph.get_node(edge.parent_id)
        child = graph.get_node(edge.child_id)
        lines.extend(
            [
                (
                    f"Edge e{edge.id}: p{parent.id} -> p{child.id} "
                    f"[operator={edge.operator}]"
                ),
                f"  action: {edge.action}",
                (
                    f"  fitness: {format_fitness(parent.fitness)} -> "
                    f"{format_fitness(child.fitness)} "
                    f"(delta_parent={edge.delta_parent!s}, outcome={edge.outcome})"
                ),
                (
                    f"  LOC: {parent.program_loc} -> {child.program_loc} "
                    f"(delta={edge.delta_loc:+d})"
                ),
            ]
        )
    return lines


def render_global_experience(entries: tuple[GlobalExperienceEntry, ...]) -> str:
    if not entries:
        return ""
    labels = {
        "effective": "Effective",
        "pitfall": "Pitfall",
        "explore": "Explore",
    }
    return "\n".join(
        (
            f"- [{labels[entry.kind.value]}] {entry.statement} "
            f"Condition: {entry.condition or 'unspecified'}. "
            f"Evidence: {', '.join(f'e{edge}' for edge in entry.evidence_edge_ids)}"
        )
        for entry in entries
    )


def build_action_prompt(
    *,
    graph: DerivationGraph,
    trajectory: Trajectory,
    base_node_id: int,
    base_reason: str,
    operator_name: str,
    operator_constraint: str,
    task_description: str,
    template_function: Function,
    action_count: int,
    maximize: bool,
    max_steps: int = 8,
    global_experience: tuple[GlobalExperienceEntry, ...] = (),
    reference_trajectory: Trajectory | None = None,
    reference_node_id: int | None = None,
) -> str:
    base_node = graph.get_node(base_node_id)
    target = copy.deepcopy(template_function)
    target.body = ""
    primary = trajectory_history(
        graph, trajectory, base_node_id=base_node_id, max_steps=max_steps
    )
    sections = [
        "[Task Description]",
        task_description.strip(),
        fitness_direction_hint(maximize),
    ]
    rendered_global = render_global_experience(global_experience)
    if rendered_global:
        sections.extend(
            [
                "",
                "[Global Experience: fallible task-level evidence]",
                rendered_global,
            ]
        )
    sections.extend(
        [
            "",
            "[Primary Trajectory]",
            primary.text,
            "",
            "[Primary Anchor Program]",
            f"Node p{base_node.id}; anchor_role={base_reason}",
            f"Idea claim: {base_node.idea}",
            "```python",
            base_node.code.rstrip(),
            "```",
        ]
    )
    if reference_trajectory is not None and reference_node_id is not None:
        reference_node = graph.get_node(reference_node_id)
        reference = trajectory_history(
            graph,
            reference_trajectory,
            base_node_id=reference_node_id,
            max_steps=max_steps,
        )
        sections.extend(
            [
                "",
                "[Reference Trajectory: knowledge provenance, not a parent]",
                reference.text,
                "",
                "[Reference Program]",
                f"Node p{reference_node.id}",
                f"Idea claim: {reference_node.idea}",
                "```python",
                reference_node.code.rstrip(),
                "```",
            ]
        )
    sections.extend(
        [
            "",
            "[Operator]",
            f"name={operator_name}",
            f"Constraint: {operator_constraint}",
            "",
            "[Target Function Contract]",
            str(target).rstrip(),
            "",
            "[Action Contract]",
            (
                "Use the selected operator and all provided context to decide the next "
                "algorithmic modification."
            ),
            (
                f"Propose exactly {action_count} concrete, self-contained action lines. "
                "Each action must state what to change and how it differs from relevant "
                "attempts already shown in the histories."
            ),
            (
                "For synthesize or transfer, state which reference principle to use and "
                "how it should be adapted in the primary program."
            ),
            (
                f"Return only a numbered list of exactly {action_count} single-line "
                "actions, without JSON, code, evidence ids, or separate rationale."
            ),
        ]
    )
    return "\n".join(sections).strip()


__all__ = [
    "RenderedHistory",
    "build_action_prompt",
    "render_global_experience",
    "trajectory_history",
]
