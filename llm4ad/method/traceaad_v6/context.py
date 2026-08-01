"""Render bounded TraceAAD v6 trajectories for model context."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ...base import Function
from .derivation_graph import DerivationGraph
from .prompt import fitness_direction_hint, format_fitness
from .schema import AnchorAttempt, ProgramNode, Trajectory
from .attempts import select_tested_attempts


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
    if len(trajectory.edge_ids) <= max_steps:
        selected_before = before
        selected_after = after
    else:
        selected_ids = set(
            _select_evidence_edges(
                graph,
                trajectory.edge_ids,
                limit=max_steps,
                positive_threshold=positive_threshold,
            )
        )
        selected_before = tuple(edge_id for edge_id in before if edge_id in selected_ids)
        selected_after = tuple(edge_id for edge_id in after if edge_id in selected_ids)
    sections: list[str] = []
    if selected_before:
        sections.append("[How This Program Was Reached]")
        sections.extend(_render_edges(graph, selected_before, highlight_supported=False))
    if selected_after:
        sections.append("[Later Attempts From This Program]")
        sections.extend(_render_edges(graph, selected_after, highlight_supported=False))
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
    rendered = trajectory_history(
        graph,
        trajectory,
        base_node_id=base_node_id,
        max_steps=max_steps,
        positive_threshold=positive_threshold,
    )
    if not rendered.formation_edge_ids and not rendered.tested_after_edge_ids:
        return rendered
    sections: list[str] = []
    if rendered.formation_edge_ids:
        sections.append("[How This Program Was Reached]")
        sections.extend(
            _render_edges(
                graph,
                rendered.formation_edge_ids,
                highlight_supported=True,
                positive_threshold=positive_threshold,
            )
        )
    if rendered.tested_after_edge_ids:
        sections.append("[Later Attempts From This Program]")
        sections.extend(
            _render_edges(
                graph,
                rendered.tested_after_edge_ids,
                highlight_supported=False,
                positive_threshold=positive_threshold,
            )
        )
    return RenderedHistory(
        text="\n".join(sections),
        edge_ids=rendered.edge_ids,
        formation_edge_ids=rendered.formation_edge_ids,
        tested_after_edge_ids=rendered.tested_after_edge_ids,
    )


def render_tested_attempts(
    attempts: tuple[AnchorAttempt, ...],
    *,
    excluded_edge_ids: set[int] | frozenset[int] = frozenset(),
    limit: int = 6,
) -> str:
    selected = select_tested_attempts(
        attempts, excluded_edge_ids=excluded_edge_ids, limit=limit
    )
    if not selected:
        return "No additional tested attempts from this anchor."
    lines = ["[Tested Attempts From This Anchor]"]
    for position, attempt in enumerate(selected, start=1):
        if attempt.status == "valid":
            detail = (
                f"{attempt.outcome}; fitness {format_fitness(attempt.fitness)}; "
                f"route_delta={_fmt_delta(attempt.delta_route_best)}"
            )
        else:
            detail = f"{attempt.status}; {attempt.failure_kind or 'failed'}"
        markers = []
        if attempt.new_global_best:
            markers.append("global-best update")
        if attempt.new_route_best:
            markers.append("route-best update")
        marker_text = f" ({', '.join(markers)})" if markers else ""
        lines.extend(
            [
                f"Attempt {position}: {detail}{marker_text}",
                f"  Planned: {_one_line(attempt.action, 300)}",
                f"  Implemented: {_one_line(attempt.idea or '(none)', 300)}",
            ]
        )
    return "\n".join(lines)


def build_action_prompt(
    *,
    base_node: ProgramNode,
    primary_history: RenderedHistory,
    tested_attempts_text: str,
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
        tested_attempts_text,
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
                "If a reference program is shown, each action must state which supported "
                "reference idea is used and how it is adapted to or made to interact "
                "with the primary program."
            ),
            (
                f"Return exactly {action_count} numbered single-line {action_label} and "
                "nothing else."
            ),
        ]
    )
    return "\n".join(sections).strip()


def _render_edges(
    graph: DerivationGraph,
    edge_ids: tuple[int, ...],
    *,
    highlight_supported: bool,
    positive_threshold: float = 0.0,
) -> list[str]:
    lines: list[str] = []
    for position, edge_id in enumerate(edge_ids, start=1):
        edge = graph.get_edge(edge_id)
        parent = graph.get_node(edge.parent_id)
        child = graph.get_node(edge.child_id)
        supported = ""
        if highlight_supported and (
            (
                edge.delta_route_best is not None
                and edge.delta_route_best > positive_threshold
            )
            or edge.new_global_best
        ):
            supported = " [supported transferable update]"
        lines.extend(
            [
                (
                    f"Step {position}: {edge.outcome}; fitness "
                    f"{format_fitness(parent.fitness)} -> "
                    f"{format_fitness(child.fitness)}{supported}"
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


def _select_evidence_edges(
    graph: DerivationGraph,
    edge_ids: tuple[int, ...],
    *,
    limit: int,
    positive_threshold: float,
) -> tuple[int, ...]:
    if limit <= 0:
        return ()

    def priority(item: tuple[int, int]) -> tuple[int, int]:
        position, edge_id = item
        edge = graph.get_edge(edge_id)
        supported = (
            edge.delta_route_best is not None
            and edge.delta_route_best > positive_threshold
        ) or edge.new_global_best
        boundary = edge.outcome in {"regress", "plateau", "unknown"}
        tier = 0 if supported else 1 if boundary else 2
        return tier, -position

    indexed = list(enumerate(edge_ids))
    selected = sorted(indexed, key=priority)[:limit]
    return tuple(edge_id for _, edge_id in sorted(selected))


def _fmt_delta(delta: float | None) -> str:
    return "unknown" if delta is None else f"{delta:.6g}"


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


__all__ = [
    "RenderedHistory",
    "build_action_prompt",
    "reference_history",
    "render_tested_attempts",
    "trajectory_history",
]
