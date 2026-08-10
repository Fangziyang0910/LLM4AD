"""Canonical anchor-local trajectory window for TraceAAD V9.4."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .schema import EventStatus, FailureObservation, GenerationEvent
from .tree import FactGraph


@dataclass(frozen=True, slots=True)
class CanonicalWindow:
    text: str
    formation_event_ids: tuple[int, ...]
    downstream_event_ids: tuple[int, ...]


def verified_failure_memory(
    observations: tuple[FailureObservation, ...] | list[FailureObservation],
    *,
    max_patterns: int = 5,
) -> str:
    """Render bounded, exact evaluator facts shared across this task run."""
    grouped: dict[tuple[str, str | None, str | None], list[FailureObservation]] = (
        defaultdict(list)
    )
    for observation in observations:
        grouped[
            (
                observation.failure_kind,
                observation.error_type,
                observation.error_message,
            )
        ].append(observation)
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            -item[1][-1].budget_order,
            tuple("" if part is None else part for part in item[0]),
        ),
    )[:max_patterns]
    if not ranked:
        return ""
    lines = [
        "[Verified Failure Feedback From This Run]",
        (
            "These are exact parser or evaluator outcomes from earlier candidates "
            "for this task. Apply an entry only when the same implementation risk is "
            "relevant; do not treat it as a judgment on the broader algorithmic idea."
        ),
    ]
    for (failure_kind, error_type, error_message), occurrences in ranked:
        detail = failure_kind
        if error_type:
            detail += f"; {error_type}"
        if error_message:
            detail += f": {_one_line(error_message)}"
        lines.append(f"- observed {len(occurrences)} time(s): {detail}")
    return "\n".join(lines)


def canonical_window(
    graph: FactGraph,
    anchor_id: int,
    *,
    max_events: int = 8,
    formation_quota: int = 4,
    downstream_quota: int = 4,
    downstream_depth: int = 3,
) -> CanonicalWindow:
    """Build the unique window without fitness-based event selection."""
    formation = list(graph.formation_event_ids(anchor_id))
    downstream = list(graph.downstream_events(anchor_id, max_depth=downstream_depth))

    formation_limit = min(formation_quota, len(formation))
    downstream_limit = min(downstream_quota, len(downstream))
    if formation_limit < formation_quota:
        downstream_limit = min(len(downstream), max_events - formation_limit)
    elif downstream_limit < downstream_quota:
        formation_limit = min(len(formation), max_events - downstream_limit)

    selected_formation = tuple(formation[-formation_limit:]) if formation_limit else ()
    selected_downstream = (
        tuple(downstream[-downstream_limit:]) if downstream_limit else ()
    )
    lines = [
        "[Local Trajectory Evidence]",
        (
            f"This is a bounded local view, not the complete search history. It "
            f"contains at most {max_events} recent Idea-result events, prioritizing "
            f"up to {formation_quota} formation steps and {downstream_quota} nearby "
            f"tests within {downstream_depth} descendant steps; unused slots are "
            "filled from the other section. Nearby tests may come from different "
            "branches. Events within each section are ordered from earlier to later."
        ),
        "",
        "[How This Program Was Formed]",
    ]
    if not selected_formation:
        lines.append("This anchor is an initial program with no formation events.")
    else:
        for position, event_id in enumerate(selected_formation, start=1):
            event = graph.get_event(event_id)
            lines.extend(_render_formation_event(graph, event, position))

    lines.append("")
    lines.append("[Tested Near This Anchor]")
    if not selected_downstream:
        lines.append("No local continuation has been tested from this anchor yet.")
    else:
        branch_ids = tuple(
            dict.fromkeys(
                branch_id
                for _, _, branch_id in selected_downstream
                if branch_id is not None
            )
        )
        branch_labels = {
            branch_id: _branch_label(position)
            for position, branch_id in enumerate(branch_ids)
        }
        for position, (event, depth, branch_id) in enumerate(
            selected_downstream, start=1
        ):
            lines.extend(
                _render_downstream_event(
                    graph,
                    event,
                    position,
                    depth,
                    None if branch_id is None else branch_labels[branch_id],
                )
            )

    return CanonicalWindow(
        text="\n".join(lines),
        formation_event_ids=selected_formation,
        downstream_event_ids=tuple(item[0].id for item in selected_downstream),
    )


def _render_formation_event(
    graph: FactGraph, event: GenerationEvent, position: int
) -> list[str]:
    parent = graph.get_node(event.anchor_id)
    child = graph.get_node(event.child_id)  # type: ignore[arg-type]
    return [
        (
            f"Formation step {position}: {event.outcome}; "
            f"fitness {format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}"
        ),
        f"  Idea implemented: {_one_line(event.idea)}",
    ]


def _render_downstream_event(
    graph: FactGraph,
    event: GenerationEvent,
    position: int,
    depth: int,
    branch_label: str | None,
) -> list[str]:
    parent = graph.get_node(event.anchor_id)
    location = (
        "direct attempt from anchor"
        if branch_label is None
        else f"result depth {depth}, branch {branch_label}"
    )
    header = (
        f"Local test {position} [{location}]: {event.outcome}; "
        f"parent fitness {format_fitness(parent.fitness)}"
    )
    if event.status is EventStatus.INVALID:
        failure = event.failure_kind or "unknown"
        if event.error_type:
            failure = f"{failure}; {event.error_type}"
        lines = [
            f"{header}; result invalid ({failure})",
            f"  Attempted idea: {_one_line(event.idea or 'unavailable')}",
        ]
        if event.error_message:
            lines.append(f"  Evaluator feedback: {_one_line(event.error_message)}")
        return lines
    child = graph.get_node(event.child_id)  # type: ignore[arg-type]
    return [
        f"{header}; result fitness {format_fitness(child.fitness)}",
        f"  Idea implemented: {_one_line(event.idea)}",
    ]


def _branch_label(position: int) -> str:
    label = ""
    current = position
    while True:
        current, remainder = divmod(current, 26)
        label = chr(ord("A") + remainder) + label
        if current == 0:
            return label
        current -= 1


def format_fitness(fitness: float) -> str:
    return f"{fitness:.6g}"


def _one_line(text: str, limit: int = 360) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "CanonicalWindow",
    "canonical_window",
    "format_fitness",
    "verified_failure_memory",
]
