"""Periodic global reflection over recent trajectory-search experience."""

from __future__ import annotations

import json

from .derivation_graph import DerivationGraph
from .schema import ExperienceKind, GlobalExperienceEntry


def build_reflection_prompt(
    *,
    task_description: str,
    maximize: bool,
    old_entries: tuple[GlobalExperienceEntry, ...],
    recent_round_edge_ids: tuple[tuple[int, ...], ...],
    graph: DerivationGraph,
) -> str:
    old = [
        {
            "kind": entry.kind.value,
            "statement": entry.statement,
            "condition": entry.condition,
            "evidence_edges": [f"e{edge}" for edge in entry.evidence_edge_ids],
        }
        for entry in old_entries
    ]
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            "Fitness direction: "
            + ("higher is better" if maximize else "lower is better"),
            "",
            "[Previous Global Experience]",
            json.dumps(old, ensure_ascii=False),
            "",
            "[Recent Search Rounds]",
            _render_recent_rounds(graph, recent_round_edge_ids),
            "",
            "[Reflection Questions]",
            "1. What major algorithmic directions were tried in this stage?",
            (
                "2. Which insights received repeated support across trajectories, "
                "and under what conditions?"
            ),
            (
                "3. Which directions repeatedly failed, stagnated, or appeared "
                "saturated?"
            ),
            (
                "4. Which conflicts or weak signals remain worth exploring in the "
                "next stage?"
            ),
            "",
            "[Output Contract]",
            (
                "Rewrite the complete global experience state. Retain, narrow, "
                "revise, or delete previous insights according to the recent results."
            ),
            "Use only the experience shown above; do not infer unobserved results.",
            (
                "Scores and outcomes are observations. State and attempted-change "
                "descriptions are recorded model claims, not verified causal facts."
            ),
            "Return a JSON array with at most 5 short entries and no markdown.",
            (
                'Each entry: {"kind":"effective|pitfall|explore",'
                '"statement":"...","condition":"...",'
                '"evidence_edges":["e1","e2"]}.'
            ),
            (
                "effective and pitfall require at least two cited edges with distinct "
                "trajectory lineages and distinct evaluated children."
            ),
            (
                "A single event or conflicting evidence may only be kind=explore. "
                "Treat improve, regress, plateau, repair, continuation, and redirection "
                "as useful search experience."
            ),
        ]
    ).strip()


def parse_global_experience(
    response: str,
    *,
    graph: DerivationGraph,
    allowed_edge_ids: set[int],
    max_entries: int = 5,
    max_chars: int = 800,
) -> tuple[GlobalExperienceEntry, ...] | None:
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or len(payload) > max_entries:
        return None
    entries: list[GlobalExperienceEntry] = []
    for raw in payload:
        if not isinstance(raw, dict):
            return None
        try:
            kind = ExperienceKind(str(raw["kind"]))
            statement = str(raw["statement"]).strip()
            condition = str(raw.get("condition", "")).strip()
            evidence = tuple(_parse_edge_id(item) for item in raw["evidence_edges"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not statement
            or not evidence
            or any(edge_id not in allowed_edge_ids for edge_id in evidence)
        ):
            return None
        if kind in (ExperienceKind.EFFECTIVE, ExperienceKind.PITFALL):
            lineages = {graph.get_edge(edge_id).root_lineage_id for edge_id in evidence}
            hashes = {
                graph.get_node(graph.get_edge(edge_id).child_id).code_hash
                for edge_id in evidence
            }
            if len(lineages) < 2 or len(hashes) < 2:
                return None
            outcomes = {graph.get_edge(edge_id).outcome for edge_id in evidence}
            if kind == ExperienceKind.EFFECTIVE and outcomes != {"improve"}:
                return None
            if kind == ExperienceKind.PITFALL and not outcomes <= {
                "plateau",
                "regress",
            }:
                return None
        entries.append(
            GlobalExperienceEntry(
                kind=kind,
                statement=statement,
                condition=condition,
                evidence_edge_ids=evidence,
            )
        )
    rendered_chars = sum(
        len(entry.statement) + len(entry.condition) for entry in entries
    )
    return None if rendered_chars > max_chars else tuple(entries)


def support_edge_ids(
    entries: tuple[GlobalExperienceEntry, ...],
) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(
            edge_id for entry in entries for edge_id in entry.evidence_edge_ids
        )
    )


def _parse_edge_id(value) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip().lower()
    if text.startswith("e"):
        text = text[1:]
    return int(text)


def _render_recent_rounds(
    graph: DerivationGraph,
    recent_round_edge_ids: tuple[tuple[int, ...], ...],
) -> str:
    lines: list[str] = []
    for position, edge_ids in enumerate(recent_round_edge_ids, start=1):
        if not edge_ids:
            lines.append(
                f"Round {position} (search iteration no-evaluated-child):"
            )
            lines.append("- No valid child was evaluated in this round.")
            continue
        first_edge = graph.get_edge(edge_ids[0])
        anchor = graph.get_node(first_edge.parent_id)
        lines.append(
            f"Round {position} (search iteration {first_edge.iteration}; "
            f"lineage={first_edge.root_lineage_id}; "
            f"trajectory={first_edge.primary_trajectory_id}; "
            f"anchor_state={_one_line(anchor.idea, 100)}):"
        )
        for edge_id in edge_ids:
            lines.append(_render_edge_experience(graph, edge_id))
    return "\n".join(lines)


def _render_edge_experience(graph: DerivationGraph, edge_id: int) -> str:
    edge = graph.get_edge(edge_id)
    parent = graph.get_node(edge.parent_id)
    child = graph.get_node(edge.child_id)
    reference = (
        "none"
        if edge.reference_trajectory_id is None
        else (
            f"trajectory={edge.reference_trajectory_id},"
            f"program={edge.reference_program_id}"
        )
    )
    return (
        f"- e{edge.id} | operator={edge.operator.value}; reference={reference} | "
        f"tried={_one_line(edge.action, 120)} | "
        f"result={edge.outcome}; parent={_score(parent.fitness)} "
        f"child={_score(child.fitness)}; "
        f"delta_parent={_score(edge.delta_parent)} "
        f"delta_route={_score(edge.delta_route_best)} "
        f"delta_global={_score(edge.delta_global_best)}"
    )


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6g}"


__all__ = [
    "build_reflection_prompt",
    "parse_global_experience",
    "support_edge_ids",
]
