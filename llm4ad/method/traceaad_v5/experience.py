"""Bounded, evidence-linked global experience for TraceAAD v5."""

from __future__ import annotations

import json

from .derivation_graph import DerivationGraph
from .schema import ExperienceKind, GlobalExperienceEntry


def build_reflection_prompt(
    *,
    task_description: str,
    maximize: bool,
    old_entries: tuple[GlobalExperienceEntry, ...],
    support_edge_ids: tuple[int, ...],
    new_edge_ids: tuple[int, ...],
    graph: DerivationGraph,
) -> str:
    relevant = tuple(dict.fromkeys((*support_edge_ids, *new_edge_ids)))
    facts = [_edge_fact(graph, edge_id) for edge_id in relevant]
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
            "[Old Global Experience]",
            json.dumps(old, ensure_ascii=False),
            "",
            "[Evidence Facts]",
            json.dumps(facts, ensure_ascii=False),
            "",
            "[Instruction]",
            "Rewrite the complete bounded global experience using only these facts.",
            "Correlation is not causation. Action/change fields are model claims.",
            "Return a JSON array with at most 6 entries and no markdown.",
            (
                'Each entry: {"kind":"effective|pitfall|explore",'
                '"statement":"...","condition":"...",'
                '"evidence_edges":["e1","e2"]}.'
            ),
            (
                "effective and pitfall require at least two cited edges with distinct "
                "root_lineage_id and distinct child_code_hash."
            ),
            "A single strict global-best event may only be kind=explore.",
        ]
    ).strip()


def parse_global_experience(
    response: str,
    *,
    graph: DerivationGraph,
    allowed_edge_ids: set[int],
    max_entries: int = 6,
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


def _edge_fact(graph: DerivationGraph, edge_id: int) -> dict:
    edge = graph.get_edge(edge_id)
    parent = graph.get_node(edge.parent_id)
    child = graph.get_node(edge.child_id)
    reference = (
        None
        if edge.reference_program_id is None
        else graph.get_node(edge.reference_program_id)
    )
    return {
        "edge_id": f"e{edge.id}",
        "root_lineage_id": edge.root_lineage_id,
        "primary_trajectory_id": edge.primary_trajectory_id,
        "anchor_role": edge.anchor_role,
        "reference_trajectory_id": edge.reference_trajectory_id,
        "reference_program_id": edge.reference_program_id,
        "reference_fitness": None if reference is None else reference.fitness,
        "reference_code_hash": None if reference is None else reference.code_hash,
        "evidence_edge_ids": [f"e{evidence}" for evidence in edge.evidence_edge_ids],
        "reference_evidence_edge_ids": [
            f"e{evidence}" for evidence in edge.reference_evidence_edge_ids
        ],
        "operator": edge.operator.value,
        "relation": edge.relation.value,
        "change_claim": edge.change,
        "novel_difference": edge.novel_difference,
        "parent_fitness": parent.fitness,
        "child_fitness": child.fitness,
        "delta_parent": edge.delta_parent,
        "delta_route_best": edge.delta_route_best,
        "delta_global_best": edge.delta_global_best,
        "global_best_update_reason": edge.global_best_update_reason,
        "parent_loc": parent.program_loc,
        "child_loc": child.program_loc,
        "parent_code_hash": parent.code_hash,
        "child_code_hash": child.code_hash,
        "outcome": edge.outcome,
    }


__all__ = [
    "build_reflection_prompt",
    "parse_global_experience",
    "support_edge_ids",
]
