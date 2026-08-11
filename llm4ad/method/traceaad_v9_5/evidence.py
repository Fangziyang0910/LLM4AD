"""Anchor-centered correction evidence selection and minimal rendering."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .forest import SearchForest
from .schema import AttemptRecord, DirectOutcome
from .source import text_hash

OUTCOME_COVERAGE_ORDER = (
    DirectOutcome.IMPROVE,
    DirectOutcome.PLATEAU,
    DirectOutcome.REGRESS,
    DirectOutcome.INVALID,
)


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    formation_attempt_ids: tuple[int, ...]
    direct_attempt_ids: tuple[int, ...]
    direct_coverage_ids: tuple[int, ...]
    direct_pool_ids: tuple[int, ...]
    formation_pool_ids: tuple[int, ...]
    folded_attempt_ids: dict[int, tuple[int, ...]]
    removed_reasons: tuple[str, ...] = ()

    @property
    def event_count(self) -> int:
        return len(self.formation_attempt_ids) + len(self.direct_attempt_ids)


@dataclass(frozen=True, slots=True)
class RenderedEvidence:
    text: str
    excerpt_hashes: dict[int, str]
    truncated_attempt_ids: tuple[int, ...]


def select_evidence(
    forest: SearchForest, anchor_state_id: int, *, max_items: int = 8
) -> EvidenceSelection:
    direct_pool = tuple(
        sorted(
            forest.direct_attempt_ids(anchor_state_id),
            key=lambda attempt_id: forest.get_attempt(attempt_id).candidate_order,
        )
    )
    representatives, folded = _deduplicate(forest, direct_pool)

    coverage: list[int] = []
    for outcome in OUTCOME_COVERAGE_ORDER:
        matching = [
            attempt_id
            for attempt_id in representatives
            if forest.get_attempt(attempt_id).direct_outcome is outcome
        ]
        if matching and len(coverage) < max_items:
            coverage.append(matching[-1])

    selected_direct = list(coverage)
    for attempt_id in reversed(representatives):
        if len(selected_direct) >= max_items:
            break
        if attempt_id not in selected_direct:
            selected_direct.append(attempt_id)
    selected_direct.sort(key=lambda item: forest.get_attempt(item).candidate_order)

    formation_pool = forest.formation_attempt_ids(anchor_state_id)
    remaining = max_items - len(selected_direct)
    selected_formation = formation_pool[-remaining:] if remaining else ()
    return EvidenceSelection(
        formation_attempt_ids=tuple(selected_formation),
        direct_attempt_ids=tuple(selected_direct),
        direct_coverage_ids=tuple(coverage),
        direct_pool_ids=direct_pool,
        formation_pool_ids=formation_pool,
        folded_attempt_ids=folded,
    )


def remove_oldest_formation(selection: EvidenceSelection) -> EvidenceSelection:
    removed = selection.formation_attempt_ids[0]
    return replace(
        selection,
        formation_attempt_ids=selection.formation_attempt_ids[1:],
        removed_reasons=selection.removed_reasons
        + (f"formation_attempt_{removed}:context_limit",),
    )


def remove_oldest_direct_supplement(
    selection: EvidenceSelection,
) -> EvidenceSelection:
    coverage = set(selection.direct_coverage_ids)
    removed = next(
        item for item in selection.direct_attempt_ids if item not in coverage
    )
    return replace(
        selection,
        direct_attempt_ids=tuple(
            item for item in selection.direct_attempt_ids if item != removed
        ),
        removed_reasons=selection.removed_reasons
        + (f"direct_attempt_{removed}:context_limit",),
    )


def render_evidence(
    forest: SearchForest,
    selection: EvidenceSelection,
    *,
    diff_excerpt_chars: int,
) -> RenderedEvidence:
    lines = ["[Recent Formation Corrections]"]
    excerpt_hashes: dict[int, str] = {}
    truncated: list[int] = []
    if not selection.formation_attempt_ids:
        lines.append("No retained formation correction.")
    else:
        for attempt_id in selection.formation_attempt_ids:
            rendered, excerpt, was_truncated = _render_attempt(
                forest.get_attempt(attempt_id), diff_excerpt_chars
            )
            lines.extend(rendered)
            excerpt_hashes[attempt_id] = text_hash(excerpt)
            if was_truncated:
                truncated.append(attempt_id)

    lines.extend(["", "[Direct Attempts from This Exact Anchor State]"])
    if not selection.direct_attempt_ids:
        lines.append("No direct attempt has been completed from this state.")
    else:
        for attempt_id in selection.direct_attempt_ids:
            rendered, excerpt, was_truncated = _render_attempt(
                forest.get_attempt(attempt_id), diff_excerpt_chars
            )
            lines.extend(rendered)
            excerpt_hashes[attempt_id] = text_hash(excerpt)
            if was_truncated:
                truncated.append(attempt_id)
    return RenderedEvidence(
        text="\n".join(lines),
        excerpt_hashes=excerpt_hashes,
        truncated_attempt_ids=tuple(truncated),
    )


def _deduplicate(
    forest: SearchForest, attempt_ids: tuple[int, ...]
) -> tuple[tuple[int, ...], dict[int, tuple[int, ...]]]:
    groups: dict[tuple[object, ...], list[int]] = {}
    for attempt_id in attempt_ids:
        attempt = forest.get_attempt(attempt_id)
        groups.setdefault(_evidence_key(attempt), []).append(attempt_id)
    representatives: list[int] = []
    folded: dict[int, tuple[int, ...]] = {}
    for group in groups.values():
        representative = max(
            group, key=lambda item: forest.get_attempt(item).candidate_order
        )
        representatives.append(representative)
        folded[representative] = tuple(item for item in group if item != representative)
    representatives.sort(key=lambda item: forest.get_attempt(item).candidate_order)
    return tuple(representatives), folded


def _evidence_key(attempt: AttemptRecord) -> tuple[object, ...]:
    if attempt.evaluator_input_hash is not None:
        return (
            "evaluator_input",
            attempt.evaluator_input_hash,
            attempt.direct_outcome,
            attempt.failure_category,
        )
    if attempt.raw_code_hash is not None:
        return (
            "raw_code",
            attempt.raw_code_hash,
            attempt.direct_outcome,
            attempt.failure_category,
        )
    return (
        "failure",
        attempt.attempt_kind,
        attempt.failure_category,
        text_hash(attempt.failure_feedback or ""),
    )


def _render_attempt(
    attempt: AttemptRecord, diff_excerpt_chars: int
) -> tuple[list[str], str, bool]:
    idea = _one_line(attempt.declared_idea or "unavailable", 300)
    if attempt.direct_outcome is DirectOutcome.INVALID:
        failure = attempt.failure_category or "invalid"
        if attempt.failure_feedback:
            failure += f": {_one_line(attempt.failure_feedback, 360)}"
        return [f"Idea: {idea}", f"Failure: {failure}"], failure, False

    excerpt, truncated = _diff_excerpt(attempt.actual_diff, diff_excerpt_chars)
    parent = format_fitness(attempt.parent_fitness)
    child = format_fitness(attempt.child_fitness)
    outcome = (
        "unknown" if attempt.direct_outcome is None else attempt.direct_outcome.value
    )
    return (
        [
            f"Idea: {idea}",
            f"Change: {excerpt}",
            f"Result: {outcome}; fitness {parent} -> {child}",
        ],
        excerpt,
        truncated,
    )


def _diff_excerpt(diff: str | None, limit: int) -> tuple[str, bool]:
    if not diff:
        return "No executable code change.", False
    one_line = " ".join(diff.split())
    if limit <= 0:
        return "[diff truncated]", True
    if len(one_line) <= limit:
        return one_line, False
    return one_line[:limit].rstrip() + " [diff truncated]", True


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "EvidenceSelection",
    "RenderedEvidence",
    "format_fitness",
    "remove_oldest_direct_supplement",
    "remove_oldest_formation",
    "render_evidence",
    "select_evidence",
]
