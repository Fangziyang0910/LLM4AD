"""The V10 valuation critic: relative judgement over the opportunity set.

The critic is called once before every primary evaluation.  It never writes
code and never outputs an uncalibrated numeric Q.  It returns at most
``K_c`` opportunities that may still be the best use of the next slot; when
its output cannot be parsed and validated, the caller falls back to the
highest-quality Develop opportunities on the shortlist.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .opportunity import g_summary, mid_rank, operator_observations, pair_counts
from .schema import (
    COMPETITIVE_SET_SIZE,
    FORMATION_WINDOW,
    OPERATORS,
    Opportunity,
    ProgramNode,
    Thread,
    AttemptRecord,
    CompetitiveEntry,
    CriticResult,
)

CODE_MAX_CHARS = 6000
REFERENCE_CODE_MAX_CHARS = 4000
IDEA_MAX_CHARS = 300
HORIZONS = ("short", "medium", "long")

# The critic prompt must stay inside the serving context window with room for
# the model's reply.  Token counts are estimated from characters; 3 chars per
# token is deliberately conservative so the estimate never undercounts.
CHARS_PER_TOKEN = 3.0
CONTEXT_SAFETY_TOKENS = 2000
MIN_TOTAL_CHAR_BUDGET = 20000
MIN_CODE_CHARS = 600


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.8g}"


def edge_outcome(parent_fitness: float, child_fitness: float) -> str:
    if child_fitness > parent_fitness:
        return "improve"
    if child_fitness < parent_fitness:
        return "regress"
    return "plateau"


@dataclass(frozen=True, slots=True)
class CriticPrompt:
    prompt: str
    valid_labels: frozenset[str]
    char_budget: int
    dropped_reference_codes: tuple[int, ...] = ()

    @property
    def clipped(self) -> bool:
        return bool(self.dropped_reference_codes)


def critic_char_budget(context_token_limit: int, max_tokens: int) -> int:
    """Conservative character budget for the critic prompt.

    The serving window must hold the prompt plus the model's reply; 3 chars
    per token keeps the estimate on the safe side.
    """
    usable = context_token_limit - max_tokens - CONTEXT_SAFETY_TOKENS
    return max(MIN_TOTAL_CHAR_BUDGET, int(usable * CHARS_PER_TOKEN))


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n# [code truncated for the critic]"


def _formation_lines(
    nodes: dict[int, ProgramNode], node_id: int, label_prefix: str
) -> list[str]:
    chain = []
    current = node_id
    while current is not None and current in nodes:
        node = nodes[current]
        if node.parent_id is None or node.parent_id not in nodes:
            break
        parent = nodes[node.parent_id]
        chain.append((parent, node))
        current = parent.id
    chain.reverse()
    chain = chain[-FORMATION_WINDOW:]
    if not chain:
        return [f"[{label_prefix}-H0] no earlier formation edge is available"]
    lines = []
    for index, (parent, child) in enumerate(chain, 1):
        idea = _clip(child.idea or "unavailable", IDEA_MAX_CHARS)
        lines.append(
            f"[{label_prefix}-H{index}] {edge_outcome(parent.fitness, child.fitness)} | "
            f"Idea: {idea} | fitness {format_fitness(parent.fitness)} -> "
            f"{format_fitness(child.fitness)}"
        )
    return lines


def _ledger_lines(
    node: ProgramNode,
    attempts: Sequence[AttemptRecord],
    pairs: dict[tuple[int, int], int],
    references: dict[int, list[int]] | None,
    label_prefix: str,
) -> list[str]:
    lines = []
    for operator in OPERATORS:
        records = [
            record
            for record in attempts
            if record.start_id == node.id and record.operator == operator
        ]
        label = f"[{label_prefix}-L{operator}]"
        if not records:
            lines.append(f"{label} total 0")
            continue
        parts = [f"{label} total {len(records)}"]
        if operator == "transfer" and references:
            pair_text = ", ".join(
                f"S{reference} x{pairs.get((node.id, reference), 0)}"
                for reference in references.get(node.id, [])
            )
            if pair_text:
                parts.append(f"pairs: {pair_text}")
        latest = records[-1]
        child = "no valid child" if latest.child_id is None else f"node S{latest.child_id}"
        parts.append(
            f"most recent: slot {latest.slot}, outcome {latest.outcome}, "
            f"{format_fitness(latest.start_fitness)} -> "
            f"{format_fitness(latest.child_fitness)} ({child})"
        )
        lines.append("; ".join(parts))
    return lines


def _operator_observation_lines(observations: dict) -> list[str]:
    lines = []
    for operator in OPERATORS:
        payload = observations.get(operator, {})
        fields = []
        for key, value in payload.items():
            if value is None:
                rendered = "unobserved"
            elif isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = str(value)
            fields.append(f"{key}={rendered}")
        lines.append(f"[OP-{operator}] " + ", ".join(fields))
    return lines


def build_critic_prompt(
    *,
    task_description: str,
    slot: int,
    remaining_budget: int,
    primary_evaluations: int,
    best_node: ProgramNode | None,
    nodes: dict[int, ProgramNode],
    threads: dict[int, Thread],
    attempts: Sequence[AttemptRecord],
    shortlist: Sequence[int],
    opportunities: Sequence[Opportunity],
    references: dict[int, list[int]],
    char_budget: int | None = None,
) -> CriticPrompt:
    """Assemble the critic view; the archive's allocation side only.

    Code is the only unbounded part.  It is fitted to ``char_budget`` in two
    stages: proportional clipping of every shown program, then dropping
    reference code (least valuable first) while start code keeps a floor.
    """
    if char_budget is None:
        char_budget = critic_char_budget(32768, 8192)
    code_limits = _fit_code_limits(nodes, shortlist, references, char_budget, _skeleton(
        task_description=task_description,
        slot=slot,
        remaining_budget=remaining_budget,
        primary_evaluations=primary_evaluations,
        best_node=best_node,
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=shortlist,
        opportunities=opportunities,
        references=references,
        code_bodies={},
    )[0])
    sections, labels, dropped = _skeleton(
        task_description=task_description,
        slot=slot,
        remaining_budget=remaining_budget,
        primary_evaluations=primary_evaluations,
        best_node=best_node,
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=shortlist,
        opportunities=opportunities,
        references=references,
        code_bodies=code_limits,
    )
    prompt = "\n".join(sections)
    return CriticPrompt(
        prompt=prompt,
        valid_labels=frozenset(labels),
        char_budget=char_budget,
        dropped_reference_codes=tuple(dict.fromkeys(dropped)),
    )


def _skeleton(
    *,
    task_description: str,
    slot: int,
    remaining_budget: int,
    primary_evaluations: int,
    best_node: ProgramNode | None,
    nodes: dict[int, ProgramNode],
    threads: dict[int, Thread],
    attempts: Sequence[AttemptRecord],
    shortlist: Sequence[int],
    opportunities: Sequence[Opportunity],
    references: dict[int, list[int]],
    code_bodies: dict[int, int | None],
) -> tuple[list[str], set[str], tuple[int, ...]]:
    """Render the critic prompt; ``code_bodies`` maps node id to clip limit.

    A limit of ``None`` renders the omission notice for that reference.  The
    first rendering pass is called with an empty map to measure the size of
    everything except code.
    """
    values = [node.fitness for node in nodes.values()]
    observations = operator_observations(attempts, threads)
    pairs = pair_counts(attempts)
    labels: set[str] = {"GLOBAL"}

    sections = [
        "[Task]",
        task_description.strip(),
        "Fitness is the task score; higher is better.",
        "",
        "[Search State]",
        f"[GLOBAL] Primary evaluations used: {primary_evaluations}. "
        f"Remaining budget: {remaining_budget}.",
        f"Next primary slot: {slot}. Archive nodes: {len(nodes)}.",
    ]
    if best_node is not None:
        sections.append(
            f"[GLOBAL] Global best fitness: {format_fitness(best_node.fitness)} "
            f"(node S{best_node.id})."
        )
    else:
        sections.append("[GLOBAL] Global best fitness: unavailable.")
    sections.append("")
    sections.append("[Operator Observations]")
    for line in _operator_observation_lines(observations):
        sections.append(line)
        labels.add(line.split("]")[0].lstrip("["))
    sections.append("")

    printed_code: set[int] = set()
    sections.append("[Screened Start States]")
    for start_id in shortlist:
        node = nodes[start_id]
        label = f"S{node.id}"
        labels.add(label)
        parent = "none" if node.parent_id is None else f"S{node.parent_id}"
        rank = mid_rank(node.fitness, values)
        sections.append(
            f"--- {label} (q={format_fitness(node.fitness)}, mid-rank={rank:.4f}, "
            f"created at slot {node.slot}, parent={parent}, thread T{node.thread_id}) ---"
        )
        thread = threads[node.thread_id]
        g_lines = []
        for key, value in g_summary(thread).items():
            if value is None:
                rendered = "unobserved" if key.startswith("G") else "none"
            elif isinstance(value, float):
                rendered = f"{value:.6g}"
            else:
                rendered = _clip(str(value), IDEA_MAX_CHARS)
            g_lines.append(f"{key}={rendered}")
        labels.add(f"{label}-G")
        sections.append(f"Thread T{thread.id} [{label}-G]: " + ", ".join(g_lines))
        sections.append(f"Code {label}:")
        sections.append("```python")
        sections.append(_clip(node.code, code_bodies.get(node.id, 0)))
        sections.append("```")
        sections.append(f"Formation path {label}:")
        formation = _formation_lines(nodes, node.id, label)
        sections.extend(formation)
        for line in formation:
            labels.add(line.split("]")[0].lstrip("["))
        sections.append(f"Ledger {label}:")
        ledger = _ledger_lines(node, attempts, pairs, references, label)
        sections.extend(ledger)
        for line in ledger:
            labels.add(line.split("]")[0].lstrip("["))
        printed_code.add(node.id)
        sections.append("")

    dropped: list[int] = []
    sections.append("[Transfer References]")
    for start_id in shortlist:
        for reference_id in references.get(start_id, []):
            reference = nodes[reference_id]
            label = f"S{start_id}-R{reference_id}"
            labels.add(label)
            rank = mid_rank(reference.fitness, values)
            thread = threads[reference.thread_id]
            sections.append(
                f"--- [{label}] reference for transfers from S{start_id}: node S{reference_id} ---"
            )
            sections.append(
                f"q={format_fitness(reference.fitness)}, mid-rank={rank:.4f}, "
                f"thread T{thread.id} (origin_action={thread.origin_action})."
            )
            sections.append(
                f"Idea summary: {_clip(reference.idea or 'unavailable', IDEA_MAX_CHARS)}; "
                f"thread origin idea: {_clip(thread.origin_idea, IDEA_MAX_CHARS)}."
            )
            if reference_id in printed_code:
                sections.append(f"Code S{reference_id}: already printed above.")
            else:
                limit = code_bodies.get(reference_id)
                if limit is None:
                    dropped.append(reference_id)
                    sections.append(
                        f"Code S{reference_id}: omitted to keep the valuation prompt "
                        "within the context window; rely on the idea summary above."
                    )
                else:
                    printed_code.add(reference_id)
                    sections.append(f"Code S{reference_id}:")
                    sections.append("```python")
                    sections.append(_clip(reference.code, limit))
                    sections.append("```")
    sections.append("")

    sections.append("[Opportunities]")
    for opportunity in opportunities:
        if opportunity.operator == "restart":
            description = "restart: propose a new hypothesis, no parent; the reference quality is the global best"
        elif opportunity.operator == "transfer":
            description = (
                f"transfer: S{opportunity.start_id} <- S{opportunity.reference_id} "
                "(migrate one evaluated mechanism from the reference)"
            )
        else:
            description = f"{opportunity.operator}: on S{opportunity.start_id}"
        sections.append(f"{opportunity.opportunity_id}: {description}")
    sections.append("")

    sections.extend(
        [
            "[Instruction]",
            "You are the valuation critic for the next primary evaluation slot.",
            "Judge which experiments may still be the best use of this slot, using",
            "immediate payoff, follow-up development potential, information value,",
            "realization delay, and the remaining budget. Information value means:",
            "an evaluation that would resolve uncertainty and thereby materially",
            "change later allocation can enter even when its immediate payoff is",
            "uncertain. A single step back or the current quality alone never",
            "decides inclusion.",
            "Return the plausibly optimal set: at most "
            f"{COMPETITIVE_SET_SIZE} opportunities, ranked, that remain worth buying now.",
            "A semantic_repair opportunity may enter only together with a concrete",
            "semantic_mismatch between the intended idea and the implementation,",
            "evidenced in that start's code, formation path, or ledger; otherwise move",
            "it to not_applicable. Cite evidence only with the labels shown above",
            "(e.g. S17-H3, S17-Lpivot, S17-G, OP-pivot, GLOBAL).",
            "Return exactly one JSON object and nothing else:",
            '{"competitive_set": ['
            '{"opportunity_id": "O17", "rank": 1, "reason": "...", '
            '"evidence_refs": ["S17-H3"], "expected_payoff_horizon": "short", '
            '"semantic_mismatch": null}'
            "], "
            '"not_applicable": [{"opportunity_id": "O11", "reason": "..."}]}',
        ]
    )
    return sections, labels, tuple(dropped)


def _fit_code_limits(
    nodes: dict[int, ProgramNode],
    shortlist: Sequence[int],
    references: dict[int, list[int]],
    char_budget: int,
    skeleton_sections: list[str],
) -> dict[int, int | None]:
    """Fit every shown code body inside the remaining character budget.

    Stage one scales the per-code clip limits proportionally with a floor;
    stage two drops reference code, least valuable first, while start code
    keeps its floor.  A limit of None means the reference code is omitted.
    """
    base = len("\n".join(skeleton_sections))
    available = char_budget - base
    shown: list[tuple[int, int, str, float]] = []  # (node id, size, kind, fitness)
    for start_id in shortlist:
        node = nodes[start_id]
        shown.append((node.id, min(len(node.code), CODE_MAX_CHARS), "start", node.fitness))
    printed = set(shortlist)
    for start_id in shortlist:
        for reference_id in references.get(start_id, []):
            if reference_id in printed:
                continue
            printed.add(reference_id)
            node = nodes[reference_id]
            shown.append(
                (node.id, min(len(node.code), REFERENCE_CODE_MAX_CHARS), "reference", node.fitness)
            )
    if not shown:
        return {}

    if available <= 0:
        limits: dict[int, int | None] = {
            node_id: (MIN_CODE_CHARS if kind == "start" else None)
            for node_id, _, kind, _ in shown
        }
    else:
        total = sum(size for _, size, _, _ in shown)
        if total <= available:
            return {
                node_id: (CODE_MAX_CHARS if kind == "start" else REFERENCE_CODE_MAX_CHARS)
                for node_id, _, kind, _ in shown
            }
        scale = available / total
        limits = {
            node_id: max(MIN_CODE_CHARS, int(size * scale))
            for node_id, size, _, _ in shown
        }

    used = sum(limit for limit in limits.values() if limit is not None)
    if used > available:
        # Least valuable donors first: ascending fitness, ties by larger id.
        droppable = sorted(
            (fitness, -node_id)
            for node_id, _, kind, fitness in shown
            if kind == "reference" and limits.get(node_id) is not None
        )
        for _, negative_id in droppable:
            if used <= available:
                break
            node_id = -negative_id
            used -= int(limits[node_id])
            limits[node_id] = None
    used = sum(limit for limit in limits.values() if limit is not None)
    if used > available and available > 0:
        start_total = sum(
            limits[node_id] for node_id, _, kind, _ in shown if kind == "start"
        )
        if start_total > 0:
            factor = max(0.05, available / start_total)
            for node_id, _, kind, _ in shown:
                if kind == "start":
                    limits[node_id] = max(200, int(limits[node_id] * factor))
    return limits


def parse_critic_response(
    response: str,
    opportunities: Sequence[Opportunity],
    valid_labels: frozenset[str],
) -> CriticResult | None:
    """Parse and validate one critic response; None when unusable."""
    payload = _extract_json(response)
    if not isinstance(payload, dict):
        return None
    raw_competitive = payload.get("competitive_set")
    if not isinstance(raw_competitive, list) or not raw_competitive:
        return None
    if len(raw_competitive) > COMPETITIVE_SET_SIZE:
        return None
    by_id = {opportunity.opportunity_id: opportunity for opportunity in opportunities}
    seen_ranks: set[int] = set()
    entries: list[CompetitiveEntry] = []
    for item in raw_competitive:
        if not isinstance(item, dict):
            return None
        opportunity = by_id.get(str(item.get("opportunity_id")))
        if opportunity is None:
            return None
        rank = item.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            return None
        if rank in seen_ranks:
            return None
        seen_ranks.add(rank)
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            return None
        raw_refs = item.get("evidence_refs", [])
        if raw_refs is None:
            raw_refs = []
        if not isinstance(raw_refs, list) or any(
            not isinstance(ref, str) or ref not in valid_labels for ref in raw_refs
        ):
            return None
        horizon = item.get("expected_payoff_horizon")
        if horizon is not None and str(horizon).lower() not in HORIZONS:
            return None
        mismatch = item.get("semantic_mismatch")
        if opportunity.operator == "semantic_repair":
            if not isinstance(mismatch, str) or not mismatch.strip():
                return None
        elif mismatch is not None and not isinstance(mismatch, str):
            return None
        entries.append(
            CompetitiveEntry(
                opportunity=opportunity,
                rank=rank,
                reason=reason.strip(),
                evidence_refs=tuple(raw_refs),
                payoff_horizon=None if horizon is None else str(horizon).lower(),
                semantic_mismatch=mismatch if isinstance(mismatch, str) else None,
            )
        )
    entries.sort(key=lambda entry: entry.rank)

    raw_not_applicable = payload.get("not_applicable", [])
    if raw_not_applicable is None:
        raw_not_applicable = []
    if not isinstance(raw_not_applicable, list):
        return None
    not_applicable: list[tuple[str, str]] = []
    competitive_ids = {entry.opportunity.opportunity_id for entry in entries}
    for item in raw_not_applicable:
        if not isinstance(item, dict):
            return None
        opportunity = by_id.get(str(item.get("opportunity_id")))
        if opportunity is None:
            return None
        if opportunity.opportunity_id in competitive_ids:
            return None
        reason = item.get("reason", "")
        if not isinstance(reason, str):
            return None
        not_applicable.append((opportunity.opportunity_id, reason.strip()))
    return CriticResult(
        entries=tuple(entries),
        not_applicable=tuple(not_applicable),
        invalid=False,
    )


def fallback_result(
    nodes: dict[int, ProgramNode],
    shortlist: Sequence[int],
    opportunities: Sequence[Opportunity],
) -> CriticResult:
    """Conservative fallback: highest-quality Develop on the shortlist."""
    develop = {
        opportunity.start_id: opportunity
        for opportunity in opportunities
        if opportunity.operator == "develop"
    }
    ranked = sorted(shortlist, key=lambda node_id: (-nodes[node_id].fitness, node_id))
    entries = []
    for rank, node_id in enumerate(ranked[:COMPETITIVE_SET_SIZE], 1):
        entries.append(
            CompetitiveEntry(
                opportunity=develop[node_id],
                rank=rank,
                reason="critic unavailable; conservative Develop on the highest-quality shortlist state",
                evidence_refs=(),
                payoff_horizon=None,
                semantic_mismatch=None,
            )
        )
    return CriticResult(entries=tuple(entries), not_applicable=(), invalid=True)


def _extract_json(response: str) -> object | None:
    text = str(response).strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = fences if fences else [text]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


__all__ = [
    "CriticPrompt",
    "build_critic_prompt",
    "edge_outcome",
    "fallback_result",
    "format_fitness",
    "parse_critic_response",
]
