"""Direct program generation and parsing protocol for TraceAAD V9-Core."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from ...base import Function

IDEA_MAX_CHARS = 300


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    declared_idea: str | None
    code: str


def format_fitness(fitness: float | None) -> str:
    return "unknown" if fitness is None else f"{fitness:.6g}"


def fitness_direction_hint(maximize: bool) -> str:
    return (
        "Fitness is this task's score and higher is better."
        if maximize
        else "Fitness is this task's metric and lower is better."
    )


def build_initial_prompt(
    *,
    task_description: str,
    template_function: Function,
    diversity_hint: str,
    maximize: bool = True,
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            (
                "Generate a complete implementation for the target Python function. "
                f"{diversity_hint}"
            ),
            "Keep the function name, arguments, return type, and contract unchanged.",
            "Include every required import and helper in the returned program.",
            "",
            "Output format:",
            (
                "Idea: <one sentence describing the implemented algorithm, "
                f"no more than {IDEA_MAX_CHARS} characters>"
            ),
            "Code:",
            "```python",
            str(target).rstrip(),
            "```",
        ]
    ).strip()


def parse_program_response(response: str) -> ParsedCandidate:
    """Lenient extraction: last fenced block, else text after Code:, else the response."""
    text = str(response)
    first_fence = text.find("```")
    blocks = _extract_code_blocks(text)
    if blocks:
        return ParsedCandidate(
            declared_idea=_short_idea_or_none(text[:first_fence]),
            code=blocks[-1],
        )

    code_marker = re.search(r"^\s*Code\s*:\s*", text, re.IGNORECASE | re.MULTILINE)
    if code_marker is not None:
        return ParsedCandidate(
            declared_idea=_short_idea_or_none(text[: code_marker.start()]),
            code=text[code_marker.end() :].strip(),
        )
    return ParsedCandidate(
        declared_idea=_short_idea_or_none(text), code=text.strip()
    )


def _short_idea_or_none(text: str) -> str | None:
    idea = _extract_idea(text)
    return None if idea is None else _short_idea(idea)


def _extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>.+?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else match.group("idea").strip()


def _short_idea(idea: str) -> str:
    compact = " ".join(str(idea).split())
    if len(compact) <= IDEA_MAX_CHARS:
        return compact
    return compact[: IDEA_MAX_CHARS - 3].rstrip() + "..."


def _extract_code_blocks(response: str) -> tuple[str, ...]:
    return tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )

__all__ = [
    "IDEA_MAX_CHARS",
    "ParsedCandidate",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
    "parse_program_response",
]
