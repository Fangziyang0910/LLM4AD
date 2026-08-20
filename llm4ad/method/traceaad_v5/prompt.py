"""Prompt construction for independent TraceAAD v5."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from ...base import Function
from .schema import ProgramNode

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
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            task_description.strip(),
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


def build_code_prompt(
    *,
    current_node: ProgramNode,
    action: str,
    task_description: str,
    template_function: Function,
    history: str,
    reference_node: ProgramNode | None = None,
    reference_history: str = "",
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    sections = [
        "[Task]",
        task_description.strip(),
        "",
        "[Current Program History]",
        history.strip(),
        "",
        "[Current Program]",
        "```python",
        current_node.code.rstrip(),
        "```",
    ]
    if reference_node is not None:
        sections.extend(
            [
                "",
                "[Reference Program History]",
                reference_history.strip(),
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
            "[Requested Modification]",
            action.strip(),
            "",
            "[Target Function]",
            str(target).rstrip(),
            "",
            "[Instruction]",
            (
                "Use the histories as evidence about tested directions, then implement "
                "the requested modification from the primary program."
            ),
            (
                "When a reference program is present, use it only in the way specified "
                "by the requested modification."
            ),
            (
                "Realize the modification in code rather than reproduce the current "
                "program unchanged."
            ),
            "Keep the target function signature and contract unchanged.",
            "Return exactly one complete implementation.",
            "Include every required import and helper in the returned program.",
            "Output only:",
            (
                "Idea: <one sentence describing the implemented change, "
                f"no more than {IDEA_MAX_CHARS} characters>"
            ),
            "Code:",
            "```python",
            "<complete function implementation>",
            "```",
        ]
    )
    return "\n".join(sections).strip()


def parse_actions(
    response: str,
    *,
    expected_count: int,
) -> tuple[list[str], list[str]]:
    actions: list[str] = []
    for line in str(response).strip().splitlines():
        match = re.match(
            r"^(?P<number>\d+)[\.\)]\s+(?P<action>\S.*)$",
            line.strip(),
        )
        if match is None or int(match.group("number")) != len(actions) + 1:
            continue
        actions.append(match.group("action").strip())
    parsed_count = len(actions)
    errors = (
        []
        if parsed_count == expected_count
        else [f"expected_{expected_count}_actions_got_{parsed_count}"]
    )
    return actions[:expected_count], errors


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
    return ParsedCandidate(declared_idea=_short_idea_or_none(text), code=text.strip())


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
    return compact[: IDEA_MAX_CHARS - 1].rstrip() + "…"


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
    "build_code_prompt",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
    "parse_actions",
    "parse_program_response",
]
