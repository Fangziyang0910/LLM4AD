"""Current algorithm, parent improvement path, and Refine/Explore instructions (unchanged from V9.14)."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from ...base import Function
from .history import format_fitness
from .schema import Intent

IDEA_TARGET_MAX_CHARS = 500

INTENT_INSTRUCTIONS: dict[Intent, str] = {
    Intent.REFINE: (
        "Continue improving the current algorithm within its existing design. "
        "Make one focused modification based on the current algorithm and its "
        "improvement history."
    ),
    Intent.EXPLORE: (
        "Seek a materially different way to improve the current algorithm. "
        "Do not merely tune parameters or make a small local modification. "
        "You may replace or substantially restructure an important part of "
        "the current design."
    ),
}


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    declared_idea: str | None
    code: str


def fitness_direction_hint(maximize: bool) -> str:
    return (
        "Fitness is this task's score and higher is better."
        if maximize
        else "Fitness is this task's metric and lower is better."
    )


def build_root_prompt(
    *, task_description: str, template_function: Function, maximize: bool
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Target Function]",
            str(target).rstrip(),
            "Keep the function name, arguments, return type, and contract unchanged.",
            "Include every required import and helper in the returned program.",
            "",
            "[Instruction]",
            "Create one complete, valid, and competitive initial algorithm.",
            "Avoid a placeholder or trivial baseline.",
            _output_contract(),
        ]
    )


def build_generation_prompt(
    *,
    task_description: str,
    code: str,
    fitness: float,
    history_text: str,
    intent: Intent,
    maximize: bool,
) -> str:
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Current Algorithm]",
            f"Fitness: {format_fitness(fitness)}",
            "```python",
            code.rstrip(),
            "```",
            "",
            history_text.strip(),
            "",
            "[Instruction]",
            INTENT_INSTRUCTIONS[intent],
            "Keep the target function signature and contract unchanged.",
            "Return one complete, self-contained implementation.",
            _output_contract(),
        ]
    )


def _output_contract() -> str:
    return "\n".join(
        [
            "Output one concise Idea and one complete Python program:",
            f"Idea: <one sentence, within {IDEA_TARGET_MAX_CHARS} characters>",
            "Code:",
            "```python",
            "<complete executable implementation>",
            "```",
        ]
    )


def parse_program_response(response: str) -> ParsedCandidate:
    text = str(response)
    first_fence = text.find("```")
    blocks = tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )
    if blocks:
        return ParsedCandidate(
            declared_idea=extract_idea(text[:first_fence]),
            code=blocks[-1],
        )

    code_marker = re.search(r"^\s*Code\s*:\s*", text, re.IGNORECASE | re.MULTILINE)
    if code_marker is not None:
        return ParsedCandidate(
            declared_idea=extract_idea(text[: code_marker.start()]),
            code=text[code_marker.end() :].strip(),
        )
    return ParsedCandidate(declared_idea=extract_idea(text), code=text.strip())


def extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>\S[^\r\n]*)$",
        str(response),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    return " ".join(match.group("idea").split())


__all__ = [
    "IDEA_TARGET_MAX_CHARS",
    "INTENT_INSTRUCTIONS",
    "ParsedCandidate",
    "build_generation_prompt",
    "build_root_prompt",
    "extract_idea",
    "fitness_direction_hint",
    "parse_program_response",
]
