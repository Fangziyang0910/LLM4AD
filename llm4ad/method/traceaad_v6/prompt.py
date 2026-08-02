"""Prompt construction for independent TraceAAD v6."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from ...base import (
    Function,
    Program,
    SampleTrimmer,
    TextFunctionProgramConverter,
)
from .schema import ProgramNode

IDEA_MAX_CHARS = 300


@dataclass(frozen=True, slots=True)
class ParsedProgram:
    idea: str
    program: Program


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
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            task_description.strip(),
            "",
            "Generate a simple, complete, and valid implementation for the target Python function.",
            "The function name, arguments, return type, and contract should remain unchanged.",
            (
                "Imports from the task template remain available. "
                "You may add small top-level helper functions when they clarify the implementation."
            ),
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
                "The histories describe what has been tried; they may help explain the "
                "requested modification from the primary program."
            ),
            (
                "When a reference program is present, the requested modification can "
                "adapt an idea from it."
            ),
            "The target function signature and contract remain unchanged.",
            "Return one complete implementation.",
            (
                "Imports from the task template remain available. Include any additional "
                "imports and top-level helper functions required by this implementation."
            ),
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
    numbered: dict[int, str] = {}
    errors: list[str] = []
    for line in str(response).strip().splitlines():
        match = re.match(
            r"^(?P<number>\d+)[\.\)]\s+(?P<action>\S.*)$",
            line.strip(),
        )
        if match is None:
            continue
        number = int(match.group("number"))
        if number < 1 or number > expected_count:
            errors.append(f"action_number_out_of_range_{number}")
            continue
        if number in numbered:
            errors.append(f"duplicate_action_number_{number}")
            continue
        numbered[number] = match.group("action").strip()
    actions = [numbered[number] for number in sorted(numbered)]
    if not actions:
        errors.append(f"no_valid_actions_up_to_{expected_count}")
    return actions, errors


def parse_program_response(
    response: str,
    template_program: Program,
    function_name: str,
) -> ParsedProgram | None:
    idea = (
        _extract_idea(response) or _extract_boxed_text(response) or "Generated program"
    )
    blocks = _extract_code_blocks(response)
    candidates = reversed(blocks) if blocks else (response,)
    program = next(
        (
            parsed
            for code in candidates
            if (
                parsed := _parse_program_candidate(
                    code, template_program, function_name
                )
            )
            is not None
        ),
        None,
    )
    return (
        None
        if program is None
        else ParsedProgram(idea=_short_idea(idea), program=program)
    )


def _parse_program_candidate(
    code: str,
    template_program: Program,
    function_name: str,
) -> Program | None:
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is not None:
        completed = _complete_program(parsed, template_program, function_name)
        if completed is not None:
            return completed
    trimmed = SampleTrimmer.sample_to_program(code, template_program)
    if trimmed is None:
        return None
    return _complete_program(trimmed, template_program, function_name)


def _complete_program(
    parsed: Program,
    template_program: Program,
    function_name: str,
) -> Program | None:
    try:
        parsed.get_function(function_name)
    except ValueError:
        return None
    return Program(
        preface=_merge_prefaces(template_program.preface, parsed.preface),
        functions=parsed.functions,
    )


def _merge_prefaces(template_preface: str, generated_preface: str) -> str:
    future: list[str] = []
    ordinary: list[str] = []
    for source in (template_preface, generated_preface):
        kept: list[str] = []
        for line in source.splitlines():
            if line.lstrip().startswith("from __future__"):
                if line not in future:
                    future.append(line)
            else:
                kept.append(line)
        block = "\n".join(kept).strip()
        if block and block not in ordinary:
            ordinary.append(block)
    return "\n".join(future + ordinary)


def _extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>.+?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else match.group("idea").strip()


def _extract_boxed_text(response: str) -> str | None:
    match = re.search(
        r"(?:\\)?boxed\s*\{(?P<idea>[^{}]+)\}",
        response,
        flags=re.IGNORECASE,
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
    "ParsedProgram",
    "build_code_prompt",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
    "parse_actions",
    "parse_program_response",
]
