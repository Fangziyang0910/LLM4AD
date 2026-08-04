"""Prompt construction for independent TraceAAD v7."""

from __future__ import annotations

import copy
import json
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
ACTION_MAX_CHARS = 600
ACTION_OUTPUT_MODE = "json_schema"


@dataclass(frozen=True, slots=True)
class ParsedProgram:
    idea: str
    program: Program


def format_fitness(fitness: float) -> str:
    return f"{fitness:.6g}"


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
    reference_node: ProgramNode | None = None,
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    sections = [
        "[Task]",
        task_description.strip(),
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
                "When a reference program is present, the requested modification can "
                "use its code as implementation evidence; do not copy unrelated code."
            ),
            "The target function signature and contract remain unchanged.",
            "Implement exactly the requested modification; do not redesign the search decision.",
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


def action_response_format(expected_count: int) -> dict:
    """Return the strict response schema used by the Action LLM call."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "traceaad_actions",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": ACTION_MAX_CHARS,
                        },
                        "minItems": 1,
                        "maxItems": expected_count,
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        },
    }


def parse_actions(
    response: str,
    *,
    expected_count: int,
) -> tuple[list[str], list[str]]:
    """Parse and validate a schema-constrained Action response."""
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    try:
        payload = json.loads(str(response))
    except (TypeError, json.JSONDecodeError):
        return [], ["invalid_json"]

    if not isinstance(payload, dict):
        return [], ["json_root_not_object"]
    if set(payload) != {"actions"}:
        return [], ["json_object_must_only_contain_actions"]
    raw_actions = payload["actions"]
    if not isinstance(raw_actions, list):
        return [], ["actions_not_array"]
    if not 1 <= len(raw_actions) <= expected_count:
        return [], [f"action_count_out_of_range_{len(raw_actions)}"]
    actions: list[str] = []
    for index, action in enumerate(raw_actions, start=1):
        if not isinstance(action, str):
            return [], [f"action_{index}_not_string"]
        compact = " ".join(action.split())
        if not compact:
            return [], [f"action_{index}_empty"]
        if len(compact) > ACTION_MAX_CHARS:
            return [], [f"action_{index}_too_long"]
        actions.append(compact)
    return actions, []


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
    "ACTION_OUTPUT_MODE",
    "ACTION_MAX_CHARS",
    "IDEA_MAX_CHARS",
    "ParsedProgram",
    "action_response_format",
    "build_code_prompt",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
    "parse_actions",
    "parse_program_response",
]
