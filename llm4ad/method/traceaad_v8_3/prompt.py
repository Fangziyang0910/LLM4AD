"""V8.3 text prompts and the deliberately small output parser."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from ...base import Function, Program, SampleTrimmer, TextFunctionProgramConverter


@dataclass(frozen=True, slots=True)
class ParsedCall1:
    design_idea: str
    program: Program


def format_fitness(value: float) -> str:
    return f"{value:.6g}"


def _target(function: Function) -> str:
    target = copy.deepcopy(function)
    target.body = ""
    return str(target).rstrip()


def build_initial_prompt(
    *,
    task_description: str,
    target_function: Function,
    operator: str,
    references: tuple[tuple[str, str], ...],
    maximize: bool,
) -> str:
    lines = [
        "[Task]",
        task_description.strip(),
        "Fitness direction: higher is better." if maximize else "Fitness direction: lower is better.",
        "",
        f"[Initialization Operator: {operator}]",
        "Generate a complete algorithm for the target function.",
    ]
    for index, (idea, code) in enumerate(references, start=1):
        lines.extend(["", f"[Reference Algorithm {index}]", f"Design Idea: {idea}", "Code:", "```python", code.rstrip(), "```"])
    lines.extend([
        "",
        "[Target Function]",
        _target(target_function),
        "",
        "Output exactly:",
        "Design Idea: <one concise sentence>",
        "Code:",
        "```python",
        "<complete implementation>",
        "```",
    ])
    return "\n".join(lines).strip()


def build_description_prompt(
    *, task_description: str, design_idea: str, code: str
) -> str:
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            "",
            "[Generated Design Idea]",
            design_idea.strip(),
            "",
            "[Generated Code]",
            "```python",
            code.rstrip(),
            "```",
            "",
            "Describe the actual behavior of this code factually and concisely.",
            "Do not propose changes and do not restate an intention that the code does not implement.",
            "Output exactly:",
            "Description: <concise factual description>",
        ]
    ).strip()


def build_search_prompt(
    *,
    task_description: str,
    current_code: str,
    current_description: str,
    current_fitness: float,
    history: str,
    operator_name: str,
    operator_instruction: str,
    target_function: Function,
    reference: tuple[str, str, str, float] | None,
    maximize: bool,
) -> str:
    lines = [
        "[Task]",
        task_description.strip(),
        "Fitness direction: higher is better." if maximize else "Fitness direction: lower is better.",
        "",
        "[Current Algorithm]",
        f"Fitness: {format_fitness(current_fitness)}",
        f"Description: {current_description.strip()}",
        "Code:",
        "```python",
        current_code.rstrip(),
        "```",
        "",
        "[Local Exploration Context]",
        history.strip(),
        "",
        "[Operator]",
        f"{operator_name}: {operator_instruction}",
    ]
    if reference is not None:
        if len(reference) == 4:
            _ref_idea, ref_description, ref_code, ref_fitness = reference
        else:
            ref_description, ref_code, ref_fitness = reference
        lines.extend(
            [
                "",
                "[Reference Algorithm]",
                f"Fitness: {format_fitness(ref_fitness)}",
                f"Description: {ref_description.strip()}",
                "Code:",
                "```python",
                ref_code.rstrip(),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "[Generation Requirements]",
            "Use the current algorithm as the direct basis for one meaningful change.",
            "Use the local exploration context to identify effective ideas and already tested directions; avoid repeating a known attempt without a reason.",
            "Allow necessary parameter, condition, data-flow, or local structural changes needed to implement the main intention.",
            "Keep the target function signature and task contract unchanged.",
            "Return one complete executable implementation with no placeholders.",
            "Make the design idea concise, actionable, and consistent with the code.",
            "Output exactly:",
            "Design Idea: <one concise sentence>",
            "Code:",
            "```python",
            "<complete implementation>",
            "```",
            "",
            "[Target Function]",
            _target(target_function),
        ]
    )
    return "\n".join(lines).strip()


def parse_call1(response: str, template: Program, function_name: str) -> ParsedCall1 | None:
    text = str(response)
    match = re.fullmatch(
        r"\s*Design Idea\s*:\s*(?P<idea>[^\r\n]+)\r?\n"
        r"(?:[ \t]*\r?\n)*"
        r"Code\s*:\s*```python[ \t]*\r?\n(?P<code>.*?)\r?\n```\s*",
        text,
        re.I | re.S,
    )
    if match is None:
        return None
    candidate = match.group("code")
    program = TextFunctionProgramConverter.text_to_program(candidate)
    if program is None:
        program = SampleTrimmer.sample_to_program(candidate, template)
    if program is None:
        return None
    try:
        program.get_function(function_name)
    except ValueError:
        return None
    return ParsedCall1(match.group("idea").strip(), program)


def parse_description(response: str) -> str | None:
    match = re.fullmatch(r"\s*Description\s*:\s*([^\r\n]+)\s*", str(response), re.I)
    return None if match is None else match.group(1).strip()


__all__ = [
    "ParsedCall1",
    "build_description_prompt",
    "build_initial_prompt",
    "build_search_prompt",
    "format_fitness",
    "parse_call1",
    "parse_description",
]
