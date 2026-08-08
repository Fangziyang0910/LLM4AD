"""Direct program generation and parsing protocol for TraceAAD V9.1."""

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


def parse_program_response(
    response: str,
    template_program: Program,
    function_name: str,
    signature_template: Program | None = None,
) -> ParsedProgram | None:
    """Require an explicit Idea because it is V9.1's only semantic edge record."""
    text = str(response)
    first_fence = text.find("```")
    if first_fence < 0:
        return None
    if re.search(
        r"^\s*Idea\s*:\s*\S[^\r\n]*$",
        text[:first_fence],
        flags=re.IGNORECASE | re.MULTILINE,
    ) is None:
        return None
    idea = _extract_idea(text)
    blocks = _extract_code_blocks(text)
    candidates = reversed(blocks) if blocks else (text,)
    program = next(
        (
            parsed
            for code in candidates
            if (
                parsed := _parse_program_candidate(
                    code,
                    template_program,
                    function_name,
                    signature_template=signature_template,
                )
            )
            is not None
        ),
        None,
    )
    return (
        None
        if program is None
        else ParsedProgram(
            idea=_short_idea(idea or "Generated program"),
            program=program,
        )
    )


def _parse_program_candidate(
    code: str,
    template_program: Program,
    function_name: str,
    *,
    signature_template: Program | None = None,
) -> Program | None:
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is not None:
        names = [function.name for function in parsed.functions]
        if function_name in names:
            return _complete_program(
                parsed,
                template_program,
                function_name,
                signature_template=signature_template,
            )
        # A syntactically complete program without the target may still be a
        # body-only response wrapped in extra prose; let SampleTrimmer recover it.
    trimmed = SampleTrimmer.sample_to_program(code, template_program)
    if trimmed is None:
        return None
    return _complete_program(
        trimmed,
        template_program,
        function_name,
        signature_template=signature_template,
    )


def _complete_program(
    parsed: Program,
    template_program: Program,
    function_name: str,
    *,
    signature_template: Program | None = None,
) -> Program | None:
    try:
        generated_target = parsed.get_function(function_name)
        contract_program = (
            template_program if signature_template is None else signature_template
        )
        template_target = contract_program.get_function(function_name)
    except ValueError:
        return None

    if (
        generated_target.args != template_target.args
        or generated_target.return_type != template_target.return_type
    ):
        return None

    generated_by_name = {function.name: function for function in parsed.functions}
    functions: list[Function] = []
    for template_function in template_program.functions:
        replacement = generated_by_name.pop(template_function.name, None)
        if replacement is not None:
            functions.append(replacement)
        else:
            functions.append(copy.deepcopy(template_function))
    functions.extend(generated_by_name.values())
    return Program(
        preface=_merge_prefaces(template_program.preface, parsed.preface),
        functions=functions,
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
    "ParsedProgram",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
    "parse_program_response",
]
