"""History-conditioned Idea + full Code protocol for V9.7.

The output contract, root prompt, and response parsing are identical to
V9.6; the anchor prompt keeps the V9.6 structure and history block and only
swaps the instruction section by generation intent (Refine / Explore).
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from ...base import Function, Program, SampleTrimmer, TextFunctionProgramConverter
from .history import format_fitness
from .schema import GenerationIntent, ProgramArtifact

IDEA_MAX_CHARS = 300
PROMPT_RENDERER_VERSION = "traceaad_v97_prompt_renderer_v1"

INTENT_INSTRUCTIONS: dict[GenerationIntent, str] = {
    GenerationIntent.REFINE: (
        "Continue improving the current algorithm within its existing design. "
        "Make one focused modification based on the current algorithm and its "
        "improvement history."
    ),
    GenerationIntent.EXPLORE: (
        "Seek a materially different way to improve the current algorithm. "
        "Do not merely tune parameters or make a small local modification. "
        "You may replace or substantially restructure an important part of "
        "the current design."
    ),
}


def prompt_renderer_hash() -> str:
    source = Path(__file__).read_bytes()
    return hashlib.sha256(source).hexdigest()


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    declared_idea: str | None
    raw_code: str
    program: Program


class ProgramResponseError(ValueError):
    """The completed response does not contain mandatory executable Code."""

    def __init__(
        self,
        message: str,
        *,
        declared_idea: str | None = None,
        raw_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.declared_idea = declared_idea
        self.raw_code = raw_code


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
            "Imports from the task template remain available; small helpers are allowed.",
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
    anchor: ProgramArtifact,
    history_text: str,
    intent: GenerationIntent,
    maximize: bool,
) -> str:
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Current Algorithm]",
            f"Fitness: {format_fitness(anchor.fitness)}",
            "```python",
            anchor.evaluator_input_code.rstrip(),
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
            "Output only one optional short Idea and one mandatory full Code block:",
            f"Idea: <optional semantic label, at most {IDEA_MAX_CHARS} characters>",
            "Code:",
            "```python",
            "<complete executable implementation>",
            "```",
            "Do not output reasoning, evidence analysis, an operator label, or a patch.",
        ]
    )


def parse_program_response(
    response: str, template_program: Program, function_name: str
) -> ParsedCandidate:
    text = str(response)
    first_fence = text.find("```")
    declared_idea = extract_idea(text if first_fence < 0 else text[:first_fence])
    blocks = tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )
    if not blocks:
        raise ProgramResponseError(
            "missing a fenced Python code block", declared_idea=declared_idea
        )
    for raw_code in reversed(blocks):
        program = _parse_candidate(raw_code, template_program, function_name)
        if program is not None:
            return ParsedCandidate(
                declared_idea=(
                    None if declared_idea is None else declared_idea[:IDEA_MAX_CHARS]
                ),
                raw_code=raw_code,
                program=program,
            )
    raise ProgramResponseError(
        "no code block contained a complete target-function implementation with the "
        "required signature",
        declared_idea=declared_idea,
        raw_code=blocks[-1],
    )


def extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>\S[^\r\n]*)$",
        str(response),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else " ".join(match.group("idea").split())


def _parse_candidate(
    code: str, template_program: Program, function_name: str
) -> Program | None:
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is None or function_name not in [item.name for item in parsed.functions]:
        parsed = SampleTrimmer.sample_to_program(code, template_program)
    if parsed is None:
        return None
    try:
        generated_target = parsed.get_function(function_name)
        template_target = template_program.get_function(function_name)
    except ValueError:
        return None
    if (
        generated_target.args != template_target.args
        or generated_target.return_type != template_target.return_type
    ):
        return None

    generated = {function.name: function for function in parsed.functions}
    functions: list[Function] = []
    for template_function in template_program.functions:
        functions.append(
            generated.pop(template_function.name, copy.deepcopy(template_function))
        )
    functions.extend(generated.values())
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


__all__ = [
    "IDEA_MAX_CHARS",
    "INTENT_INSTRUCTIONS",
    "PROMPT_RENDERER_VERSION",
    "ParsedCandidate",
    "ProgramResponseError",
    "build_generation_prompt",
    "build_root_prompt",
    "extract_idea",
    "fitness_direction_hint",
    "parse_program_response",
    "prompt_renderer_hash",
]
