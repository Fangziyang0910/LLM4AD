"""Trajectory decision followed by isolated code implementation for V9.3."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from ...base import Function, Program, SampleTrimmer, TextFunctionProgramConverter
from .complexity import comment_free_source
from .context import format_fitness
from .schema import ProgramNode

IDEA_MAX_CHARS = 300
STRATEGY_MAX_CHARS = 400


@dataclass(frozen=True, slots=True)
class ParsedProgram:
    idea: str
    program: Program


def fitness_direction_hint(maximize: bool) -> str:
    return (
        "Fitness is this task's score and higher is better."
        if maximize
        else "Fitness is this task's metric and lower is better."
    )


def build_strategy_plan_prompt(
    *,
    task_description: str,
    template_function: Function,
    maximize: bool,
    strategy_count: int,
) -> str:
    return "\n".join(
        [
            _root_prompt_prefix(task_description, template_function, maximize),
            "",
            "[Instruction]",
            (
                f"Propose exactly {strategy_count} complementary, task-grounded "
                "algorithmic strategies for implementing the target function."
            ),
            (
                "Each strategy must state a distinct computational mechanism, not a "
                "cosmetic rewrite or isolated parameter value."
            ),
            "Do not write code, estimate fitness, rank the strategies, or discuss search.",
            "Output only one non-empty line per strategy in this exact form:",
            *[
                f"Strategy {index}: <one sentence>"
                for index in range(1, strategy_count + 1)
            ],
        ]
    )


def parse_strategy_plan(response: str, strategy_count: int) -> tuple[str, ...] | None:
    matches = re.findall(
        r"^\s*Strategy\s+(\d+)\s*:\s*(\S.*?)\s*$",
        str(response),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if [int(index) for index, _ in matches] != list(range(1, strategy_count + 1)):
        return None
    strategies = tuple(
        " ".join(text.split())[:STRATEGY_MAX_CHARS] for _, text in matches
    )
    if len({strategy.casefold() for strategy in strategies}) != strategy_count:
        return None
    return strategies


def build_strategy_root_prompt(
    *,
    task_description: str,
    template_function: Function,
    maximize: bool,
    strategy_index: int,
    strategy: str,
) -> str:
    return "\n".join(
        [
            _root_prompt_prefix(task_description, template_function, maximize),
            "",
            f"[Assigned Initial Strategy {strategy_index}]",
            strategy.strip(),
            "",
            "[Instruction]",
            (
                "Implement the assigned strategy as one complete, valid, and "
                "competitive initial algorithm."
            ),
            "Avoid a placeholder or trivial baseline.",
            "Keep the target function signature and contract unchanged.",
            _code_representation_contract(),
            _output_contract(),
        ]
    )


def build_trajectory_decision_prompt(
    *,
    task_description: str,
    anchor: ProgramNode,
    window_text: str,
    maximize: bool,
    initial_strategy: str | None = None,
) -> str:
    strategy_section = (
        []
        if initial_strategy is None
        else [
            "[Initial Route Strategy]",
            initial_strategy.strip(),
            "",
        ]
    )
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            *strategy_section,
            window_text.strip(),
            "",
            "[Current Executable Anchor]",
            f"Anchor fitness: {format_fitness(anchor.fitness)}",
            "```python",
            anchor.code.rstrip(),
            "```",
            "",
            "[Decision]",
            (
                "Decide exactly one coherent next algorithmic Idea for the current "
                "executable anchor, using the local trajectory evidence."
            ),
            (
                "Treat each recorded result as evidence about that specific "
                "implementation; it does not by itself prove that the broader idea "
                "is good or bad."
            ),
            (
                "If revisiting a tested idea, require a materially different "
                "realization and state that difference."
            ),
            "The Idea must be specific enough to implement from the anchor code alone.",
            "Do not write code, estimate fitness, summarize the whole history, or discuss search.",
            "Output only:",
            f"Idea: <one sentence, no more than {IDEA_MAX_CHARS} characters>",
        ]
    )


def build_code_implementation_prompt(
    *,
    task_description: str,
    anchor: ProgramNode,
    idea: str,
    maximize: bool,
) -> str:
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Current Executable Anchor]",
            f"Anchor fitness: {format_fitness(anchor.fitness)}",
            "```python",
            anchor.code.rstrip(),
            "```",
            "",
            "[Approved Next Idea]",
            idea.strip(),
            "",
            "[Implementation]",
            "Implement exactly the approved Idea from the current anchor.",
            "Keep the current target function signature and contract unchanged.",
            "Return one complete, self-contained implementation.",
            _code_representation_contract(),
            "Output only:",
            "Code:",
            "```python",
            "<complete function implementation>",
            "```",
        ]
    )


def _output_contract() -> str:
    return "\n".join(
        [
            (
                "The Idea must accurately summarize the algorithmic mechanism "
                "actually implemented in Code."
            ),
            "Output only:",
            f"Idea: <one sentence, no more than {IDEA_MAX_CHARS} characters>",
            "Code:",
            "```python",
            "<complete function implementation>",
            "```",
        ]
    )


def _code_representation_contract() -> str:
    return (
        "Code must contain executable implementation only: do not include comments, "
        "docstrings, commented-out alternatives, or prose inside the code block."
    )


def _root_prompt_prefix(
    task_description: str, template_function: Function, maximize: bool
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Target Function]",
            str(target).rstrip(),
            "Keep the function name, arguments, return type, and contract unchanged.",
            "Imports from the task template remain available; small helpers are allowed.",
        ]
    )


def parse_program_response(
    response: str,
    template_program: Program,
    function_name: str,
    *,
    signature_template: Program | None = None,
) -> ParsedProgram | None:
    text = str(response)
    first_fence = text.find("```")
    if first_fence < 0:
        return None
    idea = extract_idea(text[:first_fence])
    if idea is None:
        return None
    blocks = tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )
    program = next(
        (
            parsed
            for code in reversed(blocks)
            if (
                parsed := _parse_candidate(
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
    if program is None:
        return None
    return ParsedProgram(idea=idea[:IDEA_MAX_CHARS], program=program)


def parse_code_response(
    response: str,
    idea: str,
    template_program: Program,
    function_name: str,
    *,
    signature_template: Program | None = None,
) -> ParsedProgram | None:
    blocks = tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            str(response),
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )
    program = next(
        (
            parsed
            for code in reversed(blocks)
            if (
                parsed := _parse_candidate(
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
    if program is None:
        return None
    return ParsedProgram(idea=idea[:IDEA_MAX_CHARS], program=program)


def extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>\S[^\r\n]*)$",
        str(response),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else " ".join(match.group("idea").split())


def _parse_candidate(
    code: str,
    template_program: Program,
    function_name: str,
    *,
    signature_template: Program | None,
) -> Program | None:
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is None or function_name not in [item.name for item in parsed.functions]:
        parsed = SampleTrimmer.sample_to_program(code, template_program)
    if parsed is None:
        return None
    try:
        generated_target = parsed.get_function(function_name)
        contract = (
            template_program if signature_template is None else signature_template
        )
        template_target = contract.get_function(function_name)
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
    combined = Program(
        preface=_merge_prefaces(template_program.preface, parsed.preface),
        functions=functions,
    )
    cleaned = TextFunctionProgramConverter.text_to_program(
        comment_free_source(str(combined))
    )
    return cleaned


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
    "ParsedProgram",
    "build_code_implementation_prompt",
    "build_strategy_plan_prompt",
    "build_strategy_root_prompt",
    "build_trajectory_decision_prompt",
    "extract_idea",
    "fitness_direction_hint",
    "parse_strategy_plan",
    "parse_code_response",
    "parse_program_response",
]
