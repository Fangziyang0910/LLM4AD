"""Conditioned generation prompts for the five V10 design operators.

The generator sees strictly less than the critic: task, current algorithm,
formation path, and the operator instruction.  Transfer additionally sees the
reference code and path, SemanticRepair the identified inconsistency, and
Restart only verified improvement cards without any code.
"""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass

from ...base import Function
from .critic import _formation_lines, format_fitness

IDEA_MAX_CHARS = 500
TRACEBACK_TAIL_CHARS = 1500

OPERATOR_INSTRUCTIONS: dict[str, str] = {
    "develop": (
        "**Develop.** Preserve the current algorithm's core design hypothesis. "
        "Improve it coherently. You may tune parameters, strengthen a local or "
        "deep mechanism, simplify harmful details, or restructure supporting "
        "logic, but do not replace the main algorithmic idea."
    ),
    "pivot": (
        "**Pivot.** Use the current algorithm as a starting point, but do not "
        "assume its core design hypothesis is correct. Replace or substantially "
        "redesign one central decision mechanism and create a coherent "
        "alternative direction."
    ),
    "transfer": (
        "**Transfer.** Preserve the useful structure of the source algorithm. "
        "Identify one mechanism in the donor that is supported by its evaluator "
        "history and integrate that mechanism coherently. Do not copy code "
        "mechanically."
    ),
    "restart": (
        "**Restart.** Propose a new algorithmic hypothesis not anchored to an "
        "existing parent."
    ),
    "semantic_repair": (
        "**SemanticRepair.** Preserve the intended algorithmic hypothesis. "
        "Correct the identified semantic inconsistency between the intended "
        "mechanism and its actual implementation. Do not redesign unrelated "
        "parts."
    ),
}


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    idea: str | None
    code: str


def _task_intro(task_description: str) -> list[str]:
    return ["[Task]", task_description.strip(), "Fitness is the task score; higher is better.", ""]


def _reliability() -> list[str]:
    return [
        "Keep execution bounded; do not use unbounded loops or recursion.",
        "Return a value satisfying the target function contract on every call.",
        "Do not mutate input objects unless the task explicitly permits it.",
    ]


def _output_contract() -> str:
    return "\n".join(
        [
            "Return one concise Idea line followed by a complete Python program in one fenced block:",
            "Idea: <one concise design idea>",
            "Code:",
            "```python",
            "<complete program>",
            "```",
            "Do not write a module or function docstring, and do not write comments.",
        ]
    )


def _target_function(function: Function) -> str:
    target = copy.deepcopy(function)
    target.body = ""
    target.docstring = None
    return str(target).rstrip()


def _current_algorithm(code: str, fitness: float | None) -> list[str]:
    fitness_line = (
        "Fitness: unavailable" if fitness is None else f"Fitness: {format_fitness(fitness)}"
    )
    return [
        "[Current Algorithm]",
        fitness_line,
        "```python",
        code.rstrip(),
        "```",
    ]


def _formation_section(nodes: dict, node_id: int, *, title: str = "[Formation Path]") -> list[str]:
    lines = _formation_lines(nodes, node_id, f"S{node_id}")
    return ["", title, *(f"{line}" for line in lines)]


def build_root_prompt(*, task_description: str, template_function: Function) -> str:
    return "\n".join(
        [
            *_task_intro(task_description),
            "[Target Function]",
            _target_function(template_function),
            "Keep the function name, arguments, and return type unchanged.",
            "Include every required import and helper.",
            "",
            "[Instruction]",
            "Propose one concise algorithmic idea and implement a competitive initial algorithm.",
            *_reliability(),
            _output_contract(),
        ]
    )


def build_generation_prompt(
    *,
    task_description: str,
    operator: str,
    nodes: dict,
    start_id: int | None,
    reference_id: int | None = None,
    semantic_mismatch: str | None = None,
    restart_cards: list[str] | None = None,
    template_function: Function | None = None,
) -> str:
    """Build the conditioned generation prompt for one selected opportunity."""
    if operator not in OPERATOR_INSTRUCTIONS:
        raise ValueError(f"unknown V10 operator: {operator}")
    sections = [*_task_intro(task_description)]
    if operator == "restart":
        sections.append("[Target Function]")
        sections.append(_target_function(template_function))
        sections.append("Keep the function name, arguments, and return type unchanged.")
        sections.append("Include every required import and helper.")
        sections.append("")
        sections.append("[Verified Improvements So Far]")
        if restart_cards:
            sections.extend(f"- {card}" for card in restart_cards)
        else:
            sections.append("none recorded yet")
    else:
        start = nodes[start_id]
        sections.extend(_current_algorithm(start.code, start.fitness))
        sections.extend(_formation_section(nodes, start.id))
        if operator == "transfer":
            reference = nodes[reference_id]
            sections.append("")
            sections.append("[Reference Algorithm (donor)]")
            sections.append(f"Fitness: {format_fitness(reference.fitness)}")
            sections.append(f"Idea: {reference.idea or 'unavailable'}")
            sections.append("```python")
            sections.append(reference.code.rstrip())
            sections.append("```")
            sections.extend(_formation_section(nodes, reference.id, title="[Reference Formation Path]"))
        if operator == "semantic_repair":
            sections.append("")
            sections.append("[Identified Semantic Inconsistency]")
            sections.append(str(semantic_mismatch))
    sections.append("")
    sections.append("[Instruction]")
    sections.append(OPERATOR_INSTRUCTIONS[operator])
    sections.append("Write the Idea line first, then the complete program.")
    sections.extend(_reliability())
    sections.append(_output_contract())
    return "\n".join(sections)


def build_repair_prompt(
    *,
    task_description: str,
    idea: str,
    base_code: str,
    failed_code: str,
    error: str,
) -> str:
    base_section = (
        ["```python", base_code.rstrip(), "```"] if base_code.strip() else ["none (new algorithm)"]
    )
    return "\n".join(
        [
            *_task_intro(task_description),
            "[Idea Under Test]",
            idea,
            "",
            "[Base Implementation]",
            *base_section,
            "",
            "[Failed Realization]",
            "```python",
            failed_code.rstrip(),
            "```",
            f"Failure: {error}",
            "",
            "[Instruction]",
            "Repair the failed realization with the smallest change that addresses the failure.",
            "Preserve the Idea and target function signature.",
            *_reliability(),
            _output_contract(),
        ]
    )


def extract_idea(text: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>[^\r\n]+)",
        str(text),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match is None:
        return None
    idea = " ".join(match.group("idea").split())
    return idea[:IDEA_MAX_CHARS].rstrip() or None


def parse_candidate_response(response: str) -> ParsedCandidate:
    text = str(response)
    matches = re.findall(
        r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        code = matches[-1].strip()
    else:
        marker = re.search(r"^\s*Code\s*:\s*", text, flags=re.IGNORECASE | re.MULTILINE)
        code = text[marker.end():].strip() if marker else text.strip()
    return ParsedCandidate(idea=extract_idea(text), code=_executable_source(code))


def _executable_source(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree = _DropDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _DropDocstrings(ast.NodeTransformer):
    def _drop(self, node):
        self.generic_visit(node)
        node.body = _without_leading_string(node.body)
        return node

    visit_Module = _drop
    visit_FunctionDef = _drop
    visit_AsyncFunctionDef = _drop
    visit_ClassDef = _drop


def _without_leading_string(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:] or [ast.Pass()]
    return body


def format_failure(error_type: str | None, error: str, traceback: str | None = None) -> str:
    parts = [error if not error_type else f"{error_type}: {error}"]
    if traceback:
        parts.append(f"Traceback (innermost last):\n{traceback[-TRACEBACK_TAIL_CHARS:]}")
    return "\n".join(parts)


__all__ = [
    "OPERATOR_INSTRUCTIONS",
    "ParsedCandidate",
    "build_generation_prompt",
    "build_repair_prompt",
    "build_root_prompt",
    "extract_idea",
    "format_failure",
    "parse_candidate_response",
]
