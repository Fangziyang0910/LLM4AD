"""Prompts and conservative parsing for the V9.21 hypothesis experiment."""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass

from ...base import Function

IDEA_MAX_CHARS = 500
TRACEBACK_TAIL_CHARS = 1500


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    idea: str | None
    code: str


def _task_intro(task_description: str) -> list[str]:
    return ["[Task]", task_description.strip(), "Fitness is the task score; higher is better.", ""]


def _working_section(working_code: str | None, working_fitness: float | None, base_code: str) -> list[str]:
    if working_code is None or working_code.strip() == base_code.strip():
        return []
    return [
        "[Current Working Implementation]",
        f"Fitness: {working_fitness:.8g}" if working_fitness is not None else "Fitness: unavailable",
        "```python",
        working_code.rstrip(),
        "```",
    ]


def _evidence_tail(formation_history: str, ledger: str, card_header: str, public_card: str | None) -> list[str]:
    sections = ["", formation_history.strip(), "", "[Implementation Evidence]", ledger.strip()]
    if public_card:
        sections.extend(["", f"[{card_header}]", public_card.strip()])
    return sections


def _reliability() -> list[str]:
    return [
        "Keep execution bounded; do not use unbounded loops or recursion.",
        "Return a value satisfying the target function contract on every call.",
        "Do not mutate input objects unless the task explicitly permits it.",
    ]


def _output_contract() -> str:
    return "\n".join(
        [
            "Return a complete Python program in one fenced block:",
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
            "Idea: <one concise design idea>",
            *_reliability(),
            _output_contract(),
        ]
    )


def build_idea_prompt(
    *,
    task_description: str,
    base_code: str,
    base_fitness: float,
    working_code: str | None = None,
    working_fitness: float | None = None,
    entry_idea: str | None,
    formation_history: str,
    ledger: str,
    proposal: str,
    public_card: str | None = None,
) -> str:
    if proposal not in {"continue", "branch"}:
        raise ValueError(f"unknown V9.21 proposal: {proposal}")
    if proposal == "continue":
        instruction = (
            "State the same algorithmic hypothesis in a more precise form, choosing one "
            "repair, reimplementation, or refinement justified by the evidence. Do not "
            "introduce an unrelated hypothesis."
        )
    else:
        instruction = (
            "Propose one materially different algorithmic hypothesis for this task. "
            "Use the public experiment only as a possible source of a compatible mechanism; "
            "do not copy it blindly."
        )
    sections = [
        *_task_intro(task_description),
        "[Stable Scaffold]",
        f"Fitness: {base_fitness:.8g}",
        "```python",
        base_code.rstrip(),
        "```",
        *_working_section(working_code, working_fitness, base_code),
        "[Entry Hypothesis]",
        entry_idea or "none (new direction)",
        *_evidence_tail(formation_history, ledger, "One Public Experiment Card", public_card),
        "",
        "[Instruction]",
        instruction,
        f"Write only one Idea line, at most {IDEA_MAX_CHARS} characters; do not write code.",
        "Idea: <one concise algorithmic hypothesis>",
    ]
    return "\n".join(sections)


def build_realization_prompt(
    *,
    task_description: str,
    idea: str,
    base_code: str,
    base_fitness: float,
    working_code: str | None = None,
    working_fitness: float | None = None,
    formation_history: str,
    ledger: str,
    proposal: str,
    public_card: str | None = None,
) -> str:
    sections = [
        *_task_intro(task_description),
        "[Idea Under Test]",
        idea.strip(),
        "",
        "[Base Implementation]",
        f"Fitness: {base_fitness:.8g}",
        "```python",
        base_code.rstrip(),
        "```",
        *_working_section(working_code, working_fitness, base_code),
        *_evidence_tail(formation_history, ledger, "Optional Public Evidence", public_card),
        "",
        "[Instruction]",
        "Implement the Idea under test as one coherent, executable change. "
        "This is an independent realization: do not refer to another response "
        "and do not assume it exists.",
        "Keep the target function signature unchanged.",
        *_reliability(),
        _output_contract(),
    ]
    return "\n".join(sections)


def build_repair_prompt(
    *,
    task_description: str,
    idea: str,
    base_code: str,
    failed_code: str,
    error: str,
) -> str:
    return "\n".join(
        [
            *_task_intro(task_description),
            "[Idea Under Test]",
            idea,
            "",
            "[Base Implementation]",
            "```python",
            base_code.rstrip(),
            "```",
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


def parse_idea_response(response: str) -> str | None:
    idea = extract_idea(str(response))
    if idea is not None:
        return idea
    for line in str(response).splitlines():
        clean = " ".join(line.strip().split())
        if clean and not clean.startswith("``"):
            return clean[:IDEA_MAX_CHARS].rstrip() or None
    return None


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
    "ParsedCandidate",
    "build_idea_prompt",
    "build_realization_prompt",
    "build_repair_prompt",
    "build_root_prompt",
    "extract_idea",
    "format_failure",
    "parse_candidate_response",
    "parse_idea_response",
]
