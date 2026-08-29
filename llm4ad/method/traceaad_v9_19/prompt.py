"""Trajectory-conditioned prompts for TraceAAD V9.19."""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass

from ...base import Function
from .history import format_fitness
from .schema import Action

IDEA_TARGET_MAX_CHARS = 500

ACTION_INSTRUCTIONS: dict[Action, str] = {
    Action.DEVELOP: (
        "Continue developing the current algorithm. Preserve its main "
        "framework and make one coherent modification with a clear "
        "performance rationale. Use the formation path to identify what has "
        "already worked, what has been revisited, and which recent direction "
        "deserves refinement. You may improve a local rule or one "
        "substantive mechanism, but avoid redesigning unrelated parts."
    ),
    Action.EXPLORE: (
        "Propose a materially different algorithmic direction for the task. "
        "Change the main decision logic rather than making a cosmetic "
        "variation. Use the formation path to avoid repeating behavior that "
        "has already been revisited without improvement, while keeping the "
        "new design coherent and executable."
    ),
    Action.CROSSOVER: (
        "Combine the current algorithm with the provided reference algorithm. "
        "Identify one useful mechanism or decision rule in the reference and "
        "integrate it coherently into the current framework. Preserve the "
        "strong parts of the current algorithm, avoid copying code blindly, "
        "and return one executable hybrid design."
    ),
}

_FITNESS_HINT = "Fitness is this task's score and higher is better."

_RELIABILITY_CONTRACT = (
    "Reliability constraints:",
    "Keep execution bounded; do not use unbounded loops or recursion.",
    "Return a value that satisfies the target function contract on every call.",
    "Do not mutate input objects unless the task explicitly permits it.",
)

_TIMEOUT_GUIDANCE = (
    "The failure is the per-call time limit: reduce the computational cost of"
    " each call (vectorize, precompute tables, shrink candidate lists) and keep"
    " the algorithmic idea."
)

TRACEBACK_TAIL_CHARS = 1500


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    declared_idea: str | None
    code: str


def _output_contract() -> str:
    return "\n".join(
        [
            "Output one concise Idea and one complete Python program:",
            f"Idea: <one concise design idea, <= {IDEA_TARGET_MAX_CHARS} chars>",
            "Code:",
            "```python",
            "<complete program>",
            "```",
            "The program may contain only imports, the target signature, and executable statements.",
            "Do not write a module or function docstring, and do not write comments.",
        ]
    )


def build_root_prompt(*, task_description: str, template_function: Function) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    target.docstring = None
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            _FITNESS_HINT,
            "",
            "[Target Function]",
            str(target).rstrip(),
            "Keep the function name, arguments, and return type unchanged.",
            "Include every required import and helper in the returned program.",
            "",
            "[Instruction]",
            "Create one complete, valid, and competitive initial algorithm.",
            "Avoid a placeholder or trivial baseline.",
            *_RELIABILITY_CONTRACT,
            _output_contract(),
        ]
    )


def build_generation_prompt(
    *,
    task_description: str,
    code: str,
    fitness: float,
    history_text: str,
    action: Action,
    reference_code: str | None = None,
    reference_fitness: float | None = None,
    reference_behavior: str | None = None,
    reference_distance: float | None = None,
) -> str:
    sections = [
            "[Task]",
            task_description.strip(),
            _FITNESS_HINT,
            "",
            "[Current Algorithm]",
            f"Fitness: {format_fitness(fitness)}",
            "```python",
            code.rstrip(),
            "```",
    ]
    if action is Action.CROSSOVER:
        if reference_code is None or reference_fitness is None:
            raise ValueError("crossover prompt requires a reference algorithm")
        sections.extend(
            [
                "",
                "[Crossover Reference Algorithm]",
                f"Fitness: {format_fitness(reference_fitness)}",
                f"Behavior tag: {reference_behavior or 'unavailable'}",
                (
                    "Behavior distance from current algorithm: "
                    + ("unavailable" if reference_distance is None else f"{reference_distance:.6g}")
                ),
                "```python",
                reference_code.rstrip(),
                "```",
            ]
        )
    sections.extend(
        [
            "",
            history_text.strip(),
            "",
            "[Instruction]",
            ACTION_INSTRUCTIONS[action],
            "Keep the target function signature unchanged.",
            "Return one complete, self-contained implementation.",
            *_RELIABILITY_CONTRACT,
            _output_contract(),
        ]
    )
    return "\n".join(sections)


def build_repair_prompt(
    *,
    task_description: str,
    parent_code: str | None,
    failed_code: str,
    error: str,
    action: Action | None,
    reference_code: str | None = None,
    reference_fitness: float | None = None,
) -> str:
    intent_text = "initialization" if action is None else action.value
    parent_block = (
        "No evaluated parent code is available; repair the failed initial candidate."
        if parent_code is None
        else "[Evaluated Parent]\n```python\n" + parent_code.rstrip() + "\n```"
    )
    reference_block = ""
    if reference_code is not None:
        reference_block = "\n[Crossover Reference]\nFitness: " + format_fitness(reference_fitness) + "\n```python\n" + reference_code.rstrip() + "\n```"
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            _FITNESS_HINT,
            "",
            parent_block,
            reference_block,
            "",
            "[Failed Candidate]",
            "```python",
            failed_code.rstrip(),
            "```",
            f"Failure during {intent_text}: {error}",
            "",
            "[Instruction]",
            "Repair the failed candidate with the smallest change that addresses the reported failure.",
            "Preserve the intended algorithmic idea and the target function signature.",
            "Do not add a docstring or comments.",
            *_RELIABILITY_CONTRACT,
            _output_contract(),
        ]
    )


_INFEASIBLE_GUIDANCE = {
    "vrptw_construct": (
        "The route became infeasible. Usual causes: the depot was returned "
        "while the vehicle was already at the depot, or the returned node is "
        "outside the given feasible set. Always return one ID taken from "
        "unvisited_nodes; return the depot only when current_node != depot, "
        "to close the current route."
    ),
    "tsp_construct": (
        "The route became invalid because the returned node was already "
        "visited. Always return one ID taken from unvisited_nodes."
    ),
}


def format_failure_feedback(
    *,
    error_type: str | None,
    error: str,
    traceback: str | None = None,
    task_key: str | None = None,
) -> str:
    """Compose the failure report for the repair prompt."""
    parts = [error if not error_type else f"{error_type}: {error}"]
    if error_type == "TimeoutError":
        parts.append(_TIMEOUT_GUIDANCE)
    if error_type == "InvalidEvaluationResult":
        guidance = _INFEASIBLE_GUIDANCE.get(task_key or "")
        if guidance is not None:
            parts.append(guidance)
    if traceback:
        parts.append(
            f"Traceback (innermost last):\n{traceback[-TRACEBACK_TAIL_CHARS:]}"
        )
    return "\n".join(parts)


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
            code=_executable_source(blocks[-1]),
        )

    code_marker = re.search(r"^\s*Code\s*:\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if code_marker is not None:
        return ParsedCandidate(
            declared_idea=extract_idea(text[: code_marker.start()]),
            code=_executable_source(text[code_marker.end() :].strip()),
        )
    return ParsedCandidate(
        declared_idea=extract_idea(text),
        code=_executable_source(text.strip()),
    )


def _executable_source(code: str) -> str:
    """Keep imports, signatures, and statements; drop docstrings and comments."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    tree = _DropDocstrings().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class _DropDocstrings(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body = _without_leading_string_expr(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        node.body = _without_leading_string_expr(node.body)
        return node

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> ast.AsyncFunctionDef:
        self.generic_visit(node)
        node.body = _without_leading_string_expr(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        self.generic_visit(node)
        node.body = _without_leading_string_expr(node.body)
        return node


def _without_leading_string_expr(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        rest = body[1:]
        return rest if rest else [ast.Pass()]
    return body


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
    "ACTION_INSTRUCTIONS",
    "IDEA_TARGET_MAX_CHARS",
    "ParsedCandidate",
    "build_generation_prompt",
    "build_repair_prompt",
    "build_root_prompt",
    "extract_idea",
    "format_failure_feedback",
    "parse_program_response",
]
