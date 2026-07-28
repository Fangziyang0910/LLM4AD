"""Prompt construction for independent TraceAAD v5."""

from __future__ import annotations

import copy

from ...base import Function
from .schema import ProgramNode

IDEA_MAX_CHARS = 300


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
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            task_description.strip(),
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
                "Use the histories as evidence about tested directions, then implement "
                "the requested modification from the primary program."
            ),
            (
                "When a reference program is present, use it only in the way specified "
                "by the requested modification."
            ),
            (
                "Realize the modification in code rather than reproduce the current "
                "program unchanged."
            ),
            "Keep the target function signature and contract unchanged.",
            "Return exactly one complete implementation.",
            (
                "Imports from the current program remain available. "
                "You may retain or add small top-level helper functions."
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


__all__ = [
    "IDEA_MAX_CHARS",
    "build_code_prompt",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
]
