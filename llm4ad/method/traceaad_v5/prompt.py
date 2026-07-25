"""Prompt construction for independent TraceAAD v5."""

from __future__ import annotations

import copy

from ...base import Function
from .schema import ProgramNode


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
            "",
            "Output format:",
            "Idea: <brief algorithm idea>",
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
    primary_history: str,
    operator_constraint: str,
    reference_node: ProgramNode | None = None,
    reference_history: str = "",
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    sections = [
        "[Task Description]",
        task_description.strip(),
        "",
        "[Primary Program: the only structural parent]",
        f"Node p{current_node.id}",
        f"Idea claim: {current_node.idea}",
        "```python",
        current_node.code.rstrip(),
        "```",
        "",
        "[Primary Trajectory Context]",
        primary_history.strip() or "No previous modification history.",
    ]
    if reference_node is not None:
        sections.extend(
            [
                "",
                "[Reference Program: knowledge only, never a parent]",
                f"Node p{reference_node.id}",
                f"Idea claim: {reference_node.idea}",
                "```python",
                reference_node.code.rstrip(),
                "```",
                "",
                "[Reference Trajectory Context]",
                reference_history.strip() or "No displayed reference history.",
            ]
        )
    sections.extend(
        [
            "",
            "[Requested Modification]",
            action.strip(),
            "",
            "[Operator Constraint]",
            operator_constraint,
            "",
            "[Target Function Contract]",
            str(target).rstrip(),
            "",
            "[Instruction]",
            "Implement exactly the requested change in the primary program.",
            "Do not merely cite an edge or global experience.",
            "Return one complete implementation with the unchanged target contract.",
            "Output only:",
            "Idea: <brief implementation claim>",
            "Code:",
            "```python",
            "<complete function implementation>",
            "```",
        ]
    )
    return "\n".join(sections).strip()


__all__ = [
    "build_code_prompt",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
]
