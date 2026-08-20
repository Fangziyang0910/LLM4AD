"""底层 prompt 片段：初始程序生成、单 action→code 生成、共享格式化 helper。"""
from __future__ import annotations

import copy

from ...base import Function
from .schema import ProgramNode


def format_fitness(fitness: float | None) -> str:
    return "unknown" if fitness is None else f"{fitness:.6g}"


def fitness_direction_hint(maximize: bool) -> str:
    if maximize:
        return "Fitness is this task's score and higher is better; a positive change is an improvement."
    return "Fitness is this task's metric and lower is better; a negative raw change is an improvement."


def build_initial_prompt(*, task_description: str, template_function: Function, diversity_hint: str) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join([
        task_description.strip(),
        "",
        f"Generate a complete implementation for the target Python function. {diversity_hint}",
        "Keep the function name, arguments, return type, and output contract unchanged.",
        "Include every required import and helper in the returned program.",
        "",
        "Output format:",
        "Idea: <brief algorithm idea>",
        "Code:",
        "```python",
        str(target).rstrip(),
        "```",
    ]).strip()


def build_code_prompt(*, current_node: ProgramNode, action: str,
                       task_description: str, template_function: Function,
                       history: str = "", operator_constraint: str = "") -> str:
    action = action.strip()
    if not action:
        raise ValueError("action must not be empty")
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join([
        "[Task Description]",
        task_description.strip(),
        "",
        "[Current Program]",
        f"Node p{current_node.id}",
        f"Idea: {current_node.idea}",
        "Code:",
        "```python",
        current_node.code.rstrip(),
        "```",
        "",
        "[Requested Modification]",
        action,
        "",
        "[History Available During Implementation]",
        history.strip() or "No previous modification history.",
        "",
        "[Operator Constraint]",
        operator_constraint.strip() or "Preserve the target contract and implement the requested change.",
        "",
        "[Target Function Contract]",
        str(target).rstrip(),
        "",
        "[Instruction]",
        "Implement the requested modification as a new complete implementation of the target function.",
        "Keep the function name, arguments, return type, and output contract unchanged.",
        "Include every required import and helper in the returned program.",
        "Return only the new idea and complete code in this format:",
        "Idea: <brief algorithm idea>",
        "Code:",
        "```python",
        "<complete function implementation>",
        "```",
        "Do not include rationale, analysis, tests, or extra text.",
    ]).strip()
