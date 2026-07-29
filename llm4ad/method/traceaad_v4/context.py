"""将单条轨迹的 MDP 历史组织为 LLM 的下一步 action 提示。"""
from __future__ import annotations

import copy

from ...base import Function
from .derivation_graph import DerivationGraph
from .prompt import fitness_direction_hint, format_fitness
from .schema import Trajectory


def trajectory_history(
    graph: DerivationGraph,
    trajectory: Trajectory,
    *,
    base_node_id: int | None = None,
    max_steps: int = 8,
) -> str:
    """Render the ordered path, including attempts after the selected anchor."""
    if not trajectory.edge_ids:
        return "No previous modification exists; this is an initial program."
    edge_ids = trajectory.edge_ids[-max_steps:]
    node_ids = trajectory.node_ids[-(len(edge_ids) + 1):]
    lines: list[str] = []
    for index, edge_id in enumerate(edge_ids, start=1):
        parent = graph.get_node(node_ids[index - 1])
        child = graph.get_node(node_ids[index])
        edge = graph.get_edge(edge_id)
        marker = " [selected anchor]" if base_node_id == parent.id else ""
        lines.extend(
            [
                f"Step {index}: p{parent.id} -> p{child.id}{marker} [operator={edge.operator}]",
                f"  action: {edge.action}",
                (
                    f"  fitness: {format_fitness(parent.fitness)} -> "
                    f"{format_fitness(child.fitness)} "
                    f"(delta={edge.delta!s}, outcome={edge.outcome})"
                ),
            ]
        )
    return "\n".join(lines)


def build_action_prompt(
    *,
    graph: DerivationGraph,
    trajectory: Trajectory,
    base_node_id: int,
    base_reason: str,
    operator_name: str,
    operator_constraint: str,
    task_description: str,
    template_function: Function,
    action_count: int,
    maximize: bool,
    max_steps: int = 8,
) -> str:
    base_node = graph.get_node(base_node_id)
    target = copy.deepcopy(template_function)
    target.body = ""
    history = trajectory_history(
        graph, trajectory, base_node_id=base_node_id, max_steps=max_steps
    )
    sections = [
        "[Task Description]",
        task_description.strip(),
        "",
        "[MDP State History]",
        "[Algorithm Improvement History]",
        "The selected trajectory is the design state history. Improvements are useful evidence; plateaus and regressions are tested boundaries, not automatic deletion.",
        fitness_direction_hint(maximize),
        history,
        "",
        "[Anchor Program]",
        f"Continue from Node p{base_node.id}. Selection reason: {base_reason}.",
        f"Idea: {base_node.idea}",
        "Code:",
        "```python",
        base_node.code.rstrip(),
        "```",
        "",
        "[Operator]",
        f"name={operator_name}",
        f"Constraint: {operator_constraint}",
        "",
        "[Target Function Contract]",
        str(target).rstrip(),
        "",
        "[Instruction]",
        "Use the complete history as reasoning context for the next action.",
        f"Propose exactly {action_count} concrete next-step modifications.",
        f"Return only a numbered list of exactly {action_count} action lines, without code or rationale.",
    ]
    return "\n".join(sections).strip()


__all__ = ["build_action_prompt", "trajectory_history"]
