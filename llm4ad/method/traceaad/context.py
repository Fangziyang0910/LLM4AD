"""把当前轨迹和少量跨轨迹成败经验组织为下一步修改提示。"""
from __future__ import annotations

import copy

from ...base import Function
from .derivation_graph import DerivationGraph
from .experience_memory import ExperienceMemory
from .prompt import fitness_direction_hint, format_fitness
from .schema import ExperienceBatch, Trajectory

_MAX_ACTION_CHARS = 300


def build_action_prompt(
    *,
    graph: DerivationGraph,
    trajectory: Trajectory,
    base_node_id: int,
    base_reason: str,
    operator_name: str,
    operator_constraint: str,
    experience_memory: ExperienceMemory,
    task_description: str,
    template_function: Function,
    action_count: int,
    maximize: bool,
    max_steps: int = 5,
    positive_k: int = 2,
    negative_k: int = 2,
) -> str:
    base_node = graph.get_node(base_node_id)
    experiences = experience_memory.examples(
        operator=operator_name,
        positive_k=positive_k,
        negative_k=negative_k,
    )
    target = copy.deepcopy(template_function)
    target.body = ""
    sections = [
        "[Task Description]",
        task_description.strip(),
        "",
        "[Algorithm Improvement History]",
        "The selected trajectory records the modifications that led to the current program.",
        fitness_direction_hint(maximize),
        _trajectory_history(graph, trajectory, max_steps=max_steps),
        "",
        "[Cross-Trajectory Action Evidence]",
        _experience_block(experiences),
        "",
        "[Operator]",
        f"name={operator_name}",
        f"Constraint: {operator_constraint}",
        "",
        "[Base Program To Modify]",
        f"Continue from Node p{base_node.id}. Selection reason: {base_reason}.",
        f"Idea: {base_node.idea}",
        "Code:",
        "```python",
        base_node.code.rstrip(),
        "```",
        "",
        "[Target Function Contract]",
        f"Only evolve:\n```python\n{target}\n```",
        "",
        "[Instruction]",
        "Use the selected trajectory as the main account of how the current program was formed.",
        "Use cross-trajectory actions only as supporting evidence of what worked or failed.",
        f"Propose exactly {action_count} concrete next-step modifications.",
        "Each modification must change one main algorithmic idea and follow the operator constraint.",
        "Do not output code or rationale.",
        f"Return only a numbered list of exactly {action_count} ideas, one per line.",
    ]
    return "\n".join(sections).strip()


def _trajectory_history(
    graph: DerivationGraph,
    trajectory: Trajectory,
    *,
    max_steps: int,
) -> str:
    if not trajectory.edge_ids:
        return "No previous modification exists; this is an initial program."
    edge_ids = trajectory.edge_ids[-max_steps:]
    node_ids = trajectory.node_ids[-(len(edge_ids) + 1):]
    lines: list[str] = []
    for index, edge_id in enumerate(edge_ids, start=1):
        parent = graph.get_node(node_ids[index - 1])
        child = graph.get_node(node_ids[index])
        edge = graph.get_edge(edge_id)
        lines.extend(
            [
                f"Step {index}: p{parent.id} -> p{child.id} [operator={edge.operator}]",
                f"  action: {edge.action}",
                (
                    f"  fitness: {format_fitness(parent.fitness)} -> "
                    f"{format_fitness(child.fitness)} "
                    f"(delta={edge.delta!s}, outcome={edge.outcome})"
                ),
            ]
        )
    return "\n".join(lines)


def _experience_block(batch: ExperienceBatch) -> str:
    if not batch.positives and not batch.negatives:
        return "No cross-trajectory action evidence is available yet."
    lines = ["Successful actions:"]
    lines.extend(_format_example(example) for example in batch.positives)
    if not batch.positives:
        lines.append("- (none)")
    lines.append("Failed actions:")
    lines.extend(_format_example(example) for example in batch.negatives)
    if not batch.negatives:
        lines.append("- (none)")
    return "\n".join(lines)


def _format_example(example) -> str:
    action = example.action
    if len(action) > _MAX_ACTION_CHARS:
        action = action[: _MAX_ACTION_CHARS - 3].rstrip() + "..."
    return (
        f"- [operator={example.operator}] action={action} "
        f"delta={example.delta:+.4g}"
    )


__all__ = ["build_action_prompt"]
