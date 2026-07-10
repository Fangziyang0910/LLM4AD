"""因果叙事 Context 构造（design §6）—— action prompt 三段式。

A. 因果叙事：trajectory 近 N 步的 (operator/mechanism/Δ/outcome/泛化) —— 比 lineage 多了 Δ 因果。
B. 蒸馏模式：PatternMemory 的跨路径机制 + 教训（ReEvo reflection / meta-scratchpad 的 verbal signal）。
C. 对比反馈：best vs worst（PathWise critic 的对比，但单次、无多 agent）。

novelty/init 算子走 initial-style（主循环分支），不经此函数。
"""
from __future__ import annotations

from ...base import Function
from .derivation_graph import DerivationGraph
from .pattern_memory import PatternMemory
from .prompt import fitness_direction_hint, format_fitness
from .schema import Trajectory


def build_action_prompt(
    *,
    graph: DerivationGraph,
    trajectory: Trajectory,
    base_node_id: int,
    base_reason: str,
    operator_name: str,
    operator_role: str,
    operator_constraint: str,
    pattern_memory: PatternMemory,
    contrast: dict | None,
    task_description: str,
    template_function: Function,
    action_count: int,
    maximize: bool,
    max_steps: int = 5,
) -> str:
    base_node = graph.get_node(base_node_id)
    sections: list[str] = []
    sections += ["[Task Description]", task_description.strip(), ""]
    sections += [
        "[Algorithm Improvement Context]",
        "The selected trajectory records attempted modifications and observed outcomes.",
        fitness_direction_hint(maximize),
        _causal_narrative(graph, trajectory, max_steps),
        "",
    ]
    sections += [
        "[Distilled Patterns]",
        _patterns_block(pattern_memory, operator=operator_name),
        "",
    ]
    sections += ["[Contrast Feedback]", _contrast_block(contrast), ""]
    sections += [
        "[Operator]",
        f"name={operator_name} role={operator_role}",
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
        _contract(template_function),
        "",
        "[Instruction]",
        "Use the trajectory, patterns, and contrast as a record of what worked and what did not.",
        f"Propose {action_count} next-step modifications for the base program above:",
        "- each modification must change one main algorithmic mechanism only;",
        "- follow the operator constraint;",
        "- avoid repeating directions that regressed or stayed unchanged.",
        "Each modification must be concrete and implementable. Do not output code or rationale.",
        f"Return only a numbered list of exactly {action_count} ideas, one per line.",
    ]
    return "\n".join(sections).strip()


def _causal_narrative(graph: DerivationGraph, trajectory: Trajectory, max_steps: int) -> str:
    if not trajectory.edge_ids:
        return "No previous improvement actions exist yet. This is an initial program."
    edge_ids = trajectory.edge_ids[-max_steps:]
    node_ids = trajectory.node_ids[-(len(edge_ids) + 1):]
    lines: list[str] = []
    for i, eid in enumerate(edge_ids):
        parent = graph.get_node(node_ids[i])
        child = graph.get_node(node_ids[i + 1])
        edge = graph.get_edge(eid)
        delta = edge.delta if edge.delta is not None else 0.0
        lines.append(
            f"Step {i}: p{parent.id} -> p{child.id}  [op={edge.operator} mech={edge.mechanism_tag}]"
        )
        lines.append(f"  action: {edge.action}")
        lines.append(
            f"  fitness: {format_fitness(parent.fitness)} -> {format_fitness(child.fitness)} "
            f"(Δ={delta:+.4g}, outcome={edge.outcome}, transfer_signal={edge.generalization_signal:.2f})"
        )
    return "\n".join(lines)


def _patterns_block(pattern_memory: PatternMemory, *, operator: str | None = None) -> str:
    mechs = pattern_memory.top_mechanisms(k=4)
    lessons = pattern_memory.top_lessons(operator=operator, k=3)
    lines: list[str] = []
    if mechs:
        lines.append("Recurring mechanisms (unique graph-edge evidence):")
        for m in mechs:
            detail = (
                f"  - {m.mechanism_tag}: "
                f"aggregate_improve_rate={m.generalization_score:.2f} "
                f"unique_support={len(m.support_ids)}"
            )
            attempts = pattern_memory.mechanism_attempts(
                m.mechanism_tag,
                operator=operator,
            )
            if operator is not None and attempts:
                rate = pattern_memory.mechanism_improve_rate(
                    m.mechanism_tag,
                    operator=operator,
                )
                detail += (
                    f" operator_improve_rate={rate:.2f}"
                    f" operator_support={attempts}"
                )
            lines.append(detail)
    else:
        lines.append("No distilled mechanisms yet.")
    if lessons:
        lines.append("Recent lessons:")
        for p in lessons:
            tag = "ANTI" if p.kind == "anti_pattern" else "OK"
            lines.append(f"  - ({tag}, {p.mechanism_tag}) {p.text}")
    return "\n".join(lines) if lines else "(none)"


def _contrast_block(contrast: dict | None) -> str:
    if not contrast:
        return "(not enough samples for contrast yet)"
    best, worst = contrast["best"], contrast["worst"]
    return (
        f"Recent best: mech={best['mechanism_tag']} fitness={format_fitness(best['fitness'])} "
        f"idea='{best['idea'][:60]}'\n"
        f"Recent worst: mech={worst['mechanism_tag']} fitness={format_fitness(worst['fitness'])} "
        f"idea='{worst['idea'][:60]}'"
    )


def _contract(template_function: Function) -> str:
    import copy
    target = copy.deepcopy(template_function)
    target.body = ""
    return str(target).rstrip()
