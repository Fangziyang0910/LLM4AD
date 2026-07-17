"""因果叙事 Context 构造 —— action prompt 三段式。

A. 因果叙事：trajectory 近 N 步的 (operator/Δf/ΔC/ΔR/outcome)。
B. 边级经验：ExperienceMemory 的成功/失败 action 示例。
C. 对比反馈：best vs worst（idea + fitness）。

novelty/init 算子走 initial-style（主循环分支），不经此函数。
"""
from __future__ import annotations

from ...base import Function
from .derivation_graph import DerivationGraph
from .experience_memory import ExperienceMemory
from .prompt import fitness_direction_hint, format_fitness
from .schema import ExperienceBatch, ProgramNode, Trajectory

_DEFAULT_ACTION_CHARS = 300


def build_action_prompt(
    *,
    graph: DerivationGraph,
    trajectory: Trajectory,
    base_node_id: int,
    base_reason: str,
    operator_name: str,
    operator_role: str,
    operator_constraint: str,
    experience_memory: ExperienceMemory,
    contrast: dict | None,
    task_description: str,
    template_function: Function,
    action_count: int,
    maximize: bool,
    max_steps: int = 5,
    positive_k: int = 2,
    negative_k: int = 2,
    max_action_chars: int = _DEFAULT_ACTION_CHARS,
) -> str:
    base_node = graph.get_node(base_node_id)
    batch = experience_memory.examples(
        operator=operator_name,
        positive_k=positive_k,
        negative_k=negative_k,
    )
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
        "[Past Action Evidence]",
        _experience_block(batch, max_action_chars=max_action_chars),
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
        _structure_summary(base_node),
        "Code:",
        "```python",
        base_node.code.rstrip(),
        "```",
        "",
        "[Target Function Contract]",
        _contract(template_function),
        "",
        "[Instruction]",
        "Use the trajectory, past action evidence, and contrast as a record of what worked and what did not.",
        f"Propose {action_count} next-step modifications for the base program above:",
        "- each modification must change one main algorithmic idea only;",
        "- follow the operator constraint;",
        "- avoid repeating directions that regressed or stayed unchanged;",
        "- prefer changes that improve fitness without unnecessary complexity or runtime growth.",
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
        delta_c = child.complexity - parent.complexity
        delta_r = child.runtime - parent.runtime
        lines.append(
            f"Step {i}: p{parent.id} -> p{child.id}  [op={edge.operator}]"
        )
        lines.append(f"  action: {edge.action}")
        lines.append(
            f"  fitness: {format_fitness(parent.fitness)} -> {format_fitness(child.fitness)} "
            f"(Δ={delta:+.4g}, outcome={edge.outcome})"
        )
        lines.append(
            f"  structure: {_short_metrics(parent)} -> {_short_metrics(child)} "
            f"(ΔC={delta_c:+.3g})"
        )
        lines.append(
            f"  runtime: {_format_runtime(parent.runtime)} -> {_format_runtime(child.runtime)} "
            f"(ΔR={delta_r:+.3g}s)"
        )
    return "\n".join(lines)


def _structure_summary(node: ProgramNode) -> str:
    return (
        f"Structure/runtime: {_short_metrics(node)}; "
        f"runtime={_format_runtime(node.runtime)}; "
        f"complexity_score={node.complexity:.3g}"
    )


def _short_metrics(node: ProgramNode) -> str:
    metrics = node.complexity_metrics
    cc = metrics.get("cyclomatic_complexity")
    nest = metrics.get("max_nesting_depth")
    loc = metrics.get("lines_of_code")
    parts: list[str] = []
    if cc is not None:
        parts.append(f"cc={cc:.3g}")
    if nest is not None:
        parts.append(f"nest={nest:.3g}")
    if loc is not None:
        parts.append(f"loc={loc:.3g}")
    if not parts:
        parts.append(f"score={node.complexity:.3g}")
    return " ".join(parts)


def _format_runtime(runtime: float) -> str:
    if runtime is None or runtime <= 0:
        return "n/a"
    if runtime < 0.001:
        return f"{runtime * 1000:.2g}ms"
    return f"{runtime:.3g}s"


def _experience_block(
    batch: ExperienceBatch,
    *,
    max_action_chars: int = _DEFAULT_ACTION_CHARS,
) -> str:
    if not batch.positives and not batch.negatives:
        return "No successful or failed past actions recorded yet."
    lines: list[str] = []
    lines.append("Successful past actions:")
    if batch.positives:
        for example in batch.positives:
            lines.append(_format_example(example, max_action_chars=max_action_chars))
    else:
        lines.append("- (none)")
    lines.append("")
    lines.append("Failed past actions:")
    if batch.negatives:
        for example in batch.negatives:
            lines.append(_format_example(example, max_action_chars=max_action_chars))
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _format_example(example, *, max_action_chars: int) -> str:
    action = example.action
    if len(action) > max_action_chars:
        action = action[: max_action_chars - 3].rstrip() + "..."
    return (
        f"- [operator={example.operator}] action={action} "
        f"delta={example.delta:+.4g}"
    )


def _contrast_block(contrast: dict | None) -> str:
    if not contrast:
        return "(not enough samples for contrast yet)"
    best, worst = contrast["best"], contrast["worst"]
    return (
        f"Recent best: fitness={format_fitness(best['fitness'])} "
        f"idea='{best['idea'][:60]}'\n"
        f"Recent worst: fitness={format_fitness(worst['fitness'])} "
        f"idea='{worst['idea'][:60]}'"
    )


def _contract(template_function: Function) -> str:
    import copy
    signature = copy.deepcopy(template_function)
    signature.body = ""
    return f"Only evolve:\n```python\n{signature}\n```"
