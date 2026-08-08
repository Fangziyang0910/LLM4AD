"""Render bounded tree histories and direct-branch evidence for V9.1."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from ...base import Function
from .prompt import fitness_direction_hint, format_fitness
from .schema import ProgramNode
from .tree import SearchTree, is_node_better


@dataclass(frozen=True, slots=True)
class RenderedHistory:
    text: str
    formation_edge_ids: tuple[int, ...]
    direct_child_edge_ids: tuple[int, ...]


def formation_history(
    tree: SearchTree,
    node_id: int,
    *,
    max_edges: int = 8,
) -> RenderedHistory:
    edge_ids = tree.ancestor_edge_ids(node_id)
    selected = edge_ids[-max_edges:] if max_edges > 0 else ()
    lines = ["[How This Program Was Reached]"]
    if not selected:
        lines.append("This is an initial program with no previous changes.")
    else:
        for position, edge_id in enumerate(selected, start=1):
            lines.extend(_render_formation_edge(tree, edge_id, position))
    return RenderedHistory("\n".join(lines), tuple(selected), ())


def select_direct_children(
    tree: SearchTree,
    node_id: int,
    *,
    limit: int = 8,
    top_count: int = 4,
) -> tuple[int, ...]:
    children = [
        tree.get_node(child_id) for child_id in tree.get_node(node_id).child_ids
    ]
    if not children or limit <= 0:
        return ()
    ranked: list[ProgramNode] = []
    for child in children:
        inserted = False
        child_best = tree.subtree_best(child.id)
        for index, incumbent in enumerate(ranked):
            if is_node_better(child_best, tree.subtree_best(incumbent.id)):
                ranked.insert(index, child)
                inserted = True
                break
        if not inserted:
            ranked.append(child)
    chosen = ranked[: min(top_count, limit)]
    chosen_ids = {child.id for child in chosen}
    for child in sorted(children, key=lambda item: item.creation_order, reverse=True):
        if len(chosen) >= limit:
            break
        if child.id not in chosen_ids:
            chosen.append(child)
            chosen_ids.add(child.id)
    return tuple(
        child.id for child in sorted(chosen, key=lambda item: item.creation_order)
    )


def node_history(
    tree: SearchTree,
    node_id: int,
    *,
    ancestor_limit: int = 8,
    direct_child_limit: int = 8,
    direct_child_top_count: int = 4,
) -> RenderedHistory:
    formation = formation_history(tree, node_id, max_edges=ancestor_limit)
    direct_children = select_direct_children(
        tree,
        node_id,
        limit=direct_child_limit,
        top_count=direct_child_top_count,
    )
    lines = [formation.text, "[Previously Tested From This Program]"]
    if not direct_children:
        lines.append(
            "No direct modifications have been evaluated from this program yet."
        )
    else:
        for position, child_id in enumerate(direct_children, start=1):
            child = tree.get_node(child_id)
            edge = tree.get_edge(child.incoming_edge_id)  # type: ignore[arg-type]
            best = tree.subtree_best(child_id)
            lines.extend(
                [
                    (
                        f"Branch {position}: {edge.outcome}; immediate fitness "
                        f"{format_fitness(child.fitness)}; subtree-best fitness "
                        f"{format_fitness(best.fitness)}; depth to subtree best "
                        f"{best.depth - child.depth}"
                    ),
                    f"  Implemented change: {_one_line(child.idea, 360)}",
                ]
            )
    return RenderedHistory(
        text="\n".join(lines),
        formation_edge_ids=formation.formation_edge_ids,
        direct_child_edge_ids=tuple(
            tree.get_node(child_id).incoming_edge_id  # type: ignore[misc]
            for child_id in direct_children
        ),
    )


def build_code_prompt(
    *,
    current_node: ProgramNode,
    current_history: str,
    operator_constraint: str,
    task_description: str,
    template_function: Function,
    maximize: bool,
    candidate_index: int,
    candidate_count: int,
    reference_node: ProgramNode | None = None,
    reference_history: str = "",
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    sections = [
        "[Task]",
        task_description.strip(),
        fitness_direction_hint(maximize),
        "",
        current_history.strip(),
        "",
        "[Current Program]",
        f"Current fitness: {format_fitness(current_node.fitness)}",
        "```python",
        current_node.code.rstrip(),
        "```",
    ]
    if reference_node is not None:
        sections.extend(
            [
                "",
                "[Reference Root Branch History]",
                reference_history.strip(),
                "",
                "[Reference Program]",
                f"Reference fitness: {format_fitness(reference_node.fitness)}",
                "```python",
                reference_node.code.rstrip(),
                "```",
            ]
        )
    sections.extend(
        [
            "",
            "[Improvement Direction]",
            operator_constraint.strip(),
            "",
            "[Target Function]",
            str(target).rstrip(),
            "",
            "[Instruction]",
            (
                f"Generate candidate {candidate_index + 1} of {candidate_count} by directly "
                "improving the current program."
            ),
            "Use the histories as evidence and do not repeat tested changes.",
            "Choose one concrete algorithmic change and implement it completely.",
            "When a reference is shown, adapt it only according to the improvement direction.",
            "Keep the target function signature and contract unchanged.",
            "Return exactly one complete implementation.",
            "Imports from the current program remain available; small top-level helpers are allowed.",
            "Output only:",
            "Idea: <one sentence describing the implemented change>",
            "Code:",
            "```python",
            "<complete function implementation>",
            "```",
        ]
    )
    return "\n".join(sections).strip()


def _render_formation_edge(tree: SearchTree, edge_id: int, position: int) -> list[str]:
    edge = tree.get_edge(edge_id)
    parent = tree.get_node(edge.parent_id)
    child = tree.get_node(edge.child_id)
    return [
        (
            f"Step {position}: {edge.outcome}; fitness "
            f"{format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}; "
            f"global breakthrough={'yes' if edge.new_global_best else 'no'}"
        ),
        f"  Implemented change: {_one_line(edge.implemented_idea, 360)}",
        (
            f"  Code change: {edge.code_change_ratio:.0%}; "
            f"LOC {parent.program_loc} -> {child.program_loc}"
        ),
    ]


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "RenderedHistory",
    "build_code_prompt",
    "formation_history",
    "node_history",
    "select_direct_children",
]
