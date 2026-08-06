"""Minimal local exploration context for V8.3."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import TreeNode
from .tree import SearchTree
from .value import directed, subtree_directed


@dataclass(frozen=True, slots=True)
class LocalContext:
    text: str
    formation_edges: tuple[tuple[int, int], ...]
    direct_edges: tuple[tuple[int, int], ...]


def build_local_context(
    tree: SearchTree,
    node: TreeNode,
    *,
    maximize: bool,
    max_formation_edges: int = 3,
    max_direct_children: int = 3,
) -> LocalContext:
    all_formation = list(tree.ancestor_edge_ids(node.id))
    formation = (
        all_formation[-max_formation_edges:]
        if max_formation_edges > 0
        else []
    )
    children = [tree.get_node(child_id) for child_id in node.child_ids]
    selected: list[TreeNode] = []

    if children:
        # For an exactly tied subtree best value, expose the newer branch as
        # specified by the local-context protocol.
        best = max(
            children,
            key=lambda child: (subtree_directed(tree, child.id, maximize), child.creation_order),
        )
        selected.append(best)
        recent_unimproved = [
            child for child in sorted(children, key=lambda item: item.creation_order, reverse=True)
            if directed(child, maximize) <= directed(node, maximize)
        ]
        for child in recent_unimproved:
            if child.id not in {item.id for item in selected}:
                selected.append(child)
                break
        for child in sorted(children, key=lambda item: item.creation_order, reverse=True):
            if child.id not in {item.id for item in selected}:
                selected.append(child)
            if len(selected) >= max_direct_children:
                break
    selected = selected[:max_direct_children]

    lines = ["[Recent Formation History]"]
    if not formation:
        if all_formation:
            lines.append("Formation history was shortened to fit the context limit.")
        else:
            lines.append("Initial algorithm; no previous structural modification.")
    for position, (parent_id, child_id) in enumerate(formation, start=1):
        parent = tree.get_node(parent_id)
        child = tree.get_node(child_id)
        edge = tree.get_edge(parent_id, child_id)
        outcome = _outcome(directed(child, maximize) - directed(parent, maximize))
        lines.extend([
            f"Step {position}: {edge.operator}; {parent.algorithm.fitness:.6g} -> {child.algorithm.fitness:.6g}; {outcome}",
            f"  Idea: {child.algorithm.design_idea}",
            f"  Description: {child.algorithm.description}",
        ])
    lines.append("[Representative Tested Branches]")
    if not selected:
        if children:
            lines.append("Representative branches were shortened to fit the context limit.")
        else:
            lines.append("No valid direct child has been evaluated from this algorithm.")
    direct_edges: list[tuple[int, int]] = []
    for child in selected:
        edge = tree.get_edge(node.id, child.id)
        direct_edges.append((node.id, child.id))
        outcome = _outcome(directed(child, maximize) - directed(node, maximize))
        lines.extend([
            f"Branch: {edge.operator}; {node.algorithm.fitness:.6g} -> {child.algorithm.fitness:.6g}; {outcome}",
            f"  Idea: {child.algorithm.design_idea}",
            f"  Description: {child.algorithm.description}",
        ])
        best_directed = subtree_directed(tree, child.id, maximize)
        if best_directed > directed(child, maximize):
            best_fitness = tree.subtree_value(child.id, maximize=maximize)
            lines.append(f"  Later subtree-best fitness: {best_fitness:.6g}")
    return LocalContext("\n".join(lines), tuple(formation), tuple(direct_edges))


def _outcome(delta: float) -> str:
    if delta > 0:
        return "improve"
    if delta < 0:
        return "regress"
    return "plateau"


__all__ = ["LocalContext", "build_local_context"]
