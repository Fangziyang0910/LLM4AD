"""MCTS-AHD-aligned UCT selection and progressive widening for V9.1."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schema import ProgramNode, SelectionStep
from .tree import SearchTree


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_node_id: int
    path: tuple[int, ...]
    steps: tuple[SelectionStep, ...]


def quality_bounds(tree: SearchTree) -> tuple[float | None, float | None]:
    return tree.q_min, tree.q_max


def normalize_value(
    value: float,
    q_min: float | None,
    q_max: float | None,
) -> float:
    """Match MCTS-AHD's min-max normalization of continuation values."""
    if q_min is None or q_max is None or abs(q_max - q_min) < 1e-10:
        return 0.0
    return (value - q_min) / (q_max - q_min)


def remaining_budget_ratio(total: int | None, used: int) -> float:
    if total is None:
        return 1.0
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, (total - used) / total))


def uct_score(
    *,
    child: ProgramNode,
    parent_visits: int,
    q_min: float | None,
    q_max: float | None,
    exploration_constant: float,
    budget_ratio: float,
) -> float:
    exploitation = normalize_value(child.subtree_value, q_min, q_max)
    exploration = exploration_constant * budget_ratio * math.sqrt(
        math.log(parent_visits + 1) / child.visit_count
    )
    return exploitation + exploration


def progressive_widening_allowed(node: ProgramNode, alpha: float) -> bool:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return int(node.visit_count**alpha) > len(node.child_ids)


def select_expansion_node(
    tree: SearchTree,
    *,
    rng: random.Random,
    total_budget: int | None,
    used_budget: int,
    exploration_constant: float,
    alpha: float,
) -> SelectionResult:
    """Select a leaf or a node whose progressive-widening budget is open."""
    if not tree.root.child_ids:
        raise ValueError("cannot select from an empty tree")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    q_min, q_max = quality_bounds(tree)
    ratio = remaining_budget_ratio(total_budget, used_budget)
    steps: list[SelectionStep] = []
    path = [tree.root.id]
    current = tree.root

    while True:
        if not current.child_ids or progressive_widening_allowed(current, alpha):
            quality = (
                0.0
                if current.subtree_value is None
                else normalize_value(current.subtree_value, q_min, q_max)
            )
            steps.append(
                SelectionStep(
                    decision_node_id=current.id,
                    option="expand",
                    target_node_id=None,
                    quality=quality,
                    raw_value=current.subtree_value,
                    option_visits=getattr(current, "expansion_count", current.visit_count),
                    score=quality,
                )
            )
            return SelectionResult(
                current.id,
                tuple(path),
                tuple(steps),
            )

        scored = [
            (
                child_id,
                uct_score(
                    child=tree.get_node(child_id),
                    parent_visits=current.visit_count,
                    q_min=q_min,
                    q_max=q_max,
                    exploration_constant=exploration_constant,
                    budget_ratio=ratio,
                ),
            )
            for child_id in current.child_ids
        ]
        target_id, score = _choose_tied(scored, rng)
        child = tree.get_node(target_id)
        steps.append(
            SelectionStep(
                decision_node_id=current.id,
                option="descend",
                target_node_id=child.id,
                quality=normalize_value(child.subtree_value, q_min, q_max),
                raw_value=child.subtree_value,
                option_visits=child.visit_count,
                score=score,
            )
        )
        current = child
        path.append(child.id)


def _choose_tied(
    scored: list[tuple[int, float]], rng: random.Random
) -> tuple[int, float]:
    maximum = max(score for _, score in scored)
    tied = [target_id for target_id, score in scored if score == maximum]
    return rng.choice(tied), maximum


def reference_candidates(
    tree: SearchTree, main_node_id: int
) -> tuple[tuple[int, ProgramNode], ...]:
    main = tree.get_node(main_node_id)
    main_branch = tree.root_branch_id(main_node_id)
    candidates: list[tuple[int, ProgramNode]] = []
    for branch_id in tree.root.child_ids:
        if branch_id == main_branch:
            continue
        representative = tree.subtree_best(branch_id)
        if representative.code_hash == main.code_hash:
            continue
        candidates.append((branch_id, representative))
    return tuple(candidates)


def sample_reference(
    tree: SearchTree,
    main_node_id: int,
    *,
    temperature: float,
    rng: random.Random,
) -> tuple[int, ProgramNode] | None:
    candidates = reference_candidates(tree, main_node_id)
    if not candidates:
        return None
    q_min, q_max = quality_bounds(tree)
    values = [
        normalize_value(tree.get_node(branch).subtree_value, q_min, q_max)
        for branch, _ in candidates
    ]
    maximum = max(values)
    weights = [math.exp((value - maximum) / temperature) for value in values]
    threshold = rng.random() * sum(weights)
    cumulative = 0.0
    for candidate, weight in zip(candidates, weights, strict=True):
        cumulative += weight
        if threshold <= cumulative:
            return candidate
    return candidates[-1]


__all__ = [
    "SelectionResult",
    "normalize_value",
    "progressive_widening_allowed",
    "quality_bounds",
    "reference_candidates",
    "remaining_budget_ratio",
    "sample_reference",
    "select_expansion_node",
    "uct_score",
]
