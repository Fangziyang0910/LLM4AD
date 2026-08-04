"""UCT, progressive widening, and branch-reference sampling for V8."""

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


def fitness_bounds(tree: SearchTree) -> tuple[float, float]:
    values = [node.directed_fitness for node in tree.nodes()]
    if not values:
        raise ValueError("fitness bounds require at least one program node")
    return min(values), max(values)


def normalize_value(value: float, lower: float, upper: float) -> float:
    if upper - lower <= 1e-12:
        return 0.5
    return (value - lower) / (upper - lower)


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
    lower: float,
    upper: float,
    exploration_constant: float,
    budget_ratio: float,
) -> float:
    exploitation = normalize_value(child.subtree_value, lower, upper)
    exploration = (
        exploration_constant
        * budget_ratio
        * math.sqrt(math.log(1 + parent_visits) / child.visit_count)
    )
    return exploitation + exploration


def widening_capacity(
    visit_count: int,
    *,
    actions_per_iteration: int = 2,
    alpha: float = 0.5,
) -> int:
    return max(actions_per_iteration, math.floor(visit_count**alpha))


def available_child_slots(
    node: ProgramNode,
    *,
    actions_per_iteration: int = 2,
    alpha: float = 0.5,
) -> int:
    capacity = widening_capacity(
        node.visit_count,
        actions_per_iteration=actions_per_iteration,
        alpha=alpha,
    )
    return max(0, min(actions_per_iteration, capacity - len(node.child_ids)))


def select_expansion_node(
    tree: SearchTree,
    *,
    rng: random.Random,
    total_budget: int | None,
    used_budget: int,
    exploration_constant: float,
    actions_per_iteration: int,
    widening_alpha: float,
) -> SelectionResult:
    if not tree.root.child_ids:
        raise ValueError("cannot select from an empty tree")
    lower, upper = fitness_bounds(tree)
    ratio = remaining_budget_ratio(total_budget, used_budget)
    path = [tree.root.id]
    steps: list[SelectionStep] = []
    parent_id = tree.root.id
    parent_visits = tree.root.visit_count
    child_ids = tree.root.child_ids
    while True:
        scored = [
            (
                child_id,
                uct_score(
                    child=tree.get_node(child_id),
                    parent_visits=parent_visits,
                    lower=lower,
                    upper=upper,
                    exploration_constant=exploration_constant,
                    budget_ratio=ratio,
                ),
            )
            for child_id in child_ids
        ]
        maximum = max(score for _, score in scored)
        tied = [child_id for child_id, score in scored if score == maximum]
        selected_id = rng.choice(tied)
        selected = tree.get_node(selected_id)
        selected_score = next(
            score for child_id, score in scored if child_id == selected_id
        )
        steps.append(
            SelectionStep(
                parent_id=parent_id,
                child_id=selected_id,
                normalized_value=normalize_value(selected.subtree_value, lower, upper),
                subtree_value=selected.subtree_value,
                visit_count=selected.visit_count,
                uct=selected_score,
            )
        )
        path.append(selected_id)
        if available_child_slots(
            selected,
            actions_per_iteration=actions_per_iteration,
            alpha=widening_alpha,
        ):
            return SelectionResult(selected_id, tuple(path), tuple(steps))
        parent_id = selected_id
        parent_visits = selected.visit_count
        child_ids = selected.child_ids
        if not child_ids:
            raise AssertionError(
                "a node without children must have an available child slot"
            )


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
    lower, upper = fitness_bounds(tree)
    values = [
        normalize_value(tree.get_node(branch).subtree_value, lower, upper)
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
    "available_child_slots",
    "fitness_bounds",
    "normalize_value",
    "reference_candidates",
    "remaining_budget_ratio",
    "sample_reference",
    "select_expansion_node",
    "uct_score",
    "widening_capacity",
]
