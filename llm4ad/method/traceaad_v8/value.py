"""UCT selection, adaptive expansion, and branch references for V8."""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .schema import ProgramNode, SelectionStep
from .tree import SearchTree


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_node_id: int
    path: tuple[int, ...]
    steps: tuple[SelectionStep, ...]


def fitness_reference_values(tree: SearchTree) -> tuple[float, ...]:
    values = sorted(node.directed_fitness for node in tree.nodes())
    if not values:
        raise ValueError("fitness ranks require at least one program node")
    return tuple(values)


def normalize_value(value: float, reference_values: tuple[float, ...]) -> float:
    """Map quality to a midrank percentile without sensitivity to outliers."""
    if len(reference_values) <= 1 or reference_values[0] == reference_values[-1]:
        return 0.5
    left = bisect_left(reference_values, value)
    right = bisect_right(reference_values, value)
    midrank = (left + right - 1) / 2
    return midrank / (len(reference_values) - 1)


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
    reference_values: tuple[float, ...],
    exploration_constant: float,
    budget_ratio: float,
) -> float:
    exploitation = normalize_value(child.subtree_value, reference_values)
    exploration = (
        exploration_constant
        * budget_ratio
        * math.sqrt(math.log(1 + parent_visits) / child.visit_count)
    )
    return exploitation + exploration


def expansion_batch_rewards(
    tree: SearchTree,
    node: ProgramNode,
    reference_values: tuple[float, ...],
) -> tuple[float, ...]:
    """Return the best eventual subtree quality opened by each successful batch."""
    rewards: dict[int, float] = {}
    for child_id in node.child_ids:
        child = tree.get_node(child_id)
        if child.batch_id is None:
            raise ValueError("non-root children must belong to an expansion batch")
        reward = normalize_value(child.subtree_value, reference_values)
        rewards[child.batch_id] = max(reward, rewards.get(child.batch_id, 0.0))
    if len(rewards) > node.expansion_count:
        raise ValueError("successful batches exceed recorded expansion attempts")
    return tuple(rewards[batch_id] for batch_id in sorted(rewards))


def expansion_quality(
    tree: SearchTree,
    node: ProgramNode,
    reference_values: tuple[float, ...],
    *,
    prior_weight: float = 1.0,
) -> float:
    """Estimate the value of asking the LLM for another child batch at this node."""
    if prior_weight <= 0:
        raise ValueError("expansion prior weight must be positive")
    prior = normalize_value(node.directed_fitness, reference_values)
    rewards = expansion_batch_rewards(tree, node, reference_values)
    return (prior_weight * prior + sum(rewards)) / (
        prior_weight + node.expansion_count
    )


def expansion_score(
    tree: SearchTree,
    node: ProgramNode,
    reference_values: tuple[float, ...],
    *,
    exploration_constant: float,
    budget_ratio: float,
    prior_weight: float = 1.0,
) -> tuple[float, float]:
    quality = expansion_quality(
        tree,
        node,
        reference_values,
        prior_weight=prior_weight,
    )
    exploration = (
        exploration_constant
        * budget_ratio
        * math.sqrt(
            math.log(1 + node.visit_count) / (1 + node.expansion_count)
        )
    )
    return quality, quality + exploration


def select_expansion_node(
    tree: SearchTree,
    *,
    rng: random.Random,
    total_budget: int | None,
    used_budget: int,
    exploration_constant: float,
    expansion_prior_weight: float = 1.0,
) -> SelectionResult:
    """Recursively compare opening a new branch with descending existing branches."""
    if not tree.root.child_ids:
        raise ValueError("cannot select from an empty tree")
    reference_values = fitness_reference_values(tree)
    ratio = remaining_budget_ratio(total_budget, used_budget)
    steps: list[SelectionStep] = []

    root_scored = _scored_children(
        tree,
        tree.root.child_ids,
        parent_visits=tree.root.visit_count,
        reference_values=reference_values,
        exploration_constant=exploration_constant,
        budget_ratio=ratio,
    )
    selected_id, selected_score = _choose_tied(root_scored, rng)
    selected = tree.get_node(selected_id)
    steps.append(
        SelectionStep(
            decision_node_id=tree.root.id,
            option="descend",
            target_node_id=selected.id,
            quality=normalize_value(selected.subtree_value, reference_values),
            raw_value=selected.subtree_value,
            option_visits=selected.visit_count,
            score=selected_score,
        )
    )
    path = [tree.root.id, selected.id]
    current = selected

    while True:
        new_quality, new_score = expansion_score(
            tree,
            current,
            reference_values,
            exploration_constant=exploration_constant,
            budget_ratio=ratio,
            prior_weight=expansion_prior_weight,
        )
        choices: list[tuple[int | None, float]] = [(None, new_score)]
        choices.extend(
            _scored_children(
                tree,
                current.child_ids,
                parent_visits=current.visit_count,
                reference_values=reference_values,
                exploration_constant=exploration_constant,
                budget_ratio=ratio,
            )
        )
        target_id, selected_score = _choose_tied(choices, rng)
        if target_id is None:
            steps.append(
                SelectionStep(
                    decision_node_id=current.id,
                    option="expand",
                    target_node_id=None,
                    quality=new_quality,
                    raw_value=None,
                    option_visits=current.expansion_count,
                    score=selected_score,
                )
            )
            return SelectionResult(current.id, tuple(path), tuple(steps))

        current = tree.get_node(target_id)
        path.append(current.id)
        steps.append(
            SelectionStep(
                decision_node_id=current.parent_id,
                option="descend",
                target_node_id=current.id,
                quality=normalize_value(current.subtree_value, reference_values),
                raw_value=current.subtree_value,
                option_visits=current.visit_count,
                score=selected_score,
            )
        )


def _scored_children(
    tree: SearchTree,
    child_ids: list[int],
    *,
    parent_visits: int,
    reference_values: tuple[float, ...],
    exploration_constant: float,
    budget_ratio: float,
) -> list[tuple[int | None, float]]:
    return [
        (
            child_id,
            uct_score(
                child=tree.get_node(child_id),
                parent_visits=parent_visits,
                reference_values=reference_values,
                exploration_constant=exploration_constant,
                budget_ratio=budget_ratio,
            ),
        )
        for child_id in child_ids
    ]


def _choose_tied(
    scored: list[tuple[int | None, float]], rng: random.Random
) -> tuple[int | None, float]:
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
    reference_values = fitness_reference_values(tree)
    values = [
        normalize_value(tree.get_node(branch).subtree_value, reference_values)
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
    "expansion_batch_rewards",
    "expansion_quality",
    "expansion_score",
    "fitness_reference_values",
    "normalize_value",
    "reference_candidates",
    "remaining_budget_ratio",
    "sample_reference",
    "select_expansion_node",
    "uct_score",
]
