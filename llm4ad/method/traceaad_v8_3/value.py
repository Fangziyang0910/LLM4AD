"""V8.3 directed fitness, trajectory signals, and recursive selection."""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right

from .schema import SelectionResult, SelectionStep, TreeNode
from .tree import SearchTree


def directed(node: TreeNode, maximize: bool) -> float:
    value = node.algorithm.fitness
    return value if maximize else -value


def rank_value(value: float, values: tuple[float, ...]) -> float:
    if len(values) <= 1 or values[0] == values[-1]:
        return 0.5
    left = bisect_left(values, value)
    right = bisect_right(values, value)
    return (left + right - 1) / (2 * (len(values) - 1))


def fitness_ranks(tree: SearchTree, maximize: bool) -> tuple[float, ...]:
    return tuple(sorted(directed(node, maximize) for node in tree.nodes()))


def node_fitness(tree: SearchTree, node: TreeNode, maximize: bool) -> float:
    return rank_value(directed(node, maximize), fitness_ranks(tree, maximize))


def subtree_directed(tree: SearchTree, node_id: int, maximize: bool) -> float:
    node = tree.get_node(node_id)
    values = [directed(node, maximize)]
    values.extend(subtree_directed(tree, child_id, maximize) for child_id in node.child_ids)
    return max(values)


def subtree_rank(tree: SearchTree, node_id: int, maximize: bool) -> float:
    return rank_value(subtree_directed(tree, node_id, maximize), fitness_ranks(tree, maximize))


def trajectory_progress(
    tree: SearchTree,
    node_id: int,
    maximize: bool,
    window: int = 8,
) -> tuple[float, float, float]:
    path = tree.ancestor_node_ids(node_id)
    values = [node_fitness(tree, tree.get_node(item), maximize) for item in path]
    start = max(0, len(values) - window - 1)
    prefix_best = values[0]
    bests = [prefix_best]
    for value in values[1:]:
        prefix_best = max(prefix_best, value)
        bests.append(prefix_best)
    base = bests[start]
    end = bests[-1]
    steps = len(bests) - 1 - start
    if steps <= 0:
        return 0.0, 0.0, 0.0
    progress = end - base
    count = sum(bests[index] > bests[index - 1] for index in range(start + 1, len(bests)))
    frequency = count / steps
    return progress, frequency, math.sqrt(max(0.0, progress * frequency))


def expansion_prior(
    tree: SearchTree,
    node: TreeNode,
    maximize: bool,
    kappa: float,
    window: int,
) -> float:
    _, _, momentum = trajectory_progress(tree, node.id, maximize, window)
    return min(1.0, node_fitness(tree, node, maximize) + kappa * momentum)


def expansion_reward(
    tree: SearchTree,
    parent: TreeNode,
    child: TreeNode,
    maximize: bool,
    rho: float,
) -> float:
    # ``new_child(parent)`` is an action whose outcome is the child created by
    # that attempt.  Descendants belong to the separate ``descend(child)``
    # action and must not retroactively improve this expansion arm.
    edge = tree.get_edge(parent.id, child.id)
    child_quality = (
        edge.child_quality
        if edge.child_quality is not None
        else node_fitness(tree, child, maximize)
    )
    parent_quality = (
        edge.parent_quality
        if edge.parent_quality is not None
        else node_fitness(tree, parent, maximize)
    )
    development = max(0.0, child_quality - parent_quality)
    return (1.0 - rho) * child_quality + rho * development


def new_child_quality(
    tree: SearchTree,
    node: TreeNode,
    maximize: bool,
    beta: float,
    rho: float,
    kappa: float,
    window: int,
) -> float:
    prior = expansion_prior(tree, node, maximize, kappa, window)
    rewards = [
        expansion_reward(tree, node, tree.get_node(child), maximize, rho)
        for child in node.child_ids
    ]
    return (beta * prior + sum(rewards)) / (beta + node.expansion_attempts)


def budget_ratio(total: int | None, used: int) -> float:
    if total is None:
        return 1.0
    if total <= 0:
        return 0.0
    return min(1.0, max(0.0, (total - used) / total))


def _exploration(parent: TreeNode, visits: int, constant: float, ratio: float) -> float:
    return constant * ratio * math.sqrt(math.log(1 + visits) / max(1, parent.visit_count))


def select_expansion_node(
    tree: SearchTree,
    *,
    maximize: bool,
    total_budget: int | None,
    used_budget: int,
    exploration_constant: float,
    beta: float,
    rho: float,
    kappa: float,
    window: int,
    rng: random.Random,
    max_depth: int = 10,
    widening_alpha: float = 0.5,
) -> SelectionResult:
    if not tree.root.child_ids:
        raise ValueError("cannot select from an empty tree")
    if not 0 < widening_alpha <= 1:
        raise ValueError("widening_alpha must be in (0, 1]")
    ratio = budget_ratio(total_budget, used_budget)
    root_scores = []
    for child_id in tree.root.child_ids:
        child = tree.get_node(child_id)
        if child.depth >= max_depth:
            continue
        quality = subtree_rank(tree, child.id, maximize)
        explore = exploration_constant * ratio * math.sqrt(
            math.log(1 + tree.root.visit_count) / max(1, child.visit_count)
        )
        root_scores.append((child.id, quality + explore, quality, explore))
    current_id, score, quality, explore = _choose(root_scores, rng)
    path = [tree.root.id, current_id]
    steps = [SelectionStep(tree.root.id, "descend", current_id, score, quality, explore)]

    while True:
        current = tree.get_node(current_id)
        # Keep the original MCTS-AHD progressive-widening invariant: a node
        # must receive a new direct-child attempt before its existing subtree
        # can consume all later visits.  The value competition remains active
        # once the width quota is satisfied.
        width_target = max(1, int(current.visit_count**widening_alpha))
        if len(current.child_ids) < width_target:
            new_quality = new_child_quality(tree, current, maximize, beta, rho, kappa, window)
            new_explore = exploration_constant * ratio * math.sqrt(
                math.log(1 + current.visit_count) / (1 + current.expansion_attempts)
            )
            steps.append(
                SelectionStep(
                    current.id,
                    "new_child",
                    None,
                    new_quality + new_explore,
                    new_quality,
                    new_explore,
                )
            )
            return SelectionResult(current.id, tuple(path), tuple(steps))
        new_quality = new_child_quality(tree, current, maximize, beta, rho, kappa, window)
        new_explore = exploration_constant * ratio * math.sqrt(
            math.log(1 + current.visit_count) / (1 + current.expansion_attempts)
        )
        choices = [(None, new_quality + new_explore, new_quality, new_explore)]
        for child_id in current.child_ids:
            child = tree.get_node(child_id)
            if child.depth >= max_depth:
                continue
            child_quality = subtree_rank(tree, child.id, maximize)
            child_explore = exploration_constant * ratio * math.sqrt(
                math.log(1 + current.visit_count) / max(1, child.visit_count)
            )
            choices.append((child.id, child_quality + child_explore, child_quality, child_explore))
        target, score, quality, explore = _choose(choices, rng)
        if target is None:
            steps.append(SelectionStep(current.id, "new_child", None, score, quality, explore))
            return SelectionResult(current.id, tuple(path), tuple(steps))
        steps.append(SelectionStep(current.id, "descend", target, score, quality, explore))
        current_id = target
        path.append(target)


def _choose(choices, rng: random.Random):
    maximum = max(item[1] for item in choices)
    tied = [item for item in choices if item[1] == maximum]
    return rng.choice(tied)


def reference_candidates(tree: SearchTree, current_id: int) -> tuple[TreeNode, ...]:
    return tuple(node for node in tree.nodes() if node.id != current_id)


def sample_reference(
    candidates: tuple[TreeNode, ...],
    maximize: bool,
    temperature: float,
    rng: random.Random,
) -> TreeNode:
    if not candidates:
        raise ValueError("reference candidates cannot be empty")
    if len(candidates) == 1:
        return candidates[0]
    values = [directed(node, maximize) for node in candidates]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    if scale == 0.0:
        return rng.choice(list(candidates))
    logits = [(value - mean) / (scale + 1e-12) / temperature for value in values]
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    threshold = rng.random() * sum(weights)
    for node, weight in zip(candidates, weights, strict=True):
        threshold -= weight
        if threshold <= 0:
            return node
    return candidates[-1]


__all__ = [
    "budget_ratio",
    "directed",
    "expansion_prior",
    "expansion_reward",
    "fitness_ranks",
    "new_child_quality",
    "node_fitness",
    "reference_candidates",
    "sample_reference",
    "select_expansion_node",
    "subtree_directed",
    "subtree_rank",
    "trajectory_progress",
]
