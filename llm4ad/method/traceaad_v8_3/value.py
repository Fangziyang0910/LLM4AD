"""Trajectory prior, route credit, and recursive selection for TraceAAD V8.3."""

from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from .schema import SelectionResult, SelectionStep, TreeNode
from .tree import SearchTree


@dataclass(frozen=True, slots=True)
class SearchWeights:
    endpoint_quality: float = 0.7
    path_best_quality: float = 0.3
    history_quality: float = 0.8
    recent_trend: float = 0.2
    trend_discount: float = 0.8
    trend_window: int = 8
    positive_threshold: float = 1e-12
    reward_quality: float = 0.75
    reward_gain: float = 0.25
    prior_weight: float = 1.0
    exploration_constant: float = 0.25
    selection_temperature: float = 0.2
    widening_alpha: float = 0.5

    def __post_init__(self) -> None:
        pairs = (
            (self.endpoint_quality, self.path_best_quality, "history"),
            (self.history_quality, self.recent_trend, "prior"),
            (self.reward_quality, self.reward_gain, "reward"),
        )
        for left, right, name in pairs:
            if left < 0 or right < 0 or left + right <= 0:
                raise ValueError(
                    f"{name} weights must be non-negative with a positive sum"
                )
        if not 0 < self.trend_discount <= 1:
            raise ValueError("trend_discount must be in (0, 1]")
        if self.trend_window <= 0:
            raise ValueError("trend_window must be positive")
        if self.positive_threshold < 0:
            raise ValueError("positive_threshold must be non-negative")
        if self.prior_weight <= 0:
            raise ValueError("prior_weight must be positive")
        if self.exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        if not 0 < self.widening_alpha <= 1:
            raise ValueError("widening_alpha must be in (0, 1]")


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
    values.extend(
        subtree_directed(tree, child_id, maximize) for child_id in node.child_ids
    )
    return max(values)


def subtree_rank(tree: SearchTree, node_id: int, maximize: bool) -> float:
    return rank_value(
        subtree_directed(tree, node_id, maximize), fitness_ranks(tree, maximize)
    )


def path_best_rank(tree: SearchTree, node_id: int, maximize: bool) -> float:
    path = tree.ancestor_node_ids(node_id)
    best = max(directed(tree.get_node(item), maximize) for item in path)
    return rank_value(best, fitness_ranks(tree, maximize))


def trajectory_trend(
    tree: SearchTree,
    node_id: int,
    maximize: bool,
    weights: SearchWeights,
) -> float:
    path = tree.ancestor_node_ids(node_id)
    if len(path) <= 1:
        return 0.5
    path = path[-(weights.trend_window + 1) :]
    signals: list[float] = []
    for parent_id, child_id in zip(path, path[1:]):
        delta = directed(tree.get_node(child_id), maximize) - directed(
            tree.get_node(parent_id), maximize
        )
        if abs(delta) <= weights.positive_threshold:
            signals.append(0.5)
        else:
            signals.append(1.0 if delta > 0 else 0.0)
    discounts = [
        weights.trend_discount ** (len(signals) - 1 - index)
        for index in range(len(signals))
    ]
    return sum(
        weight * signal for weight, signal in zip(discounts, signals, strict=True)
    ) / sum(discounts)


def trajectory_prior(
    tree: SearchTree,
    node: TreeNode,
    maximize: bool,
    weights: SearchWeights,
) -> float:
    history = (
        weights.endpoint_quality * node_fitness(tree, node, maximize)
        + weights.path_best_quality * path_best_rank(tree, node.id, maximize)
    ) / (weights.endpoint_quality + weights.path_best_quality)
    trend = trajectory_trend(tree, node.id, maximize, weights)
    return (weights.history_quality * history + weights.recent_trend * trend) / (
        weights.history_quality + weights.recent_trend
    )


def route_quality(
    tree: SearchTree,
    node: TreeNode,
    maximize: bool,
    weights: SearchWeights,
) -> float:
    prior = trajectory_prior(tree, node, maximize, weights)
    return (weights.prior_weight * prior + node.credit_sum) / (
        weights.prior_weight + node.credit_count
    )


def expansion_reward(
    tree: SearchTree,
    parent: TreeNode,
    child: TreeNode,
    maximize: bool,
    weights: SearchWeights,
) -> float:
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
    gain = max(0.0, child_quality - parent_quality)
    return (weights.reward_quality * child_quality + weights.reward_gain * gain) / (
        weights.reward_quality + weights.reward_gain
    )


def _node_step(
    tree: SearchTree,
    *,
    decision_node_id: int,
    child: TreeNode,
    parent_visits: int,
    maximize: bool,
    weights: SearchWeights,
) -> SelectionStep:
    prior = trajectory_prior(tree, child, maximize, weights)
    credit_mean = child.credit_sum / child.credit_count if child.credit_count else 0.0
    quality = route_quality(tree, child, maximize, weights)
    exploration = weights.exploration_constant * math.sqrt(
        math.log(1 + parent_visits) / max(1, child.visit_count)
    )
    return SelectionStep(
        decision_node_id=decision_node_id,
        option="descend",
        target_node_id=child.id,
        score=quality + exploration,
        quality=quality,
        exploration=exploration,
        probability=0.0,
        prior=prior,
        credit_mean=credit_mean,
        credit_count=child.credit_count,
    )


def _choose_child(
    choices: tuple[SelectionStep, ...],
    temperature: float,
    rng: random.Random,
) -> SelectionStep:
    maximum = max(choice.score for choice in choices)
    raw = [math.exp((choice.score - maximum) / temperature) for choice in choices]
    total = sum(raw)
    probabilities = [weight / total for weight in raw]
    needle = rng.random()
    selected_index = len(choices) - 1
    for index, probability in enumerate(probabilities):
        needle -= probability
        if needle <= 0:
            selected_index = index
            break
    selected = choices[selected_index]
    return SelectionStep(
        decision_node_id=selected.decision_node_id,
        option=selected.option,
        target_node_id=selected.target_node_id,
        score=selected.score,
        quality=selected.quality,
        exploration=selected.exploration,
        probability=probabilities[selected_index],
        prior=selected.prior,
        credit_mean=selected.credit_mean,
        credit_count=selected.credit_count,
    )


def _widening_step(
    tree: SearchTree, node_id: int, maximize: bool, weights: SearchWeights
) -> SelectionStep:
    if node_id == tree.root.id:
        prior = quality = credit_mean = 0.0
        credit_count = 0
    else:
        node = tree.get_node(node_id)
        prior = trajectory_prior(tree, node, maximize, weights)
        quality = route_quality(tree, node, maximize, weights)
        credit_mean = node.credit_sum / node.credit_count if node.credit_count else 0.0
        credit_count = node.credit_count
    return SelectionStep(
        decision_node_id=node_id,
        option="new_child",
        target_node_id=None,
        score=quality,
        quality=quality,
        exploration=0.0,
        probability=1.0,
        prior=prior,
        credit_mean=credit_mean,
        credit_count=credit_count,
    )


def select_expansion_node(
    tree: SearchTree,
    *,
    maximize: bool,
    weights: SearchWeights,
    rng: random.Random,
) -> SelectionResult:
    if not tree.root.child_ids:
        raise ValueError("cannot select from an empty tree")

    root_target = max(1, int(tree.root.visit_count**weights.widening_alpha))
    if len(tree.root.child_ids) < root_target:
        step = _widening_step(tree, tree.root.id, maximize, weights)
        return SelectionResult(tree.root.id, (tree.root.id,), (step,))

    root_choices = tuple(
        _node_step(
            tree,
            decision_node_id=tree.root.id,
            child=child,
            parent_visits=tree.root.visit_count,
            maximize=maximize,
            weights=weights,
        )
        for child in tree.children(tree.root.id)
    )
    selected = _choose_child(root_choices, weights.selection_temperature, rng)
    assert selected.target_node_id is not None
    current_id = selected.target_node_id
    path = [tree.root.id, current_id]
    steps = [selected]

    while True:
        current = tree.get_node(current_id)
        width_target = max(1, int(current.visit_count**weights.widening_alpha))
        if len(current.child_ids) < width_target:
            steps.append(_widening_step(tree, current.id, maximize, weights))
            return SelectionResult(current.id, tuple(path), tuple(steps))

        choices = tuple(
            _node_step(
                tree,
                decision_node_id=current.id,
                child=child,
                parent_visits=current.visit_count,
                maximize=maximize,
                weights=weights,
            )
            for child in tree.children(current.id)
        )
        selected = _choose_child(choices, weights.selection_temperature, rng)
        assert selected.target_node_id is not None
        steps.append(selected)
        current_id = selected.target_node_id
        path.append(current_id)


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
    "SearchWeights",
    "directed",
    "expansion_reward",
    "fitness_ranks",
    "node_fitness",
    "path_best_rank",
    "reference_candidates",
    "route_quality",
    "sample_reference",
    "select_expansion_node",
    "subtree_directed",
    "subtree_rank",
    "trajectory_prior",
    "trajectory_trend",
]
