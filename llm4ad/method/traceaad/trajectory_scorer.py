from __future__ import annotations

import math

from .derivation_graph import DerivationGraph
from .schema import Trajectory


def compute_trajectory_score(
        *,
        trajectory: Trajectory,
        graph: DerivationGraph,
        total_visits: int,
        iteration: int,
        max_iterations: int,
        w_end: float = 0.45,
        w_path: float = 0.55,
        w_consistency: float = 0.25,
        w_downside: float = 0.5,
        discount: float = 0.8,
        positive_threshold: float = 1e-6,
        c0: float = 0.4,
        fitness_min: float | None = None,
        fitness_max: float | None = None,
        maximize: bool = True,
) -> float:
    if len(trajectory.node_ids) == 0:
        raise ValueError("trajectory must have at least one node")
    if w_end < 0 or w_path < 0 or w_consistency < 0 or w_downside < 0 or c0 < 0:
        raise ValueError("score weights and c0 must be non-negative")
    if not (0 < discount <= 1):
        raise ValueError("discount must be in (0, 1]")
    if positive_threshold < 0:
        raise ValueError("positive_threshold must be non-negative")

    endpoint = graph.get_node(trajectory.endpoint_id)
    if not endpoint.is_valid or endpoint.fitness is None:
        return -1e9

    qualities = _trajectory_qualities(
        trajectory=trajectory,
        graph=graph,
        fitness_min=fitness_min,
        fitness_max=fitness_max,
        maximize=maximize,
    )
    if qualities is None:
        return -1e9

    endpoint_quality = qualities[-1]
    path_value = _compute_path_value(
        qualities=qualities,
        discount=discount,
        positive_threshold=positive_threshold,
        w_consistency=w_consistency,
        w_downside=w_downside,
    )
    explore_bonus = math.sqrt(math.log(total_visits + 1) / (trajectory.visit_count + 1))
    c_t = c0 * (1.0 - iteration / max(max_iterations, 1))
    c_t = max(c_t, 0.0)
    return w_end * endpoint_quality + w_path * path_value + c_t * explore_bonus


def compute_fitness_range(graph: DerivationGraph) -> tuple[float | None, float | None]:
    values = [
        node.fitness
        for node in graph.nodes()
        if node.is_valid and node.fitness is not None
    ]
    if not values:
        return None, None
    return min(values), max(values)


def _trajectory_qualities(
        *,
        trajectory: Trajectory,
        graph: DerivationGraph,
        fitness_min: float | None,
        fitness_max: float | None,
        maximize: bool,
) -> list[float] | None:
    qualities: list[float] = []
    for node_id in trajectory.node_ids:
        node = graph.get_node(node_id)
        if not node.is_valid or node.fitness is None:
            return None
        qualities.append(
            _normalize_fitness(
                node.fitness,
                fitness_min=fitness_min,
                fitness_max=fitness_max,
                maximize=maximize,
            )
        )
    return qualities


def _compute_path_value(
        *,
        qualities: list[float],
        discount: float,
        positive_threshold: float,
        w_consistency: float,
        w_downside: float,
) -> float:
    if len(qualities) <= 1:
        return 0.0

    rewards = [
        current - previous
        for previous, current in zip(qualities, qualities[1:])
    ]
    step_count = len(rewards)
    weights = [discount ** (step_count - index - 1) for index in range(step_count)]
    weight_sum = sum(weights)

    discounted_return = sum(weight * reward for weight, reward in zip(weights, rewards)) / weight_sum
    discounted_downside = sum(weight * max(-reward, 0.0) for weight, reward in zip(weights, rewards)) / weight_sum
    positive_step_ratio = sum(1 for reward in rewards if reward > positive_threshold) / step_count
    return discounted_return + w_consistency * positive_step_ratio - w_downside * discounted_downside


def _normalize_fitness(
        fitness: float,
        *,
        fitness_min: float | None,
        fitness_max: float | None,
        maximize: bool,
) -> float:
    if fitness_min is None or fitness_max is None:
        return fitness if maximize else -fitness
    if not maximize:
        fitness = -fitness
        fitness_min, fitness_max = -fitness_max, -fitness_min
    if abs(fitness_max - fitness_min) < 1e-9:
        return 0.5
    normalized = (fitness - fitness_min) / (fitness_max - fitness_min)
    return max(0.0, min(1.0, normalized))
