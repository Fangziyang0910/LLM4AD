from __future__ import annotations

import math
import random

from .derivation_graph import DerivationGraph
from .schema import Trajectory
from .trajectory_library import TrajectoryLibrary
from .trajectory_scorer import compute_fitness_range, compute_trajectory_score


def sample_trajectory(
        *,
        library: TrajectoryLibrary,
        graph: DerivationGraph,
        iteration: int,
        max_iterations: int,
        top_k: int = 5,
        temperature: float = 0.8,
        w_end: float = 0.45,
        w_path: float = 0.55,
        w_consistency: float = 0.25,
        w_downside: float = 0.5,
        discount: float = 0.8,
        positive_threshold: float = 1e-6,
        c0: float = 0.4,
        maximize: bool = True,
) -> Trajectory:
    active_trajectories = library.active()
    if len(active_trajectories) == 0:
        raise ValueError("no active trajectories available for sampling")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    fitness_min, fitness_max = compute_fitness_range(graph)
    total_visits = sum(t.visit_count for t in active_trajectories)
    scored: list[tuple[Trajectory, float]] = []
    for trajectory in active_trajectories:
        score = compute_trajectory_score(
            trajectory=trajectory,
            graph=graph,
            total_visits=total_visits,
            iteration=iteration,
            max_iterations=max_iterations,
            w_end=w_end,
            w_path=w_path,
            w_consistency=w_consistency,
            w_downside=w_downside,
            discount=discount,
            positive_threshold=positive_threshold,
            c0=c0,
            fitness_min=fitness_min,
            fitness_max=fitness_max,
            maximize=maximize,
        )
        scored.append((library.set_score(trajectory.id, score), score))

    scored.sort(key=lambda item: item[1], reverse=True)
    candidates = scored[: min(top_k, len(scored))]
    if len(candidates) == 1:
        return candidates[0][0]
    return _softmax_sample(candidates, temperature=temperature)


def archive_low_score_trajectories(
        *,
        library: TrajectoryLibrary,
        max_active: int = 1000,
) -> int:
    if max_active <= 0:
        raise ValueError("max_active must be positive")
    active = library.active()
    if len(active) <= max_active:
        return 0

    archive_budget = len(active) - max_active
    sorted_trajectories = sorted(active, key=lambda t: t.score if t.score is not None else float("-inf"))
    archived_count = 0
    for trajectory in sorted_trajectories[:archive_budget]:
        library.archive(trajectory.id)
        archived_count += 1
    return archived_count


def sample_best_node(
        *,
        library: TrajectoryLibrary,
        graph: DerivationGraph,
        maximize: bool = True,
) -> Trajectory:
    active = library.active()
    if len(active) == 0:
        raise ValueError("no active trajectories available")

    def fitness_key(t: Trajectory) -> float:
        node = graph.get_node(t.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return float("-inf") if maximize else float("inf")
        return node.fitness

    return max(active, key=fitness_key) if maximize else min(active, key=fitness_key)


def sample_random_trajectory(*, library: TrajectoryLibrary) -> Trajectory:
    active = library.active()
    if len(active) == 0:
        raise ValueError("no active trajectories available")
    return random.choice(active)


def _softmax_sample(scored_trajectories: list[tuple[Trajectory, float]], temperature: float) -> Trajectory:
    trajectories = [trajectory for trajectory, _ in scored_trajectories]
    scores = [score for _, score in scored_trajectories]
    max_score = max(scores)
    exp_scores = [math.exp((score - max_score) / temperature) for score in scores]
    total = sum(exp_scores)

    r = random.random()
    cumulative = 0.0
    for trajectory, exp_score in zip(trajectories, exp_scores):
        cumulative += exp_score / total
        if r <= cumulative:
            return trajectory
    return trajectories[-1]
