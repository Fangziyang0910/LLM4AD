"""多维 Trajectory Value + trajectory-UCB 选择（design §4）。

ValueVec 不塌缩成单标量（MEoH 教训）：survival 用 non-dominated（见 trajectory_manager），
采样用 scalarize + UCB。V_potential/V_generalization 是相对各家的真正增量。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .credit import (
    compute_path_value,
    normalize_fitness,
    trajectory_generalization,
)
from .derivation_graph import DerivationGraph
from .pattern_memory import PatternMemory
from .schema import Trajectory, ValueVec
from .similarity import (
    code_similarity,
    mechanism_profile,
    mechanism_similarity,
    trajectory_pattern,
    trajectory_pattern_similarity,
)
from .trajectory_memory import TrajectoryMemory


@dataclass(frozen=True)
class ValueWeights:
    w_quality: float = 0.50
    w_potential: float = 0.20
    w_diversity: float = 0.15
    w_novelty: float = 0.15
    w_generalization: float = 0.0
    w_sim_code: float = 0.4
    w_sim_mechanism: float = 0.4
    w_sim_trajectory: float = 0.2
    discount: float = 0.8
    w_consistency: float = 0.25
    w_downside: float = 0.5
    positive_threshold: float = 1e-6
    c0: float = 0.4
    top_k: int = 12
    temperature: float = 0.8
    fitness_clip_quantile: float = 0.10
    potential_quality_floor: float = 0.50
    ucb_floor: float = 0.05
    stagnation_ucb_boost: float = 0.20
    elite_sampling_prob: float = 0.15
    island_top_k: int = 1


def robust_active_fitness_bounds(
    *,
    trajectories: tuple[Trajectory, ...],
    graph: DerivationGraph,
    clip_quantile: float = 0.10,
) -> tuple[float | None, float | None]:
    """Return clipped fitness bounds over unique active endpoints.

    Selection should not let archived programs or repeated trajectories distort
    endpoint quality. Quantile clipping also prevents one active failure from
    compressing the useful part of the pool into an indistinguishable band.
    """
    values: list[float] = []
    seen_endpoints: set[int] = set()
    for trajectory in trajectories:
        if trajectory.endpoint_id in seen_endpoints:
            continue
        seen_endpoints.add(trajectory.endpoint_id)
        endpoint = graph.get_node(trajectory.endpoint_id)
        if endpoint.is_valid and endpoint.fitness is not None and math.isfinite(endpoint.fitness):
            values.append(float(endpoint.fitness))
    if not values:
        return None, None
    values.sort()
    if len(values) == 1:
        return values[0], values[0]
    q = max(0.0, min(float(clip_quantile), 0.49))
    return _linear_quantile(values, q), _linear_quantile(values, 1.0 - q)


def _linear_quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def _sim_pair(graph, a: Trajectory, b: Trajectory, w: ValueWeights) -> float:
    sim_code = code_similarity(
        graph.get_node(a.endpoint_id).code, graph.get_node(b.endpoint_id).code
    )
    sim_mech = mechanism_similarity(mechanism_profile(graph, a), mechanism_profile(graph, b))
    sim_pat = trajectory_pattern_similarity(trajectory_pattern(graph, a), trajectory_pattern(graph, b))
    total = w.w_sim_code + w.w_sim_mechanism + w.w_sim_trajectory
    return (w.w_sim_code * sim_code + w.w_sim_mechanism * sim_mech + w.w_sim_trajectory * sim_pat) / total


def diversity_and_novelty(*, graph, target: Trajectory,
                           others: tuple[Trajectory, ...], w: ValueWeights) -> tuple[float, float]:
    """返回 (边际多样性=1-mean_sim, 新颖度=1-max_sim)。"""
    if not others:
        return 1.0, 1.0
    unique_others = {
        other.path_key: other
        for other in others
        if other.path_key != target.path_key
    }
    sims = [_sim_pair(graph, target, other, w) for other in unique_others.values()]
    if not sims:
        return 1.0, 1.0
    return 1.0 - (sum(sims) / len(sims)), 1.0 - max(sims)


def compute_value_vec(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    pattern_memory: PatternMemory,
    active_others: tuple[Trajectory, ...],
    fmin: float | None,
    fmax: float | None,
    maximize: bool,
    w: ValueWeights,
) -> ValueVec:
    endpoint = graph.get_node(trajectory.endpoint_id)
    if not endpoint.is_valid or endpoint.fitness is None:
        return ValueVec()
    quality = normalize_fitness(endpoint.fitness, fmin, fmax, maximize)
    raw_potential = compute_path_value(
        trajectory=trajectory, graph=graph, fmin=fmin, fmax=fmax, maximize=maximize,
        discount=w.discount, positive_threshold=w.positive_threshold,
        w_consistency=w.w_consistency, w_downside=w.w_downside,
    )
    quality_floor = max(0.0, min(w.potential_quality_floor, 1.0))
    if quality <= quality_floor or quality_floor >= 1.0:
        potential = 0.0
    else:
        potential = raw_potential * (quality - quality_floor) / (1.0 - quality_floor)
    diversity, novelty = diversity_and_novelty(
        graph=graph, target=trajectory, others=active_others, w=w
    )
    generalization = 0.0
    if w.w_generalization > 0.0:
        generalization = trajectory_generalization(
            trajectory=trajectory,
            graph=graph,
        )
    return ValueVec(
        quality=quality, potential=potential, diversity=diversity,
        novelty=novelty, generalization=generalization,
    )


def scalarize(value: ValueVec, w: ValueWeights) -> float:
    return (
        w.w_quality * value.quality
        + w.w_potential * value.potential
        + w.w_diversity * value.diversity
        + w.w_novelty * value.novelty
        + w.w_generalization * value.generalization
    )


def pareto_survival_order(
    trajectories: tuple[Trajectory, ...],
) -> tuple[Trajectory, ...]:
    """Order trajectories by Pareto front, using scalar only within a front."""
    remaining = list(trajectories)
    ordered: list[Trajectory] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                other.id != candidate.id and _dominates(other, candidate)
                for other in remaining
            )
        ]
        front.sort(key=_within_front_key)
        ordered.extend(front)
        front_ids = {trajectory.id for trajectory in front}
        remaining = [trajectory for trajectory in remaining if trajectory.id not in front_ids]
    return tuple(ordered)


def _dominates(left: Trajectory, right: Trajectory) -> bool:
    left_values = left.value.as_tuple() if left.value is not None else (float("-inf"),) * 5
    right_values = right.value.as_tuple() if right.value is not None else (float("-inf"),) * 5
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def _within_front_key(trajectory: Trajectory) -> tuple[float, int]:
    scalar = trajectory.scalar_value
    if scalar is None or math.isnan(scalar):
        scalar = float("-inf")
    return -scalar, trajectory.id


def ucb_bonus(
    *,
    visit_count: int,
    total_visits: int,
    c0: float,
    iteration: int,
    max_iter: int,
    ucb_floor: float = 0.0,
    stagnation: int = 0,
    stagnation_boost: float = 0.0,
) -> float:
    horizon = max(max_iter, 1)
    decayed_c = c0 * max(0.0, 1.0 - iteration / horizon)
    c_t = max(0.0, ucb_floor, decayed_c)
    c_t += max(0.0, stagnation_boost) * min(max(stagnation, 0) / horizon, 1.0)
    return c_t * math.sqrt(math.log(total_visits + 1) / (visit_count + 1))


def select_trajectory(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    pattern_memory: PatternMemory,
    maximize: bool,
    iteration: int,
    max_iter: int,
    w: ValueWeights,
    stagnation: int = 0,
    elite_trajectory_id: int | None = None,
    elite_endpoint_id: int | None = None,
    rng: random.Random | None = None,
) -> Trajectory:
    """Score active trajectories, then sample with explicit elite protection."""
    actives = memory.unique_active()
    if not actives:
        raise ValueError("no active trajectories available for sampling")
    elite = _resolve_elite(
        actives=actives,
        memory=memory,
        graph=graph,
        maximize=maximize,
        elite_trajectory_id=elite_trajectory_id,
        elite_endpoint_id=elite_endpoint_id,
    )
    fmin, fmax = robust_active_fitness_bounds(
        trajectories=actives,
        graph=graph,
        clip_quantile=w.fitness_clip_quantile,
    )
    total_visits = memory.total_visits()
    scored: list[tuple[Trajectory, ValueVec, float]] = []
    for t in actives:
        others = tuple(o for o in actives if o.id != t.id)
        value = compute_value_vec(
            trajectory=t, graph=graph, pattern_memory=pattern_memory,
            active_others=others, fmin=fmin, fmax=fmax, maximize=maximize, w=w,
        )
        scalar = scalarize(value, w) + ucb_bonus(
            visit_count=t.visit_count, total_visits=total_visits, c0=w.c0,
            iteration=iteration, max_iter=max_iter, ucb_floor=w.ucb_floor,
            stagnation=stagnation, stagnation_boost=w.stagnation_ucb_boost,
        )
        memory.set_value(t.id, value, scalar)
        scored.append((t, value, scalar))
    scored.sort(key=lambda item: item[2], reverse=True)
    candidates = scored[: min(max(w.top_k, 0), len(scored))]
    candidate_ids = {item[0].id for item in candidates}
    if w.island_top_k > 0:
        for island_id in memory.island_ids():
            island_items = [item for item in scored if item[0].island_id == island_id]
            for item in island_items[: w.island_top_k]:
                if item[0].id not in candidate_ids:
                    candidates.append(item)
                    candidate_ids.add(item[0].id)
    elite_item = next(item for item in scored if item[0].id == elite.id)
    if elite.id not in candidate_ids:
        candidates.append(elite_item)
    elite_probability = max(0.0, min(w.elite_sampling_prob, 1.0))
    random_source = rng if rng is not None else random
    if elite_probability > 0.0 and random_source.random() < elite_probability:
        return elite_item[0]
    if len(candidates) == 1:
        return candidates[0][0]
    return _softmax_sample(candidates, temperature=w.temperature, rng=rng)


def _resolve_elite(
    *,
    actives: tuple[Trajectory, ...],
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    maximize: bool,
    elite_trajectory_id: int | None,
    elite_endpoint_id: int | None,
) -> Trajectory:
    if elite_trajectory_id is not None:
        requested = memory.get_trajectory(elite_trajectory_id)
        if requested.status.value != "active":
            raise ValueError(f"elite trajectory is not active: {elite_trajectory_id}")
        representative = next(
            (
                trajectory
                for trajectory in actives
                if trajectory.path_key == requested.path_key
            ),
            None,
        )
        if representative is None:
            raise ValueError(f"elite path is not active: {elite_trajectory_id}")
        return representative
    if elite_endpoint_id is not None:
        matches = [t for t in actives if t.endpoint_id == elite_endpoint_id]
        if not matches:
            raise ValueError(f"no active trajectory for elite endpoint: {elite_endpoint_id}")
        return min(matches, key=lambda trajectory: trajectory.visit_count)
    return best_by_quality(memory=memory, graph=graph, maximize=maximize)


def best_by_quality(*, memory: TrajectoryMemory, graph: DerivationGraph, maximize: bool) -> Trajectory:
    actives = memory.unique_active()

    def key(t: Trajectory) -> float:
        node = graph.get_node(t.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return float("-inf") if maximize else float("inf")
        return node.fitness

    return max(actives, key=key) if maximize else min(actives, key=key)


def _softmax_sample(
    scored,
    *,
    temperature: float,
    rng: random.Random | None = None,
) -> Trajectory:
    trajectories = [t for t, _, _ in scored]
    scores = [s for _, _, s in scored]
    mx = max(scores)
    safe_temperature = max(float(temperature), 1e-6)
    exps = [math.exp((s - mx) / safe_temperature) for s in scores]
    total = sum(exps)
    random_source = rng if rng is not None else random
    r = random_source.random()
    cum = 0.0
    for t, e in zip(trajectories, exps):
        cum += e / total
        if r <= cum:
            return t
    return trajectories[-1]
