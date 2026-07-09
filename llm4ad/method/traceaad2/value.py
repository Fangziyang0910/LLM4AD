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
    w_quality: float = 0.30
    w_potential: float = 0.25
    w_diversity: float = 0.10
    w_novelty: float = 0.10
    w_generalization: float = 0.25
    w_sim_code: float = 0.4
    w_sim_mechanism: float = 0.4
    w_sim_trajectory: float = 0.2
    discount: float = 0.8
    w_consistency: float = 0.25
    w_downside: float = 0.5
    positive_threshold: float = 1e-6
    c0: float = 0.4
    top_k: int = 5
    temperature: float = 0.8


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
    sims = [_sim_pair(graph, target, o, w) for o in others if o.id != target.id]
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
    potential = compute_path_value(
        trajectory=trajectory, graph=graph, fmin=fmin, fmax=fmax, maximize=maximize,
        discount=w.discount, positive_threshold=w.positive_threshold,
        w_consistency=w.w_consistency, w_downside=w.w_downside,
    )
    diversity, novelty = diversity_and_novelty(
        graph=graph, target=trajectory, others=active_others, w=w
    )
    generalization = trajectory_generalization(
        trajectory=trajectory, graph=graph, pattern_memory=pattern_memory
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


def ucb_bonus(*, visit_count: int, total_visits: int, c0: float,
               iteration: int, max_iter: int) -> float:
    c_t = c0 * max(0.0, 1.0 - iteration / max(max_iter, 1))
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
) -> Trajectory:
    """trajectory-UCB：算多维 value → 写回 → top-k → softmax 采样。"""
    actives = memory.active()
    if not actives:
        raise ValueError("no active trajectories available for sampling")
    fmin, fmax = graph.fitness_range()
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
            iteration=iteration, max_iter=max_iter,
        )
        memory.set_value(t.id, value, scalar)
        scored.append((t, value, scalar))
    scored.sort(key=lambda item: item[2], reverse=True)
    candidates = scored[: min(w.top_k, len(scored))]
    if len(candidates) == 1:
        return candidates[0][0]
    return _softmax_sample(candidates, temperature=w.temperature)


def best_by_quality(*, memory: TrajectoryMemory, graph: DerivationGraph, maximize: bool) -> Trajectory:
    actives = memory.active()

    def key(t: Trajectory) -> float:
        node = graph.get_node(t.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return float("-inf") if maximize else float("inf")
        return node.fitness

    return max(actives, key=key) if maximize else min(actives, key=key)


def _softmax_sample(scored, *, temperature: float) -> Trajectory:
    trajectories = [t for t, _, _ in scored]
    scores = [s for _, _, s in scored]
    mx = max(scores)
    exps = [math.exp((s - mx) / temperature) for s in scores]
    total = sum(exps)
    r = random.random()
    cum = 0.0
    for t, e in zip(trajectories, exps):
        cum += e / total
        if r <= cum:
            return t
    return trajectories[-1]
