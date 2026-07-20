"""计算单步有向变化和一条轨迹近期的改进趋势。"""
from __future__ import annotations

from .derivation_graph import DerivationGraph
from .schema import Trajectory


def normalize_fitness(fitness: float, fmin: float | None, fmax: float | None,
                       maximize: bool) -> float:
    if fmin is None or fmax is None:
        return 1.0 if maximize else 0.0
    if not maximize:
        fitness, fmin, fmax = -fitness, -fmax, -fmin
    if abs(fmax - fmin) < 1e-12:
        return 0.5
    return max(0.0, min(1.0, (fitness - fmin) / (fmax - fmin)))


def directed_delta(parent_fitness: float | None, child_fitness: float | None,
                    maximize: bool) -> float | None:
    if parent_fitness is None or child_fitness is None:
        return None
    return child_fitness - parent_fitness if maximize else parent_fitness - child_fitness


def compute_step_qualities(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    fmin: float | None,
    fmax: float | None,
    maximize: bool,
) -> list[float] | None:
    qualities: list[float] = []
    for nid in trajectory.node_ids:
        node = graph.get_node(nid)
        if node.fitness is None:
            return None
        qualities.append(normalize_fitness(node.fitness, fmin, fmax, maximize))
    return qualities


def compute_path_value(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    fmin: float | None,
    fmax: float | None,
    maximize: bool,
    discount: float = 0.8,
    positive_threshold: float = 1e-6,
    w_positive_ratio: float = 0.25,
    w_downside: float = 0.5,
) -> float:
    """stepwise path value：近端加权的折扣回报 + 正向步比例 − 折扣回撤。"""
    qualities = compute_step_qualities(
        trajectory=trajectory, graph=graph, fmin=fmin, fmax=fmax, maximize=maximize
    )
    if qualities is None or len(qualities) <= 1:
        return 0.0
    rewards = [cur - prev for prev, cur in zip(qualities, qualities[1:])]
    n = len(rewards)
    weights = [discount ** (n - i - 1) for i in range(n)]
    z = sum(weights)
    if z <= 0:
        return 0.0
    discounted_return = sum(w * r for w, r in zip(weights, rewards)) / z
    discounted_downside = sum(w * max(-r, 0.0) for w, r in zip(weights, rewards)) / z
    positive_ratio = sum(1 for r in rewards if r > positive_threshold) / n
    return discounted_return + w_positive_ratio * positive_ratio - w_downside * discounted_downside
