"""Stepwise credit + 泛化信号（design §5）。

- stepwise attribution：每个 step 只承担自己的 Δ，后代不回传（规避 MCTS max-backprop over-credit）。
- 泛化信号只来自 evaluator 提供的 parent/child per-instance fitness vector 与 robustness。
  scalar-only task 的该维恒为 0，不再用机制成功率或 scalar delta 伪造跨实例证据。
"""
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


def step_generalization_signal(
    *,
    parent_fitness_vector: tuple[float, ...] | None,
    child_fitness_vector: tuple[float, ...] | None,
    maximize: bool,
    neutral_tolerance: float = 1e-12,
) -> float:
    """Fractional per-instance wins for one parent-child step, in [0, 1]."""
    if (
        parent_fitness_vector is None
        or child_fitness_vector is None
        or not parent_fitness_vector
        or len(parent_fitness_vector) != len(child_fitness_vector)
    ):
        return 0.0
    outcomes: list[float] = []
    for parent, child in zip(parent_fitness_vector, child_fitness_vector):
        delta = child - parent if maximize else parent - child
        if delta > neutral_tolerance:
            outcomes.append(1.0)
        elif delta < -neutral_tolerance:
            outcomes.append(0.0)
        else:
            outcomes.append(0.5)
    return sum(outcomes) / len(outcomes)


def trajectory_generalization(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
) -> float:
    """Trajectory transfer value from vector-based step evidence and robustness."""
    endpoint = graph.get_node(trajectory.endpoint_id)
    robustness = max(0.0, min(1.0, float(endpoint.robustness)))
    if not trajectory.edge_ids:
        return robustness
    gen_signals = [
        graph.get_edge(edge_id).generalization_signal
        for edge_id in trajectory.edge_ids
    ]
    avg_step_gen = sum(gen_signals) / len(gen_signals)
    return max(
        0.0,
        min(1.0, 0.6 * avg_step_gen + 0.2 * gen_signals[-1] + 0.2 * robustness),
    )


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
        if not node.is_valid or node.fitness is None:
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
    w_consistency: float = 0.25,
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
    return discounted_return + w_consistency * positive_ratio - w_downside * discounted_downside
