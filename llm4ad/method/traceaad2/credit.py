"""Stepwise credit + 泛化信号（design §5）。

- stepwise attribution：每个 step 只承担自己的 Δ，后代不回传（规避 MCTS max-backprop over-credit）。
- 泛化信号：当前平台 task 只返回标量，无法做真·跨 instance 泛化；这里用「该机制在
  PatternMemory 的跨轨迹泛化分数 + 步内改进是否被后续维持 + endpoint robustness」近似。
  EvalResult.fitness_vector 字段保留，未来 task 提供 per-instance 时升级为真·跨 instance。
"""
from __future__ import annotations

from .derivation_graph import DerivationGraph
from .pattern_memory import PatternMemory
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
    mechanism_tag: str,
    delta: float | None,
    pattern_memory: PatternMemory,
    maximize: bool,
) -> float:
    """child 刚生成时写入 edge 的泛化信号（初始估计）。∈ [0,1]。"""
    mech = pattern_memory.mechanism_pattern(mechanism_tag)
    cross_traj_score = mech.generalization_score if mech is not None else 0.0
    # 步内弱信号：正向改进给一点基础分
    step_sign = 0.0
    if delta is not None:
        step_sign = 0.5 if (delta > 0) == maximize else (0.2 if abs(delta) < 1e-9 else 0.0)
    # 跨轨迹证据权重更高
    return max(0.0, min(1.0, 0.6 * cross_traj_score + 0.4 * step_sign))


def trajectory_generalization(
    *,
    trajectory: Trajectory,
    graph: DerivationGraph,
    pattern_memory: PatternMemory,
) -> float:
    """trajectory 级泛化：步泛化信号 + endpoint robustness + 持续性 + endpoint 机制 improve-rate。

    最后一项是机制信用贯通的关键：endpoint 机制若跨轨迹 improve-rate 高（如 adaptive_exponent），
    该 trajectory 的泛化价值自动升高，被 selection 偏好 → 深化高效机制、远离低效机制。
    """
    endpoint = graph.get_node(trajectory.endpoint_id)
    mech_rate = pattern_memory.mechanism_improve_rate(endpoint.mechanism_tag)
    mech_rate = 0.5 if mech_rate is None else mech_rate
    if not trajectory.edge_ids:
        return max(0.0, min(1.0, 0.6 * float(endpoint.robustness) + 0.4 * mech_rate))
    gen_signals: list[float] = []
    for eid in trajectory.edge_ids:
        edge = graph.get_edge(eid)
        gen_signals.append(edge.generalization_signal)
    avg_step_gen = sum(gen_signals) / len(gen_signals)
    # 持续性：最后一步是否仍是改进（未被后续 revert）的弱代理
    sustained = 1.0 if gen_signals[-1] >= 0.5 else 0.5
    return max(0.0, min(1.0, 0.30 * avg_step_gen + 0.25 * float(endpoint.robustness)
                        + 0.20 * sustained + 0.25 * mech_rate))


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
