"""四种改法的共同协议与中间起点选择。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..credit import directed_delta, normalize_fitness
from ..derivation_graph import DerivationGraph
from ..schema import NodeId, OperatorName, Trajectory
from ..trajectory_memory import TrajectoryMemory


def classify_outcome(delta: float | None, positive_threshold: float = 1e-6) -> str:
    if delta is None:
        return "unknown"
    if delta > positive_threshold:
        return "improve"
    if delta < -positive_threshold:
        return "regress"
    return "plateau"


@dataclass
class OperatorContext:
    graph: DerivationGraph
    memory: TrajectoryMemory
    selected: Trajectory
    maximize: bool
    positive_threshold: float = 1e-6


class Operator(ABC):
    name: str = ""

    @abstractmethod
    def trigger(self, ctx: OperatorContext) -> bool:
        ...

    @abstractmethod
    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        """返回 (base_node_id, reason)；Novelty 返回 (None, 'fresh_start')。"""

    @abstractmethod
    def build_constraint(self, ctx: OperatorContext, base_node_id: NodeId | None) -> str:
        """注入 action prompt 的算子特定约束文本。"""

    @abstractmethod
    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        ...

    def select_trajectory(self, ctx: OperatorContext) -> Trajectory | None:
        """默认沿用 ctx.selected；backtrack 等算子可 override 从 pool 主动另选目标 trajectory。"""
        return None


class _ExtendFromEndpointOp(Operator):
    """从当前轨迹终点继续。"""

    def select_base(self, ctx: OperatorContext) -> tuple[NodeId | None, str]:
        return ctx.selected.endpoint_id, "endpoint"

    def insert(self, ctx: OperatorContext, child_id: NodeId, edge_id: int,
               base_node_id: NodeId | None) -> Trajectory:
        assert base_node_id is not None
        return ctx.memory.extend(
            trajectory_id=ctx.selected.id,
            parent_id=base_node_id,
            child_id=child_id,
            edge_id=edge_id,
        )


# ---------- base node 选择：4 规则门控 + branch_score ----------

def trajectory_step_outcomes(graph: DerivationGraph, trajectory: Trajectory,
                              maximize: bool, positive_threshold: float = 1e-6):
    outcomes = []
    for parent_id, child_id in zip(trajectory.node_ids, trajectory.node_ids[1:]):
        parent = graph.get_node(parent_id)
        child = graph.get_node(child_id)
        delta = directed_delta(parent.fitness, child.fitness, maximize)
        outcomes.append((parent_id, child_id, delta, classify_outcome(delta, positive_threshold)))
    return outcomes


def select_base_node(*, graph: DerivationGraph, trajectory: Trajectory, maximize: bool,
                      positive_threshold: float = 1e-6) -> tuple[NodeId, str]:
    """4 规则产生候选 → branch_score 打分取最大。"""
    if not trajectory.edge_ids:
        return trajectory.endpoint_id, "initial"

    candidates: list[tuple[NodeId, str]] = [(trajectory.endpoint_id, "endpoint")]
    outcomes = trajectory_step_outcomes(graph, trajectory, maximize, positive_threshold)
    last = outcomes[-1]
    if last[3] == "regress":
        candidates.append((last[0], "last_regressed"))
    if len(outcomes) >= 2 and outcomes[-1][3] == "plateau" and outcomes[-2][3] == "plateau":
        for entry in reversed(outcomes):
            if entry[3] == "improve":
                candidates.append((entry[1], "recent_plateau"))
                break
    best_id = _best_node_in_trajectory(graph, trajectory, maximize)
    if best_id is not None and best_id != trajectory.endpoint_id:
        candidates.append((best_id, "endpoint_not_best"))

    best = max(candidates, key=lambda c: branch_score(graph, trajectory, c[0], maximize))
    return best


def branch_score(graph: DerivationGraph, trajectory: Trajectory, node_id: NodeId,
                  maximize: bool) -> float:
    """normalize(q) + 局部前向正改进 − 局部后向回撤。"""
    node = graph.get_node(node_id)
    if node.fitness is None:
        return -1e9
    fmin, fmax = graph.fitness_range()
    base_quality = normalize_fitness(node.fitness, fmin, fmax, maximize)
    idx = trajectory.node_ids.index(node_id)
    outcomes = trajectory_step_outcomes(graph, trajectory, maximize)
    before_positive = sum(e[2] or 0.0 for e in outcomes[:idx] if e[3] == "improve") / max(1, idx + 1)
    after_downside = sum(-(e[2] or 0.0) for e in outcomes[idx:] if e[3] == "regress") / max(1, len(outcomes) - idx)
    return base_quality + 0.3 * before_positive - 0.3 * after_downside


def _best_node_in_trajectory(graph: DerivationGraph, trajectory: Trajectory,
                              maximize: bool) -> NodeId | None:
    best_id: NodeId | None = None
    best_fit: float | None = None
    for nid in trajectory.node_ids:
        node = graph.get_node(nid)
        if node.fitness is None:
            continue
        if best_fit is None or (node.fitness > best_fit) == maximize:
            best_id, best_fit = nid, node.fitness
    return best_id


__all__ = [
    "OperatorName",
    "OperatorContext",
    "Operator",
    "_ExtendFromEndpointOp",
    "select_base_node",
    "branch_score",
    "classify_outcome",
]
