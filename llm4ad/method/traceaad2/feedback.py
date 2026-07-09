"""鲁棒反馈：对比式相对排名 + 置信度（design §9）。

绝对 fitness 在带噪/deceptive landscape 上不可靠；维护一个 Elo 风格的相对排名
（child vs parent / child vs siblings），对尺度差异和噪声更稳健，供反思回路做
best-vs-worst 对比（context §6.C）。置信度复用 ProgramNode.confidence（来自 robustness）。
"""
from __future__ import annotations

from collections import defaultdict

from .derivation_graph import DerivationGraph
from .schema import NodeId
from .trajectory_memory import TrajectoryMemory


class RankingModel:
    def __init__(self, k: float = 16.0) -> None:
        self.k = k
        self._scores: dict[NodeId, float] = defaultdict(float)

    def _expected(self, a: NodeId, b: NodeId) -> float:
        return 1.0 / (1.0 + 10 ** ((self._scores[b] - self._scores[a]) / 400.0))

    def update_pair(self, a: NodeId, b: NodeId, a_wins: float) -> None:
        """a_wins ∈ {1: a 胜, 0.5: 平, 0: a 负}。"""
        ea = self._expected(a, b)
        self._scores[a] += self.k * (a_wins - ea)
        self._scores[b] += self.k * ((1.0 - a_wins) - (1.0 - ea))

    def update_by_fitness(self, *, a: NodeId, b: NodeId,
                           fitness_a: float, fitness_b: float, maximize: bool) -> None:
        if fitness_a == fitness_b:
            self.update_pair(a, b, 0.5)
            return
        a_better = (fitness_a > fitness_b) if maximize else (fitness_a < fitness_b)
        self.update_pair(a, b, 1.0 if a_better else 0.0)

    def rank(self, node_id: NodeId) -> float:
        return self._scores[node_id]

    def contrast(self, *, graph: DerivationGraph, memory: TrajectoryMemory,
                  maximize: bool, window: int = 20) -> dict | None:
        """返回近期 active trajectories 中 best vs worst endpoint 的对比，供反思回路。"""
        actives = memory.active()
        if len(actives) < 2:
            return None
        scored = []
        for t in actives[-window:]:
            node = graph.get_node(t.endpoint_id)
            if not node.is_valid or node.fitness is None:
                continue
            scored.append((t, node))
        if len(scored) < 2:
            return None
        scored.sort(key=lambda x: x[1].fitness, reverse=maximize)
        best_t, best = scored[0]
        worst_t, worst = scored[-1]
        return {
            "best": {"node_id": best.id, "idea": best.idea, "fitness": best.fitness,
                      "mechanism_tag": best.mechanism_tag, "trajectory_id": best_t.id},
            "worst": {"node_id": worst.id, "idea": worst.idea, "fitness": worst.fitness,
                       "mechanism_tag": worst.mechanism_tag, "trajectory_id": worst_t.id},
        }
