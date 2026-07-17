"""相对排名反馈：Elo 风格的 child-parent 比较。

维护连通比较图上的相对排名，供 refinement prompt 做 best-vs-worst 对比。
Elo 只在连通分量内排序；未连通分量之间仍由 raw fitness 决定。
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
        self._parents: dict[NodeId, NodeId] = {}

    def _expected(self, a: NodeId, b: NodeId) -> float:
        return 1.0 / (1.0 + 10 ** ((self._scores[b] - self._scores[a]) / 400.0))

    def update_pair(self, a: NodeId, b: NodeId, a_wins: float) -> None:
        """a_wins ∈ {1: a 胜, 0.5: 平, 0: a 负}。"""
        self._union(a, b)
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

    def _find(self, node_id: NodeId) -> NodeId:
        parent = self._parents.setdefault(node_id, node_id)
        if parent != node_id:
            self._parents[node_id] = self._find(parent)
        return self._parents[node_id]

    def _union(self, a: NodeId, b: NodeId) -> None:
        root_a = self._find(a)
        root_b = self._find(b)
        if root_a != root_b:
            self._parents[root_b] = root_a

    def contrast(self, *, graph: DerivationGraph, memory: TrajectoryMemory,
                  maximize: bool, window: int = 20) -> dict | None:
        """返回近期 active trajectories 中 best vs worst endpoint 的对比，供反思回路。"""
        actives = memory.active()
        if len(actives) < 2:
            return None
        scored = []
        for t in actives[-window:]:
            node = graph.get_node(t.endpoint_id)
            if node.fitness is None:
                continue
            scored.append((t, node, self.rank(node.id)))
        if len(scored) < 2:
            return None
        components: dict[NodeId, list[tuple]] = {}
        for item in scored:
            components.setdefault(self._find(item[1].id), []).append(item)

        def raw_value(item) -> float:
            fitness = item[1].fitness
            return fitness if maximize else -fitness

        groups = list(components.values())
        best_group = max(groups, key=lambda group: max(raw_value(item) for item in group))
        worst_group = min(groups, key=lambda group: min(raw_value(item) for item in group))

        # Elo is only comparable inside one connected comparison component.
        # Raw fitness chooses between disconnected components.
        best_t, best, best_rank = max(
            best_group,
            key=lambda item: (item[2], raw_value(item)),
        )
        worst_t, worst, worst_rank = min(
            worst_group,
            key=lambda item: (item[2], raw_value(item)),
        )
        return {
            "best": {
                "node_id": best.id,
                "idea": best.idea,
                "fitness": best.fitness,
                "trajectory_id": best_t.id,
                "rank": best_rank,
            },
            "worst": {
                "node_id": worst.id,
                "idea": worst.idea,
                "fitness": worst.fitness,
                "trajectory_id": worst_t.id,
                "rank": worst_rank,
            },
        }
