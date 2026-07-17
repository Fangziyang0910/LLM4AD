"""Islands —— 多岛并行搜索子流，防全局收敛到一个 basin。

- assign_least_loaded：Novelty fresh start 分到当前活跃轨迹最少的岛。
- migrate：周期性轮换每个岛的 top trajectory，促进跨岛流动而不制造 clone。
- survival 在主循环 `_survive` 内按岛做 non-dominated 截断。
"""
from __future__ import annotations

from .trajectory_memory import TrajectoryMemory


class IslandsManager:
    def __init__(self, n_islands: int = 4) -> None:
        self.n_islands = max(1, int(n_islands))

    def assign_least_loaded(self, memory: TrajectoryMemory) -> int:
        """分配到当前活跃轨迹最少的 island；并列时选编号最小者。"""
        best_island = 0
        best_count = len(memory.active_in_island(0))
        for island in range(1, self.n_islands):
            count = len(memory.active_in_island(island))
            if count < best_count:
                best_island = island
                best_count = count
        return best_island

    def migrate(
        self,
        *,
        memory: TrajectoryMemory,
        top_per_island: int = 1,
    ) -> int:
        """Rotate each island's top trajectories without creating new identities."""
        island_ids = memory.island_ids()
        count = max(0, int(top_per_island))
        if len(island_ids) <= 1 or count == 0:
            return 0

        selected_by_island: dict[int, tuple[int, ...]] = {}
        for island in island_ids:
            members = memory.active_in_island(island)
            ranked = sorted(
                members,
                key=lambda t: t.scalar_value if t.scalar_value is not None else float("-inf"),
                reverse=True,
            )[:count]
            selected_by_island[island] = tuple(trajectory.id for trajectory in ranked)

        moved = 0
        for index, source in enumerate(island_ids):
            target = island_ids[(index + 1) % len(island_ids)]
            for trajectory_id in selected_by_island[source]:
                memory.move_to_island(trajectory_id, target)
                moved += 1
        return moved
