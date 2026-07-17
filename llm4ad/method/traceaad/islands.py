"""Islands —— 多岛并行搜索子流，防全局收敛到一个 basin。

- assign：按 mechanism_tag 哈希分配岛，让同机制聚簇、跨机制分散。
- migrate：周期性轮换每个岛的 top trajectory，促进机制流动而不制造 clone。
- survival 在主循环 `_survive` 内按岛做 non-dominated 截断。
"""
from __future__ import annotations

import hashlib

from .trajectory_memory import TrajectoryMemory


class IslandsManager:
    def __init__(self, n_islands: int = 4) -> None:
        self.n_islands = max(1, int(n_islands))

    def assign(self, mechanism_tag: str) -> int:
        digest = hashlib.sha256(mechanism_tag.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.n_islands

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
