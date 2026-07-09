"""Islands —— 多岛并行搜索子流，防全局收敛到一个 basin（design §8）。

- assign：按 mechanism_tag 哈希分配岛，让同机制聚簇、跨机制分散。
- migrate：周期性把每个岛的 top trajectory 复制到其他岛，促进机制流动。
- survival 在 trajectory_manager 内按岛做 non-dominated（见 manager.py）。
"""
from __future__ import annotations

import random

from .trajectory_memory import TrajectoryMemory


class IslandsManager:
    def __init__(self, n_islands: int = 4) -> None:
        self.n_islands = max(1, int(n_islands))

    def assign(self, mechanism_tag: str, salt: int = 0) -> int:
        return (hash((mechanism_tag, salt)) % self.n_islands + self.n_islands) % self.n_islands

    def migrate(
        self,
        *,
        memory: TrajectoryMemory,
        top_per_island: int = 1,
    ) -> int:
        """每个岛取 top trajectory，fork 到一个随机其它岛。返回迁移次数。"""
        island_ids = memory.island_ids()
        if len(island_ids) <= 1:
            return 0
        moved = 0
        for island in island_ids:
            members = memory.active_in_island(island)
            if not members:
                continue
            ranked = sorted(
                members,
                key=lambda t: t.scalar_value if t.scalar_value is not None else float("-inf"),
                reverse=True,
            )[:top_per_island]
            targets = [i for i in island_ids if i != island]
            for traj in ranked:
                target = random.choice(targets)
                memory.fork(traj.id, target)
                moved += 1
        return moved
