"""TrajectoryMemory —— 路径层记忆（采样对象）。

存储有界 trajectory，支持 create_initial / extend / branch_from（含滑窗截断）、
按 island 分组、归档。value/scalar_value 由 value.py 计算后写入。
"""
from __future__ import annotations

from dataclasses import replace

from .schema import (
    EdgeId,
    IslandId,
    NodeId,
    Trajectory,
    TrajectoryId,
    TrajectoryStatus,
    ValueVec,
)


class TrajectoryMemory:
    def __init__(self, *, max_trajectory_length: int = 8) -> None:
        if max_trajectory_length < 2:
            raise ValueError("max_trajectory_length must be at least 2")
        self.max_trajectory_length = max_trajectory_length
        self._next_id = 0
        self._trajectories: dict[TrajectoryId, Trajectory] = {}

    def create_initial(self, *, node_id: NodeId, island_id: IslandId = 0) -> Trajectory:
        traj = Trajectory(
            id=self._next_id,
            node_ids=(node_id,),
            edge_ids=(),
            endpoint_id=node_id,
            base_id=node_id,
            island_id=island_id,
        )
        self._trajectories[traj.id] = traj
        self._next_id += 1
        return traj

    def extend(self, *, trajectory_id: TrajectoryId, parent_id: NodeId, child_id: NodeId,
               edge_id: EdgeId, island_id: IslandId | None = None) -> Trajectory:
        parent = self.get_trajectory(trajectory_id)
        if parent.endpoint_id != parent_id:
            raise ValueError(
                f"extend parent must be endpoint: {parent_id} != {parent.endpoint_id}"
            )
        return self.branch_from(
            trajectory_id=trajectory_id,
            base_node_id=parent.endpoint_id,
            child_id=child_id,
            edge_id=edge_id,
            island_id=island_id,
        )

    def branch_from(self, *, trajectory_id: TrajectoryId, base_node_id: NodeId,
                    child_id: NodeId, edge_id: EdgeId, island_id: IslandId | None = None) -> Trajectory:
        parent = self.get_trajectory(trajectory_id)
        if parent.status != TrajectoryStatus.ACTIVE:
            raise ValueError(f"cannot branch from archived trajectory: {trajectory_id}")
        if base_node_id not in parent.node_ids:
            raise ValueError(f"base node must belong to the trajectory: {base_node_id}")

        base_index = parent.node_ids.index(base_node_id)
        node_ids = (*parent.node_ids[: base_index + 1], child_id)
        edge_ids = (*parent.edge_ids[:base_index], edge_id)
        overflow = len(node_ids) - self.max_trajectory_length
        if overflow > 0:
            node_ids = node_ids[overflow:]
            edge_ids = edge_ids[overflow:]
        # 滑窗后 base 可能被截掉，endpoint 永远是最后一个节点
        endpoint_id = node_ids[-1]
        new_base = node_ids[0]

        traj = Trajectory(
            id=self._next_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            endpoint_id=endpoint_id,
            base_id=new_base,
            island_id=parent.island_id if island_id is None else island_id,
        )
        self._trajectories[traj.id] = traj
        self._next_id += 1
        return traj

    def fork(self, trajectory_id: TrajectoryId, island_id: IslandId) -> Trajectory:
        """复制一条 trajectory 到指定 island（共享 graph 节点，仅 island 标签不同）。
        供 islands migration 使用——让同一条高价值路径在多个岛活跃，促进机制流动。"""
        t = self.get_trajectory(trajectory_id)
        new = replace(t, id=self._next_id, island_id=island_id, visit_count=0)
        self._trajectories[new.id] = new
        self._next_id += 1
        return new

    def record_visit(self, trajectory_id: TrajectoryId) -> Trajectory:
        t = self.get_trajectory(trajectory_id)
        updated = replace(t, visit_count=t.visit_count + 1)
        self._trajectories[trajectory_id] = updated
        return updated

    def set_value(self, trajectory_id: TrajectoryId, value: ValueVec, scalar: float) -> Trajectory:
        t = self.get_trajectory(trajectory_id)
        updated = replace(t, value=value, scalar_value=scalar)
        self._trajectories[trajectory_id] = updated
        return updated

    def archive(self, trajectory_id: TrajectoryId) -> Trajectory:
        t = self.get_trajectory(trajectory_id)
        updated = replace(t, status=TrajectoryStatus.ARCHIVED)
        self._trajectories[trajectory_id] = updated
        return updated

    def get_trajectory(self, trajectory_id: TrajectoryId) -> Trajectory:
        return self._trajectories[trajectory_id]

    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(self._trajectories.values())

    def active(self) -> tuple[Trajectory, ...]:
        return tuple(t for t in self._trajectories.values() if t.status == TrajectoryStatus.ACTIVE)

    def active_in_island(self, island_id: IslandId) -> tuple[Trajectory, ...]:
        return tuple(t for t in self.active() if t.island_id == island_id)

    def island_ids(self) -> tuple[IslandId, ...]:
        return tuple(sorted({t.island_id for t in self.active()}))

    def total_visits(self) -> int:
        return sum(t.visit_count for t in self.active())
