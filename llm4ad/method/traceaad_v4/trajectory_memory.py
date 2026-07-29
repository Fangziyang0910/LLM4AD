"""保存、扩展和筛选有界算法改进轨迹。"""

from __future__ import annotations

from dataclasses import replace

from .schema import (
    EdgeId,
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

    def create_initial(self, *, node_id: NodeId) -> Trajectory:
        traj = Trajectory(
            id=self._next_id,
            node_ids=(node_id,),
            edge_ids=(),
            endpoint_id=node_id,
        )
        self._trajectories[traj.id] = traj
        self._next_id += 1
        return traj

    def branch_from(
        self,
        *,
        trajectory_id: TrajectoryId,
        base_node_id: NodeId,
        child_id: NodeId,
        edge_id: EdgeId,
    ) -> Trajectory:
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
        endpoint_id = node_ids[-1]

        traj = Trajectory(
            id=self._next_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            endpoint_id=endpoint_id,
        )
        self._trajectories[traj.id] = traj
        self._next_id += 1
        return traj

    def record_visit(self, trajectory_id: TrajectoryId) -> Trajectory:
        t = self.get_trajectory(trajectory_id)
        updated = replace(t, visit_count=t.visit_count + 1)
        self._trajectories[trajectory_id] = updated
        return updated

    def set_value(
        self, trajectory_id: TrajectoryId, value: ValueVec, scalar: float
    ) -> Trajectory:
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
        return tuple(
            t
            for t in self._trajectories.values()
            if t.status == TrajectoryStatus.ACTIVE
        )
