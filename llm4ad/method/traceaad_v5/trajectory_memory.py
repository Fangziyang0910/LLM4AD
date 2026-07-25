"""Full trajectory state for TraceAAD v5; prompt windows never truncate state."""

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
    def __init__(self, *, prompt_window: int = 8) -> None:
        if prompt_window < 1:
            raise ValueError("prompt_window must be positive")
        self.prompt_window = int(prompt_window)
        # Compatibility name used by generic parameter/profiler code.
        self.max_trajectory_length = self.prompt_window
        self._next_id = 0
        self._trajectories: dict[TrajectoryId, Trajectory] = {}

    def create_initial(self, *, node_id: NodeId) -> Trajectory:
        trajectory = Trajectory(
            id=self._next_id,
            node_ids=(node_id,),
            edge_ids=(),
            endpoint_id=node_id,
            compact_best_id=node_id,
        )
        self._trajectories[trajectory.id] = trajectory
        self._next_id += 1
        return trajectory

    def branch_from(
        self,
        *,
        trajectory_id: TrajectoryId,
        base_node_id: NodeId,
        child_id: NodeId,
        edge_id: EdgeId,
        compact_best_id: NodeId,
    ) -> Trajectory:
        parent = self.get_trajectory(trajectory_id)
        if parent.status != TrajectoryStatus.ACTIVE:
            raise ValueError(f"cannot branch from archived trajectory: {trajectory_id}")
        if base_node_id not in parent.node_ids:
            raise ValueError(f"base node must belong to the trajectory: {base_node_id}")
        base_index = parent.node_ids.index(base_node_id)
        node_ids = (*parent.node_ids[: base_index + 1], child_id)
        edge_ids = (*parent.edge_ids[:base_index], edge_id)
        if len(node_ids) != len(edge_ids) + 1:
            raise AssertionError("trajectory path is inconsistent")
        trajectory = Trajectory(
            id=self._next_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            endpoint_id=child_id,
            compact_best_id=compact_best_id,
        )
        self._trajectories[trajectory.id] = trajectory
        self._next_id += 1
        return trajectory

    def record_visit(self, trajectory_id: TrajectoryId) -> Trajectory:
        return self._replace(
            trajectory_id,
            visit_count=self.get_trajectory(trajectory_id).visit_count + 1,
        )

    def record_reference_use(self, trajectory_id: TrajectoryId) -> Trajectory:
        return self._replace(
            trajectory_id,
            reference_use_count=(
                self.get_trajectory(trajectory_id).reference_use_count + 1
            ),
        )

    def set_value(
        self, trajectory_id: TrajectoryId, value: ValueVec, scalar: float
    ) -> Trajectory:
        return self._replace(trajectory_id, value=value, scalar_value=scalar)

    def archive(self, trajectory_id: TrajectoryId) -> Trajectory:
        return self._replace(trajectory_id, status=TrajectoryStatus.ARCHIVED)

    def get_trajectory(self, trajectory_id: TrajectoryId) -> Trajectory:
        return self._trajectories[trajectory_id]

    def trajectories(self) -> tuple[Trajectory, ...]:
        return tuple(self._trajectories.values())

    def active(self) -> tuple[Trajectory, ...]:
        return tuple(
            trajectory
            for trajectory in self._trajectories.values()
            if trajectory.status == TrajectoryStatus.ACTIVE
        )

    def _replace(self, trajectory_id: TrajectoryId, **changes) -> Trajectory:
        updated = replace(self.get_trajectory(trajectory_id), **changes)
        self._trajectories[trajectory_id] = updated
        return updated


__all__ = ["TrajectoryMemory"]
