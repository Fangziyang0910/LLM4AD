from __future__ import annotations

from dataclasses import replace

from .schema import ImprovementEdge, NodeId, Trajectory, TrajectoryId, TrajectoryStatus


class TrajectoryLibrary:
    def __init__(self, *, max_trajectory_length: int = 8) -> None:
        if max_trajectory_length < 2:
            raise ValueError("max_trajectory_length must be at least 2")
        self.max_trajectory_length = max_trajectory_length
        self._next_trajectory_id = 0
        self._trajectories: dict[TrajectoryId, Trajectory] = {}

    def create_initial(self, *, node_id: NodeId, score: float | None = None) -> Trajectory:
        trajectory = Trajectory(
            id=self._next_trajectory_id,
            node_ids=(node_id,),
            edge_ids=(),
            endpoint_id=node_id,
            score=score,
        )
        self._trajectories[trajectory.id] = trajectory
        self._next_trajectory_id += 1
        return trajectory

    def extend(
            self,
            *,
            trajectory_id: TrajectoryId,
            edge: ImprovementEdge,
            score: float | None = None,
    ) -> Trajectory:
        parent = self.get_trajectory(trajectory_id)
        if parent.status != TrajectoryStatus.ACTIVE:
            raise ValueError(f"cannot extend archived trajectory: {trajectory_id}")
        if edge.parent_id != parent.endpoint_id:
            raise ValueError(
                "edge parent must match the trajectory endpoint: "
                f"{edge.parent_id} != {parent.endpoint_id}"
            )
        return self.branch_from(
            trajectory_id=trajectory_id,
            base_node_id=parent.endpoint_id,
            edge=edge,
            score=score,
        )

    def branch_from(
            self,
            *,
            trajectory_id: TrajectoryId,
            base_node_id: NodeId,
            edge: ImprovementEdge,
            score: float | None = None,
    ) -> Trajectory:
        parent = self.get_trajectory(trajectory_id)
        if parent.status != TrajectoryStatus.ACTIVE:
            raise ValueError(f"cannot branch from archived trajectory: {trajectory_id}")
        if base_node_id not in parent.node_ids:
            raise ValueError(
                "base node must belong to the trajectory: "
                f"{base_node_id} not in {parent.node_ids}"
            )
        if edge.parent_id != base_node_id:
            raise ValueError(
                "edge parent must match the selected base node: "
                f"{edge.parent_id} != {base_node_id}"
            )

        base_index = parent.node_ids.index(base_node_id)
        node_ids = (*parent.node_ids[: base_index + 1], edge.child_id)
        edge_ids = (*parent.edge_ids[:base_index], edge.id)
        overflow = len(node_ids) - self.max_trajectory_length
        if overflow > 0:
            node_ids = node_ids[overflow:]
            edge_ids = edge_ids[overflow:]

        trajectory = Trajectory(
            id=self._next_trajectory_id,
            node_ids=node_ids,
            edge_ids=edge_ids,
            endpoint_id=edge.child_id,
            score=score,
        )
        self._trajectories[trajectory.id] = trajectory
        self._next_trajectory_id += 1
        return trajectory

    def record_visit(self, trajectory_id: TrajectoryId) -> Trajectory:
        trajectory = self.get_trajectory(trajectory_id)
        updated = replace(trajectory, visit_count=trajectory.visit_count + 1)
        self._trajectories[trajectory_id] = updated
        return updated

    def set_score(self, trajectory_id: TrajectoryId, score: float | None) -> Trajectory:
        trajectory = self.get_trajectory(trajectory_id)
        updated = replace(trajectory, score=score)
        self._trajectories[trajectory_id] = updated
        return updated

    def archive(self, trajectory_id: TrajectoryId) -> Trajectory:
        trajectory = self.get_trajectory(trajectory_id)
        updated = replace(trajectory, status=TrajectoryStatus.ARCHIVED)
        self._trajectories[trajectory_id] = updated
        return updated

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
