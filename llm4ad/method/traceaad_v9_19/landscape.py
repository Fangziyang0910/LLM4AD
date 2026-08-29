"""Behavior-landscape store: trajectories, distances, neighborhoods, regions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import behave
from .schema import (
    FAR_FROM_ARCHIVE,
    INTERMEDIATE,
    MIN_NEIGHBORS,
    NEAR_KNOWN,
    NEIGHBORHOOD_FRACTION,
    NOVELTY_HIGH,
    NOVELTY_LOW,
)

Profile = list[list[list[int]]]


def midrank_percentile(values: dict[int, float]) -> dict[int, float]:
    """Ascending mid-rank percentile in [0, 1]; all-tied inputs give 0.5."""
    if not values:
        return {}
    items = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[int, float] = {}
    index = 0
    while index < len(items):
        stop = index
        while stop + 1 < len(items) and items[stop + 1][1] == items[index][1]:
            stop += 1
        mid_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[items[position][0]] = mid_rank
        index = stop + 1
    span = len(items) - 1
    if span == 0:
        return {key: 0.5 for key in values}
    return {key: (rank - 1.0) / span for key, rank in ranks.items()}


def neighborhood_size(pool_size: int) -> int:
    """k_t = min(M-1, max(2, ceil(0.05 (M-1))))."""
    if pool_size <= 0:
        raise ValueError("neighborhoods need at least one profiled node")
    if pool_size == 1:
        return 0
    return min(
        pool_size - 1,
        max(MIN_NEIGHBORS, math.ceil(NEIGHBORHOOD_FRACTION * (pool_size - 1))),
    )


def behavior_tag(novelty: float) -> str:
    if novelty < NOVELTY_LOW:
        return NEAR_KNOWN
    if novelty < NOVELTY_HIGH:
        return INTERMEDIATE
    return FAR_FROM_ARCHIVE


class Landscape:
    """Incremental trajectory cache with the distance matrix."""

    def __init__(self, *, task: str, protocol: dict) -> None:
        self.task = task
        self.protocol = protocol
        self.prefix_mode = task in behave.PREFIX_TASKS
        self._ids: list[int] = []
        self._index: dict[int, int] = {}
        self._profiles: list[Profile] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._packed_states: np.ndarray | None = None
        self._packed_lengths: np.ndarray | None = None

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(self._ids)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def distances_from(self, trajectories: Profile) -> np.ndarray:
        return behave.trajectory_distances(
            trajectories, self._profiles, prefix_mode=self.prefix_mode
        )

    def add(
        self,
        node_id: int,
        trajectories: Profile,
        *,
        distances: np.ndarray | None = None,
    ) -> None:
        if node_id in self._index:
            raise ValueError(f"profile already cached for node {node_id}")
        row = (
            np.asarray(distances, dtype=np.float32)
            if distances is not None
            else self.distances_from(trajectories)
        )
        size = len(self._ids)
        if row.shape != (size,):
            raise ValueError("distance row length does not match the archive")
        matrix = np.zeros((size + 1, size + 1), dtype=np.float32)
        matrix[:size, :size] = self._matrix
        matrix[size, :size] = row
        matrix[:size, size] = row
        self._matrix = matrix
        self._index[node_id] = size
        self._ids.append(node_id)
        self._profiles.append(trajectories)
        # Keep the packed representation alongside the Python profiles so
        # checkpoint writes do not repeatedly repack the complete archive.
        new_states, new_lengths = behave.pack_profiles([trajectories])
        if self._packed_states is None:
            self._packed_states = new_states
            self._packed_lengths = new_lengths
        elif new_states.shape[2:] == self._packed_states.shape[2:]:
            self._packed_states = np.concatenate((self._packed_states, new_states), axis=0)
            self._packed_lengths = np.concatenate((self._packed_lengths, new_lengths), axis=0)
        else:
            self._packed_states, self._packed_lengths = behave.pack_profiles(self._profiles)

    def distance(self, left: int, right: int) -> float:
        return float(self._matrix[self._index[left], self._index[right]])

    def nearest_radii(self) -> dict[int, float]:
        """m_t(x): nearest-neighbor radius of each archived node."""
        if len(self._ids) < 2:
            return {}
        size = len(self._ids)
        radii: dict[int, float] = {}
        for position, node_id in enumerate(self._ids):
            row = self._matrix[position]
            radii[node_id] = float(min(row[other] for other in range(size) if other != position))
        return radii

    def archive_novelty(self, trajectories: Profile) -> tuple[float, np.ndarray]:
        """ν and the child's distance row, computed before insertion."""
        row = self.distances_from(trajectories)
        if row.size == 0:
            return 0.5, row
        ranked = dict(self.nearest_radii())
        ranked[-1] = float(np.min(row))
        return midrank_percentile(ranked)[-1], row

    def neighbors(self, node_id: int) -> tuple[int, ...]:
        """k_t nearest other nodes; distance ties break by node id."""
        if len(self._ids) < 2:
            if len(self._ids) == 1 and self._ids[0] == node_id:
                return ()
            raise ValueError("node is not present in the behavior archive")
        row = self._matrix[self._index[node_id]]
        candidates = [
            (float(row[position]), other)
            for position, other in enumerate(self._ids)
            if other != node_id
        ]
        candidates.sort()
        return tuple(other for _, other in candidates[: neighborhood_size(len(self._ids))])

    def select_crossover_reference(
        self,
        parent_id: int,
        quality: dict[int, float],
        rng,
    ) -> int | None:
        """Choose a strong, behavior-different second parent.

        Crossover is useful only when the reference contributes a genuine
        alternative behavior.  Candidates are therefore ranked by both
        quality percentile and distance from the selected parent.  Sampling
        among the top tied frontier avoids making the operator deterministic
        while retaining a quality floor.
        """
        candidates = [node_id for node_id in self._ids if node_id != parent_id]
        if not candidates:
            return None
        q_rank = midrank_percentile({node_id: quality[node_id] for node_id in candidates})
        distances = {node_id: self.distance(parent_id, node_id) for node_id in candidates}
        d_rank = midrank_percentile(distances)
        pareto = [
            node_id
            for node_id in candidates
            if q_rank[node_id] >= 0.5 and d_rank[node_id] >= 0.5
        ]
        if pareto:
            candidates = pareto
        scores = {
            node_id: 0.5 * q_rank[node_id] + 0.5 * d_rank[node_id]
            for node_id in candidates
        }
        frontier_score = max(scores.values())
        frontier = [node_id for node_id in candidates if scores[node_id] >= frontier_score - 0.10]
        return frontier[int(rng.random() * len(frontier)) % len(frontier)]

    def state_arrays(self) -> dict[str, np.ndarray]:
        if self._ids:
            if self._packed_states is None or self._packed_lengths is None:
                self._packed_states, self._packed_lengths = behave.pack_profiles(self._profiles)
            states, lengths = self._packed_states, self._packed_lengths
        else:
            states = lengths = np.zeros(0, dtype=np.int32)
        return {
            "ids": np.asarray(self._ids, dtype=np.int64),
            "matrix": self._matrix,
            "states": states,
            "lengths": lengths,
        }

    @classmethod
    def from_state_arrays(
        cls, *, task: str, protocol: dict, arrays: dict[str, np.ndarray]
    ) -> Landscape:
        landscape = cls(task=task, protocol=protocol)
        ids = [int(value) for value in arrays["ids"].tolist()]
        matrix = np.asarray(arrays["matrix"], dtype=np.float32)
        if matrix.shape != (len(ids), len(ids)):
            raise ValueError("distance matrix shape does not match profile ids")
        landscape._ids = list(ids)
        landscape._index = {node_id: position for position, node_id in enumerate(ids)}
        landscape._matrix = matrix
        if ids:
            landscape._profiles = _unpack_profiles(arrays["states"], arrays["lengths"])
            if len(landscape._profiles) != len(ids):
                raise ValueError("profile counts do not match stored node ids")
            landscape._packed_states = np.asarray(arrays["states"], dtype=np.int32)
            landscape._packed_lengths = np.asarray(arrays["lengths"], dtype=np.int32)
        return landscape


def _unpack_profiles(states: np.ndarray, lengths: np.ndarray) -> list[Profile]:
    profiles: list[Profile] = []
    for candidate_index in range(states.shape[0]):
        profile: Profile = []
        for probe_index in range(states.shape[1]):
            trajectory = []
            for point_index in range(states.shape[2]):
                length = int(lengths[candidate_index, probe_index, point_index])
                if length <= 0:
                    continue
                trajectory.append(
                    [
                        int(value)
                        for value in states[
                            candidate_index, probe_index, point_index, :length
                        ].tolist()
                    ]
                )
            if not trajectory:
                raise ValueError("stored profile contains an empty trajectory")
            profile.append(trajectory)
        profiles.append(profile)
    return profiles


@dataclass(frozen=True, slots=True)
class RegionStats:
    q: dict[int, float]
    promise: dict[int, float]
    underdevelopment: dict[int, float]
    raw_coverage: dict[int, float]
    neighbors: dict[int, tuple[int, ...]]
    pool_size: int
    neighborhood_size: int


def region_statistics(
    *,
    landscape: Landscape,
    quality: dict[int, float],
    opportunities: dict[int, int],
) -> RegionStats:
    ids = landscape.node_ids
    if set(quality) != set(ids):
        missing = sorted(set(ids) - set(quality))
        raise RuntimeError(f"profiled nodes without a quality value: {missing}")
    q = midrank_percentile(quality)
    neighbors = {node_id: landscape.neighbors(node_id) for node_id in ids}
    promise: dict[int, float] = {}
    raw_coverage: dict[int, float] = {}
    for node_id in ids:
        neighbor_qs = [q[neighbor] for neighbor in neighbors[node_id]]
        neighbor_median = q[node_id] if not neighbor_qs else float(np.median(neighbor_qs))
        promise[node_id] = (2.0 * q[node_id] + neighbor_median) / 3.0
        raw_coverage[node_id] = float(
            sum(opportunities.get(member, 0) for member in (node_id, *neighbors[node_id]))
        )
    coverage_rank = midrank_percentile(
        {node_id: math.log1p(value) for node_id, value in raw_coverage.items()}
    )
    return RegionStats(
        q=q,
        promise=promise,
        underdevelopment={node_id: 1.0 - coverage_rank[node_id] for node_id in ids},
        raw_coverage=raw_coverage,
        neighbors=neighbors,
        pool_size=len(ids),
        neighborhood_size=neighborhood_size(len(ids)),
    )


__all__ = [
    "Landscape",
    "Profile",
    "RegionStats",
    "behavior_tag",
    "midrank_percentile",
    "neighborhood_size",
    "region_statistics",
]
