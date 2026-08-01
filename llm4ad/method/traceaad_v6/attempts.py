"""Anchor-level tested attempts for TraceAAD V6 evidence."""

from __future__ import annotations

from dataclasses import asdict

from .schema import AnchorAttempt, EdgeId, NodeId


class AttemptMemory:
    """Stores successful edges and failed candidates keyed by anchor node."""

    def __init__(self) -> None:
        self._attempts: list[AnchorAttempt] = []

    def record(self, attempt: AnchorAttempt) -> AnchorAttempt:
        self._attempts.append(attempt)
        return attempt

    def for_anchor(self, anchor_node_id: NodeId) -> tuple[AnchorAttempt, ...]:
        return tuple(
            attempt
            for attempt in self._attempts
            if attempt.anchor_node_id == anchor_node_id
        )

    def all(self) -> tuple[AnchorAttempt, ...]:
        return tuple(self._attempts)

    def to_dict(self) -> dict:
        return {"attempts": [asdict(attempt) for attempt in self._attempts]}

    @classmethod
    def from_dict(cls, payload: dict) -> AttemptMemory:
        memory = cls()
        for item in payload.get("attempts", []):
            memory._attempts.append(
                AnchorAttempt(
                    anchor_node_id=int(item["anchor_node_id"]),
                    primary_trajectory_id=int(item["primary_trajectory_id"]),
                    operator=str(item["operator"]),
                    action=str(item["action"]),
                    iteration=item["iteration"],
                    status=str(item["status"]),
                    idea=str(item.get("idea", "")),
                    fitness=item.get("fitness"),
                    delta_parent=item.get("delta_parent"),
                    delta_route_best=item.get("delta_route_best"),
                    delta_global_best=item.get("delta_global_best"),
                    outcome=str(item.get("outcome", "unknown")),
                    edge_id=item.get("edge_id"),
                    child_id=item.get("child_id"),
                    program_loc=item.get("program_loc"),
                    failure_kind=item.get("failure_kind"),
                    new_route_best=bool(item.get("new_route_best", False)),
                    new_global_best=bool(item.get("new_global_best", False)),
                )
            )
        return memory


def select_tested_attempts(
    attempts: tuple[AnchorAttempt, ...],
    *,
    excluded_edge_ids: set[EdgeId] | frozenset[EdgeId] = frozenset(),
    limit: int = 6,
) -> tuple[AnchorAttempt, ...]:
    """Prefer best-updates, then failures/regressions, then recent attempts."""
    if limit <= 0:
        return ()
    filtered = [
        attempt
        for attempt in attempts
        if attempt.edge_id is None or attempt.edge_id not in excluded_edge_ids
    ]
    if not filtered:
        return ()

    def priority(attempt: AnchorAttempt) -> tuple[int, int]:
        if attempt.new_global_best or attempt.new_route_best:
            tier = 0
        elif attempt.status != "valid" or attempt.outcome in {
            "regress",
            "plateau",
            "unknown",
        }:
            tier = 1
        else:
            tier = 2
        # Later attempts first inside a tier (iteration then insertion order).
        order = -1 if attempt.iteration is None else -int(attempt.iteration)
        return (tier, order)

    ranked = sorted(enumerate(filtered), key=lambda item: (*priority(item[1]), -item[0]))
    selected: list[AnchorAttempt] = []
    seen: set[int] = set()
    for index, attempt in ranked:
        if index in seen:
            continue
        selected.append(attempt)
        seen.add(index)
        if len(selected) >= limit:
            break
    return tuple(selected)


__all__ = ["AttemptMemory", "select_tested_attempts"]
