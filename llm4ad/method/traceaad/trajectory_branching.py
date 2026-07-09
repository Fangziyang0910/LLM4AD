from __future__ import annotations

from dataclasses import dataclass

from .derivation_graph import DerivationGraph
from .schema import NodeId, Trajectory


@dataclass(frozen=True, slots=True)
class BaseNodeSelection:
    node_id: NodeId
    reason: str


@dataclass(frozen=True, slots=True)
class StepOutcome:
    parent_id: NodeId
    child_id: NodeId
    delta: float | None
    outcome: str


def select_base_node(
        *,
        graph: DerivationGraph,
        trajectory: Trajectory,
        maximize: bool,
        positive_threshold: float = 1e-6,
) -> BaseNodeSelection:
    if len(trajectory.node_ids) == 0:
        raise ValueError("trajectory must contain at least one node")
    if positive_threshold < 0:
        raise ValueError("positive_threshold must be non-negative")
    if not trajectory.edge_ids:
        return BaseNodeSelection(node_id=trajectory.endpoint_id, reason="initial")

    outcomes = trajectory_step_outcomes(
        graph=graph,
        trajectory=trajectory,
        maximize=maximize,
        positive_threshold=positive_threshold,
    )
    last = outcomes[-1]
    if last.outcome == "regressed":
        return BaseNodeSelection(node_id=last.parent_id, reason="last_regressed")
    if (
            len(outcomes) >= 2
            and outcomes[-1].outcome == "unchanged"
            and outcomes[-2].outcome == "unchanged"
    ):
        for outcome in reversed(outcomes):
            if outcome.outcome == "improved":
                return BaseNodeSelection(node_id=outcome.child_id, reason="recent_plateau")

    best_node_id = _best_node_id_in_trajectory(graph=graph, trajectory=trajectory, maximize=maximize)
    if best_node_id is not None and best_node_id != trajectory.endpoint_id:
        return BaseNodeSelection(node_id=best_node_id, reason="endpoint_not_best")
    return BaseNodeSelection(node_id=trajectory.endpoint_id, reason="endpoint")


def trajectory_step_outcomes(
        *,
        graph: DerivationGraph,
        trajectory: Trajectory,
        maximize: bool,
        positive_threshold: float = 1e-6,
) -> tuple[StepOutcome, ...]:
    if len(trajectory.edge_ids) != len(trajectory.node_ids) - 1:
        raise ValueError("trajectory edge count must equal node count minus one")

    outcomes: list[StepOutcome] = []
    for parent_id, child_id in zip(trajectory.node_ids, trajectory.node_ids[1:]):
        parent = graph.get_node(parent_id)
        child = graph.get_node(child_id)
        delta = _directed_delta(parent.fitness, child.fitness, maximize=maximize)
        if delta is None:
            outcome = "unknown"
        elif delta > positive_threshold:
            outcome = "improved"
        elif delta < -positive_threshold:
            outcome = "regressed"
        else:
            outcome = "unchanged"
        outcomes.append(
            StepOutcome(
                parent_id=parent_id,
                child_id=child_id,
                delta=delta,
                outcome=outcome,
            )
        )
    return tuple(outcomes)


def _best_node_id_in_trajectory(
        *,
        graph: DerivationGraph,
        trajectory: Trajectory,
        maximize: bool,
) -> NodeId | None:
    best_id: NodeId | None = None
    best_fitness: float | None = None
    for node_id in trajectory.node_ids:
        node = graph.get_node(node_id)
        if not node.is_valid or node.fitness is None:
            continue
        if best_fitness is None or _is_better(node.fitness, best_fitness, maximize=maximize):
            best_id = node_id
            best_fitness = node.fitness
    return best_id


def _directed_delta(
        parent_fitness: float | None,
        child_fitness: float | None,
        *,
        maximize: bool,
) -> float | None:
    if parent_fitness is None or child_fitness is None:
        return None
    return child_fitness - parent_fitness if maximize else parent_fitness - child_fitness


def _is_better(candidate: float, incumbent: float, *, maximize: bool) -> bool:
    return candidate > incumbent if maximize else candidate < incumbent
