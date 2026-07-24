from __future__ import annotations

from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad.value import (
    ValueWeights,
    compute_value_vec,
    score_active_trajectories,
)


def test_trajectory_selection_prefers_better_program_when_visits_match() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    low = graph.add_node(code="low", idea="low", fitness=1.0)
    high = graph.add_node(code="high", idea="high", fitness=3.0)
    memory.create_initial(node_id=low.id)
    high_trajectory = memory.create_initial(node_id=high.id)

    selected = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(),
    )[0]

    assert selected.id == high_trajectory.id


def test_path_trend_distinguishes_improving_from_regressing_history() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    root = graph.add_node(code="root", idea="root", fitness=1.0)
    better = graph.add_node(code="better", idea="better", fitness=3.0)
    worse = graph.add_node(code="worse", idea="worse", fitness=0.0)
    initial = memory.create_initial(node_id=root.id)
    improve_edge = graph.add_edge(
        parent_id=root.id,
        child_id=better.id,
        action="improve",
        delta=2.0,
        outcome="improve",
    )
    regress_edge = graph.add_edge(
        parent_id=root.id,
        child_id=worse.id,
        action="regress",
        delta=-1.0,
        outcome="regress",
    )
    improving = memory.branch_from(
        trajectory_id=initial.id,
        base_node_id=root.id,
        child_id=better.id,
        edge_id=improve_edge.id,
    )
    regressing = memory.branch_from(
        trajectory_id=initial.id,
        base_node_id=root.id,
        child_id=worse.id,
        edge_id=regress_edge.id,
    )
    weights = ValueWeights()

    improve_value = compute_value_vec(
        trajectory=improving,
        graph=graph,
        fmin=0.0,
        fmax=3.0,
        maximize=True,
        w=weights,
    )
    regress_value = compute_value_vec(
        trajectory=regressing,
        graph=graph,
        fmin=0.0,
        fmax=3.0,
        maximize=True,
        w=weights,
    )

    assert improve_value.trend > regress_value.trend


def test_minimization_direction_is_respected() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    good = graph.add_node(code="good", idea="good", fitness=2.0)
    bad = graph.add_node(code="bad", idea="bad", fitness=5.0)
    good_trajectory = memory.create_initial(node_id=good.id)
    memory.create_initial(node_id=bad.id)

    selected = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=False,
        w=ValueWeights(),
    )[0]

    assert selected.id == good_trajectory.id
