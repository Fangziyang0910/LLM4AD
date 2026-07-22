from __future__ import annotations

from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.operators import (
    BacktrackBranchOp,
    EndpointRefineOp,
    NoveltyJumpOp,
    OperatorContext,
)
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory


def _regressing_trajectory():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    root = graph.add_node(code="root", idea="root", fitness=10.0)
    child = graph.add_node(code="child", idea="bad change", fitness=8.0)
    edge = graph.add_edge(
        parent_id=root.id,
        child_id=child.id,
        action="bad change",
        operator="endpoint_refine",
        delta=-2.0,
        outcome="regress",
    )
    trajectory = memory.create_initial(node_id=root.id)
    trajectory = memory.extend(
        trajectory_id=trajectory.id,
        parent_id=root.id,
        child_id=child.id,
        edge_id=edge.id,
    )
    return graph, memory, trajectory


def _context(graph, memory, trajectory):
    return OperatorContext(
        graph=graph,
        memory=memory,
        selected=trajectory,
        maximize=True,
    )


def test_backtrack_selects_an_earlier_program_after_regression() -> None:
    graph, memory, trajectory = _regressing_trajectory()
    operator = BacktrackBranchOp()
    context = _context(graph, memory, trajectory)

    target = operator.select_trajectory(context)
    assert target is not None
    context.selected = target
    base_id, reason = operator.select_base(context)

    assert base_id == target.node_ids[0]
    assert reason in {"last_regressed", "endpoint_not_best"}


def test_endpoint_extends_the_selected_trajectory() -> None:
    graph, memory, trajectory = _regressing_trajectory()
    context = _context(graph, memory, trajectory)
    child = graph.add_node(code="better", idea="repair", fitness=11.0)
    edge = graph.add_edge(
        parent_id=trajectory.endpoint_id,
        child_id=child.id,
        action="repair",
    )

    extended = EndpointRefineOp().insert(
        context,
        child.id,
        edge.id,
        trajectory.endpoint_id,
    )

    assert extended.node_ids[-2:] == (trajectory.endpoint_id, child.id)


def test_novelty_starts_a_new_single_node_trajectory() -> None:
    graph, memory, trajectory = _regressing_trajectory()
    context = _context(graph, memory, trajectory)
    node = graph.add_node(code="new", idea="new route", fitness=7.0)

    fresh = NoveltyJumpOp().insert(context, node.id, None, None)

    assert fresh.node_ids == (node.id,)
    assert fresh.edge_ids == ()
