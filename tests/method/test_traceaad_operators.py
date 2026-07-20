from __future__ import annotations

from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.operators import (
    BacktrackBranchOp,
    EndpointRefineOp,
    MechanismCrossoverOp,
    NoveltyJumpOp,
    OperatorContext,
)
from llm4ad.method.traceaad.schema import ValueVec
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


def test_crossover_uses_a_different_trajectory_as_idea_donor() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    base_node = graph.add_node(code="def f(): return 1", idea="base", fitness=10.0)
    donor_node = graph.add_node(code="def g(): return 2", idea="donor idea", fitness=9.0)
    base = memory.create_initial(node_id=base_node.id)
    donor = memory.create_initial(node_id=donor_node.id)
    memory.set_value(base.id, ValueVec(quality=1.0), 1.0)
    memory.set_value(donor.id, ValueVec(quality=0.8), 0.8)
    context = _context(graph, memory, base)

    operator = MechanismCrossoverOp()
    assert operator.trigger(context)
    operator.select_base(context)

    assert context.donor_idea == "donor idea"


def test_crossover_is_unavailable_without_a_donor_trajectory() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    node = graph.add_node(code="def f(): return 1", idea="only route", fitness=10.0)
    trajectory = memory.create_initial(node_id=node.id)
    context = _context(graph, memory, trajectory)

    assert not MechanismCrossoverOp().trigger(context)


def test_crossover_uses_the_configured_similarity_weights() -> None:
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    selected_node = graph.add_node(code="def f(): return 1", idea="selected", fitness=5.0)
    quality_node = graph.add_node(code="def f(): return 1", idea="quality donor", fitness=10.0)
    different_node = graph.add_node(code="while x: y()", idea="different donor", fitness=1.0)
    selected = memory.create_initial(node_id=selected_node.id)
    quality = memory.create_initial(node_id=quality_node.id)
    different = memory.create_initial(node_id=different_node.id)
    memory.set_value(quality.id, ValueVec(quality=1.0), 1.0)
    memory.set_value(different.id, ValueVec(quality=0.0), 0.0)
    operator = MechanismCrossoverOp()

    code_context = OperatorContext(
        graph=graph,
        memory=memory,
        selected=selected,
        maximize=True,
        similarity_weights=(1.0, 0.0),
    )
    operator.select_base(code_context)
    assert code_context.donor_idea == "different donor"

    trajectory_context = OperatorContext(
        graph=graph,
        memory=memory,
        selected=selected,
        maximize=True,
        similarity_weights=(0.0, 1.0),
    )
    operator.select_base(trajectory_context)
    assert trajectory_context.donor_idea == "quality donor"


def test_novelty_starts_a_new_single_node_trajectory() -> None:
    graph, memory, trajectory = _regressing_trajectory()
    context = _context(graph, memory, trajectory)
    node = graph.add_node(code="new", idea="new route", fitness=7.0)

    fresh = NoveltyJumpOp().insert(context, node.id, None, None)

    assert fresh.node_ids == (node.id,)
    assert fresh.edge_ids == ()
