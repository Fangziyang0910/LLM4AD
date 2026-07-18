"""EliteCurriculum 的事实等级、构造和反馈测试。"""
from __future__ import annotations

from llm4ad.base import Function
from llm4ad.method.traceaad.context import _curriculum_block, build_action_prompt
from llm4ad.method.traceaad.curriculum import EliteCurriculum
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.experience_memory import ExperienceMemory
from llm4ad.method.traceaad.schema import OperatorName
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory


def _graph() -> tuple[DerivationGraph, int, int, int]:
    graph = DerivationGraph()
    root = graph.add_node(code="root", idea="root", fitness=0.0)
    first = graph.add_node(code="first", idea="first", fitness=1.0)
    second = graph.add_node(code="second", idea="second", fitness=2.0)
    failed = graph.add_node(code="failed", idea="failed", fitness=0.5)
    graph.add_edge(
        parent_id=root.id,
        child_id=first.id,
        action="add a stable ranking signal",
        operator=OperatorName.ENDPOINT,
        delta=1.0,
        outcome="improve",
        iteration=1,
    )
    graph.add_edge(
        parent_id=first.id,
        child_id=second.id,
        action="calibrate the ranking signal",
        operator=OperatorName.ENDPOINT,
        delta=1.0,
        outcome="improve",
        iteration=2,
    )
    graph.add_edge(
        parent_id=first.id,
        child_id=failed.id,
        action="add an uncontrolled bonus",
        operator=OperatorName.ENDPOINT,
        delta=-0.5,
        outcome="regress",
        iteration=3,
    )
    return graph, root.id, first.id, second.id


def test_curriculum_preserves_jump_semantics_and_builds_repair() -> None:
    graph, root_id, first_id, second_id = _graph()
    curriculum = EliteCurriculum(graph)
    curriculum.record_best_event(
        previous_best_node_id=None,
        new_best_node_id=root_id,
        operator="init",
        sample_order=1,
    )
    curriculum.record_best_event(
        previous_best_node_id=root_id,
        new_best_node_id=second_id,
        operator=OperatorName.ENDPOINT,
        iteration=2,
        sample_order=3,
    )

    packet = curriculum.build(
        operator=OperatorName.BACKTRACK,
        base_node_id=first_id,
        selected_trajectory_id=None,
        iteration=3,
        stagnation=1,
    )

    assert packet.trace_ids
    assert packet.repair_trace is not None
    assert packet.repair_trace.kind == "prefix_repair"
    assert packet.positive_traces[0].kind == "champion"
    assert packet.positive_traces[0].steps[-1].causal_status == "jump"
    assert len(graph.edges()) == 3


def test_curriculum_feedback_is_packet_level() -> None:
    graph, root_id, _, second_id = _graph()
    curriculum = EliteCurriculum(graph)
    curriculum.record_best_event(
        previous_best_node_id=None,
        new_best_node_id=root_id,
        operator="init",
    )
    curriculum.record_best_event(
        previous_best_node_id=root_id,
        new_best_node_id=second_id,
        operator=OperatorName.ENDPOINT,
    )
    packet = curriculum.build(
        operator=OperatorName.ENDPOINT,
        base_node_id=second_id,
        selected_trajectory_id=None,
        iteration=1,
        stagnation=0,
    )
    curriculum.record_outcome(packet, outcome="improve", global_best=True)
    snapshot = curriculum.snapshot()
    assert snapshot["champion_events"] == 2
    assert all(value > 0 for value in snapshot["trace_reward"].values())


def test_curriculum_prompt_labels_evidence_level() -> None:
    graph, root_id, _, second_id = _graph()
    curriculum = EliteCurriculum(graph)
    curriculum.record_best_event(
        previous_best_node_id=None,
        new_best_node_id=root_id,
        operator="init",
    )
    curriculum.record_best_event(
        previous_best_node_id=root_id,
        new_best_node_id=second_id,
        operator=OperatorName.ENDPOINT,
    )
    packet = curriculum.build(
        operator=OperatorName.ENDPOINT,
        base_node_id=second_id,
        selected_trajectory_id=None,
        iteration=1,
        stagnation=0,
    )
    block = _curriculum_block(packet)
    assert "champion" in block
    assert "causal_status=jump" in block

    trajectory = TrajectoryMemory().create_initial(node_id=second_id)
    prompt = build_action_prompt(
        graph=graph,
        trajectory=trajectory,
        base_node_id=second_id,
        base_reason="endpoint",
        operator_name=OperatorName.ENDPOINT,
        operator_role="exploit",
        operator_constraint="refine",
        experience_memory=ExperienceMemory(graph),
        contrast=None,
        task_description="task",
        template_function=Function(name="f", args="x", body="    pass\n"),
        action_count=1,
        maximize=True,
        curriculum=packet,
    )
    assert "[Elite Curriculum]" in prompt
    assert "not guaranteed rules" in prompt
