from __future__ import annotations

import inspect

import llm4ad.method.traceaad_v4 as traceaad_package
from llm4ad.method.traceaad_v4 import TraceAADV4, ValueWeights
from llm4ad.method.traceaad_v4.context import build_action_prompt
from llm4ad.method.traceaad_v4.derivation_graph import DerivationGraph
from llm4ad.method.traceaad_v4.operators import DEFAULT_OPERATORS
from llm4ad.method.traceaad_v4.trajectory_memory import TrajectoryMemory


def _history_fixture():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=4)
    root = graph.add_node(code="def f(x):\n    return x", idea="root", fitness=1.0)
    child = graph.add_node(
        code="def f(x):\n    return x + 1", idea="add one", fitness=2.0
    )
    edge = graph.add_edge(
        parent_id=root.id,
        child_id=child.id,
        action="add a unit improvement",
        operator="trace_refine",
        delta=1.0,
        outcome="improve",
        iteration=0,
    )
    trajectory = memory.create_initial(node_id=root.id)
    trajectory = memory.branch_from(
        trajectory_id=trajectory.id,
        base_node_id=root.id,
        child_id=child.id,
        edge_id=edge.id,
    )
    return graph, memory, trajectory


def test_public_configuration_exposes_only_search_mechanism_controls():
    parameters = inspect.signature(TraceAADV4).parameters
    assert "resume_mode" not in parameters
    assert "elite_count" not in parameters
    assert "num_evaluators" not in parameters
    assert "multi_thread_or_process_eval" not in parameters
    assert "n_islands" not in parameters
    assert "migration_interval" not in parameters
    assert "sampling_strategy" not in parameters
    assert "random_seed" not in parameters

    assert tuple(ValueWeights.__dataclass_fields__) == (
        "w_quality",
        "w_trend",
        "discount",
        "positive_threshold",
        "ucb_c",
    )
    assert set(traceaad_package.__all__) == {
        "RunArtifacts",
        "TraceAADV4",
        "TraceAADRunResult",
        "ValueWeights",
    }


def test_action_prompt_uses_single_trajectory_history_only():
    graph, _, trajectory = _history_fixture()
    template_function = type(
        "FunctionLike",
        (),
        {
            "__deepcopy__": lambda self, memo: self,
            "__str__": lambda self: "def f(x): ...",
            "body": "",
        },
    )()

    prompt = build_action_prompt(
        graph=graph,
        trajectory=trajectory,
        base_node_id=trajectory.endpoint_id,
        base_reason="endpoint",
        operator_name="trace_refine",
        operator_constraint="continue the current direction",
        task_description="Improve f.",
        template_function=template_function,
        action_count=1,
        maximize=True,
    )

    assert "add a unit improvement" in prompt
    assert "[Algorithm Improvement History]" in prompt
    assert "[Cross-Trajectory Action Evidence]" not in prompt
    assert "Elite Curriculum" not in prompt
    assert "Contrast Feedback" not in prompt
    assert "complexity" not in prompt.lower()
    assert "runtime" not in prompt.lower()


def test_v4_uses_two_uniform_single_parent_semantic_operators():
    assert {operator.name for operator in DEFAULT_OPERATORS} == {
        "trace_ideate",
        "trace_refine",
    }
