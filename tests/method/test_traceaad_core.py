from __future__ import annotations

import inspect

import llm4ad.method.traceaad as traceaad_package
from llm4ad.method.traceaad import PortfolioWeights, TraceAAD, ValueWeights
from llm4ad.method.traceaad.context import build_action_prompt
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.experience_memory import ExperienceMemory
from llm4ad.method.traceaad.operators import EndpointRefineOp, NoveltyJumpOp, OperatorContext
from llm4ad.method.traceaad.portfolio import OperatorPortfolio
from llm4ad.method.traceaad.schema import ValueVec
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory


def _history_fixture():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=4)
    root = graph.add_node(code="def f(x):\n    return x", idea="root", fitness=1.0)
    child = graph.add_node(code="def f(x):\n    return x + 1", idea="add one", fitness=2.0)
    edge = graph.add_edge(
        parent_id=root.id,
        child_id=child.id,
        action="add a unit improvement",
        operator="endpoint_refine",
        delta=1.0,
        outcome="improve",
        iteration=0,
    )
    trajectory = memory.create_initial(node_id=root.id)
    trajectory = memory.extend(
        trajectory_id=trajectory.id,
        parent_id=root.id,
        child_id=child.id,
        edge_id=edge.id,
    )
    return graph, memory, trajectory


def test_public_configuration_exposes_only_search_mechanism_controls():
    parameters = inspect.signature(TraceAAD).parameters
    assert "resume_mode" not in parameters
    assert "num_evaluators" not in parameters
    assert "multi_thread_or_process_eval" not in parameters
    assert "n_islands" not in parameters
    assert "migration_interval" not in parameters
    assert "sampling_strategy" not in parameters
    assert "random_seed" not in parameters

    assert tuple(ValueWeights.__dataclass_fields__) == (
        "w_quality",
        "w_potential",
        "w_diversity",
        "w_sim_code",
        "w_sim_trajectory",
        "discount",
        "positive_threshold",
        "ucb_c",
    )
    assert tuple(PortfolioWeights.__dataclass_fields__) == ("ucb_c",)
    assert set(traceaad_package.__all__) == {
        "TraceAAD",
        "TraceAADRunResult",
        "TraceAADProfiler",
        "ValueWeights",
        "PortfolioWeights",
    }


def test_action_prompt_uses_current_trajectory_and_cross_trajectory_actions_only():
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
        operator_name="endpoint_refine",
        operator_constraint="continue the current direction",
        experience_memory=ExperienceMemory(graph),
        task_description="Improve f.",
        template_function=template_function,
        action_count=1,
        maximize=True,
    )

    assert "add a unit improvement" in prompt
    assert "[Algorithm Improvement History]" in prompt
    assert "[Cross-Trajectory Action Evidence]" in prompt
    assert "Elite Curriculum" not in prompt
    assert "Contrast Feedback" not in prompt
    assert "complexity" not in prompt.lower()
    assert "runtime" not in prompt.lower()


def test_operator_selection_uses_empirical_reward_plus_ucb():
    graph, memory, trajectory = _history_fixture()
    ctx = OperatorContext(
        graph=graph,
        memory=memory,
        selected=trajectory,
        maximize=True,
    )
    endpoint = EndpointRefineOp()
    novelty = NoveltyJumpOp()
    portfolio = OperatorPortfolio(
        (endpoint, novelty),
        PortfolioWeights(ucb_c=0.5),
    )

    first = portfolio.choose(ctx)
    portfolio.record(first.operator, reward=-1.0)
    second = portfolio.choose(ctx)

    assert first.operator.name != second.operator.name
    assert set(second.scores) == {"endpoint_refine", "novelty_jump"}


def test_trajectory_value_has_only_quality_potential_and_diversity():
    value = ValueVec(quality=0.8, potential=0.3, diversity=0.6)
    assert value.as_tuple() == (0.8, 0.3, 0.6)
