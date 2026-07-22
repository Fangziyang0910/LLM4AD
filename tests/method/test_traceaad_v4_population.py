from __future__ import annotations

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad import TraceAAD
from llm4ad.method.traceaad.operators import TraceSynthesizeOp
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad.value import (
    ValueWeights,
    compute_value_vec,
    score_active_trajectories,
    select_diverse_trajectories,
)


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class _V4LLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "Requested Modification" not in prompt:
            return "1. Combine the first complete trajectory principle with the second one.\n2. Preserve both tested mechanisms."
        return self._program(self.calls)

    @staticmethod
    def _program(value: int) -> str:
        return (
            f"Idea: candidate {value}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {value}\n"
            "```"
        )


class _V4Evaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return float(self.calls)


def _method(*, budget: int, population: int = 3, n_init: int = 3, operators=None) -> TraceAAD:
    return TraceAAD(
        llm=_V4LLM(),
        evaluation=_V4Evaluation(),
        max_sample_nums=budget,
        n_init=n_init,
        actions_per_iteration=2,
        max_active_trajectories=population,
        operators=(TraceSynthesizeOp,) if operators else None,
    ) if operators else TraceAAD(
        llm=_V4LLM(),
        evaluation=_V4Evaluation(),
        max_sample_nums=budget,
        n_init=n_init,
        actions_per_iteration=2,
        max_active_trajectories=population,
    )


def test_expansion_pool_is_not_managed_before_two_times_population():
    method = _method(budget=3, population=3, n_init=3)
    method._initialize()
    assert len(method._memory.active()) == 3
    node = method._graph.add_node(code="def choose(value): return value + 1", idea="child", fitness=-1.0)
    method._memory.create_initial(node_id=node.id)

    method._maybe_manage_population()

    assert len(method._memory.active()) == 4
    assert len(method._memory.trajectories()) == 4


def test_population_management_returns_to_exact_target_size():
    method = _method(budget=3, population=3, n_init=3)
    method._initialize()
    for index in range(3):
        node = method._graph.add_node(
            code=f"def choose(value): return value + {index}",
            idea=f"child {index}",
            fitness=float(index),
        )
        method._memory.create_initial(node_id=node.id)

    assert len(method._memory.active()) == 6
    method._maybe_manage_population()

    assert len(method._memory.active()) == 3
    assert len(method._memory.trajectories()) == 6


def test_synthesis_evaluates_one_child_and_writes_two_trajectory_histories():
    method = _method(budget=4, population=2, n_init=2, operators=True)
    result = method.run()

    assert result.n_samples == 4
    assert result.n_total_nodes == 4
    assert result.n_trajectories == 6
    assert result.n_edges == 4
    assert len(method.active_trajectories()) == 2
    assert sum(trajectory.status.value == "archived" for trajectory in method._memory.trajectories()) == 4


def test_trajectory_quality_keeps_a_small_credit_for_best_historical_node():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    root = graph.add_node(code="root", idea="root", fitness=1.0)
    peak = graph.add_node(code="peak", idea="valuable idea", fitness=10.0)
    regress = graph.add_node(code="regress", idea="failed follow-up", fitness=2.0)
    current = graph.add_node(code="current", idea="ordinary", fitness=3.0)
    bad = graph.add_node(code="bad", idea="same endpoint without a valuable history", fitness=2.0)
    route = memory.create_initial(node_id=root.id)
    edge1 = graph.add_edge(parent_id=root.id, child_id=peak.id, action="peak", delta=9.0, outcome="improve")
    route = memory.extend(trajectory_id=route.id, parent_id=root.id, child_id=peak.id, edge_id=edge1.id)
    edge2 = graph.add_edge(parent_id=peak.id, child_id=regress.id, action="regress", delta=-8.0, outcome="regress")
    route = memory.extend(trajectory_id=route.id, parent_id=peak.id, child_id=regress.id, edge_id=edge2.id)
    memory.create_initial(node_id=current.id)
    bad_route = memory.create_initial(node_id=bad.id)

    scored = {trajectory.id: trajectory for trajectory in score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(),
    )}

    assert scored[route.id].value.quality > scored[bad_route.id].value.quality
    assert scored[route.id].scalar_value > scored[bad_route.id].scalar_value


def test_absolute_quality_normalization_respects_minimization_direction():
    graph = DerivationGraph()
    node = graph.add_node(code="candidate", idea="candidate", fitness=2.0)
    memory = TrajectoryMemory()
    trajectory = memory.create_initial(node_id=node.id)

    value = compute_value_vec(
        trajectory=trajectory,
        graph=graph,
        active_others=(),
        fmin=1.0,
        fmax=5.0,
        maximize=False,
        w=ValueWeights(),
    )

    assert value.quality == pytest.approx(0.75)


def test_diversity_representatives_keep_routes_with_different_signatures():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    nodes = [
        graph.add_node(code="def f(x): return x + offset", idea="a", fitness=10.0),
        graph.add_node(code="def f(x): return x + offset", idea="b", fitness=9.0),
        graph.add_node(code="def f(items):\n    return max(items)", idea="c", fitness=2.0),
    ]
    tuple(memory.create_initial(node_id=node.id) for node in nodes)
    scored = score_active_trajectories(memory=memory, graph=graph, maximize=True, w=ValueWeights())

    diverse = select_diverse_trajectories(
        candidates=scored,
        graph=graph,
        count=2,
    )

    assert len(diverse) == 2
    assert {graph.get_node(trajectory.endpoint_id).code for trajectory in diverse} == {
        "def f(x): return x + offset",
        "def f(items):\n    return max(items)",
    }
