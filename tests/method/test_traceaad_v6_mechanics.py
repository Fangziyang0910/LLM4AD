"""Mechanism contract tests for TraceAAD V6."""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v6 import TraceAADProfiler, TraceAADV6, ValueWeights
from llm4ad.method.traceaad_v6.attempts import select_tested_attempts
from llm4ad.method.traceaad_v6.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v6.derivation_graph import DerivationGraph
from llm4ad.method.traceaad_v6.operators import (
    DEFAULT_OPERATORS,
    OperatorName,
    select_operator,
)
from llm4ad.method.traceaad_v6.prompt import parse_actions
from llm4ad.method.traceaad_v6.schema import AnchorAttempt, ValueVec
from llm4ad.method.traceaad_v6.similarity import route_difference, trajectory_similarity
from llm4ad.method.traceaad_v6.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad_v6.value import (
    deduplicate_by_endpoint_hash,
    edge_credit,
    is_mature_trajectory,
    program_quality_key,
    qualified_reference_candidates,
    route_credit,
    score_active_trajectories,
    trajectory_sampling_distribution,
)


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedV6LLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return (
            "1. Adjust the deterministic offset using trajectory evidence.\n"
            "2. Try a second local rule grounded in the tested history."
        )

    @staticmethod
    def _program(value: int) -> str:
        return (
            f"Idea: deterministic candidate {value}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {value}\n"
            "```"
        )


class IncreasingEvaluation(Evaluation):
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


def _make_method(**kwargs) -> TraceAADV6:
    defaults = {
        "llm": ScriptedV6LLM(),
        "evaluation": IncreasingEvaluation(),
        "max_sample_nums": 12,
        "n_init": 3,
        "actions_per_iteration": 2,
        "max_trajectory_length": 8,
        "max_active_trajectories": 4,
        "softmax_temperature": 0.2,
        "random_seed": 0,
        "max_consecutive_sample_failures": 20,
        "max_stalled_iterations": 20,
        "context_token_limit": 4096,
    }
    defaults.update(kwargs)
    return TraceAADV6(**defaults)


def _seed_graph_with_route(
    *,
    fitnesses: list[float],
    codes: list[str] | None = None,
    ideas: list[str] | None = None,
    route_deltas: list[float] | None = None,
    edge_credits: list[float] | None = None,
) -> tuple[DerivationGraph, TrajectoryMemory, object]:
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    codes = codes or [f"def choose(value: int) -> int:\n    return value + {i}\n" for i in range(len(fitnesses))]
    ideas = ideas or [f"idea-{i}" for i in range(len(fitnesses))]
    nodes = [
        graph.add_node(code=code, idea=idea, fitness=fitness)
        for code, idea, fitness in zip(codes, ideas, fitnesses)
    ]
    route = memory.create_initial(node_id=nodes[0].id)
    for index in range(1, len(nodes)):
        delta = None if route_deltas is None else route_deltas[index - 1]
        credit = 0.0 if edge_credits is None else edge_credits[index - 1]
        edge = graph.add_edge(
            parent_id=nodes[index - 1].id,
            child_id=nodes[index].id,
            action=f"action-{index}",
            operator=OperatorName.REFINE,
            anchor_role="endpoint",
            primary_trajectory_id=route.id,
            delta_parent=delta,
            delta_route_best=delta,
            outcome="improve" if delta and delta > 0 else "plateau",
            edge_credit=credit,
            iteration=index,
        )
        compact = max(
            nodes[: index + 1],
            key=lambda node: program_quality_key(node, True),
        )
        route = memory.branch_from(
            trajectory_id=route.id,
            base_node_id=nodes[index - 1].id,
            child_id=nodes[index].id,
            edge_id=edge.id,
            compact_best_id=compact.id,
        )
    return graph, memory, route


def test_public_package_exports_v6_only():
    import llm4ad.method.traceaad_v6 as package

    assert package.TraceAADV6 is TraceAADV6
    assert set(package.__all__) == {
        "TraceAADV6",
        "TraceAADRunResult",
        "TraceAADProfiler",
        "ValueWeights",
    }


def test_value_weights_use_credit_not_trend():
    weights = ValueWeights()
    assert weights.search_quality == 0.8
    assert weights.search_credit == 0.2
    assert weights.endpoint_quality == 0.7
    assert weights.best_quality == 0.3
    assert not hasattr(weights, "search_trend")


def test_edge_credit_only_for_meaningful_route_progress():
    graph = DerivationGraph()
    child = graph.add_node(
        code="def choose(value: int) -> int:\n    return value + 1\n",
        idea="better",
        fitness=10.0,
    )
    weak = graph.add_node(
        code="def choose(value: int) -> int:\n    return value\n",
        idea="weak",
        fitness=1.0,
    )
    keys = [
        program_quality_key(weak, True),
        program_quality_key(child, True),
    ]
    assert edge_credit(
        child=child, batch_keys=keys, route_improved=True, maximize=True
    ) > 0
    assert (
        edge_credit(
            child=child, batch_keys=keys, route_improved=False, maximize=True
        )
        == 0.0
    )


def test_route_credit_is_discounted_average():
    graph, memory, route = _seed_graph_with_route(
        fitnesses=[1.0, 2.0, 3.0],
        route_deltas=[1.0, 1.0],
        edge_credits=[1.0, 0.0],
    )
    credit = route_credit(route, graph, discount=0.8)
    expected = (0.8 * 1.0 + 1.0 * 0.0) / (0.8 + 1.0)
    assert math.isclose(credit, expected)


def test_single_node_route_credit_is_zero():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    node = graph.add_node(
        code="def choose(value: int) -> int:\n    return value\n",
        idea="init",
        fitness=1.0,
    )
    route = memory.create_initial(node_id=node.id)
    assert route_credit(route, graph, discount=0.8) == 0.0


def test_search_value_mixes_quality_and_credit():
    graph, memory, _ = _seed_graph_with_route(
        fitnesses=[1.0, 5.0, 9.0],
        route_deltas=[4.0, 4.0],
        edge_credits=[0.9, 0.8],
    )
    # Add a second low-credit route so quality differs.
    low = graph.add_node(
        code="def choose(value: int) -> int:\n    return value - 1\n",
        idea="low",
        fitness=0.5,
    )
    memory.create_initial(node_id=low.id)
    scored = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(),
    )
    high = next(route for route in scored if len(route.edge_ids) == 2)
    assert high.value is not None
    assert high.scalar_value is not None
    expected = 0.8 * high.value.quality + 0.2 * high.value.credit
    assert math.isclose(high.scalar_value, expected)


def test_ucb_uses_global_batch_count_not_visit_sum():
    graph, memory, route_a = _seed_graph_with_route(fitnesses=[1.0])
    node_b = graph.add_node(
        code="def choose(value: int) -> int:\n    return value + 7\n",
        idea="b",
        fitness=1.0,
    )
    route_b = memory.create_initial(node_id=node_b.id)
    memory.record_visit(route_a.id)
    memory.record_visit(route_a.id)
    memory.record_visit(route_a.id)
    distribution = trajectory_sampling_distribution(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(ucb_c=10.0),
        temperature=0.2,
        batch_count=10,
    )
    adjusted = {route.id: score for route, score, _ in distribution}
    # Low-visit route gets a larger UCB bonus under the same batch count.
    assert adjusted[route_b.id] > adjusted[route_a.id]


def test_maturity_requires_top_quality_edges_and_route_progress():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    w = ValueWeights()

    def add_route(fitness: float, *, improving: bool, code_tag: str):
        start = graph.add_node(
            code=f"def choose(value: int) -> int:\n    {code_tag}_0 = value\n    return {code_tag}_0\n",
            idea=f"{code_tag}-0",
            fitness=fitness,
        )
        route = memory.create_initial(node_id=start.id)
        parent = start
        for step in range(2):
            child_fitness = fitness + (1.0 if improving else 0.0) * (step + 1)
            child = graph.add_node(
                code=(
                    f"def choose(value: int) -> int:\n"
                    f"    {code_tag}_{step + 1} = value\n"
                    f"    return {code_tag}_{step + 1}\n"
                ),
                idea=f"{code_tag}-{step + 1}",
                fitness=child_fitness,
            )
            delta = child_fitness - parent.fitness
            edge = graph.add_edge(
                parent_id=parent.id,
                child_id=child.id,
                action=f"{code_tag}-action-{step}",
                operator=OperatorName.REFINE,
                anchor_role="endpoint",
                primary_trajectory_id=route.id,
                delta_parent=delta,
                delta_route_best=delta if improving else 0.0,
                outcome="improve" if improving and delta > 0 else "plateau",
                edge_credit=0.5 if improving else 0.0,
            )
            compact = child if improving else start
            route = memory.branch_from(
                trajectory_id=route.id,
                base_node_id=parent.id,
                child_id=child.id,
                edge_id=edge.id,
                compact_best_id=compact.id,
            )
            parent = child
        return route

    # Five active routes; only the highest-quality improving ones can be mature.
    for index in range(5):
        add_route(float(index + 1), improving=True, code_tag=f"r{index}")
    weak = add_route(0.1, improving=False, code_tag="weak")

    scored = score_active_trajectories(
        memory=memory, graph=graph, maximize=True, w=w
    )
    mature_ids = {
        route.id
        for route in scored
        if is_mature_trajectory(route, active=scored, graph=graph, w=w)
    }
    # Top 30% of 6 => ceil(1.8)=2, with boundary ties retained.
    assert len(mature_ids) >= 1
    assert weak.id not in mature_ids
    for route in scored:
        if route.id in mature_ids:
            assert len(route.edge_ids) >= 2
            assert any(
                (graph.get_edge(edge_id).delta_route_best or 0.0) > 0
                for edge_id in route.edge_ids
            )


def test_operator_scheduling_gates_dual_and_refine_ratio():
    operators = tuple(op() for op in DEFAULT_OPERATORS)
    rng = random.Random(0)
    dual_hits = 0
    for _ in range(200):
        decision = select_operator(
            operators=operators,
            mature=True,
            has_qualified_reference=True,
            anchor_role="endpoint",
            recent_progress=True,
            prefer_trim_refine=False,
            rng=rng,
            dual_probability=0.25,
        )
        dual_hits += int(decision.use_dual)
    assert 20 <= dual_hits <= 80

    single = select_operator(
        operators=operators,
        mature=False,
        has_qualified_reference=True,
        anchor_role="endpoint",
        recent_progress=True,
        prefer_trim_refine=False,
        rng=random.Random(1),
        dual_probability=0.25,
    )
    assert not single.use_dual
    assert single.operator.name in {OperatorName.IDEATE, OperatorName.REFINE}

    refine_hits = 0
    for _ in range(300):
        decision = select_operator(
            operators=operators,
            mature=False,
            has_qualified_reference=False,
            anchor_role="endpoint",
            recent_progress=True,
            prefer_trim_refine=False,
            rng=rng,
            dual_probability=0.25,
        )
        refine_hits += int(decision.operator.name == OperatorName.REFINE)
    # endpoint + progress => refine:ideate = 2:1
    assert 160 <= refine_hits <= 240


def test_reference_requires_difference_and_median_gate():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    codes = [
        "def choose(value: int) -> int:\n    alpha = value\n    return alpha + 1\n",
        "def choose(value: int) -> int:\n    beta = value\n    return beta + 2\n",
        "def choose(value: int) -> int:\n    gamma = value\n    return gamma + 3\n",
        "def choose(value: int) -> int:\n    alpha = value\n    return alpha + 1\n",
    ]
    routes = []
    for index, code in enumerate(codes):
        node = graph.add_node(code=code, idea=f"idea-{index}-{code[20:40]}", fitness=float(10 - index))
        route = memory.create_initial(node_id=node.id)
        # Give each route two improving edges so they can be mature.
        parent = node
        for step in range(2):
            child = graph.add_node(
                code=code + f"# step {step}\n",
                idea=f"idea-{index}-step-{step}-unique-{step*index}",
                fitness=float(10 - index + step + 1),
            )
            edge = graph.add_edge(
                parent_id=parent.id,
                child_id=child.id,
                action=f"unique-action-{index}-{step}-{'alpha' if index == 0 else 'other'}",
                operator=OperatorName.IDEATE if step == 0 else OperatorName.REFINE,
                anchor_role="endpoint",
                primary_trajectory_id=route.id,
                delta_parent=1.0,
                delta_route_best=1.0,
                outcome="improve",
                edge_credit=0.5,
            )
            route = memory.branch_from(
                trajectory_id=route.id,
                base_node_id=parent.id,
                child_id=child.id,
                edge_id=edge.id,
                compact_best_id=child.id,
            )
            parent = child
        routes.append(route)

    scored = score_active_trajectories(
        memory=memory, graph=graph, maximize=True, w=ValueWeights()
    )
    primary = scored[0]
    qualified = qualified_reference_candidates(
        primary=primary,
        anchor_id=primary.endpoint_id,
        active=scored,
        graph=graph,
        w=ValueWeights(),
    )
    # Identical compact-best hash routes are excluded.
    assert all(
        graph.get_node(route.compact_best_id).code_hash
        != graph.get_node(primary.compact_best_id).code_hash
        for route, _ in qualified
    )
    if len(qualified) >= 2:
        diffs = sorted(diff for _, diff in qualified)
        median = diffs[len(diffs) // 2]
        assert all(diff + 1e-12 >= median for _, diff in qualified)


def test_similarity_uses_compact_best_code():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    left_end = graph.add_node(
        code="def choose(value: int) -> int:\n    return value + 1\n",
        idea="left-end",
        fitness=1.0,
    )
    left_best = graph.add_node(
        code="def choose(value: int) -> int:\n    special_left = value\n    return special_left\n",
        idea="left-best",
        fitness=5.0,
    )
    right_end = graph.add_node(
        code="def choose(value: int) -> int:\n    return value + 1\n",
        idea="right-end",
        fitness=1.0,
    )
    right_best = graph.add_node(
        code="def choose(value: int) -> int:\n    special_right = value\n    return special_right\n",
        idea="right-best",
        fitness=5.0,
    )
    left = memory.create_initial(node_id=left_best.id)
    # Force endpoint different from compact best.
    edge = graph.add_edge(
        parent_id=left_best.id,
        child_id=left_end.id,
        action="regress",
        operator=OperatorName.REFINE,
        anchor_role="endpoint",
        primary_trajectory_id=left.id,
        delta_parent=-4.0,
        delta_route_best=-4.0,
        outcome="regress",
    )
    left = memory.branch_from(
        trajectory_id=left.id,
        base_node_id=left_best.id,
        child_id=left_end.id,
        edge_id=edge.id,
        compact_best_id=left_best.id,
    )
    right = memory.create_initial(node_id=right_best.id)
    edge_r = graph.add_edge(
        parent_id=right_best.id,
        child_id=right_end.id,
        action="regress",
        operator=OperatorName.REFINE,
        anchor_role="endpoint",
        primary_trajectory_id=right.id,
        delta_parent=-4.0,
        delta_route_best=-4.0,
        outcome="regress",
    )
    right = memory.branch_from(
        trajectory_id=right.id,
        base_node_id=right_best.id,
        child_id=right_end.id,
        edge_id=edge_r.id,
        compact_best_id=right_best.id,
    )
    # Endpoints are identical; compact-best codes differ, so Sim < 1.
    assert left.endpoint_id != left.compact_best_id
    assert trajectory_similarity(graph=graph, left=left, right=right) < 1.0
    assert route_difference(graph=graph, left=left, right=right) > 0.0


def test_population_dedup_keeps_best_anchor():
    graph = DerivationGraph()
    memory = TrajectoryMemory()
    shared_code = "def choose(value: int) -> int:\n    return value + 3\n"
    nodes = [
        graph.add_node(code=shared_code, idea=f"dup-{i}", fitness=float(i + 1))
        for i in range(3)
    ]
    routes = [memory.create_initial(node_id=node.id) for node in nodes]
    scored = []
    for route, quality in zip(routes, (0.2, 0.9, 0.5)):
        scored.append(
            memory.set_value(route.id, ValueVec(quality=quality, credit=0.1), quality)
        )
    kept = deduplicate_by_endpoint_hash(
        routes=tuple(scored),
        graph=graph,
        best_trajectory_id=routes[0].id,
    )
    assert len(kept) == 1
    assert kept[0].id == routes[0].id


def test_tested_attempts_prefer_best_updates_and_failures():
    attempts = (
        AnchorAttempt(
            anchor_node_id=1,
            primary_trajectory_id=0,
            operator="trace_refine",
            action="a0",
            iteration=0,
            status="valid",
            outcome="improve",
            edge_id=10,
        ),
        AnchorAttempt(
            anchor_node_id=1,
            primary_trajectory_id=0,
            operator="trace_refine",
            action="a1",
            iteration=1,
            status="parse_failed",
            failure_kind="parse_failed",
        ),
        AnchorAttempt(
            anchor_node_id=1,
            primary_trajectory_id=0,
            operator="trace_refine",
            action="a2",
            iteration=2,
            status="valid",
            outcome="improve",
            edge_id=12,
            new_route_best=True,
        ),
        AnchorAttempt(
            anchor_node_id=1,
            primary_trajectory_id=0,
            operator="trace_refine",
            action="a3",
            iteration=3,
            status="valid",
            outcome="regress",
            edge_id=13,
        ),
    )
    selected = select_tested_attempts(attempts, excluded_edge_ids={12}, limit=2)
    # Best-update a2 is excluded; remaining prefer recent failure/regress.
    assert [item.action for item in selected] == ["a3", "a1"]


def test_reference_is_not_structural_parent():
    method = _make_method(max_sample_nums=20, n_init=4, max_active_trajectories=4)
    result = method.run()
    assert result.n_valid_nodes >= 4
    for edge in method._graph.edges():
        assert edge.reference_trajectory_id is None or edge.parent_id is not None
        # Every child has exactly one structural parent edge.
    children = [edge.child_id for edge in method._graph.edges()]
    assert len(children) == len(set(children))


def test_batch_visit_increments_even_without_valid_child(tmp_path: Path):
    class FailingCodeLLM(ScriptedV6LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            self.prompts.append(str(prompt))
            if "Generate a complete implementation" in prompt:
                return self._program(self.calls)
            if "[Requested Modification]" in prompt:
                return "Idea: broken\n```python\nnot valid python\n```"
            return "1. Try one change.\n2. Try another change."

    method = _make_method(
        llm=FailingCodeLLM(),
        max_sample_nums=6,
        n_init=3,
        checkpoint_dir=tmp_path,
    )
    before_batches = method._batch_count
    method._initialize()
    method._initialization_complete = True
    active_before = method.active_trajectories()
    assert active_before
    visits_before = {route.id: route.visit_count for route in active_before}
    method._run_iteration(0)
    assert method._batch_count == before_batches + 1
    visited = [
        route
        for route in method.active_trajectories()
        if route.visit_count > visits_before.get(route.id, 0)
    ]
    assert visited


def test_end_to_end_smoke_and_checkpoint(tmp_path: Path):
    method = _make_method(
        max_sample_nums=16,
        n_init=4,
        max_active_trajectories=4,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1,
        random_seed=7,
    )
    result = method.run()
    assert result.best_node is not None
    assert result.n_samples > 0
    assert (tmp_path / "latest.json").is_file()
    resumed = _make_method(
        max_sample_nums=16,
        n_init=4,
        max_active_trajectories=4,
        resume_from=tmp_path / "latest.json",
        checkpoint_interval=1,
        random_seed=7,
    )
    assert resumed._tot_sample_nums == method._tot_sample_nums
    assert resumed._batch_count == method._batch_count
    assert resumed._best_node is not None
    assert resumed._best_node.id == method._best_node.id


def test_single_parent_invariant_after_branching():
    method = _make_method(max_sample_nums=18, n_init=3, max_active_trajectories=6)
    method.run()
    parents = {}
    for edge in method._graph.edges():
        assert edge.child_id not in parents
        parents[edge.child_id] = edge.parent_id


def test_action_parser_requires_exactly_two_actions():
    assert parse_actions("1. only one", expected_count=2)[0] == []
    assert parse_actions("1. first\n2. second", expected_count=2)[0] == [
        "first",
        "second",
    ]
    assert parse_actions("1. first\n2. second\n3. third", expected_count=2)[0] == []


def test_batch_global_best_marking_is_order_independent():
    class ScoreByProgram(Evaluation):
        def __init__(self) -> None:
            super().__init__(
                template_program=TEMPLATE,
                task_description="Improve choose.",
                safe_evaluate=False,
            )

        def evaluate_program(self, program_str, callable_func, **kwargs):
            values = re.findall(r"return value \+ (\d+)", program_str)
            return float(values[-1]) if values else 0.0

    class OrderedLLM(ScriptedV6LLM):
        def __init__(self, order: tuple[str, str]) -> None:
            super().__init__()
            self.order = order

        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            self.prompts.append(str(prompt))
            if "Generate a complete implementation" in prompt:
                return self._program(0)
            if "[Requested Modification]" not in prompt:
                return "\n".join(
                    f"{index}. {name}" for index, name in enumerate(self.order, 1)
                )
            value = 2 if "HIGH" in prompt else 1
            return self._program(value)

    def run_order(order: tuple[str, str]):
        method = _make_method(
            llm=OrderedLLM(order),
            evaluation=ScoreByProgram(),
            n_init=1,
            max_sample_nums=3,
        )
        method._initialize()
        method._initialization_complete = True
        method._run_iteration(0)
        flags = {
            method._graph.get_node(edge.child_id).fitness: edge.new_global_best
            for edge in method._graph.edges()
        }
        return flags, method._best_node.fitness

    forward, forward_best = run_order(("HIGH", "LOW"))
    reverse, reverse_best = run_order(("LOW", "HIGH"))
    assert forward == reverse == {1.0: False, 2.0: True}
    assert forward_best == reverse_best == 2.0


def test_llm_transport_failure_is_not_algorithm_evidence():
    class TransportFailureLLM(ScriptedV6LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            if "Generate a complete implementation" in prompt:
                return self._program(0)
            if "[Requested Modification]" in prompt:
                raise ConnectionError("transport unavailable")
            return "1. first change\n2. second change"

    method = _make_method(
        llm=TransportFailureLLM(),
        n_init=1,
        max_sample_nums=3,
    )
    method._initialize()
    method._initialization_complete = True
    method._run_iteration(0)
    assert method._attempts.all() == ()
    assert method._consecutive_sample_failures == 2


def test_non_finite_evaluation_never_forms_a_program_node():
    class NonFiniteEvaluation(IncreasingEvaluation):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            return float("nan")

    method = _make_method(
        evaluation=NonFiniteEvaluation(),
        n_init=1,
        max_sample_nums=1,
    )
    result = method.run()
    assert result.n_samples == 1
    assert result.n_valid_nodes == 0
    assert result.best_node is None


def test_context_limit_is_hard_for_single_trace_and_initialization():
    llm = ScriptedV6LLM()
    too_small = _make_method(
        llm=llm,
        n_init=1,
        max_sample_nums=1,
        context_token_limit=10,
    )
    result = too_small.run()
    assert result.n_samples == 0
    assert llm.calls == 0

    method = _make_method(n_init=1, max_sample_nums=1)
    method._initialize()
    route = method.active_trajectories()[0]
    method._context_token_limit = 10
    context = method._build_action_context(
        selected=route,
        anchor_id=route.endpoint_id,
        operator=method._operators[0],
        reference_route=None,
        reference_node=None,
    )
    assert context is None


def test_successful_llm_calls_record_exact_prompt_and_response(tmp_path: Path):
    profiler = TraceAADProfiler(
        log_dir=str(tmp_path),
        log_style="simple",
        create_random_path=False,
    )
    method = _make_method(
        profiler=profiler,
        n_init=1,
        max_sample_nums=3,
    )
    method.run()
    records = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {record["stage"] for record in records} == {"init", "action", "code"}
    assert all("prompt" in record and "response" in record for record in records)


def test_checkpoint_preserves_stop_state_and_rejects_config_drift():
    method = _make_method(max_sample_nums=3)
    method._stalled_iterations = 4
    method._consecutive_sample_failures = 5
    payload = dump_state(method)
    assert payload["stalled_iterations"] == 4
    assert payload["consecutive_sample_failures"] == 5
    assert payload["search_configuration"]["max_sample_nums"] == 3

    restored = _make_method(max_sample_nums=3)
    load_state(restored, payload)
    assert restored._stalled_iterations == 4
    assert restored._consecutive_sample_failures == 5

    incompatible = _make_method(max_sample_nums=4)
    with pytest.raises(ValueError, match="configuration"):
        load_state(incompatible, payload)
