"""Mechanism contract tests for TraceAAD V7."""

from __future__ import annotations

import json
import math
import re
from dataclasses import fields

import pytest

from llm4ad.method.traceaad_v7 import traceaad as traceaad_module
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v7 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV7,
    ValueWeights,
)
from llm4ad.method.traceaad_v7.checkpoint import dump_state, load_state, save_checkpoint
from llm4ad.method.traceaad_v7.context import trajectory_history
from llm4ad.method.traceaad_v7.derivation_graph import DerivationGraph
from llm4ad.method.traceaad_v7.operators import (
    DEFAULT_OPERATORS,
    TRACE_REFINE,
    TRACE_TRANSFER,
    Operator,
    OperatorDecision,
    select_operator,
)
from llm4ad.method.traceaad_v7.prompt import build_code_prompt
from llm4ad.method.traceaad_v7.schema import (
    OperatorName,
    TrajectoryStatus,
    ValueVec,
)
from llm4ad.method.traceaad_v7.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad_v7.value import (
    program_quality_key,
    reference_sampling_distribution,
    score_active_trajectories,
    trajectory_sampling_distribution,
)


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedV7LLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.initial_programs = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        prompt = str(prompt)
        if "Generate a simple, complete" in prompt:
            self.initial_programs += 1
            return self.program(self.initial_programs)
        if "[Requested Modification]" in prompt:
            return self.program(self.calls)
        return "1. Make one focused change.\n2. Test a second focused change."

    @staticmethod
    def program(value: int) -> str:
        return (
            f"Idea: deterministic candidate {value}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {value}\n"
            "```"
        )


class ScoreByProgram(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        values = re.findall(r"return value \+ (\d+)", program_str)
        return float(values[-1]) if values else 0.0


def _make_method(**kwargs) -> TraceAADV7:
    defaults = {
        "llm": ScriptedV7LLM(),
        "evaluation": ScoreByProgram(),
        "max_sample_nums": 12,
        "n_init": 3,
        "actions_per_iteration": 2,
        "max_trajectory_length": 8,
        "max_active_trajectories": 4,
        "elite_count": 2,
        "softmax_temperature": 0.2,
        "random_seed": 0,
        "max_consecutive_sample_failures": 20,
        "max_stalled_iterations": 20,
        "context_token_limit": 4096,
    }
    defaults.update(kwargs)
    return TraceAADV7(**defaults)


def _add_route(
    graph: DerivationGraph,
    memory: TrajectoryMemory,
    fitnesses: list[float],
    *,
    code_prefix: str,
):
    nodes = [
        graph.add_node(
            code=(
                "def choose(value: int) -> int:\n"
                f"    return value + {index}  # {code_prefix}\n"
            ),
            idea=f"{code_prefix}-{index}",
            fitness=fitness,
        )
        for index, fitness in enumerate(fitnesses)
    ]
    route = memory.create_initial(node_id=nodes[0].id)
    for index, child in enumerate(nodes[1:], start=1):
        parent = nodes[index - 1]
        edge = graph.add_edge(
            parent_id=parent.id,
            child_id=child.id,
            action=f"change-{index}",
            operator=OperatorName.REFINE,
            anchor_role="endpoint",
            primary_trajectory_id=route.id,
            delta_parent=child.fitness - parent.fitness,
        )
        compact = max(
            nodes[: index + 1], key=lambda node: program_quality_key(node, True)
        )
        route = memory.branch_from(
            trajectory_id=route.id,
            base_node_id=parent.id,
            child_id=child.id,
            edge_id=edge.id,
            compact_best_id=compact.id,
        )
    return route


class _IndexRng:
    def __init__(self, index: int) -> None:
        self.index = index

    def randrange(self, size: int) -> int:
        return self.index % size

    def random(self) -> float:
        return 0.0


def test_public_package_identifies_v7_protocol():
    import llm4ad.method.traceaad_v7 as package

    assert package.TraceAADV7 is TraceAADV7
    assert package.PROTOCOL_ID == "traceaad-v7-v1"
    assert package.CHECKPOINT_VERSION == 9
    assert set(package.__all__) == {
        "TraceAADV7",
        "TraceAADRunResult",
        "TraceAADProfiler",
        "ValueWeights",
        "CHECKPOINT_VERSION",
        "PROTOCOL_ID",
    }


def test_value_weights_restore_v5_quality_trend_mixture():
    assert {field.name for field in fields(ValueWeights)} == {
        "endpoint_quality",
        "best_quality",
        "search_quality",
        "search_trend",
        "ucb_c",
        "discount",
        "positive_threshold",
    }
    weights = ValueWeights()
    assert weights.search_quality == 0.8
    assert weights.search_trend == 0.2


def test_recent_trend_changes_scheduling_without_changing_fitness_order():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    improving = _add_route(graph, memory, [1.0, 2.0], code_prefix="up")
    regressing = _add_route(graph, memory, [3.0, 2.0], code_prefix="down")

    scored = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(endpoint_quality=1.0, best_quality=0.0),
        trajectories=(improving, regressing),
    )
    by_id = {route.id: route for route in scored}
    assert by_id[improving.id].value.quality == pytest.approx(0.5)
    assert by_id[regressing.id].value.quality == pytest.approx(0.5)
    assert by_id[improving.id].value.trend == pytest.approx(1.0)
    assert by_id[regressing.id].value.trend == pytest.approx(0.0)
    assert by_id[improving.id].scalar_value > by_id[regressing.id].scalar_value
    assert program_quality_key(
        graph.get_node(improving.endpoint_id), True
    ) == program_quality_key(graph.get_node(regressing.endpoint_id), True)


def test_ucb_exploration_decays_with_remaining_budget():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    visited = _add_route(graph, memory, [1.0], code_prefix="visited")
    fresh = _add_route(graph, memory, [1.0], code_prefix="fresh")
    for _ in range(4):
        memory.record_visit(visited.id)

    def gap(remaining: float) -> float:
        distribution = trajectory_sampling_distribution(
            memory=memory,
            graph=graph,
            maximize=True,
            w=ValueWeights(ucb_c=1.0),
            temperature=0.2,
            remaining_budget_ratio=remaining,
            selection_count=4,
        )
        adjusted = {route.id: score for route, score, _ in distribution}
        return adjusted[fresh.id] - adjusted[visited.id]

    assert gap(1.0) > gap(0.25) > gap(0.0)
    assert math.isclose(gap(0.0), 0.0, abs_tol=1e-12)


def test_initialization_keeps_one_active_route_per_executable_state():
    class DuplicateInitLLM(ScriptedV7LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            return self.program(1)

    method = _make_method(
        llm=DuplicateInitLLM(),
        n_init=3,
        max_sample_nums=3,
    )
    result = method.run()

    assert result.n_total_nodes == 3
    assert result.n_trajectories == 3
    assert len(method.active_trajectories()) == 1
    assert (
        len(
            {
                method._graph.get_node(route.endpoint_id).code_hash
                for route in method.active_trajectories()
            }
        )
        == 1
    )


def test_duplicate_child_keeps_fact_edges_but_not_active_slots():
    class DuplicateChildLLM(ScriptedV7LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            prompt = str(prompt)
            self.calls += 1
            if "Generate a simple, complete" in prompt:
                self.initial_programs += 1
                return self.program(self.initial_programs)
            if "[Requested Modification]" in prompt:
                return self.program(1)
            return "1. First duplicate.\n2. Second duplicate."

    method = _make_method(
        llm=DuplicateChildLLM(),
        n_init=2,
        max_sample_nums=4,
    )
    method._initialize()
    method._initialization_complete = True
    method._run_iteration(0)

    assert len(method._graph.nodes()) == 4
    assert len(method._graph.edges()) == 2
    assert len(method._memory.trajectories()) == 4
    active = method.active_trajectories()
    active_hashes = {
        method._graph.get_node(route.endpoint_id).code_hash for route in active
    }
    assert len(active) == len(active_hashes) == 2
    assert (
        sum(
            route.status == TrajectoryStatus.ARCHIVED
            for route in method._memory.trajectories()
        )
        == 2
    )


def test_batch_global_best_marking_is_order_independent():
    class OrderedLLM(ScriptedV7LLM):
        def __init__(self, order: tuple[str, str]) -> None:
            super().__init__()
            self.order = order

        def draw_sample(self, prompt, *args, **kwargs):
            prompt = str(prompt)
            self.calls += 1
            if "Generate a simple, complete" in prompt:
                return self.program(0)
            if "[Requested Modification]" not in prompt:
                return "\n".join(
                    f"{index}. {name}" for index, name in enumerate(self.order, 1)
                )
            return self.program(2 if "HIGH" in prompt else 1)

    def run_order(order: tuple[str, str]):
        method = _make_method(
            llm=OrderedLLM(order),
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


@pytest.mark.parametrize("failure_mode", ["transport", "parse", "eval"])
def test_failed_children_never_enter_graph_or_trajectory_memory(failure_mode: str):
    class FailureLLM(ScriptedV7LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            prompt = str(prompt)
            self.calls += 1
            if "Generate a simple, complete" in prompt:
                return self.program(0)
            if "[Requested Modification]" not in prompt:
                return "1. First change.\n2. Second change."
            if failure_mode == "transport":
                raise ConnectionError("transport unavailable")
            if failure_mode == "parse":
                return "Idea: invalid\n```python\nnot valid python\n```"
            return self.program(1)

    class ConditionalEvaluation(ScoreByProgram):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            score = super().evaluate_program(program_str, callable_func, **kwargs)
            return None if failure_mode == "eval" and score > 0 else score

    method = _make_method(
        llm=FailureLLM(),
        evaluation=ConditionalEvaluation(),
        n_init=1,
        max_sample_nums=3,
    )
    method._initialize()
    method._initialization_complete = True
    method._run_iteration(0)

    assert len(method._graph.nodes()) == 1
    assert len(method._graph.edges()) == 0
    assert len(method._memory.trajectories()) == 1


def test_checkpoint_round_trip_preserves_values_and_active_uniqueness():
    method = _make_method(n_init=2, max_sample_nums=2)
    method._initialize()
    score_active_trajectories(
        memory=method._memory,
        graph=method._graph,
        maximize=True,
        w=method._value_weights,
    )
    payload = dump_state(method)

    assert payload["version"] == CHECKPOINT_VERSION
    assert payload["protocol_id"] == PROTOCOL_ID
    assert "trend" in payload["memory"]["trajectories"][0]["value"]
    assert payload["memory"]["trajectories"][0]["scalar_value"] is not None

    restored = _make_method(n_init=2, max_sample_nums=2)
    load_state(restored, payload)
    active_hashes = [
        restored._graph.get_node(route.endpoint_id).code_hash
        for route in restored.active_trajectories()
    ]
    assert len(active_hashes) == len(set(active_hashes))
    assert all(
        route.scalar_value is not None for route in restored.active_trajectories()
    )


def test_checkpoint_rejects_duplicate_active_executable_states():
    method = _make_method(n_init=1, max_sample_nums=1)
    method._initialize()
    payload = json.loads(json.dumps(dump_state(method)))
    duplicate = dict(payload["memory"]["trajectories"][0])
    duplicate["id"] = 1
    payload["memory"]["trajectories"].append(duplicate)
    payload["memory"]["next_id"] = 2

    with pytest.raises(ValueError, match="duplicate active executable states"):
        load_state(_make_method(n_init=1, max_sample_nums=1), payload)


def test_ucb_clock_is_global_and_does_not_reset_when_routes_are_archived():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    fresh = _add_route(graph, memory, [1.0], code_prefix="fresh")
    visited = _add_route(graph, memory, [1.0], code_prefix="visited")
    for _ in range(10):
        memory.record_visit(visited.id)

    kwargs = {
        "memory": memory,
        "graph": graph,
        "maximize": True,
        "w": ValueWeights(
            endpoint_quality=1.0,
            best_quality=0.0,
            search_quality=1.0,
            search_trend=0.0,
            ucb_c=1.0,
        ),
        "temperature": 0.2,
        "remaining_budget_ratio": 1.0,
        "selection_count": 10,
    }
    before = dict(
        (route.id, adjusted)
        for route, adjusted, _ in trajectory_sampling_distribution(**kwargs)
    )
    memory.archive(visited.id)
    after = dict(
        (route.id, adjusted)
        for route, adjusted, _ in trajectory_sampling_distribution(**kwargs)
    )

    assert after[fresh.id] == pytest.approx(before[fresh.id])


def test_initialization_stops_after_repeated_evaluation_failures():
    class FailedEvaluation(ScoreByProgram):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            return None

    method = _make_method(
        evaluation=FailedEvaluation(),
        n_init=2,
        max_sample_nums=20,
        max_stalled_iterations=3,
    )
    result = method.run()

    assert result.n_samples == 3
    assert result.n_total_nodes == 0
    assert method._initialization_complete is True


def test_initialization_stops_after_repeated_duplicate_states():
    class DuplicateInitLLM(ScriptedV7LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            return self.program(1)

    method = _make_method(
        llm=DuplicateInitLLM(),
        n_init=2,
        max_sample_nums=20,
        max_stalled_iterations=3,
    )
    result = method.run()

    assert result.n_samples == 4
    assert result.n_total_nodes == 4
    assert len(method.active_trajectories()) == 1


def test_early_initialization_checkpoint_does_not_restart_initialization():
    class FailedEvaluation(ScoreByProgram):
        def evaluate_program(self, program_str, callable_func, **kwargs):
            return None

    method = _make_method(
        evaluation=FailedEvaluation(),
        n_init=2,
        max_sample_nums=20,
        max_stalled_iterations=2,
    )
    method.run()
    payload = dump_state(method)

    restored_llm = ScriptedV7LLM()
    restored = _make_method(
        llm=restored_llm,
        evaluation=FailedEvaluation(),
        n_init=2,
        max_sample_nums=20,
        max_stalled_iterations=2,
    )
    load_state(restored, payload)
    restored.run()

    assert restored._initialization_complete is True
    assert restored._tot_sample_nums == 2
    assert restored_llm.calls == 0


def test_graph_rejects_non_finite_program_fitness():
    graph = DerivationGraph()

    with pytest.raises(ValueError, match="fitness must be finite"):
        graph.add_node(code=TEMPLATE, idea="invalid", fitness=float("nan"))


def test_v7_operators_are_immutable_data_values():
    assert [operator.name for operator in DEFAULT_OPERATORS] == [
        OperatorName.IDEATE,
        OperatorName.REFINE,
        OperatorName.SYNTHESIZE,
        OperatorName.TRANSFER,
    ]
    assert all(isinstance(operator, Operator) for operator in DEFAULT_OPERATORS)


def test_initial_prompt_is_built_and_counted_once():
    class CountingLLM(ScriptedV7LLM):
        def __init__(self) -> None:
            super().__init__()
            self.initial_prompt_counts = 0

        def count_tokens(self, text):
            if "Generate a simple, complete" in str(text):
                self.initial_prompt_counts += 1
            return len(str(text).encode("utf-8"))

    llm = CountingLLM()
    method = _make_method(llm=llm, n_init=3, max_sample_nums=3)
    method._initialize()

    assert llm.initial_prompt_counts == 1


def test_dual_action_overflow_builds_single_fallback_once(monkeypatch):
    class MarkerCountingLLM(ScriptedV7LLM):
        def count_tokens(self, text):
            return 100 if "[Reference Program]" in str(text) else 1

    method = _make_method(
        llm=MarkerCountingLLM(),
        n_init=2,
        max_sample_nums=4,
        context_token_limit=10,
    )
    method._initialize()

    def forced_operator(*, operators, allow_dual, rng):
        del operators, rng
        operator = TRACE_TRANSFER if allow_dual else TRACE_REFINE
        return OperatorDecision(
            operator=operator,
            use_dual=allow_dual,
            reason="forced_test_operator",
        )

    prompt_kinds: list[str] = []
    real_build_action_prompt = traceaad_module.build_action_prompt

    def counted_action_prompt(**kwargs):
        prompt_kinds.append(
            "dual" if kwargs["reference_node"] is not None else "single"
        )
        return real_build_action_prompt(**kwargs)

    monkeypatch.setattr(traceaad_module, "select_operator", forced_operator)
    monkeypatch.setattr(traceaad_module, "build_action_prompt", counted_action_prompt)
    method._run_iteration(0)

    assert prompt_kinds.count("dual") == method._memory.max_trajectory_length + 1
    assert prompt_kinds.count("single") == 1


def test_route_quality_combines_endpoint_and_compact_best_percentiles():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    current_good = _add_route(graph, memory, [10.0], code_prefix="current")
    historic_good = _add_route(graph, memory, [12.0, 8.0], code_prefix="historic")

    scored = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(),
        trajectories=(current_good, historic_good),
    )
    quality = {route.id: route.value.quality for route in scored if route.value}

    assert quality[current_good.id] == pytest.approx(0.7)
    assert quality[historic_good.id] == pytest.approx(0.3)


def test_equal_fitness_route_rank_uses_the_program_loc_tie_break():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    short = graph.add_node(code="def f():\n    return 1\n", idea="short", fitness=1.0)
    long = graph.add_node(
        code="def f():\n    value = 1\n    return value\n",
        idea="long",
        fitness=1.0,
    )
    short_route = memory.create_initial(node_id=short.id)
    long_route = memory.create_initial(node_id=long.id)

    scored = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=ValueWeights(),
        trajectories=(short_route, long_route),
    )

    quality = {route.id: route.value.quality for route in scored if route.value}
    assert quality[short_route.id] == 1.0
    assert quality[long_route.id] == 0.0
    assert program_quality_key(short, True) > program_quality_key(long, True)


def test_operator_sampling_uses_all_and_single_only_available_sets():
    single = {
        select_operator(
            operators=DEFAULT_OPERATORS,
            allow_dual=False,
            rng=_IndexRng(index),
        ).operator.name
        for index in range(2)
    }
    all_names = {
        select_operator(
            operators=DEFAULT_OPERATORS,
            allow_dual=True,
            rng=_IndexRng(index),
        ).operator.name
        for index in range(4)
    }

    assert single == {OperatorName.IDEATE, OperatorName.REFINE}
    assert all_names == set(OperatorName)


def test_constructor_rejects_dual_only_operator_set():
    with pytest.raises(ValueError, match="single-trajectory operator"):
        _make_method(operators=(TRACE_TRANSFER,))


def test_reference_sampling_excludes_primary_and_prefers_higher_quality():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    primary = _add_route(graph, memory, [1.0], code_prefix="primary")
    low = _add_route(graph, memory, [1.0], code_prefix="low")
    high = _add_route(graph, memory, [1.0], code_prefix="high")
    primary = memory.set_value(primary.id, ValueVec(quality=0.4), 0.4)
    low = memory.set_value(low.id, ValueVec(quality=0.1), 0.1)
    high = memory.set_value(high.id, ValueVec(quality=0.9), 0.9)

    distribution = reference_sampling_distribution(
        primary=primary,
        active=(primary, low, high),
        temperature=0.2,
    )
    probabilities = {route.id: probability for route, _, probability in distribution}

    assert set(probabilities) == {low.id, high.id}
    assert probabilities[high.id] > probabilities[low.id]


def test_anchor_selection_keeps_endpoint_and_compact_best_available():
    method = _make_method(n_init=0, max_sample_nums=0)
    route = _add_route(
        method._graph,
        method._memory,
        [10.0, 8.0],
        code_prefix="anchor",
    )
    method._rng = _IndexRng(0)
    assert method._select_anchor(route) == (route.endpoint_id, "endpoint")
    method._rng = _IndexRng(1)
    assert method._select_anchor(route) == (route.compact_best_id, "compact_best")


def test_population_keeps_global_best_and_unique_executable_states():
    method = _make_method(
        n_init=0,
        max_sample_nums=0,
        max_active_trajectories=3,
        elite_count=1,
        random_seed=4,
    )
    routes = []
    for fitness in range(6):
        node = method._graph.add_node(
            code=(f"def choose(value: int) -> int:\n    return value + {fitness}\n"),
            idea=f"route-{fitness}",
            fitness=float(fitness),
        )
        routes.append(method._memory.create_initial(node_id=node.id))
    method._best_node = method._graph.get_node(routes[-1].endpoint_id)
    method._best_trajectory_id = routes[-1].id

    method._manage_population()
    active = method.active_trajectories()
    active_hashes = {
        method._graph.get_node(route.endpoint_id).code_hash for route in active
    }

    assert len(active) == len(active_hashes) == 3
    assert routes[-1].id in {route.id for route in active}


def test_history_is_chronological_and_separates_later_attempts():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    route = _add_route(
        graph,
        memory,
        [1.0, 0.5, 1.2],
        code_prefix="history",
    )

    history = trajectory_history(graph, route, base_node_id=route.node_ids[1])

    assert "[How This Program Was Reached]" in history.text
    assert "[Later Attempts From This Program]" in history.text
    assert "Step 1:" in history.text and "Step 2:" in history.text
    assert history.text.index("change-1") < history.text.index("change-2")
    assert history.formation_edge_ids == (route.edge_ids[0],)
    assert history.tested_after_edge_ids == (route.edge_ids[1],)


def test_action_and_code_share_primary_and_reference_histories():
    method = _make_method(n_init=2, max_sample_nums=2)
    method._initialize()
    primary, reference = method.active_trajectories()
    reference_node = method._graph.get_node(reference.compact_best_id)
    context = method._build_action_context(
        selected=primary,
        anchor_id=primary.endpoint_id,
        operator=TRACE_TRANSFER,
        reference_route=reference,
        reference_node=reference_node,
    )
    assert context is not None and context.used_dual

    code_prompt = build_code_prompt(
        current_node=method._graph.get_node(primary.endpoint_id),
        action="adapt the reference idea",
        task_description=method._task_description_str,
        template_function=method._function_to_evolve,
        history=context.primary_history,
        reference_node=reference_node,
        reference_history=context.reference_history,
    )

    assert context.primary_history in code_prompt
    assert context.reference_history in code_prompt
    assert reference_node.code.rstrip() in context.prompt
    assert reference_node.code.rstrip() in code_prompt


def test_dual_code_overflow_restarts_batch_with_single_context():
    class CapturingLLM(ScriptedV7LLM):
        def __init__(self) -> None:
            super().__init__()
            self.prompts: list[str] = []

        def draw_sample(self, prompt, *args, **kwargs):
            self.prompts.append(str(prompt))
            return super().draw_sample(prompt, *args, **kwargs)

        def count_tokens(self, text):
            prompt = str(text)
            base = len(prompt.encode("utf-8"))
            if "[Instruction]" in prompt and "[Reference Program]" in prompt:
                return base + 10_000
            return base

    llm = CapturingLLM()
    method = _make_method(
        llm=llm,
        n_init=2,
        max_sample_nums=4,
        operators=(TRACE_TRANSFER, TRACE_REFINE),
    )
    method._initialize()
    primary, reference = method.active_trajectories()
    reference_node = method._graph.get_node(reference.compact_best_id)
    dual = method._build_action_context(
        selected=primary,
        anchor_id=primary.endpoint_id,
        operator=TRACE_TRANSFER,
        reference_route=reference,
        reference_node=reference_node,
    )
    single = method._build_action_context(
        selected=primary,
        anchor_id=primary.endpoint_id,
        operator=TRACE_REFINE,
        reference_route=None,
        reference_node=None,
    )
    assert dual is not None and single is not None

    actions = ("Make one focused change.", "Test a second focused change.")

    def code_prompt(context, action, *, include_reference):
        return build_code_prompt(
            current_node=method._graph.get_node(primary.endpoint_id),
            action=action,
            task_description=method._task_description_str,
            template_function=method._function_to_evolve,
            history=context.primary_history,
            reference_node=reference_node if include_reference else None,
            reference_history=(context.reference_history if include_reference else ""),
        )

    single_code_tokens = [
        method._count_tokens(code_prompt(single, action, include_reference=False))
        for action in actions
    ]
    dual_code_tokens = [
        method._count_tokens(code_prompt(dual, action, include_reference=True))
        for action in actions
    ]
    limit = max(dual.token_count, single.token_count, *single_code_tokens)
    assert all(limit < token_count for token_count in dual_code_tokens)

    method._context_token_limit = limit
    method._rng = _IndexRng(0)
    method._run_iteration(0)

    assert method._graph.edges()
    assert all(edge.operator == OperatorName.REFINE for edge in method._graph.edges())
    assert all(edge.reference_trajectory_id is None for edge in method._graph.edges())
    action_prompts = [prompt for prompt in llm.prompts if "[Action Guidance]" in prompt]
    assert any("[Reference Program]" in prompt for prompt in action_prompts)
    assert any("[Reference Program]" not in prompt for prompt in action_prompts)


def test_context_limit_is_hard_for_initialization_and_single_route():
    llm = ScriptedV7LLM()
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
        operator=TRACE_REFINE,
        reference_route=None,
        reference_node=None,
    )
    assert context is None


def test_checkpoint_preserves_stop_state_and_rejects_protocol_drift():
    method = _make_method(max_sample_nums=3)
    method._stalled_iterations = 4
    method._consecutive_sample_failures = 5
    payload = dump_state(method)

    restored = _make_method(max_sample_nums=3)
    load_state(restored, payload)
    assert restored._stalled_iterations == 4
    assert restored._consecutive_sample_failures == 5

    with pytest.raises(ValueError, match="old checkpoints are not migrated"):
        load_state(_make_method(max_sample_nums=3), dict(payload, version=8))
    with pytest.raises(ValueError, match="protocol_id"):
        load_state(
            _make_method(max_sample_nums=3),
            dict(payload, protocol_id="traceaad-v7-drifted"),
        )
    with pytest.raises(ValueError, match="configuration"):
        load_state(_make_method(max_sample_nums=4), payload)


def test_checkpoint_rejects_runtime_identity_and_graph_metadata_drift():
    method = _make_method(n_init=1, max_sample_nums=1)
    method._initialize()
    payload = dump_state(method)

    changed_task = _make_method(n_init=1, max_sample_nums=1)
    changed_task._task_description_str = "A different algorithm-design task."
    with pytest.raises(ValueError, match="identity"):
        load_state(changed_task, payload)

    bad_metadata = json.loads(json.dumps(payload))
    bad_metadata["graph"]["nodes"][0]["code_hash"] = "not-the-code-hash"
    with pytest.raises(ValueError, match="invalid code metadata"):
        load_state(_make_method(n_init=1, max_sample_nums=1), bad_metadata)


def test_checkpoint_resume_continues_to_the_original_budget(tmp_path):
    method = _make_method(
        n_init=2,
        max_sample_nums=6,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1,
        random_seed=7,
    )
    method._initialize()
    method._initialization_complete = True
    method._run_iteration(0)
    method._next_attempt_id = 1
    save_checkpoint(method)

    resumed = _make_method(
        n_init=2,
        max_sample_nums=6,
        checkpoint_dir=tmp_path,
        checkpoint_interval=1,
        resume_from=tmp_path / "latest.json",
        random_seed=7,
    )
    result = resumed.run()

    assert result.n_samples == 6
    assert resumed._batch_count >= 2
    assert resumed._next_attempt_id >= 2
