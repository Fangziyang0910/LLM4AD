"""Mechanism contract tests for the Occam TraceAAD V6 protocol."""

from __future__ import annotations

import json
import math
import re
from dataclasses import fields
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v6 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADProfiler,
    TraceAADV6,
    ValueWeights,
)
from llm4ad.method.traceaad_v6.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v6.context import trajectory_history
from llm4ad.method.traceaad_v6.derivation_graph import DerivationGraph
from llm4ad.method.traceaad_v6.operators import (
    DEFAULT_OPERATORS,
    OperatorName,
    TraceRefineOp,
    TraceTransferOp,
    select_operator,
)
from llm4ad.method.traceaad_v6.prompt import build_code_prompt, parse_actions
from llm4ad.method.traceaad_v6.schema import ValueVec
from llm4ad.method.traceaad_v6.similarity import trajectory_similarity
from llm4ad.method.traceaad_v6.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad_v6.value import (
    program_quality_key,
    reference_sampling_distribution,
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
        if "Generate a simple, complete" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return (
            "1. Adjust the deterministic offset using trajectory evidence.\n"
            "2. Try a second rule suggested by the task and history."
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


def _add_route(
    graph: DerivationGraph,
    memory: TrajectoryMemory,
    fitnesses: list[float],
    *,
    code_prefix: str = "route",
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
            outcome="improve" if child.fitness > parent.fitness else "regress",
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


def test_public_package_identifies_occam_protocol():
    import llm4ad.method.traceaad_v6 as package

    assert package.TraceAADV6 is TraceAADV6
    assert package.PROTOCOL_ID == "traceaad-v6-occam-v1"
    assert package.CHECKPOINT_VERSION == 8
    assert set(package.__all__) == {
        "TraceAADV6",
        "TraceAADRunResult",
        "TraceAADProfiler",
        "ValueWeights",
        "CHECKPOINT_VERSION",
        "PROTOCOL_ID",
    }


def test_value_weights_contain_quality_and_ucb_only():
    assert {field.name for field in fields(ValueWeights)} == {
        "endpoint_quality",
        "best_quality",
        "ucb_c",
        "positive_threshold",
    }
    weights = ValueWeights()
    assert weights.endpoint_quality == 0.7
    assert weights.best_quality == 0.3
    assert weights.ucb_c == 0.25


def test_route_quality_is_endpoint_and_compact_best_fitness_only():
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
    values = {route.id: route.value.quality for route in scored if route.value}
    assert math.isclose(values[current_good.id], 0.7)
    assert math.isclose(values[historic_good.id], 0.3)


def test_equal_fitness_has_equal_route_rank_but_program_ties_prefer_shorter():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    short = graph.add_node(code="def f():\n    return 1\n", idea="short", fitness=1.0)
    long = graph.add_node(
        code="def f():\n    value = 1\n    return value\n", idea="long", fitness=1.0
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
    assert {route.value.quality for route in scored if route.value} == {0.5}
    assert program_quality_key(short, True) > program_quality_key(long, True)


def test_ucb_uses_global_batch_count_and_protects_low_visit_routes():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    route_a = _add_route(graph, memory, [1.0], code_prefix="a")
    route_b = _add_route(graph, memory, [1.0], code_prefix="b")
    for _ in range(3):
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
    assert adjusted[route_b.id] > adjusted[route_a.id]


def test_operator_sampling_is_uniform_over_the_available_set():
    operators = tuple(operator_type() for operator_type in DEFAULT_OPERATORS)
    single = {
        select_operator(
            operators=operators, allow_dual=False, rng=_IndexRng(index)
        ).operator.name
        for index in range(2)
    }
    all_names = {
        select_operator(
            operators=operators, allow_dual=True, rng=_IndexRng(index)
        ).operator.name
        for index in range(4)
    }
    assert single == {OperatorName.IDEATE, OperatorName.REFINE}
    assert all_names == set(OperatorName)


def test_constructor_rejects_a_dual_only_custom_operator_set():
    with pytest.raises(ValueError, match="single-trajectory operator"):
        _make_method(operators=(TraceTransferOp,))


def test_reference_sampling_uses_quality_without_hash_or_difference_gate():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    primary = _add_route(graph, memory, [1.0], code_prefix="same")
    low = _add_route(graph, memory, [1.0], code_prefix="same")
    high = _add_route(graph, memory, [1.0], code_prefix="same")
    primary = memory.set_value(primary.id, ValueVec(quality=0.4))
    low = memory.set_value(low.id, ValueVec(quality=0.1))
    high = memory.set_value(high.id, ValueVec(quality=0.9))

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
    route = _add_route(method._graph, method._memory, [10.0, 8.0])
    method._rng = _IndexRng(0)
    assert method._select_anchor(route) == (route.endpoint_id, "endpoint")
    method._rng = _IndexRng(1)
    assert method._select_anchor(route) == (route.compact_best_id, "compact_best")


def test_population_keeps_global_best_and_q_samples_without_deduplication():
    method = _make_method(
        n_init=0,
        max_sample_nums=0,
        max_active_trajectories=3,
        random_seed=4,
    )
    routes = []
    for fitness in range(6):
        node = method._graph.add_node(
            code="def choose(value: int) -> int:\n    return value\n",
            idea=f"route-{fitness}",
            fitness=float(fitness),
        )
        routes.append(method._memory.create_initial(node_id=node.id))
    method._best_node = method._graph.get_node(routes[-1].endpoint_id)
    method._best_trajectory_id = routes[-1].id

    method._manage_population()
    active = method.active_trajectories()
    assert len(active) == 3
    assert routes[-1].id in {route.id for route in active}
    assert (
        len({method._graph.get_node(route.endpoint_id).code_hash for route in active})
        == 1
    )


def test_similarity_remains_available_for_offline_diagnostics():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    left = _add_route(graph, memory, [1.0], code_prefix="left")
    right = _add_route(graph, memory, [1.0], code_prefix="right")
    score = trajectory_similarity(graph=graph, left=left, right=right)
    assert 0.0 <= score <= 1.0


def test_initialization_does_not_force_novelty_against_previous_ideas():
    llm = ScriptedV6LLM()
    method = _make_method(llm=llm, n_init=3, max_sample_nums=3)
    method._initialize()
    assert len(llm.prompts) == 3
    assert all("different" not in prompt.lower() for prompt in llm.prompts)


def test_history_is_factual_and_chronological_without_support_labels():
    graph = DerivationGraph()
    memory = TrajectoryMemory(max_trajectory_length=8)
    route = _add_route(graph, memory, [1.0, 0.5, 1.2])
    history = trajectory_history(graph, route, base_node_id=route.endpoint_id)
    assert "Step 1" in history.text and "Step 2" in history.text
    assert history.text.index("change-1") < history.text.index("change-2")
    assert "supported" not in history.text.lower()
    assert "priority" not in history.text.lower()


def test_action_and_code_share_primary_and_reference_histories():
    method = _make_method(n_init=2, max_sample_nums=2)
    method._initialize()
    primary, reference = method.active_trajectories()
    operator = next(
        item for item in method._operators if item.name == OperatorName.TRANSFER
    )
    reference_node = method._graph.get_node(reference.compact_best_id)
    context = method._build_action_context(
        selected=primary,
        anchor_id=primary.endpoint_id,
        operator=operator,
        reference_route=reference,
        reference_node=reference_node,
    )
    assert context is not None and context.used_dual
    assert reference_node.code.rstrip() in context.prompt

    code_prompt = build_code_prompt(
        current_node=method._graph.get_node(primary.endpoint_id),
        action="consider the reference idea",
        task_description=method._task_description_str,
        template_function=method._function_to_evolve,
        history=context.primary_history,
        reference_node=reference_node,
        reference_history=context.reference_history,
    )
    assert context.primary_history in code_prompt
    assert context.reference_history in code_prompt
    assert reference_node.code.rstrip() in code_prompt


def test_dual_code_overflow_restarts_the_batch_with_single_context():
    class FirstIndexRng(_IndexRng):
        def random(self) -> float:
            return 0.0

    llm = ScriptedV6LLM()
    method = _make_method(
        llm=llm,
        n_init=2,
        max_sample_nums=4,
        operators=(TraceTransferOp, TraceRefineOp),
    )
    method._initialize()
    primary, reference = method.active_trajectories()
    transfer = method._operators[0]
    refine = method._operators[1]
    reference_node = method._graph.get_node(reference.compact_best_id)
    dual = method._build_action_context(
        selected=primary,
        anchor_id=primary.endpoint_id,
        operator=transfer,
        reference_route=reference,
        reference_node=reference_node,
    )
    single = method._build_action_context(
        selected=primary,
        anchor_id=primary.endpoint_id,
        operator=refine,
        reference_route=None,
        reference_node=None,
    )
    assert dual is not None and single is not None
    action = "Adjust the deterministic offset using trajectory evidence."
    dual_code = build_code_prompt(
        current_node=method._graph.get_node(primary.endpoint_id),
        action=action,
        task_description=method._task_description_str,
        template_function=method._function_to_evolve,
        history=dual.primary_history,
        reference_node=reference_node,
        reference_history=dual.reference_history,
    )
    single_code = build_code_prompt(
        current_node=method._graph.get_node(primary.endpoint_id),
        action=action,
        task_description=method._task_description_str,
        template_function=method._function_to_evolve,
        history=single.primary_history,
    )
    limit = max(dual.token_count, single.token_count, method._count_tokens(single_code))
    assert limit < method._count_tokens(dual_code)

    method._context_token_limit = limit
    method._rng = FirstIndexRng(0)
    method._initialization_complete = True
    method._run_iteration(0)

    assert method._graph.edges()
    assert all(edge.operator == OperatorName.REFINE for edge in method._graph.edges())
    assert all(edge.reference_trajectory_id is None for edge in method._graph.edges())
    action_prompts = [prompt for prompt in llm.prompts if "[Action Guidance]" in prompt]
    assert any("[Reference Program]" in prompt for prompt in action_prompts)
    assert any("[Reference Program]" not in prompt for prompt in action_prompts)


def test_prompts_do_not_impose_quality_proxy_or_forced_trim_rules():
    method = _make_method(n_init=1, max_sample_nums=1)
    method._initialize()
    route = method.active_trajectories()[0]
    operator = next(
        item for item in method._operators if item.name == OperatorName.REFINE
    )
    context = method._build_action_context(
        selected=route,
        anchor_id=route.endpoint_id,
        operator=operator,
        reference_route=None,
        reference_node=None,
    )
    assert context is not None
    code_prompt = build_code_prompt(
        current_node=method._graph.get_node(route.endpoint_id),
        action="try an improvement",
        task_description=method._task_description_str,
        template_function=method._function_to_evolve,
        history=context.primary_history,
    )
    combined = f"{context.prompt}\n{code_prompt}".lower()
    assert "imports from the task template remain available" in combined
    assert "imports from the current program remain available" not in combined
    for forbidden in (
        "smallest executable",
        "preserve supported",
        "forced trim",
        "mature route",
        "difference threshold",
    ):
        assert forbidden not in combined


def test_action_parser_accepts_partial_batches_and_ignores_excess_actions():
    assert parse_actions("1. only one", expected_count=2)[0] == ["only one"]
    assert parse_actions("1. first\n2. second", expected_count=2)[0] == [
        "first",
        "second",
    ]
    actions, errors = parse_actions("1. first\n2. second\n3. ignored", expected_count=2)
    assert actions == ["first", "second"]
    assert "action_number_out_of_range_3" in errors


def test_reference_is_never_a_second_structural_parent():
    method = _make_method(max_sample_nums=20, n_init=4, max_active_trajectories=4)
    result = method.run()
    assert result.n_valid_nodes >= 4
    children = [edge.child_id for edge in method._graph.edges()]
    assert len(children) == len(set(children))


def test_batch_visit_increments_even_without_a_valid_child():
    class FailingCodeLLM(ScriptedV6LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            self.prompts.append(str(prompt))
            if "Generate a simple, complete" in prompt:
                return self._program(self.calls)
            if "[Requested Modification]" in prompt:
                return "Idea: broken\n```python\nnot valid python\n```"
            return "1. Try one change.\n2. Try another change."

    method = _make_method(llm=FailingCodeLLM(), max_sample_nums=6, n_init=3)
    method._initialize()
    method._initialization_complete = True
    visits_before = {
        route.id: route.visit_count for route in method.active_trajectories()
    }
    method._run_iteration(0)
    assert method._batch_count == 1
    assert any(
        route.visit_count > visits_before.get(route.id, 0)
        for route in method.active_trajectories()
    )


def test_end_to_end_smoke_and_checkpoint_resume(tmp_path: Path):
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
    assert result.n_samples == 16
    checkpoint = tmp_path / "latest.json"
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert payload["version"] == CHECKPOINT_VERSION
    assert payload["protocol_id"] == PROTOCOL_ID
    assert "attempts" not in payload

    resumed = _make_method(
        max_sample_nums=16,
        n_init=4,
        max_active_trajectories=4,
        resume_from=checkpoint,
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
            if "Generate a simple, complete" in prompt:
                return self._program(0)
            if "[Requested Modification]" not in prompt:
                return "\n".join(
                    f"{index}. {name}" for index, name in enumerate(self.order, 1)
                )
            return self._program(2 if "HIGH" in prompt else 1)

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


def test_llm_transport_failure_stays_in_observability_not_search_memory():
    class TransportFailureLLM(ScriptedV6LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            self.calls += 1
            if "Generate a simple, complete" in prompt:
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
    assert not hasattr(method, "_attempts")
    assert len(method._graph.nodes()) == 1
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


def test_context_limit_is_hard_for_initialization_and_single_trace():
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
    method = _make_method(profiler=profiler, n_init=1, max_sample_nums=3)
    method.run()
    records = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {record["stage"] for record in records} == {"init", "action", "code"}
    assert all("prompt" in record and "response" in record for record in records)


def test_checkpoint_preserves_stop_state_and_rejects_old_or_drifted_protocol():
    method = _make_method(max_sample_nums=3)
    method._stalled_iterations = 4
    method._consecutive_sample_failures = 5
    payload = dump_state(method)
    assert payload["version"] == 8
    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["stalled_iterations"] == 4
    assert payload["consecutive_sample_failures"] == 5
    assert "attempts" not in payload

    restored = _make_method(max_sample_nums=3)
    load_state(restored, payload)
    assert restored._stalled_iterations == 4
    assert restored._consecutive_sample_failures == 5

    old = dict(payload, version=7)
    with pytest.raises(ValueError, match="old checkpoints are not migrated"):
        load_state(_make_method(max_sample_nums=3), old)
    wrong_protocol = dict(payload, protocol_id="traceaad-v6-quality-gated")
    with pytest.raises(ValueError, match="protocol_id"):
        load_state(_make_method(max_sample_nums=3), wrong_protocol)
    with pytest.raises(ValueError, match="configuration"):
        load_state(_make_method(max_sample_nums=4), payload)


def test_checkpoint_rejects_task_evaluator_or_model_identity_drift():
    method = _make_method(max_sample_nums=3)
    payload = dump_state(method)

    changed_task = _make_method(max_sample_nums=3)
    changed_task._task_description_str = "A different algorithm-design task."
    with pytest.raises(ValueError, match="identity"):
        load_state(changed_task, payload)

    class DifferentEvaluation(IncreasingEvaluation):
        pass

    with pytest.raises(ValueError, match="identity"):
        load_state(
            _make_method(max_sample_nums=3, evaluation=DifferentEvaluation()), payload
        )


def test_checkpoint_rejects_malformed_graph_and_trajectory_references():
    method = _make_method(n_init=1, max_sample_nums=1)
    method._initialize()
    original = dump_state(method)
    bad_metadata = json.loads(json.dumps(original))
    bad_metadata["graph"]["nodes"][0]["code_hash"] = "not-the-code-hash"
    with pytest.raises(ValueError, match="invalid code metadata"):
        load_state(_make_method(n_init=1, max_sample_nums=1), bad_metadata)

    payload = json.loads(json.dumps(original))
    payload["memory"]["trajectories"][0]["node_ids"] = [999]
    payload["memory"]["trajectories"][0]["endpoint_id"] = 999
    payload["memory"]["trajectories"][0]["compact_best_id"] = 999
    with pytest.raises(ValueError, match="unknown node"):
        load_state(_make_method(n_init=1, max_sample_nums=1), payload)
