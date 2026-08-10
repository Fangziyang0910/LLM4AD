from __future__ import annotations

import ast
import io
import json
import tokenize
from pathlib import Path

from llm4ad.base import Evaluation, LLM, TextFunctionProgramConverter
from llm4ad.method.traceaad_artifacts import TraceAADArtifacts
from llm4ad.method.traceaad_v9_4 import PROTOCOL_ID, TraceAADV94
from llm4ad.method.traceaad_v9_4.checkpoint import (
    CHECKPOINT_VERSION,
    save_checkpoint,
)
from llm4ad.method.traceaad_v9_4.context import canonical_window
from llm4ad.method.traceaad_v9_4.complexity import comment_free_source
from llm4ad.method.traceaad_v9_4.prompt import (
    build_strategy_plan_prompt,
    parse_program_response,
    parse_strategy_plan,
)
from llm4ad.method.traceaad_v9_4.tree import FactGraph
from llm4ad.method.traceaad_v9_4.tree import (
    TRAJECTORY_CREDIT_DEPTH,
    TRAJECTORY_CREDIT_DISCOUNT,
)
from llm4ad.method.traceaad_v9_4.value import quality_pool, select_anchor

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, parse_fail_calls: set[int] | None = None) -> None:
        super().__init__()
        self.calls = 0
        self.candidate_calls = 0
        self.prompts: list[str] = []
        self.parse_fail_calls = parse_fail_calls or set()

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        if "Propose exactly" in prompt:
            count = int(prompt.split("Propose exactly ", 1)[1].split()[0])
            return "\n".join(
                f"Strategy {index}: distinct mechanism {index}"
                for index in range(1, count + 1)
            )
        self.candidate_calls += 1
        if self.candidate_calls in self.parse_fail_calls:
            return "Idea: try a malformed change\nnot a code block"
        return (
            f"Idea: candidate {self.candidate_calls}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.candidate_calls}\n"
            "```"
        )


class OneTransportFailureLLM(ScriptedLLM):
    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def draw_sample(self, prompt, *args, **kwargs):
        if not self.failed:
            self.failed = True
            self.calls += 1
            self.prompts.append(prompt)
            raise RuntimeError("temporary transport failure")
        return super().draw_sample(prompt, *args, **kwargs)


class CommentingLLM(ScriptedLLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "Propose exactly" in prompt:
            return super().draw_sample(prompt, *args, **kwargs)
        self.calls += 1
        self.candidate_calls += 1
        self.prompts.append(prompt)
        return (
            f"Idea: executable candidate {self.candidate_calls}\n"
            "```python\n"
            "# explanation that must be removed\n"
            "def choose(value: int) -> int:\n"
            '    """documentation that must be removed"""\n'
            f"    return value + {self.candidate_calls}  # inline explanation\n"
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


class SecondInvalidEvaluation(IncreasingEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return 1.0 if self.calls == 1 else None


class SequenceEvaluation(IncreasingEvaluation):
    def __init__(self, scores: list[float]) -> None:
        super().__init__()
        self.scores = scores

    def evaluate_program(self, program_str, callable_func, **kwargs):
        score = self.scores[self.calls]
        self.calls += 1
        return score


class CommentFreeEvaluation(IncreasingEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        tree = ast.parse(program_str)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        tokens = tokenize.generate_tokens(io.StringIO(program_str).readline)
        assert all(token.type != tokenize.COMMENT for token in tokens)
        assert all(ast.get_docstring(function) is None for function in functions)
        return super().evaluate_program(program_str, callable_func, **kwargs)


def add_root(graph: FactGraph, fitness: float, order: int) -> int:
    return graph.add_root(
        code=f"def choose(value):\n    return value + {order}\n",
        idea=f"root {order}",
        fitness=fitness,
        maximize=True,
        creation_order=order,
    ).id


def add_valid(
    graph: FactGraph, anchor_id: int, fitness: float, budget_order: int
) -> tuple[int, int]:
    child, event = graph.add_valid_event(
        anchor_id=anchor_id,
        code=f"def choose(value):\n    return value + {budget_order + 100}\n",
        idea=f"event {budget_order}",
        fitness=fitness,
        maximize=True,
        stage="search",
        iteration=budget_order,
        budget_order=budget_order,
        new_global_best=False,
        global_best_update_reason=None,
    )
    return child.id, event.id


def add_invalid(
    graph: FactGraph,
    anchor_id: int,
    budget_order: int,
    *,
    code: str | None = None,
) -> int:
    return graph.add_invalid_event(
        anchor_id=anchor_id,
        idea=f"failed event {budget_order}",
        code=code,
        failure_kind="parse",
        error_type=None,
        error_message=None,
        stage="search",
        iteration=budget_order,
        budget_order=budget_order,
    ).id


def test_anchor_value_uses_bounded_distance_decayed_trajectory_credit() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 7.0, 1)
    child_id, _ = add_valid(graph, root_id, 12.0, 2)
    root = graph.get_node(root_id)
    child = graph.get_node(child_id)

    assert root.budget_value == 9.5
    assert child.budget_value == 12.0

    add_valid(graph, child_id, 13.0, 3)
    assert child.budget_value == 12.5
    assert root.budget_value == 9.0

    add_root(graph, -1_000_000.0, 4)
    assert root.budget_value == 9.0

    add_valid(graph, root_id, 1.0, 5)
    assert root.budget_value == 7.0 + (2.5 + 1.5 + 0.0) / 3


def test_delayed_breakthrough_credits_intermediate_and_ancestor_differently() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    intermediate_id, _ = add_valid(graph, root_id, 5.0, 2)
    breakthrough_id, event_id = add_valid(graph, intermediate_id, 20.0, 3)

    root = graph.get_node(root_id)
    intermediate = graph.get_node(intermediate_id)
    breakthrough = graph.get_node(breakthrough_id)
    event = graph.get_event(event_id)

    assert event.anchor_credit == 7.5
    assert [
        (update.node_id, update.distance, update.credit)
        for update in event.credit_updates
    ] == [
        (intermediate.id, 1, 7.5),
        (root.id, 2, 2.5),
    ]
    assert intermediate.budget_value == 12.5
    assert root.budget_value == 11.25
    assert breakthrough.budget_value == 20.0


def test_trajectory_credit_stops_at_the_visible_downstream_depth() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 0.0, 1)
    first_id, _ = add_valid(graph, root_id, 0.0, 2)
    second_id, _ = add_valid(graph, first_id, 0.0, 3)
    third_id, _ = add_valid(graph, second_id, 0.0, 4)
    _, event_id = add_valid(graph, third_id, 100.0, 5)
    event = graph.get_event(event_id)

    assert TRAJECTORY_CREDIT_DEPTH == 3
    assert TRAJECTORY_CREDIT_DISCOUNT == 0.5
    assert [update.node_id for update in event.credit_updates] == [
        third_id,
        second_id,
        first_id,
    ]
    assert root_id not in [update.node_id for update in event.credit_updates]


def test_comment_free_source_removes_comments_and_docstrings_only() -> None:
    source = '''"""module documentation"""
def choose(value: int) -> int:
    """function documentation"""
    marker = "# data, not a comment"  # inline explanation
    # commented-out alternative: return 0
    return value + len(marker)
'''
    cleaned = comment_free_source(source)
    tree = ast.parse(cleaned)
    function = tree.body[0]

    assert ast.get_docstring(tree) is None
    assert isinstance(function, ast.FunctionDef)
    assert ast.get_docstring(function) is None
    assert "# data, not a comment" in cleaned
    tokens = tokenize.generate_tokens(io.StringIO(cleaned).readline)
    assert all(token.type != tokenize.COMMENT for token in tokens)


def test_program_parser_canonicalizes_generated_code_before_evaluation() -> None:
    template = TextFunctionProgramConverter.text_to_program(TEMPLATE)
    assert template is not None
    response = '''Idea: add a fixed offset
```python
# long explanation that must never enter the trajectory
def choose(value: int) -> int:
    """redundant documentation"""
    # another explanation
    return value + 2  # inline explanation
```
'''
    parsed = parse_program_response(response, template, "choose")
    assert parsed is not None
    source = str(parsed.program)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)

    assert ast.get_docstring(function) is None
    assert all(token.type != tokenize.COMMENT for token in tokens)
    assert source.strip().endswith("return value + 2")


def test_search_evaluates_stores_and_reuses_only_comment_free_code(
    tmp_path: Path,
) -> None:
    method = TraceAADV94(
        llm=CommentingLLM(),
        evaluation=CommentFreeEvaluation(),
        profiler=TraceAADArtifacts(run_dir=tmp_path),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    method.run()

    for node in method._graph.nodes():
        tokens = tokenize.generate_tokens(io.StringIO(node.code).readline)
        assert all(token.type != tokenize.COMMENT for token in tokens)
        assert '"""' not in node.code
    candidate_rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "candidates.jsonl")
        .read_text()
        .splitlines()
    ]
    assert all("# explanation" not in row["program"] for row in candidate_rows)
    assert "# explanation" not in method._llm.prompts[-1]


def test_minimization_uses_negated_raw_fitness_without_normalization() -> None:
    graph = FactGraph()
    root = graph.add_root(
        code="def choose(value):\n    return value\n",
        idea="root",
        fitness=10.0,
        maximize=False,
        creation_order=1,
    )
    child, _ = graph.add_valid_event(
        anchor_id=root.id,
        code="def choose(value):\n    return value + 1\n",
        idea="lower objective",
        fitness=8.0,
        maximize=False,
        stage="search",
        iteration=0,
        budget_order=2,
        new_global_best=True,
        global_best_update_reason="strict_fitness",
    )

    assert root.directed_fitness == -10.0
    assert root.budget_value == -9.0
    assert child.directed_fitness == -8.0
    assert graph.best_node() is child


def test_invalid_event_is_neutral_anchor_quality_but_consumes_evidence() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    event_id = add_invalid(graph, root_id, 2)
    root = graph.get_node(root_id)
    event = graph.get_event(event_id)

    assert root.budget_event_count == 1
    assert root.trajectory_event_count == 1
    assert root.budget_value == 10.0
    assert event.anchor_credit == 0.0
    assert event.child_id is None


def test_valid_regression_adds_zero_credit_and_remains_eligible_fact() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    child_id, event_id = add_valid(graph, root_id, 2.0, 2)
    root = graph.get_node(root_id)
    event = graph.get_event(event_id)

    assert event.child_id == child_id
    assert event.outcome == "regress"
    assert event.anchor_credit == 0.0
    assert root.budget_value == 10.0


def test_strict_breakthrough_excludes_equal_fitness_shorter_tie() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    _, strict_event = graph.add_valid_event(
        anchor_id=root_id,
        code="def choose(value):\n    return value + 1\n",
        idea="strict improvement",
        fitness=11.0,
        maximize=True,
        stage="search",
        iteration=0,
        budget_order=2,
        new_global_best=True,
        global_best_update_reason="strict_fitness",
    )
    _, tie_event = graph.add_valid_event(
        anchor_id=root_id,
        code="def choose(value):\n    return value\n",
        idea="shorter tie",
        fitness=10.0,
        maximize=True,
        stage="search",
        iteration=1,
        budget_order=3,
        new_global_best=True,
        global_best_update_reason="tie_shorter",
    )

    assert strict_event.strict_breakthrough
    assert not tie_event.strict_breakthrough


def test_evaluator_invalid_event_keeps_code_change_offline_not_in_prompt() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    event_id = add_invalid(
        graph,
        root_id,
        2,
        code="def choose(value):\n    candidate = value + 1\n    return candidate\n",
    )
    event = graph.get_event(event_id)
    window = canonical_window(graph, root_id)

    assert event.code_change_ratio is not None
    assert event.delta_loc == 1
    assert "Parsed code change" not in window.text
    assert "LOC" not in window.text


def test_selection_uses_q_top10_then_prioritizes_untested_anchor() -> None:
    graph = FactGraph()
    root_ids = [add_root(graph, 100.0 - index, index + 1) for index in range(11)]
    add_invalid(graph, root_ids[0], 20)

    pool = quality_pool(graph, pool_size=10)
    assert root_ids[-1] not in [node.id for node in pool]
    selection = select_anchor(graph, pool_size=10)
    assert selection.selected_node_id == root_ids[1]
    assert selection.mode == "basic_validation"

    for offset, node in enumerate(pool[1:], start=21):
        add_invalid(graph, node.id, offset)
    selection = select_anchor(graph, pool_size=10)
    assert selection.selected_node_id == root_ids[0]
    assert selection.mode == "anchor_productivity"


def test_every_fifth_selection_covers_the_least_tested_quality_anchor() -> None:
    graph = FactGraph()
    root_ids = [add_root(graph, 100.0 - index, index + 1) for index in range(3)]
    for order, root_id in enumerate(root_ids, start=10):
        add_invalid(graph, root_id, order)
    add_invalid(graph, root_ids[0], 20)
    add_invalid(graph, root_ids[0], 21)

    exploit = select_anchor(graph, pool_size=3, decision_order=3)
    explore = select_anchor(graph, pool_size=3, decision_order=4)

    assert exploit.selected_node_id == root_ids[0]
    assert exploit.mode == "anchor_productivity"
    assert explore.selected_node_id == root_ids[1]
    assert explore.mode == "quality_local_exploration"

    fresh_graph = FactGraph()
    waiting_id = add_root(fresh_graph, 90.0, 1)
    add_root(fresh_graph, 100.0, 2)
    periodic = select_anchor(fresh_graph, pool_size=2, decision_order=4)
    assert periodic.selected_node_id == waiting_id
    assert periodic.mode == "quality_local_exploration"


def test_strategy_plan_parser_requires_exact_distinct_numbered_routes() -> None:
    valid = "\n".join(f"Strategy {index}: mechanism {index}" for index in range(1, 4))
    assert parse_strategy_plan(valid, 3) == (
        "mechanism 1",
        "mechanism 2",
        "mechanism 3",
    )
    assert parse_strategy_plan(valid.replace("mechanism 3", "mechanism 2"), 3) is None
    assert parse_strategy_plan(valid.replace("Strategy 2", "Strategy 4"), 3) is None


def test_strategy_plan_prompt_contains_only_task_contract_and_strategy_request() -> (
    None
):
    evaluation = IncreasingEvaluation()
    template = evaluation.template_program
    program = TextFunctionProgramConverter.text_to_program(template)
    assert program is not None
    prompt = build_strategy_plan_prompt(
        task_description=evaluation.task_description,
        template_function=program.functions[0],
        maximize=True,
        strategy_count=8,
    )

    assert "Improve choose." in prompt
    assert "def choose(value: int) -> int:" in prompt
    assert "Propose exactly 8 complementary" in prompt
    assert "Do not write code" in prompt
    assert "Fitness:" not in prompt
    assert "Existing Evaluated" not in prompt


def test_canonical_window_uses_formation_and_recent_downstream_without_fitness_pick() -> (
    None
):
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    current = root_id
    formation_ids: list[int] = []
    for order, fitness in enumerate((9.0, 8.0, 7.0, 6.0, 5.0), start=2):
        current, event_id = add_valid(graph, current, fitness, order)
        formation_ids.append(event_id)
    anchor_id = current

    downstream_ids = [add_invalid(graph, anchor_id, order) for order in range(20, 26)]
    graph.get_event(downstream_ids[0])
    window = canonical_window(graph, anchor_id)

    assert window.formation_event_ids == tuple(formation_ids[-4:])
    assert window.downstream_event_ids == tuple(downstream_ids[-4:])
    assert "result invalid (parse)" in window.text
    assert "failed event 20" not in window.text
    assert "failed event 25" in window.text


def test_canonical_window_fills_unused_formation_slots_from_downstream() -> None:
    graph = FactGraph()
    root_id = add_root(graph, 10.0, 1)
    anchor_id, formation_event = add_valid(graph, root_id, 11.0, 2)
    downstream_ids = [add_invalid(graph, anchor_id, order) for order in range(10, 17)]

    window = canonical_window(graph, anchor_id)
    assert window.formation_event_ids == (formation_event,)
    assert window.downstream_event_ids == tuple(downstream_ids[-7:])
    assert len(window.formation_event_ids) + len(window.downstream_event_ids) == 8


def test_downstream_window_stops_at_depth_three_and_preserves_real_branch() -> None:
    graph = FactGraph()
    anchor_id = add_root(graph, 10.0, 1)
    first_id, event1 = add_valid(graph, anchor_id, 9.0, 2)
    second_id, event2 = add_valid(graph, first_id, 8.0, 3)
    third_id, event3 = add_valid(graph, second_id, 7.0, 4)
    _, event4 = add_valid(graph, third_id, 1000.0, 5)

    window = canonical_window(graph, anchor_id)
    assert window.downstream_event_ids == (event1, event2, event3)
    assert event4 not in window.downstream_event_ids
    assert "branch A" in window.text
    assert f"branch {first_id}" not in window.text
    assert "result depth 3" in window.text


def test_eight_strategy_routes_are_curated_to_six_anchors_before_competition() -> None:
    llm = ScriptedLLM()
    evaluation = IncreasingEvaluation()
    method = TraceAADV94(
        llm=llm,
        evaluation=evaluation,
        max_sample_nums=16,
        initial_route_pool_size=8,
        initial_anchor_count=6,
        context_token_limit=24576,
    )
    result = method.run()

    roots = [
        method._graph.get_node(node_id) for node_id in method._graph.root.child_ids
    ]
    assert result.n_samples == 16
    assert result.n_evaluations == 16
    assert result.n_root_children == 8
    assert result.n_eligible_nodes == 6
    assert result.n_events == 8
    assert method._initialization_complete
    assert method._bootstrapped_root_ids == {root.id for root in roots}
    assert [root.budget_event_count for root in roots] == [1] * 8
    assert len(method._eligible_node_ids) == 6
    assert method._eligible_node_ids == set(range(10, 16))
    assert len(method._initial_strategy_cards) == 8
    assert llm.calls == 17
    assert "Propose exactly 8 complementary" in llm.prompts[0]
    assert "Do not write code" in llm.prompts[0]
    assert "[Assigned Initial Strategy 1]" in llm.prompts[1]
    assert "[Existing Evaluated Initial Routes]" not in llm.prompts[2]
    assert "return value + 1" not in llm.prompts[2]
    assert "The Idea must accurately summarize" in llm.prompts[1]
    assert "do not include comments, docstrings" in llm.prompts[1]
    assert "[Initial Route Strategy]" in llm.prompts[9]


def test_initial_curation_uses_best_executable_endpoint_quality() -> None:
    method = TraceAADV94(
        llm=ScriptedLLM(),
        evaluation=SequenceEvaluation([100.0, 70.0, 0.0, 80.0]),
        max_sample_nums=4,
        initial_route_pool_size=2,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    method.run()

    roots = [
        method._graph.get_node(node_id) for node_id in method._graph.root.child_ids
    ]
    improving_child = method._graph.get_node(roots[1].child_ids[0])
    assert roots[0].fitness == 100.0
    assert roots[0].budget_value == 100.0
    assert improving_child.fitness == 80.0
    assert method._eligible_node_ids == {roots[0].id}


def test_one_formal_budget_event_generates_exactly_one_candidate() -> None:
    llm = ScriptedLLM()
    method = TraceAADV94(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=17,
        initial_route_pool_size=8,
        initial_anchor_count=6,
        context_token_limit=24576,
    )
    result = method.run()

    assert result.n_iterations == 1
    assert result.n_events == 9
    assert result.n_samples == 17
    assert llm.calls == 18
    assert result.n_eligible_nodes == 7
    assert method._graph.events()[-1].anchor_id in set(range(10, 16))
    assert method._graph.events()[-1].child_id in method._eligible_node_ids


def test_summary_keeps_initial_anchor_count_separate_from_later_children(
    tmp_path: Path,
) -> None:
    method = TraceAADV94(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        profiler=TraceAADArtifacts(run_dir=tmp_path),
        max_sample_nums=17,
        initial_route_pool_size=8,
        initial_anchor_count=6,
        context_token_limit=24576,
    )
    method.run()
    summary = json.loads(
        (tmp_path / "logs" / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["initial_anchor_actual"] == 6
    assert summary["n_eligible_nodes"] == 7


def test_generation_prompt_contains_only_current_anchor_full_code() -> None:
    llm = ScriptedLLM()
    method = TraceAADV94(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    method.run()

    formal_prompt = llm.prompts[3]
    assert "return value + 2" in formal_prompt
    assert "return value + 1" not in formal_prompt
    assert "Idea implemented: candidate 2" in formal_prompt
    assert "bounded local view, not the complete search history" in formal_prompt
    assert "Nearby tests may come from different branches" in formal_prompt
    assert "specific implementation" in formal_prompt
    assert "The Idea must accurately summarize" in formal_prompt
    assert "This is one complete search decision" in formal_prompt
    assert "do not include comments, docstrings" in formal_prompt
    assert "continue, correct, reorganize, or replace" not in formal_prompt
    assert "Do not repeat a tested change" not in formal_prompt
    assert "[Target Function]" not in formal_prompt
    assert "Budget " not in formal_prompt
    assert "[event " not in formal_prompt
    assert "Code change:" not in formal_prompt
    assert "LOC" not in formal_prompt


def test_parse_failure_consumes_budget_and_enters_anchor_window() -> None:
    llm = ScriptedLLM(parse_fail_calls={2})
    method = TraceAADV94(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    result = method.run()
    root = method._graph.get_node(method._graph.root.child_ids[0])
    window = canonical_window(method._graph, root.id)

    assert result.n_samples == 3
    assert result.n_evaluations == 2
    assert root.budget_event_count == 2
    assert "result invalid (parse; ProgramResponseError)" in window.text
    assert "try a malformed change" in window.text
    assert "missing a fenced Python code block" in window.text
    assert "[Verified Failure Feedback From This Run]" in method._llm.prompts[-1]
    assert "observed 1 time(s): parse; ProgramResponseError" in method._llm.prompts[-1]


def test_root_generation_failure_is_fed_back_before_retry() -> None:
    llm = ScriptedLLM(parse_fail_calls={1})
    method = TraceAADV94(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    method.run()

    retry_prompt = llm.prompts[2]
    assert "[Verified Failure Feedback From This Run]" in retry_prompt
    assert "missing a fenced Python code block" in retry_prompt
    assert "not a code block" not in retry_prompt


def test_evaluator_failure_consumes_budget_and_keeps_parsed_change_facts() -> None:
    method = TraceAADV94(
        llm=ScriptedLLM(),
        evaluation=SecondInvalidEvaluation(),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    result = method.run()
    root = method._graph.get_node(method._graph.root.child_ids[0])
    event = method._graph.events()[0]
    window = canonical_window(method._graph, root.id)

    assert result.n_evaluations == 3
    assert event.child_id is None
    assert event.code_change_ratio is not None
    assert event.failure_kind == "invalid_result"
    assert event.error_type == "InvalidEvaluationResult"
    assert event.error_message == "evaluator returned None"
    assert "InvalidEvaluationResult" in window.text
    assert "evaluator returned None" in window.text
    assert "Parsed code change" not in window.text
    assert "[Verified Failure Feedback From This Run]" in method._llm.prompts[-1]
    assert "evaluator returned None" in method._llm.prompts[-1]


def test_v94_artifacts_preserve_exact_runtime_prompts(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    method = TraceAADV94(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        profiler=TraceAADArtifacts(run_dir=tmp_path),
        max_sample_nums=2,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    method.run()

    rows = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["prompt"] for row in rows] == llm.prompts


def test_transport_failure_does_not_consume_budget_or_skip_root_bootstrap() -> None:
    llm = OneTransportFailureLLM()
    method = TraceAADV94(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
    )
    result = method.run()

    root_id = method._graph.root.child_ids[0]
    assert result.n_samples == 2
    assert result.n_evaluations == 2
    assert llm.calls == 4
    assert method._initialization_complete
    assert method._bootstrapped_root_ids == {root_id}


def test_checkpoint_round_trip_preserves_anchor_credit_and_events(
    tmp_path: Path,
) -> None:
    method = TraceAADV94(
        llm=ScriptedLLM(parse_fail_calls={3}),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    method.run()
    checkpoint = save_checkpoint(method)
    assert checkpoint is not None
    payload = json.loads(checkpoint.read_text())
    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["version"] == CHECKPOINT_VERSION

    restored = TraceAADV94(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "restored",
        resume_from=checkpoint,
    )
    assert restored._tot_sample_nums == method._tot_sample_nums
    assert restored._evaluation_count == method._evaluation_count
    assert restored._bootstrapped_root_ids == method._bootstrapped_root_ids
    assert restored._initial_selected_anchor_ids == method._initial_selected_anchor_ids
    assert restored._initial_strategy_cards == method._initial_strategy_cards
    assert restored._root_strategy_cards == method._root_strategy_cards
    assert restored._eligible_node_ids == method._eligible_node_ids
    assert restored._failure_history == method._failure_history
    assert [node.budget_value for node in restored._graph.nodes()] == [
        node.budget_value for node in method._graph.nodes()
    ]
    assert restored._graph.events() == method._graph.events()


def test_resumed_search_continues_from_next_budget_event(tmp_path: Path) -> None:
    method = TraceAADV94(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    method._initialize()
    assert method._tot_sample_nums == 2
    checkpoint = save_checkpoint(method)
    assert checkpoint is not None

    restored = TraceAADV94(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        initial_route_pool_size=1,
        initial_anchor_count=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "restored",
        resume_from=checkpoint,
    )
    result = restored.run()

    assert result.n_samples == 4
    assert result.n_iterations == 2
    assert result.n_events == 3
    assert restored._initialization_complete
