from __future__ import annotations

import random

import pytest

from llm4ad.base import Evaluation, LLM, TextFunctionProgramConverter
from llm4ad.method.traceaad_v8_3 import DEFAULT_OPERATORS, TraceAADV8_3
from llm4ad.method.traceaad_v8_3.context import build_local_context
from llm4ad.method.traceaad_v8_3.prompt import parse_call1, parse_description
from llm4ad.method.traceaad_v8_3.schema import AlgorithmRecord, OperatorName
from llm4ad.method.traceaad_v8_3.tree import SearchTree
from llm4ad.method.traceaad_v8_3.value import (
    SearchWeights,
    expansion_reward,
    node_fitness,
    route_quality,
    select_expansion_node,
    subtree_rank,
    trajectory_prior,
)


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class _Evaluation(Evaluation):
    def __init__(self, *, fail_after: int | None = None) -> None:
        super().__init__(
            TEMPLATE, task_description="Improve choose.", safe_evaluate=False
        )
        self.calls = 0
        self.fail_after = fail_after

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            return None
        return float(self.calls)


class _LLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        prompt = str(prompt)
        self.prompts.append(prompt)
        if "[Generated Code]" in prompt:
            return "Description: deterministic offset implementation."
        return (
            "Design Idea: add a deterministic offset\n"
            "Code:\n```python\n"
            f"def choose(value: int) -> int:\n    return value + {self.calls}\n"
            "```"
        )


def test_v83_initialization_and_two_call_protocol():
    llm = _LLM()
    evaluation = _Evaluation()
    method = TraceAADV8_3(llm, evaluation, n_init=2, max_sample_nums=2, random_seed=0)

    result = method.run()

    assert result.initialization_complete
    assert result.n_root_children == 2
    assert result.n_edges == 0
    assert evaluation.calls == 2
    assert llm.calls == 4
    assert method.attempts[1].operator == "e1"
    assert method.attempts[1].status == "node_created"
    assert "[Reference Algorithm 1]" in llm.prompts[2]
    assert "[Generated Code]" in llm.prompts[1]


def test_v83_search_adds_one_single_parent_child():
    llm = _LLM()
    method = TraceAADV8_3(
        llm, _Evaluation(), n_init=2, max_sample_nums=3, random_seed=1
    )

    result = method.run()

    assert result.n_total_nodes == 3
    assert result.n_edges == 1
    child = method.tree.get_node(2)
    assert child.parent_id in (0, 1)
    assert method.tree.get_edge(child.parent_id, child.id).reference_node_id in (
        None,
        0,
        1,
    )
    assert child.algorithm.description == "deterministic offset implementation."


def test_v83_evaluation_failure_consumes_budget_but_does_not_enter_tree():
    llm = _LLM()
    method = TraceAADV8_3(
        llm,
        _Evaluation(fail_after=2),
        n_init=2,
        max_sample_nums=3,
        max_consecutive_failures=1,
        random_seed=0,
    )

    result = method.run()

    assert result.n_samples == 3
    assert result.n_total_nodes == 2
    assert result.n_edges == 0
    assert result.stop_reason == "consecutive_failures"
    assert (
        method.tree.get_node(0).expansion_attempts
        + method.tree.get_node(1).expansion_attempts
        == 1
    )
    assert (
        method.tree.get_node(0).credit_count + method.tree.get_node(1).credit_count == 1
    )
    assert (
        method.tree.get_node(0).credit_sum + method.tree.get_node(1).credit_sum == 0.0
    )


def test_v83_context_distinguishes_omitted_history_from_absent_history():
    tree = SearchTree()
    parent = tree.add_initial(
        AlgorithmRecord("seed", "def f():\n return 0", "seed", 1.0, 0.0)
    )
    child, _ = tree.add_child(
        parent.id,
        AlgorithmRecord("change", "def f():\n return 1", "change", 2.0, 0.0),
        OperatorName.REFINE,
        None,
    )
    context = build_local_context(
        tree, parent, maximize=True, max_formation_edges=0, max_direct_children=0
    )
    assert "shortened to fit" in context.text
    assert "No valid direct child" not in context.text


def test_v83_expansion_credit_stops_at_direct_child():
    tree = SearchTree()
    parent = tree.add_initial(
        AlgorithmRecord("seed", "def f():\n return 0", "seed", 0.0, 0.0)
    )
    tree.add_initial(
        AlgorithmRecord("anchor", "def f():\n return 1", "anchor", 3.0, 0.0)
    )
    child, edge = tree.add_child(
        parent.id,
        AlgorithmRecord("step", "def f():\n return 2", "step", 2.0, 0.0),
        OperatorName.REFINE,
        None,
    )
    edge.parent_quality = node_fitness(tree, parent, maximize=True)
    edge.child_quality = node_fitness(tree, child, maximize=True)
    weights = SearchWeights()
    before = expansion_reward(tree, parent, child, maximize=True, weights=weights)
    tree.add_child(
        child.id,
        AlgorithmRecord(
            "breakthrough", "def f():\n return 3", "breakthrough", 100.0, 0.0
        ),
        OperatorName.REFINE,
        None,
    )

    direct_quality = edge.child_quality
    parent_quality = edge.parent_quality
    expected = 0.75 * direct_quality + 0.25 * max(0.0, direct_quality - parent_quality)

    # The existing child is allowed to inherit the remote breakthrough for
    # ``descend``; the expansion action is credited only for its direct child.
    assert subtree_rank(tree, child.id, maximize=True) == 1.0
    assert expansion_reward(
        tree, parent, child, maximize=True, weights=weights
    ) == pytest.approx(expected)
    assert expansion_reward(
        tree, parent, child, maximize=True, weights=weights
    ) == pytest.approx(before)
    assert expansion_reward(
        tree, parent, child, maximize=True, weights=weights
    ) < subtree_rank(tree, child.id, maximize=True)


def test_v83_selection_has_no_fixed_depth_horizon():
    tree = SearchTree()
    root_child = tree.add_initial(
        AlgorithmRecord("root", "def f():\n return 0", "root", 0.0, 0.0)
    )
    terminal = root_child
    for depth in range(2, 13):
        terminal, _ = tree.add_child(
            terminal.id,
            AlgorithmRecord(
                f"step-{depth}", "def f():\n return 1", "step", float(depth), 0.0
            ),
            OperatorName.REFINE,
            None,
        )

    selection = select_expansion_node(
        tree,
        maximize=True,
        weights=SearchWeights(),
        rng=random.Random(0),
    )
    assert selection.node_id == terminal.id
    assert len(selection.path) == 13
    assert tree.get_node(selection.node_id).depth == 12


def test_v83_progressive_widening_breaks_a_deep_subtree_tie():
    tree = SearchTree()
    root_child = tree.add_initial(
        AlgorithmRecord("root", "def f():\n return 0", "root", 0.0, 0.0)
    )
    direct, _ = tree.add_child(
        root_child.id,
        AlgorithmRecord("direct", "def f():\n return 1", "direct", 1.0, 0.0),
        OperatorName.REFINE,
        None,
    )
    tree.add_child(
        direct.id,
        AlgorithmRecord(
            "breakthrough", "def f():\n return 2", "breakthrough", 100.0, 0.0
        ),
        OperatorName.REFINE,
        None,
    )
    # The parent has been visited four times and has only one direct child;
    # alpha=.5 therefore requires a second lateral expansion even though the
    # existing child owns a much better historical subtree value.
    root_child.visit_count = 4
    root_child.expansion_attempts = 1

    selection = select_expansion_node(
        tree,
        maximize=True,
        weights=SearchWeights(exploration_constant=0.0),
        rng=random.Random(0),
    )
    assert selection.node_id == root_child.id
    assert selection.steps[-1].option == "new_child"


def test_v83_progressive_widening_reopens_virtual_root_routes():
    tree = SearchTree()
    for index in range(2):
        tree.add_initial(
            AlgorithmRecord(
                f"seed-{index}",
                "def f():\n return 0",
                f"seed-{index}",
                float(index),
                0.0,
            )
        )
    tree.root.visit_count = 9

    selection = select_expansion_node(
        tree,
        maximize=True,
        weights=SearchWeights(),
        rng=random.Random(0),
    )

    assert selection.node_id == tree.root.id
    assert selection.path == (tree.root.id,)
    assert selection.steps[-1].option == "new_child"


def test_v83_improving_recent_route_has_higher_prior_with_same_endpoint_and_best():
    tree = SearchTree()
    first_root = tree.add_initial(
        AlgorithmRecord("a0", "def f(): pass", "a0", 0.0, 0.0)
    )
    first_best, _ = tree.add_child(
        first_root.id,
        AlgorithmRecord("a1", "def f(): pass", "a1", 2.0, 0.0),
        OperatorName.REFINE,
        None,
    )
    regressing, _ = tree.add_child(
        first_best.id,
        AlgorithmRecord("a2", "def f(): pass", "a2", 1.0, 0.0),
        OperatorName.REFINE,
        None,
    )
    second_root = tree.add_initial(
        AlgorithmRecord("b0", "def f(): pass", "b0", 2.0, 0.0)
    )
    second_low, _ = tree.add_child(
        second_root.id,
        AlgorithmRecord("b1", "def f(): pass", "b1", 0.0, 0.0),
        OperatorName.REFINE,
        None,
    )
    improving, _ = tree.add_child(
        second_low.id,
        AlgorithmRecord("b2", "def f(): pass", "b2", 1.0, 0.0),
        OperatorName.REFINE,
        None,
    )

    weights = SearchWeights()
    assert node_fitness(tree, improving, True) == node_fitness(tree, regressing, True)
    assert trajectory_prior(tree, improving, True, weights) > trajectory_prior(
        tree, regressing, True, weights
    )


def test_v83_zero_outcomes_correct_one_lucky_route_credit_downward():
    tree = SearchTree()
    node = tree.add_initial(AlgorithmRecord("seed", "def f(): pass", "seed", 1.0, 0.0))
    weights = SearchWeights()
    path = tree.selected_path(node.id)
    tree.record_outcome(path, 1.0)
    values = [route_quality(tree, node, True, weights)]
    for _ in range(3):
        tree.record_outcome(path, 0.0)
        values.append(route_quality(tree, node, True, weights))
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)


def test_v83_softmax_favors_better_route_without_making_it_deterministic():
    tree = SearchTree()
    low = tree.add_initial(AlgorithmRecord("low", "def f(): pass", "low", 0.0, 0.0))
    high = tree.add_initial(AlgorithmRecord("high", "def f(): pass", "high", 1.0, 0.0))
    weights = SearchWeights(exploration_constant=0.0)
    selected = [
        select_expansion_node(
            tree, maximize=True, weights=weights, rng=random.Random(seed)
        )
        for seed in range(200)
    ]
    counts = {
        low.id: sum(result.path[1] == low.id for result in selected),
        high.id: sum(result.path[1] == high.id for result in selected),
    }
    assert counts[high.id] > counts[low.id] > 0
    assert all(0.0 < result.steps[0].probability < 1.0 for result in selected)


def test_v83_keeps_five_operators_equally_eligible():
    assert {operator.name for operator in DEFAULT_OPERATORS} == set(OperatorName)


def test_v83_text_protocol_rejects_extra_output():
    template = TextFunctionProgramConverter.text_to_program(TEMPLATE)
    assert template is not None
    valid = "Design Idea: add one\nCode:\n```python\ndef choose(value: int) -> int:\n    return value\n```"
    assert parse_call1(valid, template, "choose") is not None
    assert parse_call1("preface\n" + valid, template, "choose") is None
    assert (
        parse_call1(
            "Design Idea: add one\ndef choose(value):\n return value",
            template,
            "choose",
        )
        is None
    )
    assert parse_description("Description: factual behavior") == "factual behavior"
    assert parse_description("Description: factual behavior\nextra") is None
