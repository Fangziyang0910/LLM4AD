"""Mechanism tests for the TraceAAD V8 complete-tree protocol."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM, TextFunctionProgramConverter
from llm4ad.method.traceaad_v8 import RunArtifacts
from llm4ad.method.traceaad_v8 import TraceAADV8
from llm4ad.method.traceaad_v8.checkpoint import dump_state, load_state, save_checkpoint
from llm4ad.method.traceaad_v8.context import node_history, select_direct_children
from llm4ad.method.traceaad_v8.operators import TraceIdeateOp, TraceSynthesizeOp
from llm4ad.method.traceaad_v8.prompt import parse_program_response
from llm4ad.method.traceaad_v8.schema import OperatorName
from llm4ad.method.traceaad_v8.tree import SearchTree
from llm4ad.method.traceaad_v8.value import (
    expansion_batch_rewards,
    expansion_quality,
    fitness_reference_values,
    normalize_value,
    reference_candidates,
    remaining_budget_ratio,
    sample_reference,
    select_expansion_node,
    uct_score,
)

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        text = str(prompt)
        self.prompts.append(text)
        return (
            f"Idea: deterministic candidate {self.calls}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
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


class FailingEvaluation(IncreasingEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return 1.0 if self.calls == 1 else None


class BrokenCodeLLM(ScriptedLLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Improvement Direction]" in str(prompt):
            self.calls += 1
            self.prompts.append(str(prompt))
            return "not a program"
        return super().draw_sample(prompt, *args, **kwargs)


class LengthCountingLLM(ScriptedLLM):
    def count_tokens(self, prompt: str) -> int:
        return len(prompt)


class NonNumericEvaluation(IncreasingEvaluation):
    RESULTS = (1.0, "not-a-number", float("nan"), 4.0)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        result = self.RESULTS[self.calls]
        self.calls += 1
        return result


class ConfigurableEvaluation(IncreasingEvaluation):
    def __init__(self, scale: int) -> None:
        super().__init__()
        self.scale = scale


def add_child(
    tree: SearchTree,
    parent_id: int,
    fitness: float,
    *,
    code: str | None = None,
    creation_order: int | None = None,
    batch_id: int | None = None,
    record_visit: bool = True,
):
    index = tree._next_node_id
    if record_visit:
        tree.record_batch_visit(parent_id)
    child, edge, _ = tree.add_child(
        parent_id=parent_id,
        code=code or f"def choose(value):\n    return value + {index}\n",
        idea=f"child {index}",
        fitness=fitness,
        maximize=True,
        creation_order=index if creation_order is None else creation_order,
        operator=OperatorName.IDEATE,
        reference_node_id=None,
        reference_root_branch_id=None,
        global_best_directed_fitness=None,
        new_global_best=False,
        global_best_update_reason=None,
        iteration=0,
        batch_id=index + 1 if batch_id is None else batch_id,
        sibling_seq=0,
        sample_order=index + 1,
    )
    return child, edge


def make_tree(fitnesses: tuple[float, ...] = (1.0, 2.0)) -> SearchTree:
    tree = SearchTree()
    for order, fitness in enumerate(fitnesses, start=1):
        tree.add_initial(
            code=f"def choose(value):\n    return value + {order}\n",
            idea=f"initial {order}",
            fitness=fitness,
            maximize=True,
            creation_order=order,
        )
    return tree


def test_virtual_root_and_complete_single_parent_tree() -> None:
    tree = make_tree()
    root = tree.root
    assert not hasattr(root, "code")
    assert not hasattr(root, "fitness")
    child, _ = add_child(tree, root.child_ids[0], 3.0)
    grandchild, _ = add_child(tree, child.id, 4.0)

    assert all(node.parent_id == root.id for node in tree.nodes() if node.depth == 1)
    assert grandchild.parent_id == child.id
    assert len(tree.ancestor_node_ids(grandchild.id)) == 3
    assert tree.root_branch_id(grandchild.id) == root.child_ids[0]
    assert not hasattr(tree, "active")
    assert not hasattr(tree, "archive")
    assert not hasattr(tree, "population")


def test_subtree_max_backup_and_exact_tie_prefer_shorter_program() -> None:
    tree = make_tree((1.0,))
    branch = tree.root.child_ids[0]
    long, _ = add_child(
        tree,
        branch,
        5.0,
        code="def choose(value):\n    copied = value\n    return copied\n",
    )
    short, _ = add_child(
        tree,
        branch,
        5.0,
        code="def choose(value):\n    return value\n",
    )

    assert tree.get_node(branch).subtree_value == 5.0
    assert tree.get_node(branch).subtree_best_node_id == short.id
    assert tree.root.subtree_best_node_id == short.id
    assert long.id != short.id


def test_uct_normalization_budget_decay_and_equal_fitness_fallback() -> None:
    tree = make_tree((2.0, 2.0))
    child = tree.get_node(tree.root.child_ids[0])
    assert normalize_value(2.0, (2.0, 2.0)) == 0.5
    assert normalize_value(9.0, (-1000.0, 9.0, 10.0)) == 0.5
    early = uct_score(
        child=child,
        parent_visits=10,
        reference_values=(2.0, 2.0),
        exploration_constant=0.1,
        budget_ratio=1.0,
    )
    late = uct_score(
        child=child,
        parent_visits=10,
        reference_values=(2.0, 2.0),
        exploration_constant=0.1,
        budget_ratio=0.1,
    )
    assert early > late > 0.5
    assert remaining_budget_ratio(100, 75) == 0.25
    assert remaining_budget_ratio(None, 1000) == 1.0


def test_seeded_uct_tie_breaking_is_reproducible() -> None:
    first = select_expansion_node(
        make_tree((1.0, 1.0)),
        rng=random.Random(9),
        total_budget=100,
        used_budget=2,
        exploration_constant=0.5,
    )
    second = select_expansion_node(
        make_tree((1.0, 1.0)),
        rng=random.Random(9),
        total_budget=100,
        used_budget=2,
        exploration_constant=0.5,
    )
    assert first.selected_node_id == second.selected_node_id


def test_adaptive_expansion_quality_uses_batch_credit_and_failed_attempts() -> None:
    tree = make_tree((1.0,))
    node = tree.get_node(0)
    add_child(tree, node.id, 2.0, batch_id=7)
    add_child(tree, node.id, 3.0, batch_id=7, record_visit=False)
    tree.record_batch_visit(node.id)  # A failed batch has no child and zero reward.
    references = fitness_reference_values(tree)

    assert node.expansion_count == 2
    assert expansion_batch_rewards(tree, node, references) == (1.0,)
    assert expansion_quality(tree, node, references) == pytest.approx(1 / 3)


def test_recursive_selection_compares_new_branch_with_existing_children() -> None:
    tree = make_tree((1.0,))
    root_child = tree.get_node(0)
    add_child(tree, root_child.id, 2.0)
    best, _ = add_child(tree, root_child.id, 3.0)
    descended = select_expansion_node(
        tree,
        rng=random.Random(0),
        total_budget=100,
        used_budget=3,
        exploration_constant=0.0,
    )
    assert descended.selected_node_id == best.id
    assert descended.path == (-1, root_child.id, best.id)
    assert descended.steps[-1].option == "expand"

    second_tree = make_tree((3.0,))
    second_root = second_tree.get_node(0)
    add_child(second_tree, second_root.id, 1.0)
    reopened = select_expansion_node(
        second_tree,
        rng=random.Random(0),
        total_budget=100,
        used_budget=3,
        exploration_constant=0.0,
    )
    assert reopened.selected_node_id == second_root.id
    assert reopened.path == (-1, second_root.id)
    assert reopened.steps[-1].option == "expand"


def test_direct_code_parser_requires_explicit_idea() -> None:
    template = TextFunctionProgramConverter.text_to_program(TEMPLATE)
    assert template is not None
    code_only = "```python\ndef choose(value: int) -> int:\n    return value + 1\n```"
    idea_inside_code = (
        "```python\n"
        "def choose(value: int) -> int:\n"
        '    \"\"\"Idea: text inside generated code.\"\"\"\n'
        "    return value + 1\n"
        "```"
    )

    assert parse_program_response(code_only, template, "choose") is None
    assert parse_program_response(idea_inside_code, template, "choose") is None


def test_internal_node_context_uses_top_four_then_recent_children() -> None:
    tree = make_tree((0.0,))
    parent_id = tree.root.child_ids[0]
    for order in range(10):
        add_child(tree, parent_id, float(order), creation_order=order)
    selected = select_direct_children(tree, parent_id, limit=8, top_count=4)
    assert selected == tuple(range(3, 11))
    history = node_history(tree, parent_id)
    assert "[Previously Tested From This Program]" in history.text
    assert "subtree-best fitness" in history.text
    assert len(history.direct_child_edge_ids) == 8


def test_reference_is_other_root_branch_and_same_code_is_excluded() -> None:
    tree = make_tree((1.0, 2.0, 3.0))
    main_id = tree.root.child_ids[0]
    main = tree.get_node(main_id)
    duplicate_branch = tree.get_node(tree.root.child_ids[1])
    duplicate_branch.code = main.code
    duplicate_branch.code_hash = main.code_hash
    candidates = reference_candidates(tree, main_id)
    assert [branch for branch, _ in candidates] == [tree.root.child_ids[2]]
    before = [(node.id, node.visit_count, node.parent_id) for node in tree.nodes()]
    branch_id, reference = sample_reference(
        tree, main_id, temperature=0.2, rng=random.Random(0)
    )
    after = [(node.id, node.visit_count, node.parent_id) for node in tree.nodes()]
    assert branch_id != tree.root_branch_id(main_id)
    assert tree.root_branch_id(reference.id) == branch_id
    assert before == after


def test_batch_visit_counts_path_once_even_when_code_parse_fails() -> None:
    method = TraceAADV8(
        llm=BrokenCodeLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        offspring_per_iteration=2,
        context_token_limit=24576,
        max_stalled_iterations=1,
        random_seed=0,
    )
    result = method.run()
    root_child = method._tree.get_node(method._tree.root.child_ids[0])
    assert result.n_samples == 1
    assert result.n_batches == 1
    assert method._tree.root.visit_count == 2
    assert root_child.visit_count == 2
    assert root_child.expansion_count == 1
    assert not root_child.child_ids


def test_evaluation_failure_consumes_budget_but_adds_no_node() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=FailingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        offspring_per_iteration=2,
        context_token_limit=24576,
        max_stalled_iterations=1,
        random_seed=0,
    )
    result = method.run()
    assert result.n_samples == 3
    assert result.n_total_nodes == 1
    assert result.n_edges == 0
    assert result.n_batches == 1


def test_non_numeric_and_nan_evaluator_results_do_not_abort_search() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=NonNumericEvaluation(),
        max_sample_nums=4,
        n_init=1,
        offspring_per_iteration=2,
        context_token_limit=24576,
        random_seed=0,
    )
    result = method.run()
    assert result.n_samples == 4
    assert result.n_total_nodes == 2
    assert result.best_node is not None
    assert result.best_node.fitness == 4.0


def test_direct_code_context_shrinks_branch_history_to_fit() -> None:
    llm = LengthCountingLLM()
    method = TraceAADV8(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=10,
        n_init=0,
        offspring_per_iteration=1,
        context_token_limit=100_000,
        operators=(TraceIdeateOp,),
        random_seed=0,
    )
    base = method._tree.add_initial(
        code=TEMPLATE,
        idea="base",
        fitness=1.0,
        maximize=True,
        creation_order=1,
    )
    for order in range(8):
        add_child(method._tree, base.id, float(order + 2), creation_order=order + 2)
    operator = TraceIdeateOp()
    full = method._build_code_context(
        base_node=base,
        operator=operator,
        candidate_index=0,
        candidate_count=1,
        reference_branch_id=None,
        reference_node=None,
    )
    assert full is not None
    method._context_token_limit = full.prompt_tokens - 1
    narrower = method._build_code_context(
        base_node=base,
        operator=operator,
        candidate_index=0,
        candidate_count=1,
        reference_branch_id=None,
        reference_node=None,
    )
    assert narrower is not None
    assert len(narrower.direct_child_edge_ids) < len(full.direct_child_edge_ids)
    assert "[Improvement Direction]" in narrower.prompt
    assert "[Action Contract]" not in narrower.prompt


def test_dual_operator_full_run_uses_cross_branch_reference_only() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=0,
        offspring_per_iteration=2,
        context_token_limit=24576,
        operators=(TraceIdeateOp, TraceSynthesizeOp),
        random_seed=0,
    )
    for order in range(2):
        method._tree.add_initial(
            code=f"def choose(value):\n    return value + {order}\n",
            idea=f"root {order}",
            fitness=float(order - 2),
            maximize=True,
            creation_order=order + 1,
        )
    method._tot_sample_nums = 2
    method._best_node = method._tree.subtree_best(method._tree.root.child_ids[1])
    method._best_node_sample_order = 2
    method._operators = (TraceSynthesizeOp(),)
    method._run_iteration(0)

    assert len(method._tree.edges()) == 2
    assert sum(edge.new_global_best for edge in method._tree.edges()) == 1
    winner = next(edge for edge in method._tree.edges() if edge.new_global_best)
    assert method._tree.get_node(winner.child_id).fitness == 2.0
    for edge in method._tree.edges():
        assert edge.operator == OperatorName.SYNTHESIZE
        assert edge.reference_node_id is not None
        assert edge.reference_root_branch_id != method._tree.root_branch_id(
            edge.parent_id
        )


def test_minimization_direction_uses_directed_subtree_credit() -> None:
    tree = SearchTree()
    high = tree.add_initial(
        code="def choose(value):\n    return value + 10\n",
        idea="high",
        fitness=10.0,
        maximize=False,
        creation_order=1,
    )
    low = tree.add_initial(
        code="def choose(value):\n    return value + 8\n",
        idea="low",
        fitness=8.0,
        maximize=False,
        creation_order=2,
    )
    child, _, _ = tree.add_child(
        parent_id=high.id,
        code="def choose(value):\n    return value + 7\n",
        idea="lower",
        fitness=7.0,
        maximize=False,
        creation_order=3,
        operator=OperatorName.IDEATE,
        reference_node_id=None,
        reference_root_branch_id=None,
        global_best_directed_fitness=low.directed_fitness,
        new_global_best=True,
        global_best_update_reason="strict_fitness",
        iteration=0,
        batch_id=1,
        sibling_seq=0,
        sample_order=3,
    )
    assert tree.root.subtree_best_node_id == child.id
    assert tree.root.subtree_value == -7.0


def test_small_scripted_run_preserves_histories_and_has_no_population(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    method = TraceAADV8(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        artifacts=RunArtifacts(run_dir=tmp_path),
        max_sample_nums=7,
        n_init=3,
        offspring_per_iteration=2,
        context_token_limit=24576,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1,
        random_seed=4,
    )
    result = method.run()
    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))

    assert result.n_samples == 7
    assert result.n_root_children == 3
    assert result.n_total_nodes == 7
    assert result.n_edges == 4
    assert payload["protocol_id"] == "traceaad-v8.2-adaptive-expand"
    assert payload["version"] == 3
    assert "memory" not in payload
    assert "population" not in payload
    assert not hasattr(method, "_memory")
    code_prompts = [prompt for prompt in llm.prompts if "[Improvement Direction]" in prompt]
    assert len(code_prompts) == 4
    assert llm.calls == result.n_samples
    assert all("[How This Program Was Reached]" in prompt for prompt in code_prompts)
    assert all("Current fitness:" in prompt for prompt in code_prompts)
    assert all("[Action Contract]" not in prompt for prompt in llm.prompts)
    edges = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "edges.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    decisions = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(edge["implemented_idea"] for edge in edges)
    assert all("action" not in edge for edge in edges)
    assert not any(item["event"] == "actions_generated" for item in decisions)
    selected = [item for item in decisions if item["event"] == "node_selected"]
    assert selected
    assert all(item["expansion_policy"] == "adaptive_new_child_uct" for item in selected)
    assert all(item["selection_steps"][-1]["option"] == "expand" for item in selected)


def test_checkpoint_round_trip_preserves_tree_rng_and_next_selection(
    tmp_path: Path,
) -> None:
    first = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=2,
        offspring_per_iteration=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path,
        random_seed=11,
    )
    first.run()
    payload = dump_state(first)
    second = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=2,
        offspring_per_iteration=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path,
        random_seed=11,
    )
    load_state(second, payload)

    first_selection = select_expansion_node(
        first._tree,
        rng=first._rng,
        total_budget=first._max_sample_nums,
        used_budget=first._tot_sample_nums,
        exploration_constant=first._exploration_constant,
        expansion_prior_weight=first._expansion_prior_weight,
    )
    second_selection = select_expansion_node(
        second._tree,
        rng=second._rng,
        total_budget=second._max_sample_nums,
        used_budget=second._tot_sample_nums,
        exploration_constant=second._exploration_constant,
        expansion_prior_weight=second._expansion_prior_weight,
    )
    assert dump_state(first)["tree"] == dump_state(second)["tree"]
    assert first_selection == second_selection


def test_checkpoint_rejects_corrupt_subtree_credit() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=1,
        n_init=1,
        context_token_limit=24576,
    )
    method.run()
    payload = dump_state(method)
    payload["tree"]["nodes"][0]["subtree_value"] = 999.0
    target = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=1,
        n_init=1,
        context_token_limit=24576,
    )
    with pytest.raises(ValueError, match="subtree backup"):
        load_state(target, payload)


def test_checkpoint_rejects_misaligned_expansion_batch() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        offspring_per_iteration=1,
        context_token_limit=24576,
    )
    method.run()
    payload = dump_state(method)
    payload["tree"]["nodes"][1]["batch_id"] = 999
    target = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        offspring_per_iteration=1,
        context_token_limit=24576,
    )
    with pytest.raises(ValueError, match="expansion batch"):
        load_state(target, payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "checkpoint version"),
        ("protocol_id", "traceaad-v8.1-direct", "protocol"),
    ],
)
def test_checkpoint_rejects_pre_adaptive_expansion_protocol(
    field: str,
    value: object,
    message: str,
) -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=1,
        n_init=1,
        context_token_limit=24576,
    )
    method.run()
    payload = dump_state(method)
    payload[field] = value

    target = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=1,
        n_init=1,
        context_token_limit=24576,
    )
    with pytest.raises(ValueError, match=message):
        load_state(target, payload)


def test_checkpoint_rejects_changed_direct_evaluator_configuration() -> None:
    first = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=ConfigurableEvaluation(scale=1),
        max_sample_nums=1,
        n_init=1,
        context_token_limit=24576,
    )
    first.run()
    payload = dump_state(first)
    changed = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=ConfigurableEvaluation(scale=2),
        max_sample_nums=1,
        n_init=1,
        context_token_limit=24576,
    )
    with pytest.raises(ValueError, match="runtime identity"):
        load_state(changed, payload)


def test_checkpoint_resume_continues_to_the_same_budget(tmp_path: Path) -> None:
    first = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=2,
        offspring_per_iteration=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path,
        random_seed=5,
    )
    first._initialize()
    first._initialization_complete = True
    first._run_iteration(0)
    first._next_attempt_id = 1
    save_checkpoint(first)
    assert first._tot_sample_nums == 3

    resumed = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=2,
        offspring_per_iteration=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path,
        resume_from=tmp_path / "latest.json",
        random_seed=5,
    )
    result = resumed.run()
    assert result.n_samples == 5
    assert result.n_total_nodes == 5
    assert resumed._next_attempt_id >= 2
