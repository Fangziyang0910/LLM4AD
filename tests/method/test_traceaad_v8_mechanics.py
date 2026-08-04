"""Mechanism tests for the TraceAAD V8 complete-tree protocol."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_artifacts import TraceAADArtifacts
from llm4ad.method.traceaad_v8 import TraceAADV8
from llm4ad.method.traceaad_v8.checkpoint import dump_state, load_state, save_checkpoint
from llm4ad.method.traceaad_v8.context import node_history, select_direct_children
from llm4ad.method.traceaad_v8.operators import TraceIdeateOp, TraceSynthesizeOp
from llm4ad.method.traceaad_v8.schema import OperatorName
from llm4ad.method.traceaad_v8.tree import SearchTree
from llm4ad.method.traceaad_v8.value import (
    available_child_slots,
    normalize_value,
    reference_candidates,
    remaining_budget_ratio,
    sample_reference,
    select_expansion_node,
    uct_score,
    widening_capacity,
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
        if "[Action Contract]" in text:
            count = 1 if "Return exactly 1 numbered" in text else 2
            return "\n".join(
                f"{index}. Apply distinct local rule number {self.calls + index}."
                for index in range(1, count + 1)
            )
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


class BrokenActionLLM(ScriptedLLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in str(prompt):
            self.calls += 1
            self.prompts.append(str(prompt))
            return "no numbered action"
        return super().draw_sample(prompt, *args, **kwargs)


class LongActionLLM(ScriptedLLM):
    ACTION = "Apply a distinct local adjustment " + "x" * 800

    def count_tokens(self, prompt: str) -> int:
        return len(prompt)

    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in str(prompt):
            self.calls += 1
            self.prompts.append(str(prompt))
            return f"1. {self.ACTION}"
        return super().draw_sample(prompt, *args, **kwargs)


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
):
    index = tree._next_node_id
    child, edge, _ = tree.add_child(
        parent_id=parent_id,
        code=code or f"def choose(value):\n    return value + {index}\n",
        idea=f"child {index}",
        fitness=fitness,
        maximize=True,
        creation_order=index if creation_order is None else creation_order,
        operator=OperatorName.IDEATE,
        action=f"action {index}",
        reference_node_id=None,
        reference_root_branch_id=None,
        global_best_directed_fitness=None,
        new_global_best=False,
        global_best_update_reason=None,
        iteration=0,
        batch_id=1,
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
    assert normalize_value(2.0, 2.0, 2.0) == 0.5
    early = uct_score(
        child=child,
        parent_visits=10,
        lower=2.0,
        upper=2.0,
        exploration_constant=0.5,
        budget_ratio=1.0,
    )
    late = uct_score(
        child=child,
        parent_visits=10,
        lower=2.0,
        upper=2.0,
        exploration_constant=0.5,
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
        actions_per_iteration=2,
        widening_alpha=0.5,
    )
    second = select_expansion_node(
        make_tree((1.0, 1.0)),
        rng=random.Random(9),
        total_budget=100,
        used_budget=2,
        exploration_constant=0.5,
        actions_per_iteration=2,
        widening_alpha=0.5,
    )
    assert first.selected_node_id == second.selected_node_id


def test_progressive_widening_capacity_and_child_slots() -> None:
    tree = make_tree((1.0,))
    node = tree.get_node(0)
    assert widening_capacity(1) == 2
    assert widening_capacity(9) == 3
    assert widening_capacity(16) == 4
    assert available_child_slots(node) == 2
    add_child(tree, node.id, 2.0)
    add_child(tree, node.id, 3.0)
    assert available_child_slots(node) == 0
    node.visit_count = 9
    assert available_child_slots(node) == 1


def test_recursive_selection_descends_then_reopens_internal_node() -> None:
    tree = make_tree((1.0,))
    root_child = tree.get_node(0)
    first, _ = add_child(tree, root_child.id, 2.0)
    second, _ = add_child(tree, root_child.id, 3.0)
    root_child.visit_count = 4
    first.visit_count = 2
    second.visit_count = 2
    descended = select_expansion_node(
        tree,
        rng=random.Random(0),
        total_budget=100,
        used_budget=3,
        exploration_constant=0.5,
        actions_per_iteration=2,
        widening_alpha=0.5,
    )
    assert len(descended.path) == 3
    root_child.visit_count = 9
    reopened = select_expansion_node(
        tree,
        rng=random.Random(0),
        total_budget=100,
        used_budget=3,
        exploration_constant=0.5,
        actions_per_iteration=2,
        widening_alpha=0.5,
    )
    assert reopened.selected_node_id == root_child.id
    assert reopened.path == (-1, root_child.id)


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


def test_batch_visit_counts_path_once_even_when_action_parse_fails() -> None:
    method = TraceAADV8(
        llm=BrokenActionLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        actions_per_iteration=2,
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
    assert not root_child.child_ids


def test_evaluation_failure_consumes_budget_but_adds_no_node() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=FailingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        actions_per_iteration=2,
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
        actions_per_iteration=2,
        context_token_limit=24576,
        random_seed=0,
    )
    result = method.run()
    assert result.n_samples == 4
    assert result.n_total_nodes == 2
    assert result.best_node is not None
    assert result.best_node.fitness == 4.0


def test_code_overflow_shrinks_context_and_regenerates_matching_action(
    tmp_path: Path,
) -> None:
    llm = LongActionLLM()
    artifacts = TraceAADArtifacts(run_dir=tmp_path)
    method = TraceAADV8(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        profiler=artifacts,
        max_sample_nums=10,
        n_init=0,
        actions_per_iteration=1,
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
    for order in range(2):
        add_child(method._tree, base.id, float(order + 2), creation_order=order + 2)
    base.visit_count = 9
    method._tree.root.visit_count = 9
    method._best_node = method._tree.subtree_best(base.id)
    method._best_node_sample_order = 3
    method._tot_sample_nums = 3
    operator = TraceIdeateOp()
    full = method._build_action_context(
        base_node=base,
        operator=operator,
        action_count=1,
        reference_branch_id=None,
        reference_node=None,
    )
    assert full is not None
    full_code_tokens = method._prepare_code_requests(
        actions=[LongActionLLM.ACTION], base_node=base, context=full
    )[0][3]
    narrower = None
    for limit in range(full.prompt_tokens - 1, 0, -1):
        candidate = method._build_action_context(
            base_node=base,
            operator=operator,
            action_count=1,
            reference_branch_id=None,
            reference_node=None,
            prompt_token_limit=limit,
        )
        if candidate is not None and len(candidate.direct_child_edge_ids) < len(
            full.direct_child_edge_ids
        ):
            narrower = candidate
            break
    assert narrower is not None
    narrow_code_tokens = method._prepare_code_requests(
        actions=[LongActionLLM.ACTION], base_node=base, context=narrower
    )[0][3]
    method._context_token_limit = max(full.prompt_tokens, narrow_code_tokens)
    assert method._context_token_limit < full_code_tokens

    method._run_iteration(0)
    artifacts.finish()
    decisions = [
        json.loads(line)
        for line in (tmp_path / "artifacts" / "decisions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert method._tot_sample_nums > 3
    assert any(item["event"] == "context_shrunk" for item in decisions)
    generated = [item for item in decisions if item["event"] == "actions_generated"]
    assert len(generated) >= 2
    assert (
        generated[-1]["direct_child_edge_ids"] != generated[0]["direct_child_edge_ids"]
    )


def test_dual_operator_full_run_uses_cross_branch_reference_only() -> None:
    method = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=0,
        actions_per_iteration=2,
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
        action="lower score",
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
        profiler=TraceAADArtifacts(run_dir=tmp_path),
        max_sample_nums=7,
        n_init=3,
        actions_per_iteration=2,
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
    assert payload["protocol_id"] == "traceaad-v8"
    assert "memory" not in payload
    assert "population" not in payload
    assert not hasattr(method, "_memory")
    action_prompts = [prompt for prompt in llm.prompts if "[Action Contract]" in prompt]
    code_prompts = [
        prompt for prompt in llm.prompts if "[Requested Modification]" in prompt
    ]
    assert action_prompts and code_prompts
    assert all("[How This Program Was Reached]" in prompt for prompt in action_prompts)
    assert all("[How This Program Was Reached]" in prompt for prompt in code_prompts)
    assert all("Current fitness:" in prompt for prompt in action_prompts)
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
    assert any(item["event"] == "actions_generated" for item in decisions)


def test_checkpoint_round_trip_preserves_tree_rng_and_next_selection(
    tmp_path: Path,
) -> None:
    first = TraceAADV8(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=2,
        actions_per_iteration=1,
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
        actions_per_iteration=1,
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
        actions_per_iteration=first._actions_per_iteration,
        widening_alpha=first._widening_alpha,
    )
    second_selection = select_expansion_node(
        second._tree,
        rng=second._rng,
        total_budget=second._max_sample_nums,
        used_budget=second._tot_sample_nums,
        exploration_constant=second._exploration_constant,
        actions_per_iteration=second._actions_per_iteration,
        widening_alpha=second._widening_alpha,
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
        actions_per_iteration=1,
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
        actions_per_iteration=1,
        context_token_limit=24576,
        checkpoint_dir=tmp_path,
        resume_from=tmp_path / "latest.json",
        random_seed=5,
    )
    result = resumed.run()
    assert result.n_samples == 5
    assert result.n_total_nodes == 5
    assert resumed._next_attempt_id >= 2
