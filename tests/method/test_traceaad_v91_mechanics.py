from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_1 import PROTOCOL_ID, TraceAADV91
from llm4ad.method.traceaad_v9_1.checkpoint import (
    CHECKPOINT_VERSION,
    load_checkpoint,
    load_state,
    save_checkpoint,
)
from llm4ad.method.traceaad_v9_1.schema import OperatorName
from llm4ad.method.traceaad_v9_1.tree import SearchTree
from llm4ad.method.traceaad_v9_1.value import (
    select_trajectory,
    trajectory_quality_pool,
    wilson_upper_bound,
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
        self.prompts.append(prompt)
        return (
            f"Idea: candidate {self.calls}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )


class ParseFailAfterInitLLM(ScriptedLLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if self.calls >= 1:
            self.calls += 1
            self.prompts.append(prompt)
            return "not a program"
        return super().draw_sample(prompt, *args, **kwargs)


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


def make_tree(*fitnesses: float) -> SearchTree:
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


def add_child(tree: SearchTree, parent_id: int, fitness: float, batch_id: int = 1):
    node_id = tree._next_node_id
    return tree.add_child(
        parent_id=parent_id,
        code=f"def choose(value):\n    return value + {node_id + 10}\n",
        idea=f"child {node_id}",
        fitness=fitness,
        maximize=True,
        creation_order=node_id + 10,
        operator=OperatorName.IDEATE,
        reference_node_id=None,
        global_best_directed_fitness=None,
        new_global_best=False,
        global_best_update_reason=None,
        iteration=0,
        batch_id=batch_id,
        sibling_seq=0,
        sample_order=node_id + 1,
    )


def test_quality_gate_uses_raw_endpoint_fitness_without_global_normalization() -> None:
    tree = make_tree(8.0, 10.0, 9.0, -1_000_000.0)
    assert [node.fitness for node in trajectory_quality_pool(tree, pool_size=3)] == [
        10.0,
        9.0,
        8.0,
    ]
    selection = select_trajectory(tree, pool_size=3, confidence_z=1.0)
    assert selection.selected_node_id == 1
    assert selection.mode == "basic_validation"


def test_wilson_allocation_uses_only_each_trajectorys_own_evidence() -> None:
    tree = make_tree(10.0, 9.0)
    first, second = tree.nodes()
    for batch in range(1, 11):
        tree.record_verification(
            first.id,
            valid_candidate_count=1,
            route_advanced=batch <= 2,
            global_advanced=False,
            batch_id=batch,
            recent_window=4,
        )
    for batch in range(11, 13):
        tree.record_verification(
            second.id,
            valid_candidate_count=1,
            route_advanced=batch == 11,
            global_advanced=False,
            batch_id=batch,
            recent_window=4,
        )
    assert wilson_upper_bound(1, 2) > wilson_upper_bound(2, 10)
    before = select_trajectory(tree, pool_size=2, confidence_z=1.0)
    assert before.selected_node_id == second.id
    tree.add_initial(
        code="def choose(value):\n    return value - 999\n",
        idea="extreme failure",
        fitness=-1_000_000.0,
        maximize=True,
        creation_order=99,
    )
    after = select_trajectory(tree, pool_size=2, confidence_z=1.0)
    assert after.selected_node_id == before.selected_node_id
    assert after.wilson_upper == before.wilson_upper
    assert first.verification_count == 10
    assert first.route_advance_count == 2


def test_route_advance_compares_against_complete_trajectory_history() -> None:
    tree = make_tree(10.0)
    root_id = tree.root.child_ids[0]
    child, first_edge = add_child(tree, root_id, 5.0, batch_id=1)
    grandchild, second_edge = add_child(tree, child.id, 7.0, batch_id=2)
    assert not first_edge.advances_parent_trajectory
    assert not second_edge.advances_parent_trajectory
    assert child.trajectory_best_value == 10.0
    assert grandchild.trajectory_best_value == 10.0
    assert grandchild.trajectory_best_node_id == root_id


def test_verification_credit_never_backpropagates_to_ancestors() -> None:
    tree = make_tree(10.0)
    root = tree.get_node(tree.root.child_ids[0])
    child, _ = add_child(tree, root.id, 11.0)
    tree.record_verification(
        child.id,
        valid_candidate_count=2,
        route_advanced=True,
        global_advanced=True,
        batch_id=2,
        recent_window=4,
    )
    assert child.verification_count == 1
    assert child.route_advance_count == 1
    assert root.verification_count == 0
    assert root.route_advance_count == 0


def test_iteration_verifies_one_trajectory_with_two_distinct_ideas() -> None:
    method = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        context_token_limit=24576,
        random_seed=0,
    )
    method._initialize()
    root = method._tree.get_node(method._tree.root.child_ids[0])
    method._run_iteration(0)
    children = [method._tree.get_node(node_id) for node_id in root.child_ids]
    assert len(children) == 2
    assert len({child.operator for child in children}) == 2
    assert root.verification_count == 1
    assert root.valid_candidate_count == 2
    assert len(method._tree.root.child_ids) == 1


def test_failed_verification_is_evidence_on_selected_trajectory() -> None:
    method = TraceAADV91(
        llm=ParseFailAfterInitLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        context_token_limit=24576,
        random_seed=0,
    )
    method._initialize()
    root = method._tree.get_node(method._tree.root.child_ids[0])
    method._run_iteration(0)
    assert root.verification_count == 1
    assert root.valid_candidate_count == 0
    assert root.recent_advances == [False]
    assert not root.child_ids


def test_initialization_after_first_root_is_derived_from_evaluated_histories() -> None:
    llm = ScriptedLLM()
    method = TraceAADV91(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=3,
        context_token_limit=24576,
        random_seed=0,
    )
    method._initialize()
    roots = [method._tree.get_node(node_id) for node_id in method._tree.root.child_ids]
    assert "[Existing Evaluated Histories]" not in llm.prompts[0]
    assert "[Existing Evaluated Histories]" in llm.prompts[1]
    assert roots[0].bootstrap_reference_node_ids == []
    assert roots[1].bootstrap_reference_node_ids == [roots[0].id]
    assert set(roots[2].bootstrap_reference_node_ids) == {roots[0].id, roots[1].id}


def test_end_to_end_run_exhausts_budget_through_trajectory_events() -> None:
    method = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=1,
        context_token_limit=24576,
        random_seed=0,
    )
    result = method.run()
    assert result.n_samples == 5
    assert result.n_batches == 2
    assert result.n_root_children == 1
    assert sum(node.verification_count for node in method._tree.nodes()) == 2
    assert result.best_node is method._tree.best_node()


def test_checkpoint_round_trip_persists_trajectory_evidence(tmp_path: Path) -> None:
    method = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=2,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "checkpoints",
        random_seed=0,
    )
    method._initialize()
    method._run_iteration(0)
    method._initialization_complete = True
    checkpoint = save_checkpoint(method)
    assert checkpoint is not None
    payload = json.loads(checkpoint.read_text())
    assert payload["protocol_id"] == PROTOCOL_ID
    assert payload["version"] == CHECKPOINT_VERSION
    assert "q_min" not in payload["tree"]

    restored = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=2,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "restored",
        random_seed=0,
        resume_from=checkpoint,
    )
    selected = next(node for node in restored._tree.nodes() if node.verification_count)
    assert selected.valid_candidate_count == 2
    assert selected.last_verification_batch_id == 1
    load_checkpoint(restored, checkpoint)

    payload["version"] = CHECKPOINT_VERSION - 1
    with pytest.raises(ValueError, match="unsupported"):
        load_state(restored, payload)
