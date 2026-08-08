from __future__ import annotations

import json
import random
from pathlib import Path

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_1 import TraceAADV91
from llm4ad.method.traceaad_v9_1.checkpoint import load_checkpoint, save_checkpoint
from llm4ad.method.traceaad_v9_1.schema import OperatorName
from llm4ad.method.traceaad_v9_1.tree import SearchTree
from llm4ad.method.traceaad_v9_1.value import (
    progressive_widening_allowed,
    select_expansion_node,
)

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
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


def add_child(tree: SearchTree, parent_id: int, fitness: float) -> None:
    node_id = tree._next_node_id
    tree.add_child(
        parent_id=parent_id,
        code=f"def choose(value):\n    return value + {node_id + 10}\n",
        idea=f"child {node_id}",
        fitness=fitness,
        maximize=True,
        creation_order=node_id + 10,
        operator=OperatorName.IDEATE,
        reference_node_id=None,
        reference_root_branch_id=None,
        global_best_directed_fitness=None,
        new_global_best=False,
        global_best_update_reason=None,
        iteration=0,
        batch_id=node_id + 1,
        sibling_seq=0,
        sample_order=node_id + 1,
    )
    tree.record_successful_visit(parent_id)


def test_v91_uses_mcts_progressive_widening_at_root_and_nodes() -> None:
    tree = make_tree(1.0, 2.0, 3.0, 4.0)
    assert tree.root.visit_count == 4
    assert not progressive_widening_allowed(tree.root, 0.5)
    tree.root.visit_count = 25
    assert progressive_widening_allowed(tree.root, 0.5)
    selection = select_expansion_node(
        tree,
        rng=random.Random(0),
        total_budget=100,
        used_budget=4,
        exploration_constant=0.1,
        alpha=0.5,
    )
    assert selection.selected_node_id == tree.root.id
    assert selection.steps[-1].option == "expand"

    node = tree.get_node(tree.root.child_ids[0])
    add_child(tree, node.id, 5.0)
    node.visit_count = 4
    assert progressive_widening_allowed(node, 0.5)


def test_v91_backup_is_continuation_value() -> None:
    tree = make_tree(10.0)
    parent_id = tree.root.child_ids[0]
    add_child(tree, parent_id, 2.0)
    parent = tree.get_node(parent_id)
    assert parent.subtree_value == 2.0
    assert parent.subtree_best_node_id == parent.child_ids[0]


def test_failed_generation_does_not_change_effective_visits() -> None:
    method = TraceAADV91(
        llm=ParseFailAfterInitLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        context_token_limit=24576,
        random_seed=0,
    )
    method._initialize()
    node = method._tree.get_node(method._tree.root.child_ids[0])
    before = (method._tree.root.visit_count, node.visit_count, node.expansion_count)
    method._run_iteration(0)
    after = (method._tree.root.visit_count, node.visit_count, node.expansion_count)
    assert after == before
    assert not node.child_ids


def test_v91_root_expansion_has_a_real_generation_path() -> None:
    method = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=4,
        context_token_limit=24576,
        random_seed=0,
    )
    method._initialize()
    before = len(method._tree.root.child_ids)
    method._tree.root.visit_count = 25
    method._run_iteration(0)
    assert len(method._tree.root.child_ids) == before + 1
    assert method._tot_sample_nums == 5


def test_v91_checkpoint_round_trip_persists_quality_bounds(tmp_path: Path) -> None:
    method = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=2,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "checkpoints",
        random_seed=0,
    )
    method._initialize()
    method._initialization_complete = True
    checkpoint = save_checkpoint(method)
    assert checkpoint is not None
    payload = json.loads(checkpoint.read_text())
    assert payload["protocol_id"] == "traceaad-v9.1-mcts-aligned"
    assert payload["tree"]["q_min"] == 1.0
    assert payload["tree"]["q_max"] == 2.0

    restored = TraceAADV91(
        llm=ScriptedLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=2,
        context_token_limit=24576,
        checkpoint_dir=tmp_path / "restored",
        random_seed=0,
        resume_from=checkpoint,
    )
    assert restored._tree.q_min == 1.0
    assert restored._tree.q_max == 2.0
    load_checkpoint(restored, checkpoint)
