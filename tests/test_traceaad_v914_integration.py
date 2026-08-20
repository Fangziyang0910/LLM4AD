from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_14 import (
    CHECKPOINT_VERSION,
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    PROTOCOL_ID,
    REFINE_PROBABILITY,
    RunArtifacts,
    TraceAADV914,
)
from llm4ad.method.traceaad_v9_14.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v9_14.history import parent_path, render_path
from llm4ad.method.traceaad_v9_14.schema import Algorithm, Intent, Outcome, Pending
from llm4ad.method.traceaad_v9_14.selection import (
    score_algorithms_in_branch,
    score_branches,
    select,
)
from llm4ad.method.traceaad_v9_14.traceaad import draw_intent
from llm4ad.method.traceaad_v9_14.tree import Tree, is_better

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return (
            f"Idea: candidate {self.calls}\n"
            "Code:\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class FlakyLLM(LLM):
    """LLM that alternates between valid code, bad format, and code returning non-finite."""
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        if self.calls == 3:
            return "Just some chit chat without code block."
        if self.calls == 5:
            return "Idea: nan\nCode:\n```python\ndef choose(value: int) -> int:\n    return float('nan')\n```"
        return (
            f"Idea: candidate {self.calls}\n"
            "Code:\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class IncrementalEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose function.",
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        # Result depends on the value added
        if callable_func is None:
            return None
        res = callable_func(10)
        return float(res)


def test_tree_pure_node_structure_and_provenance() -> None:
    tree = Tree(maximize=True)
    assert tree.virtual_root_id == 0
    assert len(tree.valid_algorithms()) == 0

    # Add branch root
    root1 = tree.add_branch_root(code="def choose(v): return v", fitness=10.0)
    assert root1.id == 1
    assert root1.parent_id == 0
    assert root1.q == 10.0
    assert tree.branch_ids == [1]

    # Add child
    child1 = tree.add_child(
        parent_id=root1.id,
        code="def choose(v): return v + 1",
        fitness=11.0,
        intent="refine",
        idea="add 1",
        diff="--- parent.py\n+++ candidate.py\n@@ -1 +1 @@\n-def choose(v): return v\n+def choose(v): return v + 1",
        added=1,
        removed=1,
        dq=1.0,
        outcome=Outcome.IMPROVE,
        stage="search",
        iteration=0,
    )
    assert child1.id == 2
    assert child1.parent_id == 1
    assert child1.q == 11.0
    assert child1.idea == "add 1"
    assert child1.outcome == Outcome.IMPROVE

    # Ancestors
    assert tree.ancestor_ids(child1.id) == (0, 1, 2)
    assert tree.branch_id_of(child1.id) == 1

    # Serialization
    payload = tree.to_dict()
    restored = Tree.from_dict(payload)
    assert len(restored.valid_algorithms()) == 2
    assert restored.get_algorithm(2).idea == "add 1"
    assert restored.get_algorithm(2).outcome == Outcome.IMPROVE


def test_history_extraction_and_rendering() -> None:
    tree = Tree(maximize=True)
    root = tree.add_branch_root(code="def choose(v): return v", fitness=10.0)

    # Root has no history
    assert parent_path(tree, root.id) == ()
    assert "No history events" in render_path(tree, ())

    # Add two generations
    c1 = tree.add_child(
        parent_id=root.id,
        code="def choose(v): return v + 1",
        fitness=11.0,
        idea="step 1",
        diff="+1 line",
        added=1,
        removed=0,
        dq=1.0,
        outcome=Outcome.IMPROVE,
    )
    c2 = tree.add_child(
        parent_id=c1.id,
        code="def choose(v): return v + 2",
        fitness=12.0,
        idea="step 2",
        diff="+1 line",
        added=1,
        removed=0,
        dq=1.0,
        outcome=Outcome.IMPROVE,
    )

    path_ids = parent_path(tree, c2.id)
    assert path_ids == (c1.id, c2.id)
    rendered = render_path(tree, path_ids)
    assert "Idea: step 1" in rendered
    assert "Idea: step 2" in rendered
    assert "Fitness: 10 -> 11" in rendered
    assert "Fitness: 11 -> 12" in rendered


def test_two_level_selection() -> None:
    tree = Tree(maximize=True)
    r1 = tree.add_branch_root(code="def choose(v): return v + 1", fitness=11.0)
    r2 = tree.add_branch_root(code="def choose(v): return v + 2", fitness=12.0)
    r1.count = 2
    r2.count = 0

    # Branch score
    branches = score_branches(tree, s=1.0)
    assert len(branches) == 2
    # Choice
    choice = select(tree, s=1.0)
    # r2 has higher quality (12.0) and fewer counts (0), should be chosen
    assert choice.branch_id == r2.id
    assert choice.algorithm_id == r2.id


def test_draw_intent_determinism() -> None:
    i0 = draw_intent(seed=42, iteration=0)
    i1 = draw_intent(seed=42, iteration=0)
    assert i0 == i1
    # Check that draw_intent produces both Refine and Explore over many iterations
    intents = {draw_intent(seed=42, iteration=i) for i in range(100)}
    assert Intent.REFINE in intents
    assert Intent.EXPLORE in intents


def test_v914_smoke_search_and_checkpoint_resume(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    evaluation = IncrementalEvaluation()
    run_dir = tmp_path / "run_smoke"
    method = TraceAADV914(
        llm=llm,
        evaluation=evaluation,
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=20,
        n_roots=4,
        context_limit=32768,
        checkpoint_dir=run_dir / "checkpoints",
        seed=1,
    )
    method.run()

    # Verify summary
    summary_path = run_dir / "logs" / "summary.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 20
    assert summary["n_branches"] == 4
    assert summary["n_algorithms"] == 20
    assert summary["initialization_complete"] is True

    # Verify evaluations.csv
    eval_csv = (run_dir / "evaluations.csv").read_text(encoding="utf-8")
    lines = eval_csv.strip().splitlines()
    assert len(lines) == 21  # header + 20 evals

    # Verify best program
    best_prog = (run_dir / "best_program.py").read_text(encoding="utf-8")
    assert "Best Program Discovered by TraceAAD V9.14" in best_prog

    # Resume test with identical configuration
    resumed = TraceAADV914(
        llm=ScriptedLLM(start=100),
        evaluation=evaluation,
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=20,
        n_roots=4,
        context_limit=32768,
        checkpoint_dir=run_dir / "checkpoints",
        resume_from=run_dir / "checkpoints" / "latest.json",
        seed=1,
    )
    assert resumed._n_eval == 20
    assert len(resumed._tree.valid_algorithms()) == 20
    resumed.run()
    assert resumed._n_eval == 20


def test_v914_invalid_responses_do_not_pollute_tree(tmp_path: Path) -> None:
    llm = FlakyLLM()
    evaluation = IncrementalEvaluation()
    run_dir = tmp_path / "run_flaky"
    method = TraceAADV914(
        llm=llm,
        evaluation=evaluation,
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=12,
        n_roots=2,
        context_limit=32768,
        checkpoint_dir=run_dir / "checkpoints",
        seed=0,
    )
    method.run()

    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["evaluator_call_count"] == 12
    # Notice: invalid parse attempts do not call evaluator, but non-finite does
    # Tree only has valid algorithm nodes
    for algo in method._tree.valid_algorithms():
        assert algo.code is not None
        assert algo.fitness is not None
        assert algo.q is not None
