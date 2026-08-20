"""Core mechanism tests for TraceAAD V9.14."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_14 import RunArtifacts, TraceAADV914
from llm4ad.method.traceaad_v9_14.checkpoint import save_checkpoint
from llm4ad.method.traceaad_v9_14.history import parent_path, render_path
from llm4ad.method.traceaad_v9_14.prompt import parse_program_response
from llm4ad.method.traceaad_v9_14.schema import Intent, Pending
from llm4ad.method.traceaad_v9_14.selection import select, selection_score
from llm4ad.method.traceaad_v9_14.tree import Tree

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, *, start: int = 0, invalid_at: set[int] | None = None) -> None:
        super().__init__()
        self.calls = start
        self.invalid_at = invalid_at or set()

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        if self.calls in self.invalid_at:
            return "unusable response"
        return (
            f"Idea: candidate {self.calls}\n"
            "Code:\n```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )


class IncrementalEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs) -> float | None:
        return None if callable_func is None else float(callable_func(10))


def test_tree_derives_quality_best_history_and_round_trips() -> None:
    tree = Tree(maximize=False)
    root = tree.add_algorithm(code="def f(): return 10", fitness=10.0)
    child = tree.add_algorithm(
        code="def f(): return 8",
        fitness=8.0,
        parent_id=root.id,
        idea="reduce cost",
    )
    child.count = 3

    assert tree.quality(root) == -10.0
    assert tree.best() is child
    assert tree.ancestor_ids(child.id) == (0, root.id, child.id)
    assert parent_path(tree, child.id) == (child.id,)
    history = render_path(tree, (child.id,))
    assert "Idea: reduce cost" in history
    assert "Result: improve (Fitness: 10 -> 8)" in history

    restored = Tree.from_dict(tree.to_dict())
    assert restored.to_dict() == tree.to_dict()
    assert restored.best().id == child.id


def test_selection_is_driven_only_by_quality_and_parent_count() -> None:
    tree = Tree(maximize=True)
    first = tree.add_algorithm(code="a", fitness=10.0)
    second = tree.add_algorithm(code="b", fitness=10.0)

    assert select(tree) is first
    first.count = 3
    assert selection_score(tree, first) == pytest.approx(10.5)
    assert selection_score(tree, second) == pytest.approx(11.0)
    assert select(tree) is second


def test_prompt_response_parser_is_tolerant() -> None:
    fenced = parse_program_response(
        "Idea: focused step\nCode:\n```python\ndef choose(v): return v + 1\n```"
    )
    assert fenced.declared_idea == "focused step"
    assert fenced.code == "def choose(v): return v + 1"

    marked = parse_program_response(
        "Idea: direct code\nCode:\ndef choose(v): return v"
    )
    assert marked.code == "def choose(v): return v"
    assert parse_program_response("plain response").code == "plain response"


def test_search_uses_evaluator_count_as_budget_and_writes_minimal_outputs(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    method = TraceAADV914(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=6,
        n_roots=2,
        checkpoint_dir=run_dir / "checkpoints",
    )
    method.run()

    assert method._n_eval == 6
    assert len(method._tree.root_algorithms()) == 2
    assert len(method._tree.valid_algorithms()) == 6
    assert method.best.fitness == 16.0

    summary = json.loads((run_dir / "logs/summary.json").read_text())
    assert summary == {
        "status": "finished",
        "stop_reason": "evaluator_budget_exhausted",
        "best_algorithm_id": 6,
        "best_score": 16.0,
        "evaluator_call_count": 6,
        "n_algorithms": 6,
        "n_roots": 2,
        "has_pending": False,
    }
    assert (run_dir / "best_program.py").is_file()
    assert not (run_dir / "logs/errors.jsonl").exists()


def test_failed_evaluation_consumes_budget_and_parent_count(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed"
    method = TraceAADV914(
        llm=ScriptedLLM(invalid_at={2}),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=2,
        n_roots=1,
    )
    method.run()

    root = method._tree.root_algorithms()[0]
    assert method._n_eval == 2
    assert root.count == 1
    assert len(method._tree.valid_algorithms()) == 1
    with (run_dir / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "exec_error"


def test_checkpoint_contains_only_irreplaceable_runtime_state(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    method = TraceAADV914(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=3,
        n_roots=1,
        checkpoint_dir=checkpoint_dir,
    )
    root = method._tree.add_algorithm(code=TEMPLATE, fitness=10.0)
    method._n_eval = 1
    method._pending = Pending(
        parent_id=root.id,
        intent=Intent.REFINE.value,
        response=(
            "Idea: recovered step\nCode:\n```python\n"
            "def choose(value: int) -> int:\n    return value + 9\n```"
        ),
    )
    path = save_checkpoint(method)
    state = json.loads(path.read_text())
    assert set(state) == {"tree", "pending", "n_eval"}

    resumed = TraceAADV914(
        llm=ScriptedLLM(start=9),
        evaluation=IncrementalEvaluation(),
        budget=3,
        n_roots=1,
        resume_from=path,
    )
    resumed.run()

    assert resumed._n_eval == 3
    assert resumed._pending is None
    assert resumed._tree.get_algorithm(2).idea == "recovered step"
    assert sum(algorithm.count for algorithm in resumed._tree.valid_algorithms()) == 2


def test_model_transport_failure_is_immediately_visible(tmp_path: Path) -> None:
    class FailingLLM(LLM):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def draw_sample(self, prompt: str, *args, **kwargs) -> str:
            self.calls += 1
            raise ConnectionError("offline")

    llm = FailingLLM()
    method = TraceAADV914(
        llm=llm,
        evaluation=IncrementalEvaluation(),
        budget=2,
        n_roots=1,
    )
    with pytest.raises(ConnectionError, match="offline"):
        method.run()
    assert llm.calls == 1
    assert method._n_eval == 0
