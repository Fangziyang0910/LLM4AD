"""Comprehensive benchmark test suite for TraceAAD V9.12.

Structure:
1. Data Structures & Invariants (Forest, Program, Anchor, is_better, serialization)
2. Operators & Selection Mechanics (Progress-conditioned Explore probability, budget suppression, failure evidence window)
3. Robustness & Fault Tolerance (Invalid code, NaN/inf handling, budget boundaries)
4. End-to-End Search & Exact Checkpoint Replay (Artifacts validation, state equality, runner verification)
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_12 import (
    EXPLORE_PROBABILITY_MAX,
    EXPLORE_PROBABILITY_MIN,
    MIN_EXPLORE_REMAINING_EVALS,
    PROGRESS_WINDOW,
    RunArtifacts,
    TraceAADV912,
)
from llm4ad.method.traceaad_v9_12.checkpoint import load_checkpoint, save_checkpoint
from llm4ad.method.traceaad_v9_12.forest import Forest, is_better
from llm4ad.method.traceaad_v9_12.schema import Anchor, Attempt, Intent, Outcome, Program
from llm4ad.method.traceaad_v9_12.selection import (
    explore_probability,
    refine_failure_evidence,
    select,
)

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


# ==============================================================================
# Test Doubles (Stubs / Mocks)
# ==============================================================================

class ScriptedLLM(LLM):
    """Deterministic LLM producing valid Python candidates."""

    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
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
    """LLM simulating syntax error and non-finite outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        if self.calls == 2:
            return "No fenced code block."
        if self.calls == 4:
            return "Idea: bad syntax\nCode:\n```python\ndef choose(value: int) -> int:\n    return value +\n```"
        if self.calls == 6:
            return "Idea: nan fitness\nCode:\n```python\ndef choose(value: int) -> int:\n    return float('nan')\n```"
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
    """Safe evaluation returning float fitness based on return value."""

    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose function.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs) -> float | None:
        if callable_func is None:
            return None
        return float(callable_func(10))


# ==============================================================================
# 1. Data Structures & Invariants
# ==============================================================================

def test_v912_is_better_total_order() -> None:
    """Validate strict total order: q > len(code) > id."""
    p1 = Program(id=1, code="def f(): pass", code_hash="h1", fitness=10.0, q=10.0, length=13, order=1)
    p2 = Program(id=2, code="def f(): pass", code_hash="h2", fitness=12.0, q=12.0, length=13, order=2)
    assert is_better(p2, p1) is True
    assert is_better(p1, p2) is False

    p_short = Program(id=3, code="def f(): return 1", code_hash="h3", fitness=10.0, q=10.0, length=17, order=3)
    p_long = Program(id=4, code="def f():\n    return 1", code_hash="h4", fitness=10.0, q=10.0, length=24, order=4)
    assert is_better(p_short, p_long) is True
    assert is_better(p_long, p_short) is False


def test_v912_forest_hierarchy_and_lineage() -> None:
    """Validate Forest root and child anchor relationships."""
    forest = Forest(maximize=True)
    p1 = forest.add_program(code="def choose(v): return v", fitness=10.0, order=1)
    root = forest.add_root(program_id=p1.id, order=1)
    assert root.id == 0
    assert root.root_id == 0
    assert forest.root_ids == [0]

    p2 = forest.add_program(code="def choose(v): return v + 1", fitness=11.0, order=2)
    child = forest.add_child(parent_id=root.id, program_id=p2.id, attempt_id=0, order=2)
    assert child.parent_id == root.id
    assert child.root_id == root.id
    assert forest.get_anchor(child.parent_id).program_id == p1.id


def test_v912_forest_serialization_loss_free() -> None:
    """Validate that Forest serialization preserves 100% of state."""
    forest = Forest(maximize=False)
    p1 = forest.add_program(code="def choose(v): return v", fitness=10.0, order=1)
    root = forest.add_root(program_id=p1.id, order=1)
    p2 = forest.add_program(code="def choose(v): return v - 1", fitness=9.0, order=2)
    child = forest.add_child(parent_id=root.id, program_id=p2.id, attempt_id=0, order=2)
    child.n = 3

    payload = forest.to_dict()
    restored = Forest.from_dict(payload)

    assert restored.maximize is False
    assert len(restored.programs()) == 2
    assert len(restored.anchors()) == 2
    restored_child = restored.get_anchor(child.id)
    assert restored_child.n == 3
    assert restored_child.root_id == root.id


# ==============================================================================
# 2. Operators & Selection Mechanics
# ==============================================================================

def test_v912_probability_is_bounded_and_progress_conditioned() -> None:
    """Validate progress-conditioned explore probability formula."""
    assert explore_probability(0.0) == EXPLORE_PROBABILITY_MIN
    assert explore_probability(1.0) == EXPLORE_PROBABILITY_MAX
    assert explore_probability(0.25) == 0.15


def test_v912_explore_is_suppressed_when_budget_unavailable() -> None:
    """Explore is gracefully downgraded to Refine if remaining evals < 2."""
    method = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=10,
        n_roots=1,
    )
    program = method._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    method._forest.add_root(program_id=program.id, order=1)
    method._s = 0.0
    method._seed = next(
        seed
        for seed in range(1000)
        if select(method._forest, 0.0, seed=seed, iteration=0).intent is Intent.EXPLORE
    )

    method._n_eval = method._budget - MIN_EXPLORE_REMAINING_EVALS
    allowed = method._next_decision()
    assert allowed.intent is Intent.EXPLORE
    assert allowed.operator_probability_applied
    assert not allowed.explore_suppressed_for_budget

    method._n_eval += 1
    suppressed = method._next_decision()
    assert suppressed.normal_choice.intent is Intent.EXPLORE
    assert suppressed.intent is Intent.REFINE
    assert not suppressed.operator_probability_applied
    assert suppressed.explore_suppressed_for_budget


def test_v912_progress_window_resets_at_explore_child() -> None:
    """Sliding window resets on an explore child and counts followup refine evidence."""
    method = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=10,
        n_roots=1,
    )
    forest = method._forest
    program = forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    root = forest.add_root(program_id=program.id, order=1)

    # Add a failed refine attempt
    failed_id = forest.next_attempt_id()
    forest.add_attempt(
        Attempt(
            id=failed_id,
            anchor_id=root.id,
            child_id=None,
            program_id=None,
            intent=Intent.REFINE.value,
            idea=None,
            diff=None,
            added=0,
            removed=0,
            parent_fitness=1.0,
            child_fitness=None,
            dq=None,
            outcome=Outcome.INVALID,
            kind="invalid",
            order=2,
            stage="search",
            iteration=0,
        )
    )
    # Add an improved refine child
    improved_program = forest.add_program(
        code=TEMPLATE.replace("return value", "return value + 1"),
        fitness=2.0,
        order=3,
    )
    improved_id = forest.next_attempt_id()
    improved = forest.add_child(
        parent_id=root.id,
        program_id=improved_program.id,
        attempt_id=improved_id,
        order=3,
    )
    forest.add_attempt(
        Attempt(
            id=improved_id,
            anchor_id=root.id,
            child_id=improved.id,
            program_id=improved_program.id,
            intent=Intent.REFINE.value,
            idea="improve",
            diff=None,
            added=0,
            removed=0,
            parent_fitness=1.0,
            child_fitness=2.0,
            dq=1.0,
            outcome=Outcome.IMPROVE,
            kind="new",
            order=3,
            stage="search",
            iteration=1,
        )
    )
    assert refine_failure_evidence(forest, improved.id) == (1 / 8, 2)

    # Add an explore child branching off improved
    explored_program = forest.add_program(
        code=TEMPLATE.replace("return value", "return value - 1"),
        fitness=1.5,
        order=4,
    )
    explored_id = forest.next_attempt_id()
    explored = forest.add_child(
        parent_id=improved.id,
        program_id=explored_program.id,
        attempt_id=explored_id,
        order=4,
    )
    forest.add_attempt(
        Attempt(
            id=explored_id,
            anchor_id=improved.id,
            child_id=explored.id,
            program_id=explored_program.id,
            intent=Intent.EXPLORE.value,
            idea="change direction",
            diff=None,
            added=0,
            removed=0,
            parent_fitness=2.0,
            child_fitness=1.5,
            dq=-0.5,
            outcome=Outcome.REGRESS,
            kind="new",
            order=4,
            stage="search",
            iteration=2,
        )
    )
    assert refine_failure_evidence(forest, explored.id) == (0.0, 0)
    assert PROGRESS_WINDOW == 8


# ==============================================================================
# 3. Robustness & Fault Tolerance
# ==============================================================================

def test_v912_flaky_responses_handled_gracefully(tmp_path: Path) -> None:
    """Invalid and NaN responses consume eval budget but do not pollute valid programs."""
    llm = FlakyLLM()
    run_dir = tmp_path / "run_flaky"
    method = TraceAADV912(
        llm=llm,
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=10,
        n_roots=2,
        checkpoint_dir=run_dir / "checkpoints",
        seed=1,
    )
    method.run()

    assert method._n_eval == 10
    for prog in method._forest.programs():
        assert math.isfinite(prog.fitness)
        assert math.isfinite(prog.q)


def test_v912_budget_exhaustion_during_initialization(tmp_path: Path) -> None:
    """When budget is smaller than n_roots, halts gracefully."""
    run_dir = tmp_path / "run_budget_under_roots"
    method = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=3,
        n_roots=8,
        checkpoint_dir=run_dir / "checkpoints",
    )
    method.run()

    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "initialization_failure"
    assert summary["evaluator_call_count"] == 3
    assert summary["n_roots"] == 3
    assert summary["initialization_complete"] is False


# ==============================================================================
# 4. End-to-End Search & Exact Checkpoint Replay
# ==============================================================================

def test_v912_smoke_runs_and_records_artifacts(tmp_path: Path) -> None:
    """Full search run produces valid summary and evaluations.csv."""
    run_dir = tmp_path / "run_smoke"
    method = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=20,
        checkpoint_dir=run_dir / "checkpoints",
        seed=3,
    )
    method.run()

    summary = json.loads((run_dir / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 20
    assert summary["n_roots"] == 8
    assert summary["n_followup"] >= 0
    assert summary["exploration_followup_anchor_id"] is None

    eval_csv = (run_dir / "evaluations.csv").read_text()
    assert len(eval_csv.splitlines()) == 21  # header + 20 rows


def test_v912_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    """State saved and loaded via checkpoint is completely preserved."""
    source = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=100,
        n_roots=1,
        checkpoint_dir=tmp_path / "source",
    )
    program = source._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    root = source._forest.add_root(program_id=program.id, order=1)
    source._initialization_complete = True
    source._s = 0.0
    source._iteration = 8
    source._n_eval = 20
    source._exploration_followup_anchor_id = root.id
    source._n_followup = 2
    save_checkpoint(source)

    restored = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=100,
        n_roots=1,
        checkpoint_dir=tmp_path / "restored",
    )
    load_checkpoint(restored, tmp_path / "source" / "latest.json")

    assert restored._iteration == 8
    assert restored._n_eval == 20
    assert restored._exploration_followup_anchor_id == root.id
    assert restored._n_followup == 2
    assert restored.best.id == program.id


def test_v912_runner_builds_protocol(tmp_path: Path) -> None:
    """Runner correctly instantiates TraceAADV912 with spec configuration."""
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_12",
        backend="server3",
        budget=1000,
        run_name="v912",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV912)
    assert spec.n_init == 8
    assert method._max_history == PROGRESS_WINDOW == 8
    method._llm.close()
