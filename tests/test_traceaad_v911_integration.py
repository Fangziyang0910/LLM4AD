"""Comprehensive benchmark test suite for TraceAAD V9.11.

Structure:
1. Data Structures & Invariants (Forest, Program, Anchor, is_better, serialization)
2. Operators & Selection Mechanics (Regime state machine: DEVELOP -> EXPLORE -> LANDING, stagnation window, budget suppression)
3. Robustness & Fault Tolerance (Flaky LLM, NaN/inf handling, initialization boundary termination)
4. End-to-End Search & Exact Checkpoint Replay (Regime event logs, exact checkpoint equality, interrupted resume reproducibility)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_11 import (
    MIN_EXPLORE_REMAINING_EVALS,
    STAGNATION_WINDOW,
    RunArtifacts,
    TraceAADV911,
)
from llm4ad.method.traceaad_v9_11.checkpoint import load_checkpoint, save_checkpoint
from llm4ad.method.traceaad_v9_11.forest import Forest, is_better
from llm4ad.method.traceaad_v9_11.schema import Anchor, Attempt, Outcome, Program, Regime
from llm4ad.method.traceaad_v9_11.traceaad import decide_regime

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
            "Code:\n```python\n"
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
            "Code:\n```python\n"
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


class ConstantEvaluation(Evaluation):
    """Evaluation returning constant fitness to trigger stagnation clock."""

    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose function.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs) -> float | None:
        return 1.0


# ==============================================================================
# 1. Data Structures & Invariants
# ==============================================================================

def test_v911_is_better_total_order() -> None:
    """Validate strict total order: q > len(code) > id."""
    p1 = Program(id=1, code="def f(): pass", code_hash="h1", fitness=10.0, q=10.0, length=13, order=1)
    p2 = Program(id=2, code="def f(): pass", code_hash="h2", fitness=12.0, q=12.0, length=13, order=2)
    assert is_better(p2, p1) is True
    assert is_better(p1, p2) is False

    p_short = Program(id=3, code="def f(): return 1", code_hash="h3", fitness=10.0, q=10.0, length=17, order=3)
    p_long = Program(id=4, code="def f():\n    return 1", code_hash="h4", fitness=10.0, q=10.0, length=24, order=4)
    assert is_better(p_short, p_long) is True
    assert is_better(p_long, p_short) is False


def test_v911_forest_hierarchy_and_serialization() -> None:
    """Validate Forest root and child creation, and loss-free serialization."""
    forest = Forest(maximize=True)
    p1 = forest.add_program(code="def choose(v): return v", fitness=10.0, order=1)
    root = forest.add_root(program_id=p1.id, order=1)
    p2 = forest.add_program(code="def choose(v): return v + 1", fitness=11.0, order=2)
    child = forest.add_child(parent_id=root.id, program_id=p2.id, attempt_id=0, order=2)
    child.n = 2

    assert forest.root_ids == [0]
    assert forest.get_anchor(child.parent_id).program_id == p1.id

    payload = forest.to_dict()
    restored = Forest.from_dict(payload)
    assert len(restored.programs()) == 2
    assert len(restored.anchors()) == 2
    assert restored.get_anchor(child.id).n == 2


# ==============================================================================
# 2. Operators & Selection Mechanics
# ==============================================================================

def test_v911_regime_clock_and_landing_priority() -> None:
    """Validate state transitions: DEVELOP -> EXPLORE -> LANDING -> DEVELOP."""
    # 1. Before stagnation: DEVELOP
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW - 1,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=None,
            remaining_evals=100,
        )
        is Regime.DEVELOP
    )
    # 2. At stagnation window: EXPLORE
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=None,
            remaining_evals=100,
        )
        is Regime.EXPLORE
    )
    # 3. Landing override takes absolute priority: LANDING
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=42,
            remaining_evals=100,
        )
        is Regime.LANDING
    )
    # 4. Budget insufficient for explore (< 2): fallback to DEVELOP
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=None,
            remaining_evals=MIN_EXPLORE_REMAINING_EVALS - 1,
        )
        is Regime.DEVELOP
    )


# ==============================================================================
# 3. Robustness & Fault Tolerance
# ==============================================================================

def test_v911_flaky_responses_handled_gracefully(tmp_path: Path) -> None:
    """Invalid and NaN responses consume budget but do not corrupt valid programs."""
    llm = FlakyLLM()
    run_dir = tmp_path / "run_flaky"
    method = TraceAADV911(
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


def test_v911_budget_exhaustion_during_initialization(tmp_path: Path) -> None:
    """When budget is less than n_roots, halts gracefully with initialization_failure."""
    run_dir = tmp_path / "run_budget_under_roots"
    method = TraceAADV911(
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

def test_v911_smoke_runs_develop_explore_and_landing(tmp_path: Path) -> None:
    """Full search run executes DEVELOP, EXPLORE, and LANDING regimes cleanly."""
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV911(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        artifacts=artifacts,
        budget=30,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 30
    assert summary["n_roots"] == 8
    assert summary["n_develop"] >= 1
    assert summary["n_explore"] >= 1
    assert summary["landing_anchor_id"] is None

    programs = method._forest.programs()
    assert all(len(program.code_hash) == 64 for program in programs)
    assert all(program.evaluation_seconds is not None for program in programs)

    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    ]
    selected = [item for item in events if item["event"] == "regime_selected"]
    assert any(item["regime"] == "explore" for item in selected)
    assert any(item["regime"] == "landing" for item in selected)


def test_v911_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    """State dumped and loaded via checkpoint is completely preserved."""
    source = TraceAADV911(
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
    source._last_progress_order = 0
    source._last_explore_order = 0
    source._landing_anchor_id = root.id
    source._n_landing = 2
    save_checkpoint(source)

    restored = TraceAADV911(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=100,
        n_roots=1,
        checkpoint_dir=tmp_path / "restored",
    )
    load_checkpoint(restored, tmp_path / "source" / "latest.json")

    assert restored._iteration == 8
    assert restored._landing_anchor_id == root.id
    assert restored._n_landing == 2


def test_v911_interrupted_resume_reproduces_regime_trajectory(tmp_path: Path) -> None:
    """Interrupted search resumes to produce identical search trajectory."""
    common = {
        "evaluation": IncrementalEvaluation(),
        "budget": 30,
        "seed": 9,
    }
    uninterrupted = TraceAADV911(llm=ScriptedLLM(), **common)
    uninterrupted.run()

    interrupted = TraceAADV911(
        llm=ScriptedLLM(),
        checkpoint_dir=tmp_path / "interrupted" / "checkpoints",
        **common,
    )
    interrupted._initialize()
    while interrupted._n_eval < 25:
        decision = interrupted._next_decision()
        interrupted._log_choice(decision)
        interrupted._generate(
            interrupted._prompt(decision.anchor_id, decision.intent),
            anchor_id=decision.anchor_id,
            stage="search",
            iteration=interrupted._iteration,
            intent=decision.intent.value,
        )
    stopped_at = interrupted._n_candidates

    resumed = TraceAADV911(
        llm=ScriptedLLM(start=stopped_at),
        checkpoint_dir=tmp_path / "interrupted" / "checkpoints",
        resume_from=tmp_path / "interrupted" / "checkpoints" / "latest.json",
        **common,
    )
    resumed.run()

    uninterrupted_forest = uninterrupted._forest.to_dict()
    resumed_forest = resumed._forest.to_dict()
    uninterrupted_programs = [
        {key: value for key, value in item.items() if key != "evaluation_seconds"}
        for item in uninterrupted_forest["programs"]
    ]
    resumed_programs = [
        {key: value for key, value in item.items() if key != "evaluation_seconds"}
        for item in resumed_forest["programs"]
    ]
    assert uninterrupted_programs == resumed_programs
    assert uninterrupted_forest["anchors"] == resumed_forest["anchors"]
    assert resumed._last_explore_order == uninterrupted._last_explore_order
    assert resumed._landing_anchor_id is None
    assert resumed._n_eval == uninterrupted._n_eval == 30


def test_v911_runner_builds_and_validates_spec(tmp_path: Path) -> None:
    """Runner builds TraceAADV911 and enforces 8-root specification."""
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_11",
        backend="server3",
        budget=1000,
        run_name="v911",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV911)
    assert spec.n_init == 8
    method._llm.close()

    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_11",
            n_init=1,
            experiments_root=tmp_path,
        )
