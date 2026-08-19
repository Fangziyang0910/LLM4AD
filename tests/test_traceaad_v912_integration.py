from __future__ import annotations

import json
from pathlib import Path

from experiments.runners.traceaad import run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_12 import (
    CHECKPOINT_VERSION,
    EXPLORE_PROBABILITY_MAX,
    EXPLORE_PROBABILITY_MIN,
    MIN_EXPLORE_REMAINING_EVALS,
    PROGRESS_WINDOW,
    PROTOCOL_ID,
    RunArtifacts,
    TraceAADV912,
)
from llm4ad.method.traceaad_v9_12.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v9_12.schema import Attempt, Intent, Outcome
from llm4ad.method.traceaad_v9_12.selection import (
    explore_probability,
    refine_failure_evidence,
    select,
)

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return f"Idea: candidate {self.calls}\nCode:\n```python\ndef choose(value: int) -> int:\n    return value + {self.calls}\n```"

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class ConstantEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(template_program=TEMPLATE, task_description="Improve choose.", safe_evaluate=False, timeout_seconds=10)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return 1.0


def test_probability_is_bounded_and_progress_conditioned() -> None:
    assert explore_probability(0.0) == EXPLORE_PROBABILITY_MIN
    assert explore_probability(1.0) == EXPLORE_PROBABILITY_MAX
    assert explore_probability(0.25) == 0.15


def test_explore_is_suppressed_only_when_followup_budget_is_unavailable() -> None:
    method = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        budget=10,
        n_roots=1,
        context_limit=32768,
    )
    program = method._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    method._forest.add_root(program_id=program.id, order=1)
    method._s = 0.0
    method._seed = next(
        seed
        for seed in range(1000)
        if select(method._forest, 0.0, seed=seed, iteration=0).intent
        is Intent.EXPLORE
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


def test_v912_smoke_runs_and_records_followup(tmp_path: Path) -> None:
    method = TraceAADV912(
        llm=ScriptedLLM(), evaluation=ConstantEvaluation(), artifacts=RunArtifacts(tmp_path, console_output=False),
        budget=30, context_limit=32768, checkpoint_dir=tmp_path / "checkpoints", seed=3,
    )
    method.run()
    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 30
    assert summary["n_roots"] == 8
    assert summary["n_followup"] >= 0
    assert summary["exploration_followup_anchor_id"] is None
    assert summary["n_explore"] > 0
    assert summary["n_valid_explore_children"] == summary["n_followup"]


def test_progress_window_resets_at_explore_child_and_counts_followup() -> None:
    method = TraceAADV912(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        budget=10,
        n_roots=1,
        context_limit=32768,
    )
    forest = method._forest
    program = method._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    root = forest.add_root(program_id=program.id, order=1)

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

    followup_id = forest.next_attempt_id()
    forest.add_attempt(
        Attempt(
            id=followup_id,
            anchor_id=explored.id,
            child_id=None,
            program_id=None,
            intent=Intent.REFINE.value,
            idea=None,
            diff=None,
            added=0,
            removed=0,
            parent_fitness=1.5,
            child_fitness=None,
            dq=None,
            outcome=Outcome.INVALID,
            kind="invalid",
            order=5,
            stage="search",
            iteration=3,
        )
    )
    assert refine_failure_evidence(forest, explored.id) == (1 / 8, 1)
    assert PROGRESS_WINDOW == 8


def test_checkpoint_roundtrip_preserves_v912_protocol(tmp_path: Path) -> None:
    source = TraceAADV912(llm=ScriptedLLM(), evaluation=ConstantEvaluation(), budget=10, n_roots=1, context_limit=32768)
    payload = dump_state(source)
    restored = TraceAADV912(llm=ScriptedLLM(), evaluation=ConstantEvaluation(), budget=10, n_roots=1, context_limit=32768)
    load_state(restored, json.loads(json.dumps(payload)))
    assert restored.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert restored.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION


def test_runner_builds_v912_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(task="tsp_construct", version="v9_12", backend="server3", budget=1000, run_name="v912", experiments_root=tmp_path)
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV912)
    assert spec.n_init == 8
    assert method.search_configuration()["progress_window"] == 8
    method._llm.close()
