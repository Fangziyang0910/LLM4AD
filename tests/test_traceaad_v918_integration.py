"""Focused mechanism tests for TraceAAD V9.18-R0."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_18 import RunArtifacts, TraceAADV918
from llm4ad.method.traceaad_v9_18.checkpoint import save_checkpoint
from llm4ad.method.traceaad_v9_18.prompt import (
    build_generation_prompt,
    extract_diagnosis,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_18.schema import Intent, Pending
from llm4ad.method.traceaad_v9_18.selection import (
    decide,
    opportunity_value,
    robust_root_scale,
    selection_score,
)
from llm4ad.method.traceaad_v9_18.tree import Tree

TEMPLATE = """def choose(value: int) -> int:
    return value
"""

RESPONSE = """Idea: increment
Code:
```python
def choose(value: int) -> int:
    return value + 1
```
"""


class ScriptedLLM(LLM):
    def __init__(self, response: str = RESPONSE) -> None:
        super().__init__()
        self.response = response
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.response


class SequenceLLM(LLM):
    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        return next(self.responses)


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


def test_r0_opportunity_only_applies_to_explore_entries() -> None:
    tree = Tree(maximize=True)
    root = tree.add_algorithm(code="root", fitness=10.0)
    entry = tree.add_algorithm(
        code="entry", fitness=9.0, parent_id=root.id, created_by=Intent.EXPLORE.value,
        entry_id=1, is_explore_entry=True,
    )
    regression = tree.add_algorithm(
        code="regression", fitness=8.0, parent_id=root.id,
        created_by=Intent.REFINE.value,
    )
    assert opportunity_value(root) == pytest.approx(0.0)
    assert opportunity_value(entry) == pytest.approx(1.0)
    assert selection_score(tree, entry, sigma_q=10.0, allocation_mode="opportunity") == pytest.approx(10.0)
    assert selection_score(tree, regression, sigma_q=10.0, allocation_mode="opportunity") == pytest.approx(8.0)
    entry.n_after = 2
    assert opportunity_value(entry) == pytest.approx(1.0 / 2.718281828459045)


def test_duplicate_explore_candidate_does_not_create_opportunity_entry() -> None:
    method = TraceAADV918(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=2,
        n_roots=1,
        allocation_mode="opportunity",
    )
    root = method._tree.add_algorithm(code="root", fitness=1.0)
    method._sigma_q = 1.0
    method._pending = Pending(
        parent_id=root.id,
        intent=Intent.EXPLORE.value,
        response=RESPONSE,
    )
    method._process_pending()
    child = method._tree.valid_algorithms()[-1]
    assert child.is_explore_entry is True

    method._pending = Pending(
        parent_id=root.id,
        intent=Intent.EXPLORE.value,
        response=RESPONSE,
    )
    method._process_pending()
    duplicate = method._tree.valid_algorithms()[-1]
    assert duplicate.is_explore_entry is False
    assert opportunity_value(duplicate) == pytest.approx(0.0)


def test_root_scale_uses_valid_root_quality_mad() -> None:
    tree = Tree(maximize=True)
    for index, quality in enumerate((1.0, 2.0, 3.0, 4.0), start=1):
        tree.add_algorithm(code=f"root-{index}", fitness=quality)
    assert robust_root_scale(tree) == pytest.approx(1.0)


def test_diagnosis_is_optional_audit_text() -> None:
    response = """Diagnosis: the candidate lacks a fallback
Idea: add fallback
Code:
```python
def choose(value: int) -> int:
    return value + 1
```
"""
    parsed = parse_program_response(response)
    assert parsed.diagnosis == "the candidate lacks a fallback"
    assert parsed.declared_idea == "add fallback"
    assert extract_diagnosis("Idea: only") is None


def test_global_facts_are_added_only_to_facts_explore_prompt() -> None:
    prompt = build_generation_prompt(
        task_description="Improve choose.",
        code=TEMPLATE,
        fitness=1.0,
        history_text="[Recent Algorithm Improvement History]",
        intent=Intent.EXPLORE,
        maximize=True,
        global_facts="[Global Verified Facts - R0]\nGlobal best quality: 2",
    )
    assert "[Global Verified Facts - R0]" in prompt
    assert "Global best quality: 2" in prompt


def test_r0_run_uses_one_primary_slot_per_initial_candidate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    llm = ScriptedLLM()
    method = TraceAADV918(
        llm=llm,
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=6,
        n_roots=1,
        seed=3,
        checkpoint_dir=run_dir / "checkpoints",
        allocation_mode="opportunity",
        explore_context="facts",
    )
    method.run()

    assert method._n_eval == 6
    assert method._sigma_q == pytest.approx(0.0)
    with (run_dir / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert [int(row["eval_count"]) for row in rows] == list(range(1, 7))
    assert all(row["allocation_mode"] == "opportunity" for row in rows[1:])
    assert (run_dir / "mechanism_events.jsonl").is_file()
    assert method._n_calls == 6
    events = [
        json.loads(line)
        for line in (run_dir / "mechanism_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = [event for event in events if event["event"] == "pre_decision"]
    assert decisions
    assert {"q", "opportunity", "n_after", "score"}.issubset(
        decisions[0]["selection_snapshot"][0]
    )


def test_repair_audit_hash_identifies_the_actual_request(tmp_path: Path) -> None:
    bad = "Idea: broken\nCode:\n```python\ndef choose(value: int) -> int:\n    return (\n```"
    llm = SequenceLLM([bad, RESPONSE])
    run_dir = tmp_path / "repair-audit"
    method = TraceAADV918(
        llm=llm,
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=1,
        n_roots=1,
        error_retries=1,
        checkpoint_dir=run_dir / "checkpoints",
    )

    child = method._generate(
        "Original prompt",
        parent_id=0,
        intent=None,
        mode="initialization",
        entry_id=None,
        facts_hash="facts-before-request",
    )
    assert child is not None
    with (run_dir / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["prompt_hash"] != rows[1]["prompt_hash"]
    assert rows[0]["facts_hash"] == "facts-before-request"
    assert rows[1]["facts_hash"] == ""


def test_checkpoint_preserves_pre_request_decision(tmp_path: Path) -> None:
    method = TraceAADV918(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=10,
        n_roots=1,
        seed=4,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    root = method._tree.add_algorithm(code="root", fitness=1.0)
    method._sigma_q = 1.0
    method._decision = decide(
        method._tree,
        seed=4,
        n_eval=1,
        sigma_q=1.0,
        allocation_mode="q",
        decision_index=0,
    )
    method._pending = method._pending or None
    path = save_checkpoint(method)

    resumed = TraceAADV918(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=10,
        n_roots=1,
        seed=4,
        checkpoint_dir=tmp_path / "resumed-checkpoints",
        resume_from=path,
    )
    assert resumed._decision is not None
    assert resumed._decision.parent.id == root.id
    assert resumed._decision.decision_index == 0
    assert resumed._decision.selection_snapshot


def test_initialization_fork_allows_only_allocation_mode_change(tmp_path: Path) -> None:
    source = TraceAADV918(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=8,
        n_roots=1,
        seed=4,
        checkpoint_dir=tmp_path / "source",
        allocation_mode="q",
    )
    source._tree.add_algorithm(code="root", fitness=1.0)
    source._n_eval = 1
    source._sigma_q = 0.0
    path = save_checkpoint(source)

    fork = TraceAADV918(
        llm=ScriptedLLM(),
        evaluation=IncrementalEvaluation(),
        budget=3,
        n_roots=1,
        seed=4,
        checkpoint_dir=tmp_path / "fork",
        resume_from=path,
        allocation_mode="opportunity",
        fork_from_initialization=True,
    )
    assert fork._n_eval == 1
    assert fork._allocation_mode == "opportunity"
    assert fork.best is not None
