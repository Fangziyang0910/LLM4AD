"""Focused mechanism tests for TraceAAD V9.16."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_16 import RunArtifacts, TraceAADV916
from llm4ad.method.traceaad_v9_16.checkpoint import save_checkpoint
from llm4ad.method.traceaad_v9_16.schema import Intent, LandingState
from llm4ad.method.traceaad_v9_16.selection import (
    decide,
    landing_ticket,
    selection_score,
)
from llm4ad.method.traceaad_v9_16.tree import Tree

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, response: str) -> None:
        super().__init__()
        self.response = response
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        return self.response


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


RESPONSE = """Idea: fixed candidate
Code:
```python
def choose(value: int) -> int:
    return value + 1
```
"""


def test_v916_parent_score_is_quality_only() -> None:
    tree = Tree(maximize=True)
    root = tree.add_algorithm(code="a", fitness=1.0)
    child = tree.add_algorithm(code="b", fitness=3.0, parent_id=root.id)
    assert selection_score(tree, root) == pytest.approx(1.0)
    assert selection_score(tree, child) == pytest.approx(3.0)
    decision = decide(tree, seed=0, n_eval=1, decision_index=0)
    assert decision.p_explore == pytest.approx(0.3)
    assert decision.parent_q == pytest.approx(tree.quality(decision.parent))


def test_landing_ticket_is_deterministic_and_independent() -> None:
    assert landing_ticket(seed=42, entry_id=2) is True
    assert landing_ticket(seed=42, entry_id=2) is True
    assert landing_ticket(seed=0, entry_id=2) is False


def test_run_uses_exact_three_landing_slots_and_records_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm4ad.method.traceaad_v9_16.traceaad.landing_ticket", lambda **_: True
    )
    run_dir = tmp_path / "run"
    method = TraceAADV916(
        llm=ScriptedLLM(RESPONSE),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=40,
        n_roots=1,
        seed=3,
        checkpoint_dir=run_dir / "checkpoints",
    )
    method.run()

    assert method._n_eval == 40
    assert method._landing_budget == 3
    assert method._landing_slots_used == 3
    assert method._active_landing is None
    with (run_dir / "evaluations.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    landing_rows = [row for row in rows if row["mode"] == "landing"]
    assert len(landing_rows) == 3
    assert [row["landing_step"] for row in landing_rows] == ["1", "2", "3"]
    assert all(row["intent"] == Intent.REFINE.value for row in landing_rows)
    assert (run_dir / "landing_events.jsonl").is_file()


def test_checkpoint_round_trips_active_landing_state(tmp_path: Path) -> None:
    method = TraceAADV916(
        llm=ScriptedLLM(RESPONSE),
        evaluation=IncrementalEvaluation(),
        budget=40,
        n_roots=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    root = method._tree.add_algorithm(code="root", fitness=1.0, entry_id=1)
    method._next_entry_id = 2
    method._active_landing = LandingState(
        id=1,
        entry_id=1,
        origin_id=root.id,
        origin_parent_id=root.id,
        latest_valid_id=root.id,
        completed_steps=1,
        start_eval=8,
    )
    method._entry_tickets = {1: True}
    path = save_checkpoint(method)
    assert path is not None

    resumed = TraceAADV916(
        llm=ScriptedLLM(RESPONSE),
        evaluation=IncrementalEvaluation(),
        budget=40,
        n_roots=1,
        resume_from=path,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    assert resumed._active_landing is not None
    assert resumed._active_landing.completed_steps == 1
    assert resumed._entry_tickets == {1: True}
