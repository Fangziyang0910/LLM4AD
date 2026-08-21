"""Tests for TraceAAD V9.7 streamlined artifacts and logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm4ad.method.traceaad_v9_7.artifacts import (
    EVALUATIONS_CSV_HEADER,
    RunArtifacts,
)
from llm4ad.method.traceaad_v9_7.schema import Outcome


def test_artifacts_initialization(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)

    eval_path = tmp_path / "evaluations.csv"
    assert eval_path.exists()

    with eval_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == EVALUATIONS_CSV_HEADER

    artifacts.finish()


def test_artifacts_record_candidate_and_best(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)

    # Record root candidate
    artifacts.record_candidate(
        order=1,
        stage="root_generation",
        iteration=None,
        route_id=0,
        anchor_id=None,
        child_id=0,
        program_id=0,
        intent=None,
        kind="root_new",
        outcome=None,
        status="ok",
        parent_fitness=None,
        child_fitness=100.5,
        dq=None,
        is_new_best=True,
        best_fitness=100.5,
        eval_count=1,
        budget=10,
    )

    artifacts.record_best(code="def select(): return 1", fitness=100.5)

    # Check best_program.py
    best_file = tmp_path / "best_program.py"
    assert best_file.exists()
    content = best_file.read_text(encoding="utf-8")
    assert "Fitness: 100.5" in content
    assert "def select(): return 1" in content

    # Record search step
    artifacts.record_candidate(
        order=2,
        stage="search",
        iteration=0,
        route_id=0,
        anchor_id=0,
        child_id=1,
        program_id=1,
        intent="refine",
        kind="new",
        outcome=Outcome.IMPROVE,
        status="ok",
        parent_fitness=100.5,
        child_fitness=95.2,
        dq=5.3,
        is_new_best=True,
        best_fitness=95.2,
        eval_count=2,
        budget=10,
    )

    # Verify evaluations.csv has 2 records
    with (tmp_path / "evaluations.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 records
        assert rows[1][0] == "1"
        assert rows[1][9] == "root_new"
        assert rows[2][0] == "2"
        assert rows[2][8] == "refine"
        assert rows[2][10] == "improve"

    artifacts.write_summary(
        status="finished",
        best_score=95.2,
        evaluator_call_count=2,
    )
    summary_path = tmp_path / "logs" / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["best_score"] == 95.2
    assert summary["status"] == "finished"

    artifacts.finish()
