"""Tests for TraceAAD V9.7 streamlined artifacts and logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm4ad.method.traceaad_v9_7.artifacts import (
    BEST_CURVE_CSV_HEADER,
    EVALUATIONS_CSV_HEADER,
    RunArtifacts,
)
from llm4ad.method.traceaad_v9_7.schema import Outcome


def test_artifacts_initialization(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)

    eval_path = tmp_path / "evaluations.csv"
    curve_path = tmp_path / "best_curve.csv"
    err_path = tmp_path / "logs" / "errors.jsonl"

    assert eval_path.exists()
    assert curve_path.exists()
    assert err_path.exists()

    with eval_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == EVALUATIONS_CSV_HEADER

    with curve_path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == BEST_CURVE_CSV_HEADER

    artifacts.finish()


def test_artifacts_record_candidate_and_best(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)

    # Record root candidate
    artifacts.record_candidate(
        attempt_id=0,
        order=1,
        stage="root_generation",
        iteration=None,
        anchor_id=None,
        child_id=0,
        program_id=0,
        intent=None,
        kind="root_new",
        outcome=None,
        evaluator_called=True,
        status="ok",
        parent_fitness=None,
        child_fitness=100.5,
        dq=None,
        eval_count=1,
        route_id=0,
        best_fitness=100.5,
        is_new_best=True,
        budget=10,
    )

    artifacts.record_best(
        code="def select(): return 1",
        fitness=100.5,
        eval_count=1,
        iteration=None,
        order=1,
        program_id=0,
    )

    # Check best_program.py
    best_file = tmp_path / "best_program.py"
    assert best_file.exists()
    content = best_file.read_text(encoding="utf-8")
    assert "Fitness: 100.5" in content
    assert "def select(): return 1" in content

    # Record search step
    artifacts.record_candidate(
        attempt_id=1,
        order=2,
        stage="search",
        iteration=0,
        anchor_id=0,
        child_id=1,
        program_id=1,
        intent="refine",
        kind="new",
        outcome=Outcome.IMPROVE,
        evaluator_called=True,
        status="ok",
        parent_fitness=100.5,
        child_fitness=95.2,
        dq=5.3,
        eval_count=2,
        route_id=0,
        best_fitness=95.2,
        is_new_best=True,
        budget=10,
    )

    artifacts.record_best(
        code="def select(): return 2",
        fitness=95.2,
        eval_count=2,
        iteration=0,
        order=2,
        program_id=1,
    )

    # Verify best_curve.csv has 2 rows
    with (tmp_path / "best_curve.csv").open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 points
        assert rows[1][0] == "1"
        assert rows[1][3] == "100.5"
        assert rows[2][0] == "2"
        assert rows[2][3] == "95.2"

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
