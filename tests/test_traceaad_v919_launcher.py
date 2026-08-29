from __future__ import annotations

import json
from pathlib import Path

from experiments.runners.traceaad.launch_v919 import (
    BUDGET,
    INITIAL_ROOTS,
    VERSION,
    build_plan,
    command_for,
    is_finished,
    _heldout_complete,
    PRIMARY_BACKENDS,
)


def test_v919_scheduler_builds_fifteen_fresh_runs(tmp_path: Path) -> None:
    plan = build_plan(experiments_root=tmp_path)

    assert len(plan) == 15
    assert len({item.session for item in plan}) == 15
    assert len({item.run_dir for item in plan}) == 15
    assert {item.repeat for item in plan} == {1, 2, 3}
    assert len({item.task for item in plan}) == 5
    # V9.19 initializes its own roots; runs start fresh without a bootstrap.
    assert all(item.run_dir.parent.name == f"traceaad_{VERSION}" for item in plan)


def test_v919_scheduler_does_not_use_server3b() -> None:
    assert "server3b" not in PRIMARY_BACKENDS


def test_v919_scheduler_launches_fresh_run_with_frozen_budget(tmp_path: Path) -> None:
    item = build_plan(experiments_root=tmp_path)[0]

    command = command_for(item, "server3")

    assert "--version" in command and VERSION in command
    assert "--budget" in command and str(BUDGET) in command
    assert "--n-init" in command and str(INITIAL_ROOTS) in command
    assert "--seed" in command and "0" in command
    assert "--run-name" in command and item.run_name in command


def test_v919_scheduler_rejects_uncheckpointed_directory(tmp_path: Path) -> None:
    item = build_plan(experiments_root=tmp_path)[0]
    item.run_dir.mkdir(parents=True)
    try:
        command_for(item, "server3")
    except RuntimeError as error:
        assert "without a checkpoint" in str(error)
    else:
        raise AssertionError("expected RuntimeError for uncheckpointed run dir")


def test_v919_scheduler_resumes_existing_checkpoint(tmp_path: Path) -> None:
    item = build_plan(experiments_root=tmp_path)[0]
    checkpoint = item.run_dir / "checkpoints" / "latest.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n", encoding="utf-8")

    command = command_for(item, "server3")

    assert command[-2:] == ["--resume-from", str(item.run_dir)]


def test_v919_scheduler_finished_requires_full_primary_budget(tmp_path: Path) -> None:
    item = build_plan(experiments_root=tmp_path)[0]
    summary_path = item.run_dir / "logs" / "summary.json"
    summary_path.parent.mkdir(parents=True)

    summary_path.write_text(json.dumps({"status": "finished", "budget_slots": BUDGET - 1}))
    assert not is_finished(item)

    summary_path.write_text(json.dumps({"status": "finished", "budget_slots": BUDGET}))
    assert is_finished(item)


def test_v919_heldout_completeness_needs_every_run_and_unit(tmp_path: Path) -> None:
    items = build_plan(experiments_root=tmp_path)
    tsp_items = [item for item in items if item.task == "tsp_construct"]
    output_dir = tsp_items[0].method_dir / "eval_best_complete"

    partial = {
        "run_records": [{"run_name": tsp_items[0].run_name}],
        "results_by_size": {
            "50": {"summary": {"num_runs": 3, "num_successful_eval_runs": 3}}
        },
    }
    (output_dir).mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(partial), encoding="utf-8")
    assert not _heldout_complete(output_dir, tsp_items)

    complete = {
        "run_records": [{"run_name": item.run_name} for item in tsp_items],
        "results_by_size": {
            unit: {"summary": {"num_runs": 3, "num_successful_eval_runs": 3}}
            for unit in ("50", "100", "200")
        },
    }
    (output_dir / "results.json").write_text(json.dumps(complete), encoding="utf-8")
    assert _heldout_complete(output_dir, tsp_items)
