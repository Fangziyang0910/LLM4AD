from __future__ import annotations

import json
from pathlib import Path

from experiments.runners.traceaad_v9_18.launch import (
    BUDGET,
    build_plan,
    command_for,
    is_finished,
    _heldout_complete,
)


def test_v918_scheduler_builds_two_arms_and_thirty_runs(tmp_path: Path) -> None:
    plan = build_plan(experiments_root=tmp_path)

    assert len(plan) == 30
    assert len({item.session for item in plan}) == 30
    assert len({item.run_dir for item in plan}) == 30
    assert {item.arm_tag for item in plan} == {"A0q", "A1o"}
    assert {item.repeat for item in plan} == {1, 2, 3}


def test_v918_scheduler_uses_bootstrap_for_new_run(tmp_path: Path) -> None:
    plan = build_plan(experiments_root=tmp_path)
    item = plan[0]
    item.initialization_checkpoint.parent.mkdir(parents=True)
    item.initialization_checkpoint.write_text("{}\n", encoding="utf-8")

    command = command_for(item, "server3")

    assert "--initialization-checkpoint" in command
    assert str(item.initialization_checkpoint) in command
    assert "--budget" in command and str(BUDGET) in command


def test_v918_scheduler_resumes_existing_checkpoint(tmp_path: Path) -> None:
    item = build_plan(experiments_root=tmp_path)[0]
    checkpoint = item.run_dir / "checkpoints" / "latest.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}\n", encoding="utf-8")

    command = command_for(item, "server3")

    assert command[-2:] == ["--resume-from", str(item.run_dir)]


def test_v918_scheduler_finished_requires_full_primary_budget(tmp_path: Path) -> None:
    item = build_plan(experiments_root=tmp_path)[0]
    summary = item.run_dir / "logs" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps({"status": "finished", "budget_slots": BUDGET}) + "\n",
        encoding="utf-8",
    )

    assert is_finished(item)


def test_v918_scheduler_rejects_partial_heldout_results(tmp_path: Path) -> None:
    items = build_plan(experiments_root=tmp_path)[:3]
    output = items[0].method_dir / "eval_best_v918_A0q_tsp_construct_complete"
    output.mkdir(parents=True)
    payload = {
        "run_records": [{"run_name": item.run_name} for item in items],
        "eval_results_by_size": {
            "tsp50": {
                "summary": {
                    "num_runs": 3,
                    "num_successful_eval_runs": 2,
                }
            }
        },
    }
    (output / "results.json").write_text(
        json.dumps(payload) + "\n", encoding="utf-8"
    )

    assert not _heldout_complete(output, items)
