from __future__ import annotations

import json

from experiments.runners.traceaad import fill_v9_4


def test_v94_fill_plan_has_four_tasks_and_three_repeats() -> None:
    plan = fill_v9_4.build_plan(batch="20260810_120000", repeats=3)

    assert len(plan) == 12
    assert {(item.task, item.repeat) for item in plan} == {
        (task, repeat) for task in fill_v9_4.TASKS for repeat in range(1, 4)
    }
    assert plan[0].run_name == "v9_4_20260810_120000_tsp_rep1"
    assert plan[0].session == "v94_20260810_120000_tsp_r1"
    assert len({item.session for item in plan}) == 12
    assert len({item.run_dir for item in plan}) == 12


def test_v94_fill_command_freezes_formal_protocol() -> None:
    item = fill_v9_4.build_plan(batch="batch", repeats=3)[0]
    command = item.command("server1", budget=1000)

    assert command[command.index("--version") + 1] == "v9_4"
    assert command[command.index("--backend") + 1] == "server1"
    assert command[command.index("--n-init") + 1] == "8"
    assert command[command.index("--output-tokens") + 1] == "8192"
    assert command[command.index("--context-token-limit") + 1] == "24576"
    assert command[command.index("--seed") + 1] == str(item.repeat)
    assert command[command.index("--budget") + 1] == "1000"


def test_v94_fill_assigns_every_reported_free_slot() -> None:
    plan = fill_v9_4.build_plan(batch="batch", repeats=3)
    assigned = fill_v9_4.assign_pending(
        plan,
        {"zhong": 2, "server3": 2, "server3b": 2, "local": 2},
    )

    assert len(assigned) == 4
    assert [backend for _, backend in assigned] == [
        "server3",
        "server3b",
        "server3",
        "server3b",
    ]


def test_v94_fill_deduplicates_evaluator_children_by_run_name() -> None:
    first = (
        "python -m experiments.runners.traceaad.run --backend zhong --run-name run_a"
    )
    second = (
        "python -m experiments.runners.traceaad.run --backend server1 --run-name run_b"
    )
    usage = fill_v9_4.usage_from_cmdlines([first, first, second, "uv run " + second])

    assert usage == {"zhong": 1, "server1": 1, "server3": 0, "server3b": 0, "local": 0}


def test_v94_fill_reads_finished_summary_and_never_relaunches_existing_dir(
    tmp_path, monkeypatch
) -> None:
    item = fill_v9_4.V94Run(
        task="tsp_construct",
        repeat=1,
        session="test",
        run_name="test",
        run_dir=tmp_path,
    )
    summary = tmp_path / "logs" / "summary.json"
    summary.parent.mkdir()
    summary.write_text(json.dumps({"status": "finished"}), encoding="utf-8")
    monkeypatch.setattr(fill_v9_4, "session_running", lambda _: False)

    assert fill_v9_4.run_state(item) == "finished"
    summary.unlink()
    assert fill_v9_4.run_state(item) == "orphaned"
