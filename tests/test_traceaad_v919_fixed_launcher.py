from __future__ import annotations

from pathlib import Path

from experiments.runners.traceaad import launch_v919_fixed as launcher


def test_fixed_plan_uses_distinct_batch_name(tmp_path: Path) -> None:
    plan = launcher.build_plan(experiments_root=tmp_path)
    assert len(plan) == 15
    assert all("v9_19_fixed_20260829" in item.run_name for item in plan)
    assert all(item.run_dir.parent.name == "traceaad_v9_19" for item in plan)
    assert len({item.session for item in plan}) == 15


def test_fixed_scheduler_limits_server3_and_includes_server3b(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher,
        "free_slots",
        lambda: {"server3": 4, "server1": 2, "server3b": 9, "local": 3},
    )
    assert launcher.available_slots() == {
        "server3": 3,
        "server1": 2,
        "server3b": 9,
        "local": 3,
    }
    assert "server3b" in launcher.ALLOWED_BACKENDS


def test_fixed_scheduler_does_not_double_count_batch_processes(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher,
        "free_slots",
        lambda: {"server3": 3, "server1": 1, "server3b": 9, "local": 2},
    )
    expected = {"server3": 3, "server1": 1, "server3b": 9, "local": 2}
    assert launcher.available_slots("v9_19_fixed_20260829") == expected


def test_fixed_assignment_fills_each_available_slot_once(tmp_path: Path, monkeypatch) -> None:
    plan = launcher.build_plan(experiments_root=tmp_path)
    monkeypatch.setattr(
        launcher,
        "free_slots",
        lambda: {"server3": 4, "server1": 2, "server3b": 9, "local": 0},
    )
    assigned = launcher.assign_backends(plan)
    assert len(assigned) == 14
    assert [backend for _, backend in assigned].count("server3") == 3
    assert [backend for _, backend in assigned].count("server1") == 2
    assert [backend for _, backend in assigned].count("server3b") == 9
    assert all(backend in launcher.ALLOWED_BACKENDS for _, backend in assigned)


def test_fixed_command_accepts_server3b(tmp_path: Path) -> None:
    item = launcher.build_plan(experiments_root=tmp_path)[0]
    command = launcher.command_for(item, "server3b")
    assert "--backend" in command
    assert command[command.index("--backend") + 1] == "server3b"
