from __future__ import annotations

from collections import Counter

from experiments.runners.traceaad.launch_v917 import build_plan, launch_pending


def test_v917_plan_has_fifteen_unique_runs_and_requested_backend_split() -> None:
    plan = build_plan(batch="test_v917")
    assert len(plan) == 15
    assert Counter(item.backend for item in plan) == {"server3": 9, "server3b": 6}
    assert len({item.session for item in plan}) == 15
    assert len({item.run_dir for item in plan}) == 15
    assert all("traceaad_v9_17" in str(item.run_dir) for item in plan)


def test_v917_launch_pending_respects_backend_limits(monkeypatch) -> None:
    plan = build_plan(batch="test_v917_limits")
    launched: list[str] = []
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v917._done", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v917._running", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v917.launch",
        lambda item, dry_run=False: launched.append(item.backend),
    )
    count = launch_pending(
        plan,
        limits={"server3": 2, "server3b": 1},
        dry_run=True,
    )
    assert count == 3
    assert Counter(launched) == {"server3": 2, "server3b": 1}
