from __future__ import annotations

from collections import Counter

from experiments.traceaad_v9_14.launch import build_plan, launch_pending


def test_v914_plan_has_fifteen_runs_and_fixed_backend_layout() -> None:
    plan = build_plan(batch="batch")

    assert len(plan) == 15
    assert len({item.session for item in plan}) == 15
    assert len({item.run_name for item in plan}) == 15
    assert Counter(item.backend for item in plan) == {
        "server3": 8,
        "server3b": 7,
    }
    assert Counter((item.task, item.repeat) for item in plan) == {
        (task, repeat): 1
        for repeat in range(1, 4)
        for task in {
            "tsp_construct",
            "cvrp_aco",
            "op_aco",
            "online_bin_packing",
            "vrptw_construct",
        }
    }


def test_v914_launch_limits_each_backend(monkeypatch) -> None:
    plan = build_plan(batch="batch")
    launched: list[str] = []
    monkeypatch.setattr(
        "experiments.traceaad_v9_14.launch._done", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.traceaad_v9_14.launch._running", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.traceaad_v9_14.launch.launch",
        lambda item, dry_run=False: launched.append(item.backend),
    )
    count = launch_pending(
        plan,
        limits={"server3": 2, "server3b": 1},
        dry_run=True,
    )
    assert count == 3
    assert Counter(launched) == {"server3": 2, "server3b": 1}
