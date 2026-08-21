from __future__ import annotations

from collections import Counter

from experiments.runners.traceaad.launch_v915 import build_plan, launch_pending


def test_v915_plan_has_twelve_runs_balanced_across_both_servers() -> None:
    plan = build_plan(batch="batch")

    assert len(plan) == 12
    assert len({item.session for item in plan}) == 12
    assert len({item.run_name for item in plan}) == 12
    assert Counter(item.backend for item in plan) == {
        "server3": 6,
        "server3b": 6,
    }
    assert Counter((item.task, item.repeat) for item in plan) == {
        (task, repeat): 1
        for repeat in range(1, 4)
        for task in {
            "tsp_construct",
            "cvrp_aco",
            "op_aco",
            "online_bin_packing",
        }
    }
    assert all("traceaad_v9_15" in str(item.run_dir) for item in plan)


def test_v915_launch_limits_each_backend(monkeypatch) -> None:
    plan = build_plan(batch="batch")
    launched: list[str] = []
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v915._done", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v915._running", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v915.launch",
        lambda item, dry_run=False: launched.append(item.backend),
    )

    count = launch_pending(
        plan,
        limits={"server3": 4, "server3b": 2, "server1": 3},
    )

    assert count == 6
    assert Counter(launched) == {"server3": 4, "server3b": 2}
