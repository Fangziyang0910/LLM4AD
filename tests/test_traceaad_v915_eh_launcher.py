from __future__ import annotations

from collections import Counter

from experiments.runners.traceaad.launch_v915_eh import build_plan, launch_pending


def test_v915_eh_plan_is_tsp_only_and_uses_separate_artifacts() -> None:
    plan = build_plan(batch="batch")

    assert len(plan) == 3
    assert {item.task for item in plan} == {"tsp_construct"}
    assert {item.repeat for item in plan} == {1, 2, 3}
    assert Counter(item.backend for item in plan) == {
        "server3": 2,
        "server3b": 1,
    }
    assert len({item.session for item in plan}) == 3
    assert all("traceaad_v9_15_eh" in str(item.run_dir) for item in plan)
    assert all(item.run_name.startswith("v9_15_eh_") for item in plan)


def test_v915_eh_launch_respects_backend_limits(monkeypatch) -> None:
    plan = build_plan(batch="batch")
    launched: list[str] = []
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v915_eh._done", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v915_eh._running", lambda item: False
    )
    monkeypatch.setattr(
        "experiments.runners.traceaad.launch_v915_eh.launch",
        lambda item, dry_run=False: launched.append(item.backend),
    )

    count = launch_pending(
        plan,
        limits={"server3": 1, "server3b": 1, "server1": 0},
    )

    assert count == 2
    assert Counter(launched) == {"server3": 1, "server3b": 1}
