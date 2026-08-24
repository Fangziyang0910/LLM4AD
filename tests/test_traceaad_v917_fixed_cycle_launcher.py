from __future__ import annotations

from experiments.runners.traceaad.launch_v917_fixed_cycle import (
    backend_launch_capacity,
    build_plan,
)
from experiments.runners._common import TASKS


def test_fixed_cycle_plan_pairs_all_fifteen_adaptive_initializations() -> None:
    plan = build_plan(adaptive_batch="adaptive", fixed_batch="fixed")
    assert len(plan) == 15
    assert len({item.session for item in plan}) == 15
    assert len({item.run_dir for item in plan}) == 15
    assert all("traceaad_v9_17_fixed_cycle" in str(item.run_dir) for item in plan)
    assert all("paired_initialization/latest.json" in str(item.initialization_checkpoint) for item in plan)
    assert {(item.task, item.repeat) for item in plan} == {
        (task, repeat) for repeat in range(1, 4) for task in TASKS
    }


def test_fixed_cycle_uses_three_local_and_all_nine_server1_slots() -> None:
    capacity = backend_launch_capacity(
        usage={"server3": 9, "server3b": 6, "local": 1, "server1": 2},
        available={"server3": 0, "server3b": 3, "local": 2, "server1": 7},
    )
    assert capacity == {"server3": 0, "server3b": 3, "local": 2, "server1": 7}
