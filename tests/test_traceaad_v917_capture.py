from __future__ import annotations

from experiments.runners.traceaad_v9_17_fixed_cycle.capture_initializations import (
    is_fork_boundary,
)


def test_v917_initialization_capture_requires_exact_first_development_state() -> None:
    state = {
        "version": "v9_17",
        "phase": "development",
        "s_r_frozen": True,
        "cycle": 1,
        "sweep": 1,
        "initial_order": list(range(1, 9)),
        "initial_cursor": 8,
        "eligible_ids": list(range(1, 9)),
        "active_ids": list(range(1, 9)),
        "sweep_order": [],
        "sweep_cursor": 0,
        "successful_ids": [],
        "active_block": None,
        "generation": None,
        "pending": None,
        "discovery_attempted": False,
    }
    assert is_fork_boundary(state)
    for key, value in {
        "sweep_order": [1, 2],
        "active_block": {"id": 1},
        "generation": {"mode": "development"},
        "cycle": 2,
    }.items():
        changed = dict(state)
        changed[key] = value
        assert not is_fork_boundary(changed)
