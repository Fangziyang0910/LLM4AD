"""Launch the formal TraceAAD V10.1 batch.

5 tasks x 3 repeats = 15 runs. Backends are assigned by fixed rotation over
(server1, server3, local), task-major, which yields the quota
server1 x5 / server3 x5 / local x5 and puts every task's three
repeats on three different backends. The watcher relaunches a dead run in the
same run directory, where run.py resumes from tree_state.json.
"""

from __future__ import annotations

from pathlib import Path

from experiments.infra.base import BackendName
from experiments.infra.launcher import launch_batch

MODULE = "experiments.traceaad_v10_1.run"
METHOD = "v101"
BACKEND_ROTATION: tuple[BackendName, ...] = ("server1", "server3", "local")


def main() -> None:
    launch_batch(
        method=METHOD,
        module=MODULE,
        results_root=Path(__file__).resolve().parent / "results",
        backend_rotation=BACKEND_ROTATION,
        default_session_prefix="v101",
        default_watch_interval=120,
    )


if __name__ == "__main__":
    main()
