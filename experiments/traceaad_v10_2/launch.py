"""Launch the formal TraceAAD V10.2 batch.

5 tasks x 3 repeats = 15 runs: 6 on server3:8000, 6 on server3:8001,
1 local, and 2 on server1.
"""

from __future__ import annotations

from pathlib import Path

from experiments.infra.base import BackendName
from experiments.infra.launcher import launch_batch

MODULE = "experiments.traceaad_v10_2.run"
METHOD = "v102"
BACKEND_ROTATION: tuple[BackendName, ...] = (
    "server3",
    "server3b",
    "local",
    "server1",
)

BACKEND_MAP: dict[tuple[str, int], BackendName] = {
    ("tsp_construct", 1): "server3",
    ("tsp_construct", 2): "server3b",
    ("tsp_construct", 3): "server1",
    ("cvrp_aco", 1): "server3",
    ("cvrp_aco", 2): "server3b",
    ("cvrp_aco", 3): "server3",
    ("op_aco", 1): "server3",
    ("op_aco", 2): "server3b",
    ("op_aco", 3): "server1",
    ("online_bin_packing", 1): "server3",
    ("online_bin_packing", 2): "server3b",
    ("online_bin_packing", 3): "local",
    ("vrptw_construct", 1): "server3",
    ("vrptw_construct", 2): "server3b",
    ("vrptw_construct", 3): "server3b",
}


def main() -> None:
    launch_batch(
        method=METHOD,
        module=MODULE,
        results_root=Path(__file__).resolve().parent / "results",
        backend_rotation=BACKEND_ROTATION,
        backend_map=BACKEND_MAP,
        default_session_prefix="v102",
        default_watch_interval=120,
    )


if __name__ == "__main__":
    main()
