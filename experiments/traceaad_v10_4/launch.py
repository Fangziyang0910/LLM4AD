"""Launch the formal TraceAAD V10.4 batch (5 tasks x 3 repeats)."""

from __future__ import annotations

from pathlib import Path

from experiments.infra.base import BackendName
from experiments.infra.launcher import launch_batch

MODULE = "experiments.traceaad_v10_4.run"
METHOD = "v104"
BACKEND_ROTATION: tuple[BackendName, ...] = (
    "server3",
    "server3b",
    "server1",
    "local",
)

BACKEND_MAP: dict[tuple[str, int], BackendName] = {
    ("tsp_construct", 1): "server3",
    ("tsp_construct", 2): "server3b",
    ("tsp_construct", 3): "server1",
    ("cvrp_aco", 1): "local",
    ("cvrp_aco", 2): "server3",
    ("cvrp_aco", 3): "server3b",
    ("op_aco", 1): "server1",
    ("op_aco", 2): "local",
    ("op_aco", 3): "server3",
    ("online_bin_packing", 1): "server3b",
    ("online_bin_packing", 2): "server1",
    ("online_bin_packing", 3): "local",
    ("vrptw_construct", 1): "server3",
    ("vrptw_construct", 2): "server3b",
    ("vrptw_construct", 3): "server1",
}


def main() -> None:
    launch_batch(
        method=METHOD,
        module=MODULE,
        results_root=Path(__file__).resolve().parent / "results",
        backend_rotation=BACKEND_ROTATION,
        backend_map=BACKEND_MAP,
        default_session_prefix="v104",
        default_watch_interval=120,
    )


if __name__ == "__main__":
    main()
