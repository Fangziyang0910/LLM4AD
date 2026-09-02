"""Launch the formal TraceAAD V10.2 batch.

5 tasks x 3 repeats = 15 runs. Rerouted exclusively to (server1, local)
to ensure high throughput and prevent timeouts.
"""

from __future__ import annotations

from pathlib import Path

from experiments.infra.base import BackendName
from experiments.infra.launcher import launch_batch

MODULE = "experiments.traceaad_v10_2.run"
METHOD = "v102"
BACKEND_ROTATION: tuple[BackendName, ...] = ("server1", "local")

# Explicit distribution for the 7 active runs (3 on local, 4 on server1)
BACKEND_MAP: dict[tuple[str, int], BackendName] = {
    ("cvrp_aco", 3): "local",
    ("op_aco", 2): "local",
    ("tsp_construct", 2): "local",
    ("cvrp_aco", 1): "server1",
    ("online_bin_packing", 1): "server1",
    ("cvrp_aco", 2): "server1",
    ("vrptw_construct", 2): "server1",
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
