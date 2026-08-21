"""Checkpoint persistence for TraceAAD V9.13.

The region view is not serialized: it is a deterministic function of the
evaluated programs, so restoring rebuilds it from the forest in program-id
(= real evaluation) order.  Checkpoint recovery therefore cannot change
region snapshots or the intent schedule.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .forest import Forest
from .schema import Pending


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    path = Path(target) / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "forest": method._forest.to_dict(),
                "pending": None if method._pending is None else asdict(method._pending),
                "n_candidates": method._n_candidates,
                "n_eval": method._n_eval,
                "iteration": method._iteration,
                "initialization_complete": method._initialization_complete,
                "bootstrapped": sorted(method._bootstrapped),
                "bootstrap_deltas": list(method._bootstrap_deltas),
                "s": method._s,
                "treatment_counters": dict(method._treatment_counters),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    method._forest = Forest.from_dict(state["forest"])
    method._pending = (
        None if state["pending"] is None else Pending(**state["pending"])
    )
    method._n_candidates = int(state["n_candidates"])
    method._n_eval = int(state["n_eval"])
    method._iteration = int(state["iteration"])
    method._initialization_complete = bool(state["initialization_complete"])
    method._bootstrapped = {int(item) for item in state["bootstrapped"]}
    method._bootstrap_deltas = [float(item) for item in state["bootstrap_deltas"]]
    method._s = None if state["s"] is None else float(state["s"])
    method._treatment_counters = {
        str(key): int(value)
        for key, value in state["treatment_counters"].items()
    }
    method._regions = method._build_region_view()
    return checkpoint


__all__ = [
    "load_checkpoint",
    "save_checkpoint",
]
