"""Checkpoint persistence for TraceAAD V9.20."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .landscape import Landscape
from .schema import Attempt, Pending
from .tree import Tree

CHECKPOINT_VERSION = "v9_20"
CHECKPOINT_NAME = "latest.json"
BEHAVE_STATE_NAME = "behave.npz"
VIEW_NAME = "view.json"


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    target.mkdir(parents=True, exist_ok=True)
    behave_path = target / BEHAVE_STATE_NAME
    # ``save_checkpoint`` is called before and after every attempt.  Pending
    # state changes do not alter the behavior archive, so avoid an expensive
    # full archive pack/write when the profiled-node count is unchanged.
    archive_size = len(method._landscape.node_ids)
    persisted_size = getattr(method, "_checkpoint_behave_size", None)
    if persisted_size != archive_size or not behave_path.is_file():
        _atomic_np_save(behave_path, method._landscape.state_arrays())
        method._checkpoint_behave_size = archive_size
    state = {
        "version": CHECKPOINT_VERSION,
        "task": method._task_key,
        "behave_protocol": method._landscape.protocol,
        "tree": method._tree.to_dict(),
        "pending": None if method._pending is None else asdict(method._pending),
        "attempts": [asdict(attempt) for attempt in method._attempts],
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "repair_llm_calls": method._repair_llm_calls,
        "repair_eval_calls": method._repair_eval_calls,
        "ordinary_decisions": method._n_ordinary_decisions,
    }
    path = target / CHECKPOINT_NAME
    _atomic_write(path, json.dumps(state, indent=2) + "\n")
    _atomic_write(target / VIEW_NAME, json.dumps(_view_state(method), separators=(",", ":")))
    return path


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint is not a TraceAAD V9.20 checkpoint")
    if state.get("task") != method._task_key:
        raise ValueError("checkpoint task does not match the run")
    if state.get("behave_protocol") != method._landscape.protocol:
        raise ValueError("checkpoint BehaveSim protocol does not match the run")

    behave_state = checkpoint.parent / BEHAVE_STATE_NAME
    if not behave_state.is_file():
        raise FileNotFoundError(
            f"checkpoint BehaveSim state is missing: {behave_state}"
        )
    with np.load(behave_state) as payload:
        arrays = {name: payload[name] for name in payload.files}
    method._landscape = Landscape.from_state_arrays(
        task=method._task_key,
        protocol=state["behave_protocol"],
        arrays=arrays,
    )
    method._tree = Tree.from_dict(state["tree"])
    profiled = set(method._landscape.node_ids)
    valid = {algorithm.id for algorithm in method._tree.valid_algorithms()}
    if profiled != valid:
        raise RuntimeError(
            "valid nodes without a cached behavior profile: "
            f"{sorted(valid - profiled)}; unexplained profiles: "
            f"{sorted(profiled - valid)}"
        )
    pending = state.get("pending")
    method._pending = None if pending is None else Pending(**pending)
    method._attempts = [Attempt(**item) for item in state.get("attempts", [])]
    method._n_eval = int(state["n_eval"])
    method._n_calls = int(state["n_calls"])
    method._repair_llm_calls = int(state["repair_llm_calls"])
    method._repair_eval_calls = int(state["repair_eval_calls"])
    method._n_ordinary_decisions = int(state["ordinary_decisions"])
    method._checkpoint_behave_size = len(method._landscape.node_ids)
    return checkpoint


def _view_state(method) -> dict:
    nodes = []
    for algorithm in method._tree.valid_algorithms():
        item = asdict(algorithm)
        item.pop("code", None)
        nodes.append(item)
    best = method._tree.best()
    return {
        "n_eval": method._n_eval,
        "best_id": None if best is None else best.id,
        "nodes": nodes,
    }


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_np_save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


__all__ = [
    "BEHAVE_STATE_NAME",
    "CHECKPOINT_VERSION",
    "VIEW_NAME",
    "load_checkpoint",
    "save_checkpoint",
]
