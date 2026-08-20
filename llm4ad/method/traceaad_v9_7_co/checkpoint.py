"""Checkpoint persistence for TraceAAD V9.7-CO (code-only ablation arm)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .forest import Forest
from .schema import PROTOCOL_ID, Pending

CHECKPOINT_VERSION = 4


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _pending_from_dict(item: Mapping[str, Any] | None) -> Pending | None:
    if item is None:
        return None
    return Pending(
        id=int(item["id"]),
        anchor_id=None if item["anchor_id"] is None else int(item["anchor_id"]),
        stage=str(item["stage"]),
        iteration=None if item["iteration"] is None else int(item["iteration"]),
        order=int(item["order"]),
        intent=None if item["intent"] is None else str(item["intent"]),
        response=str(item["response"]),
    )


def dump_state(method) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "config": method.search_configuration(),
        "forest": method._forest.to_dict(),
        "pending": None if method._pending is None else asdict(method._pending),
        "n_candidates": method._n_candidates,
        "n_eval": method._n_eval,
        "iteration": method._iteration,
        "initialization_complete": method._initialization_complete,
        "bootstrapped": sorted(method._bootstrapped),
        "bootstrap_deltas": list(method._bootstrap_deltas),
        "s": method._s,
        "best_id": method._best_id,
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported TraceAAD V9.7-CO checkpoint version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint protocol does not match TraceAAD V9.7-CO")
    if payload.get("config") != method.search_configuration():
        raise ValueError("checkpoint configuration does not match")
    method._forest = Forest.from_dict(payload["forest"])
    method._pending = _pending_from_dict(payload["pending"])
    method._n_candidates = int(payload["n_candidates"])
    method._n_eval = int(payload["n_eval"])
    method._iteration = int(payload["iteration"])
    method._initialization_complete = bool(payload["initialization_complete"])
    method._bootstrapped = {int(item) for item in payload["bootstrapped"]}
    method._bootstrap_deltas = [float(item) for item in payload["bootstrap_deltas"]]
    method._s = None if payload["s"] is None else float(payload["s"])
    method._best_id = None if payload["best_id"] is None else int(payload["best_id"])


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    _atomic_write(latest, dump_state(method))
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "load_checkpoint",
    "save_checkpoint",
]
