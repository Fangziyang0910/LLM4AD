"""Checkpoint persistence for TraceAAD V9.16."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import LandingState, Pending
from .tree import Tree


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    path = Path(target) / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "version": "v9_16",
        "tree": method._tree.to_dict(),
        "pending": None if method._pending is None else asdict(method._pending),
        "active_landing": (
            None if method._active_landing is None else asdict(method._active_landing)
        ),
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "n_stag": method._n_stag,
        "attempt": method._attempt_number,
        "attempt_kind": method._attempt_kind,
        "ordinary_decisions": method._n_ordinary_decisions,
        "next_entry_id": method._next_entry_id,
        "next_landing_id": method._next_landing_id,
        "landing_budget": method._landing_budget,
        "landing_slots_used": method._landing_slots_used,
        "entry_tickets": {str(k): v for k, v in method._entry_tickets.items()},
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return path


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("version") not in {None, "v9_16"}:
        raise ValueError("checkpoint is not a TraceAAD V9.16 checkpoint")
    method._tree = Tree.from_dict(state["tree"])
    pending = state.get("pending")
    method._pending = None if pending is None else Pending(**pending)
    active = state.get("active_landing")
    method._active_landing = None if active is None else LandingState(**active)
    method._n_eval = state["n_eval"]
    method._n_calls = state.get("n_calls", method._n_eval)
    method._n_stag = state["n_stag"]
    method._attempt_number = state.get("attempt", 1)
    method._attempt_kind = state.get("attempt_kind", "initial")
    method._n_ordinary_decisions = state.get("ordinary_decisions", 0)
    method._next_entry_id = state.get("next_entry_id", 1)
    method._next_landing_id = state.get("next_landing_id", 1)
    method._landing_budget = state.get("landing_budget", method._landing_budget)
    method._landing_slots_used = state.get("landing_slots_used", 0)
    method._entry_tickets = {
        int(k): bool(v) for k, v in state.get("entry_tickets", {}).items()
    }
    return checkpoint


__all__ = ["load_checkpoint", "save_checkpoint"]
