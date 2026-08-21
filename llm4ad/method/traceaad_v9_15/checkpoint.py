"""Minimal checkpoint persistence for TraceAAD V9.15."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import Pending
from .tree import Tree


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    path = Path(target) / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tree": method._tree.to_dict(),
                "pending": None if method._pending is None else asdict(method._pending),
                "n_eval": method._n_eval,
                "n_stag": method._n_stag,
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
    method._tree = Tree.from_dict(state["tree"])
    method._pending = (
        None if state["pending"] is None else Pending(**state["pending"])
    )
    method._n_eval = state["n_eval"]
    method._n_stag = state["n_stag"]
    return checkpoint


__all__ = ["load_checkpoint", "save_checkpoint"]
