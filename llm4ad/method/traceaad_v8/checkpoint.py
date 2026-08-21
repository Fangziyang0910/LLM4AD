"""Checkpoint persistence for TraceAAD V8."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .schema import ImprovementEdge, OperatorName, ProgramNode, VirtualRoot
from .tree import SearchTree


def _tree_to_dict(tree: SearchTree) -> dict[str, Any]:
    return {
        "root": asdict(tree.root),
        "next_node_id": tree._next_node_id,
        "next_edge_id": tree._next_edge_id,
        "nodes": [asdict(node) for node in tree.nodes()],
        "edges": [asdict(edge) for edge in tree.edges()],
    }


def _tree_from_dict(payload: Mapping[str, Any]) -> SearchTree:
    tree = SearchTree()
    tree.root = VirtualRoot(**payload["root"])
    for item in payload["nodes"]:
        tree._nodes[int(item["id"])] = ProgramNode(**item)
    for item in payload["edges"]:
        tree._edges[int(item["id"])] = ImprovementEdge(
            **{**item, "operator": OperatorName(item["operator"])}
        )
    tree._next_node_id = int(payload["next_node_id"])
    tree._next_edge_id = int(payload["next_edge_id"])
    return tree


def dump_state(method) -> dict[str, Any]:
    return {
        "initialization_complete": method._initialization_complete,
        "total_samples": method._tot_sample_nums,
        "next_attempt_id": method._next_attempt_id,
        "batch_count": method._batch_count,
        "stalled_iterations": method._stalled_iterations,
        "best_node_id": None if method._best_node is None else method._best_node.id,
        "best_node_sample_order": method._best_node_sample_order,
        "tree": _tree_to_dict(method._tree),
        "rng_state": method._rng.getstate(),
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    method._tree = _tree_from_dict(payload["tree"])
    best_id = payload["best_node_id"]
    method._best_node = None if best_id is None else method._tree.get_node(int(best_id))
    method._best_node_sample_order = (
        None
        if payload["best_node_sample_order"] is None
        else int(payload["best_node_sample_order"])
    )
    method._tot_sample_nums = int(payload["total_samples"])
    method._next_attempt_id = int(payload["next_attempt_id"])
    method._batch_count = int(payload["batch_count"])
    method._stalled_iterations = int(payload["stalled_iterations"])
    method._initialization_complete = bool(payload["initialization_complete"])
    method._rng.setstate(_as_tuple(payload["rng_state"]))


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps(dump_state(method), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    method._last_checkpoint_batch = method._batch_count
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    method._last_checkpoint_batch = method._batch_count
    return checkpoint


__all__ = [
    "dump_state",
    "load_checkpoint",
    "load_state",
    "save_checkpoint",
]
