"""TraceAAD 搜索状态的保存与恢复。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .derivation_graph import DerivationGraph
from .schema import (
    ImprovementEdge,
    ProgramNode,
    Trajectory,
    TrajectoryStatus,
    ValueVec,
)
from .trajectory_memory import TrajectoryMemory


def _graph_to_dict(graph: DerivationGraph) -> dict[str, Any]:
    return {
        "next_node_id": graph._next_node_id,
        "next_edge_id": graph._next_edge_id,
        "nodes": [asdict(node) for node in graph.nodes()],
        "edges": [asdict(edge) for edge in graph.edges()],
    }


def _graph_from_dict(payload: Mapping[str, Any]) -> DerivationGraph:
    graph = DerivationGraph()
    for item in payload["nodes"]:
        node = ProgramNode(
            id=int(item["id"]),
            code=str(item["code"]),
            idea=str(item["idea"]),
            fitness=item["fitness"],
        )
        graph._nodes[node.id] = node
    for item in payload["edges"]:
        edge = ImprovementEdge(
            id=int(item["id"]),
            parent_id=int(item["parent_id"]),
            child_id=int(item["child_id"]),
            action=str(item["action"]),
            operator=str(item["operator"]),
            delta=item["delta"],
            outcome=str(item["outcome"]),
            iteration=item["iteration"],
        )
        graph._edges[edge.id] = edge
        graph._incoming_edge_by_child[edge.child_id] = edge.id
    graph._next_node_id = int(payload["next_node_id"])
    graph._next_edge_id = int(payload["next_edge_id"])
    return graph


def _memory_to_dict(memory: TrajectoryMemory) -> dict[str, Any]:
    trajectories = []
    for trajectory in memory.trajectories():
        item = asdict(trajectory)
        item["status"] = str(trajectory.status)
        trajectories.append(item)
    return {
        "max_trajectory_length": memory.max_trajectory_length,
        "next_id": memory._next_id,
        "trajectories": trajectories,
    }


def _memory_from_dict(payload: Mapping[str, Any]) -> TrajectoryMemory:
    memory = TrajectoryMemory(
        max_trajectory_length=int(payload["max_trajectory_length"])
    )
    memory._next_id = int(payload["next_id"])
    for item in payload["trajectories"]:
        value_payload = item.get("value")
        trajectory = Trajectory(
            id=int(item["id"]),
            node_ids=tuple(int(node_id) for node_id in item["node_ids"]),
            edge_ids=tuple(int(edge_id) for edge_id in item["edge_ids"]),
            endpoint_id=int(item["endpoint_id"]),
            visit_count=int(item.get("visit_count", 0)),
            status=TrajectoryStatus(item["status"]),
            value=(
                None
                if value_payload is None
                else ValueVec(
                    quality=float(value_payload.get("quality", 0.0)),
                    trend=float(value_payload.get("trend", 0.0)),
                )
            ),
            scalar_value=item.get("scalar_value"),
        )
        memory._trajectories[trajectory.id] = trajectory
    return memory


def dump_state(method) -> dict[str, Any]:
    return {
        "initialization_complete": method._initialization_complete,
        "total_samples": method._tot_sample_nums,
        "next_attempt_id": method._next_attempt_id,
        "best_node_id": (None if method._best_node is None else method._best_node.id),
        "best_trajectory_id": method._best_trajectory_id,
        "graph": _graph_to_dict(method._graph),
        "memory": _memory_to_dict(method._memory),
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    graph = _graph_from_dict(payload["graph"])
    method._graph = graph
    method._memory = _memory_from_dict(payload["memory"])
    method._tot_sample_nums = int(payload["total_samples"])
    method._next_attempt_id = int(payload.get("next_attempt_id", 0))
    method._initialization_complete = bool(payload.get("initialization_complete", True))
    best_id = payload.get("best_node_id")
    method._best_node = None if best_id is None else graph.get_node(int(best_id))
    method._best_trajectory_id = payload.get("best_trajectory_id")


def find_latest_checkpoint(path: str | Path) -> Path:
    source = Path(path)
    if source.is_file():
        return source
    candidates = (
        source / "latest.json",
        source / "checkpoints" / "latest.json",
        source / "logs" / "checkpoints" / "latest.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no TraceAAD checkpoint found under {source}")


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = directory
    if target is None:
        target = method._checkpoint_dir
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps(dump_state(method), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    method._last_checkpoint_sample = method._tot_sample_nums
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = find_latest_checkpoint(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    method._last_checkpoint_sample = method._tot_sample_nums
    return checkpoint
