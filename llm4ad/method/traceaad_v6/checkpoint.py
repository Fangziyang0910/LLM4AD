"""Save and restore the current TraceAAD V6 search state."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .complexity import code_hash, nonempty_loc
from .derivation_graph import DerivationGraph
from .schema import (
    ImprovementEdge,
    OperatorName,
    ProgramNode,
    PROTOCOL_ID,
    Trajectory,
    TrajectoryStatus,
    ValueVec,
)
from .trajectory_memory import TrajectoryMemory

CHECKPOINT_VERSION = 8


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
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


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
            program_loc=int(item["program_loc"]),
            code_hash=str(item["code_hash"]),
        )
        if node.id in graph._nodes:
            raise ValueError(f"checkpoint contains duplicate node id {node.id}")
        if node.code_hash != code_hash(node.code) or node.program_loc != nonempty_loc(
            node.code
        ):
            raise ValueError(f"checkpoint node {node.id} has invalid code metadata")
        if node.fitness is None or not math.isfinite(float(node.fitness)):
            raise ValueError(f"checkpoint node {node.id} has invalid fitness")
        graph._nodes[node.id] = node
    for item in payload["edges"]:
        edge = ImprovementEdge(
            id=int(item["id"]),
            parent_id=int(item["parent_id"]),
            child_id=int(item["child_id"]),
            operator=OperatorName(item["operator"]),
            action=str(item["action"]),
            anchor_role=str(item["anchor_role"]),
            primary_trajectory_id=int(item["primary_trajectory_id"]),
            reference_trajectory_id=item["reference_trajectory_id"],
            reference_program_id=item["reference_program_id"],
            delta_parent=item["delta_parent"],
            delta_route_best=item["delta_route_best"],
            delta_global_best=item["delta_global_best"],
            delta_loc=int(item["delta_loc"]),
            code_change_ratio=float(item["code_change_ratio"]),
            outcome=str(item["outcome"]),
            iteration=item["iteration"],
            new_global_best=bool(item["new_global_best"]),
            global_best_update_reason=item["global_best_update_reason"],
        )
        if edge.id in graph._edges:
            raise ValueError(f"checkpoint contains duplicate edge id {edge.id}")
        if edge.parent_id not in graph._nodes or edge.child_id not in graph._nodes:
            raise ValueError(f"checkpoint edge {edge.id} references an unknown node")
        if (
            edge.reference_program_id is not None
            and edge.reference_program_id not in graph._nodes
        ):
            raise ValueError(
                f"checkpoint edge {edge.id} references an unknown reference program"
            )
        if edge.child_id in graph._incoming_edge_by_child:
            raise ValueError(
                f"checkpoint contains multiple parents for node {edge.child_id}"
            )
        graph._edges[edge.id] = edge
        graph._incoming_edge_by_child[edge.child_id] = edge.id
    graph._next_node_id = int(payload["next_node_id"])
    graph._next_edge_id = int(payload["next_edge_id"])
    if graph._next_node_id < 0 or (
        graph._nodes and graph._next_node_id <= max(graph._nodes)
    ):
        raise ValueError("checkpoint next_node_id does not advance past existing nodes")
    if graph._next_edge_id < 0 or (
        graph._edges and graph._next_edge_id <= max(graph._edges)
    ):
        raise ValueError("checkpoint next_edge_id does not advance past existing edges")
    return graph


def _memory_to_dict(memory: TrajectoryMemory) -> dict[str, Any]:
    return {
        "max_trajectory_length": memory.max_trajectory_length,
        "next_id": memory._next_id,
        "trajectories": [asdict(route) for route in memory.trajectories()],
    }


def _memory_from_dict(
    payload: Mapping[str, Any], graph: DerivationGraph
) -> TrajectoryMemory:
    memory = TrajectoryMemory(
        max_trajectory_length=int(payload["max_trajectory_length"])
    )
    memory._next_id = int(payload["next_id"])
    for item in payload["trajectories"]:
        value_payload = item["value"]
        if value_payload is None:
            value = None
        else:
            value = ValueVec(
                quality=float(value_payload["quality"]),
            )
        route = Trajectory(
            id=int(item["id"]),
            node_ids=tuple(int(x) for x in item["node_ids"]),
            edge_ids=tuple(int(x) for x in item["edge_ids"]),
            endpoint_id=int(item["endpoint_id"]),
            compact_best_id=int(item["compact_best_id"]),
            visit_count=int(item["visit_count"]),
            status=TrajectoryStatus(item["status"]),
            value=value,
        )
        if len(route.node_ids) != len(route.edge_ids) + 1:
            raise ValueError(f"inconsistent trajectory path: {route.id}")
        if len(route.node_ids) > memory.max_trajectory_length:
            raise ValueError(f"trajectory exceeds configured length: {route.id}")
        if route.compact_best_id not in route.node_ids:
            raise ValueError(f"compact best is outside trajectory: {route.id}")
        if route.endpoint_id != route.node_ids[-1]:
            raise ValueError(f"endpoint is not the final trajectory node: {route.id}")
        if route.id in memory._trajectories:
            raise ValueError(f"checkpoint contains duplicate trajectory id {route.id}")
        for node_id in route.node_ids:
            if node_id not in graph._nodes:
                raise ValueError(
                    f"trajectory {route.id} references unknown node {node_id}"
                )
        for index, edge_id in enumerate(route.edge_ids):
            if edge_id not in graph._edges:
                raise ValueError(
                    f"trajectory {route.id} references unknown edge {edge_id}"
                )
            edge = graph.get_edge(edge_id)
            if (
                edge.parent_id != route.node_ids[index]
                or edge.child_id != route.node_ids[index + 1]
            ):
                raise ValueError(f"trajectory {route.id} contains a misaligned edge")
        memory._trajectories[route.id] = route
    if memory._next_id < 0 or (
        memory._trajectories and memory._next_id <= max(memory._trajectories)
    ):
        raise ValueError("checkpoint next trajectory id does not advance past routes")
    route_ids = set(memory._trajectories)
    for edge in graph.edges():
        if edge.primary_trajectory_id not in route_ids:
            raise ValueError(
                f"edge {edge.id} references unknown primary trajectory "
                f"{edge.primary_trajectory_id}"
            )
        if (
            edge.reference_trajectory_id is not None
            and edge.reference_trajectory_id not in route_ids
        ):
            raise ValueError(
                f"edge {edge.id} references unknown reference trajectory "
                f"{edge.reference_trajectory_id}"
            )
    return memory


def dump_state(method) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "initialization_complete": method._initialization_complete,
        "total_samples": method._tot_sample_nums,
        "next_attempt_id": method._next_attempt_id,
        "batch_count": method._batch_count,
        "stalled_iterations": method._stalled_iterations,
        "consecutive_sample_failures": method._consecutive_sample_failures,
        "search_aborted": method._search_aborted,
        "best_node_id": None if method._best_node is None else method._best_node.id,
        "best_node_sample_order": method._best_node_sample_order,
        "best_trajectory_id": method._best_trajectory_id,
        "graph": _graph_to_dict(method._graph),
        "memory": _memory_to_dict(method._memory),
        "rng_state": method._rng.getstate(),
        "search_configuration": method.search_configuration(),
        "runtime_identity": method.runtime_identity(),
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    version = int(payload.get("version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported TraceAAD checkpoint version: {version}; "
            f"expected {CHECKPOINT_VERSION}; old checkpoints are not migrated"
        )
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(
            "checkpoint protocol_id does not match TraceAAD v6 protocol"
        )
    if payload.get("search_configuration") != method.search_configuration():
        raise ValueError(
            "checkpoint search configuration does not match the current TraceAAD V6 "
            "configuration"
        )
    if payload.get("runtime_identity") != method.runtime_identity():
        raise ValueError(
            "checkpoint task, evaluator, or LLM identity does not match the current "
            "TraceAAD V6 runtime"
        )
    graph = _graph_from_dict(payload["graph"])
    method._graph = graph
    method._memory = _memory_from_dict(payload["memory"], graph)
    method._tot_sample_nums = int(payload["total_samples"])
    method._next_attempt_id = int(payload["next_attempt_id"])
    method._batch_count = int(payload.get("batch_count", 0))
    method._stalled_iterations = int(payload.get("stalled_iterations", 0))
    method._consecutive_sample_failures = int(
        payload.get("consecutive_sample_failures", 0)
    )
    method._search_aborted = bool(payload.get("search_aborted", False))
    method._initialization_complete = bool(payload["initialization_complete"])
    best_id = payload["best_node_id"]
    method._best_node = None if best_id is None else graph.get_node(int(best_id))
    best_sample_order = payload["best_node_sample_order"]
    method._best_node_sample_order = (
        None if best_sample_order is None else int(best_sample_order)
    )
    best_route = payload["best_trajectory_id"]
    method._best_trajectory_id = None if best_route is None else int(best_route)
    if (
        method._best_trajectory_id is not None
        and method._best_trajectory_id not in method._memory._trajectories
    ):
        raise ValueError("checkpoint best trajectory does not exist")
    method._rng.setstate(_as_tuple(payload["rng_state"]))
    artifacts = getattr(method, "_artifacts", None) or getattr(method, "_profiler", None)
    if artifacts is not None and hasattr(artifacts, "sync_after_resume"):
        best = method._best_node
        artifacts.sync_after_resume(
            total_samples=method._tot_sample_nums,
            best_score=None if best is None else best.fitness,
            best_sample_order=method._best_node_sample_order,
        )
    elif artifacts is not None and hasattr(artifacts, "_num_samples"):
        artifacts._num_samples = method._tot_sample_nums
        best = method._best_node
        if best is not None and best.fitness is not None:
            artifacts._best_score = best.fitness
            artifacts._best_sample_order = method._best_node_sample_order


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else directory
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    _atomic_write(latest, dump_state(method))
    method._last_checkpoint_batch = method._batch_count
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    method._last_checkpoint_batch = method._batch_count
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "dump_state",
    "load_checkpoint",
    "load_state",
    "save_checkpoint",
]
