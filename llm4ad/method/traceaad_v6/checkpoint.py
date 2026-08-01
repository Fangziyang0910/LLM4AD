"""Save and restore the current TraceAAD V6 search state."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .attempts import AttemptMemory
from .derivation_graph import DerivationGraph
from .schema import (
    ImprovementEdge,
    OperatorName,
    ProgramNode,
    Trajectory,
    TrajectoryStatus,
    ValueVec,
)
from .trajectory_memory import TrajectoryMemory

CHECKPOINT_VERSION = 6


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
            edge_credit=float(item.get("edge_credit", 0.0)),
            iteration=item["iteration"],
            new_global_best=bool(item["new_global_best"]),
            global_best_update_reason=item["global_best_update_reason"],
        )
        if edge.child_id in graph._incoming_edge_by_child:
            raise ValueError(
                f"checkpoint contains multiple parents for node {edge.child_id}"
            )
        graph._edges[edge.id] = edge
        graph._incoming_edge_by_child[edge.child_id] = edge.id
    graph._next_node_id = int(payload["next_node_id"])
    graph._next_edge_id = int(payload["next_edge_id"])
    return graph


def _memory_to_dict(memory: TrajectoryMemory) -> dict[str, Any]:
    return {
        "max_trajectory_length": memory.max_trajectory_length,
        "next_id": memory._next_id,
        "trajectories": [asdict(route) for route in memory.trajectories()],
    }


def _memory_from_dict(payload: Mapping[str, Any]) -> TrajectoryMemory:
    memory = TrajectoryMemory(
        max_trajectory_length=int(payload["max_trajectory_length"])
    )
    memory._next_id = int(payload["next_id"])
    for item in payload["trajectories"]:
        value_payload = item["value"]
        if value_payload is None:
            value = None
        else:
            credit = value_payload.get("credit", value_payload.get("trend", 0.0))
            value = ValueVec(
                quality=float(value_payload["quality"]),
                credit=float(credit),
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
            scalar_value=item["scalar_value"],
        )
        if len(route.node_ids) != len(route.edge_ids) + 1:
            raise ValueError(f"inconsistent trajectory path: {route.id}")
        if len(route.node_ids) > memory.max_trajectory_length:
            raise ValueError(f"trajectory exceeds configured length: {route.id}")
        if route.compact_best_id not in route.node_ids:
            raise ValueError(f"compact best is outside trajectory: {route.id}")
        memory._trajectories[route.id] = route
    return memory


def dump_state(method) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
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
        "attempts": method._attempts.to_dict(),
        "rng_state": method._rng.getstate(),
        "search_configuration": method.search_configuration(),
        "profiler": _dump_profiler(method),
    }


def _dump_profiler(method) -> dict[str, Any] | None:
    profiler = method._profiler
    if profiler is None:
        return None
    return {
        "started_at": profiler._process_start_time.isoformat(),
        "evaluate_success_program_num": profiler._evaluate_success_program_num,
        "evaluate_failed_program_num": profiler._evaluate_failed_program_num,
        "total_sample_time": profiler._tot_sample_time,
        "total_evaluate_time": profiler._tot_evaluate_time,
        "error_count": profiler._error_count,
        "llm_call_count": profiler._llm_call_count,
        "method_event_count": profiler._method_event_count,
        "method_state_count": profiler._method_state_count,
        "logging_degraded": profiler._logging_degraded,
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    version = int(payload.get("version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported TraceAAD checkpoint version: {version}; "
            f"expected {CHECKPOINT_VERSION}"
        )
    if payload.get("search_configuration") != method.search_configuration():
        raise ValueError(
            "checkpoint search configuration does not match the current TraceAAD V6 "
            "configuration"
        )
    graph = _graph_from_dict(payload["graph"])
    method._graph = graph
    method._memory = _memory_from_dict(payload["memory"])
    method._attempts = AttemptMemory.from_dict(payload.get("attempts", {"attempts": []}))
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
    method._rng.setstate(_as_tuple(payload["rng_state"]))
    _restore_profiler(method, payload.get("profiler"))


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


def _restore_profiler(method, payload: Mapping[str, Any] | None) -> None:
    profiler = method._profiler
    if profiler is None or payload is None:
        return
    profiler._num_samples = method._tot_sample_nums
    profiler._process_start_time = profiler._process_start_time.fromisoformat(
        payload["started_at"]
    )
    profiler._evaluate_success_program_num = int(
        payload["evaluate_success_program_num"]
    )
    profiler._evaluate_failed_program_num = int(payload["evaluate_failed_program_num"])
    profiler._tot_sample_time = float(payload["total_sample_time"])
    profiler._tot_evaluate_time = float(payload["total_evaluate_time"])
    profiler._error_count = int(payload["error_count"])
    profiler._llm_call_count = int(payload["llm_call_count"])
    profiler._method_event_count = int(payload["method_event_count"])
    profiler._method_state_count = int(payload.get("method_state_count", 0))
    profiler._logging_degraded = bool(payload["logging_degraded"])
    best = method._best_node
    if best is None or best.fitness is None:
        return
    profiler._cur_best_program_score = best.fitness
    profiler._cur_best_program_sample_order = method._best_node_sample_order


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else directory
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    _atomic_write(latest, dump_state(method))
    method._last_checkpoint_sample = method._tot_sample_nums
    method._last_checkpoint_batch = method._batch_count
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    method._last_checkpoint_sample = method._tot_sample_nums
    method._last_checkpoint_batch = method._batch_count
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "dump_state",
    "load_checkpoint",
    "load_state",
    "save_checkpoint",
]
