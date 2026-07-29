"""TraceAAD 搜索状态的原子化保存与恢复。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from ...base import TextFunctionProgramConverter
from .derivation_graph import DerivationGraph
from .schema import (
    ImprovementEdge,
    ProgramNode,
    Trajectory,
    TrajectoryStatus,
    ValueVec,
)
from .trajectory_memory import TrajectoryMemory

CHECKPOINT_VERSION = 5


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
        if edge.child_id in graph._incoming_edge_by_child:
            raise ValueError(
                f"checkpoint contains multiple parents for node {edge.child_id}"
            )
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
        "version": CHECKPOINT_VERSION,
        "task_description": method._task_description_str,
        "template_program": method._evaluation.template_program,
        "function_name": method._function_to_evolve.name,
        "maximize": method._maximize,
        "search_configuration": _search_configuration(method),
        "initialization_complete": method._initialization_complete,
        "total_samples": method._tot_sample_nums,
        "next_attempt_id": method._next_attempt_id,
        "best_node_id": (None if method._best_node is None else method._best_node.id),
        "best_trajectory_id": method._best_trajectory_id,
        "graph": _graph_to_dict(method._graph),
        "memory": _memory_to_dict(method._memory),
    }


def _search_configuration(method) -> dict[str, Any]:
    return {
        "n_init": method._n_init,
        "actions_per_iteration": method._actions_per_iteration,
        "max_trajectory_length": method._memory.max_trajectory_length,
        "max_active_trajectories": method._max_active_trajectories,
        "management_threshold": method._management_threshold,
        "elite_count": method._elite_count,
        "diversity_count": method._diversity_count,
        "softmax_temperature": method._softmax_temperature,
        "value_weights": asdict(method._value_weights),
        "operators": [operator.name for operator in method._operators],
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    if int(payload.get("version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported TraceAAD checkpoint version: {payload.get('version')}"
        )
    if payload.get("task_description") != method._task_description_str:
        raise ValueError("checkpoint task description does not match the evaluation")
    if payload.get("template_program") != method._evaluation.template_program:
        raise ValueError("checkpoint template does not match the evaluation")
    if payload.get("function_name") != method._function_to_evolve.name:
        raise ValueError("checkpoint function does not match the evaluation template")
    if bool(payload.get("maximize")) != method._maximize:
        raise ValueError("checkpoint fitness direction does not match TraceAAD")
    if payload.get("search_configuration") != _search_configuration(method):
        raise ValueError("checkpoint search configuration does not match TraceAAD")

    graph = _graph_from_dict(payload["graph"])
    memory = _memory_from_dict(payload["memory"])
    method._graph = graph
    method._memory = memory
    method._tot_sample_nums = int(payload["total_samples"])
    method._next_attempt_id = int(payload.get("next_attempt_id", 0))
    method._initialization_complete = bool(payload.get("initialization_complete", True))
    best_id = payload.get("best_node_id")
    method._best_node = None if best_id is None else graph.get_node(int(best_id))
    method._best_trajectory_id = payload.get("best_trajectory_id")
    _restore_profiler(method)


def _restore_profiler(method) -> None:
    profiler = method._profiler
    if profiler is None:
        return
    profiler._num_samples = method._tot_sample_nums
    best = method._best_node
    if best is None or best.fitness is None or getattr(profiler, "_num_objs", 1) >= 2:
        return
    function = TextFunctionProgramConverter.program_to_function(best.code)
    if function is not None:
        function.score = best.fitness
        function.algorithm = best.idea
        profiler._cur_best_function = function
    profiler._cur_best_program_score = best.fitness
    profiler._cur_best_program_sample_order = method._tot_sample_nums


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
    root = Path(target)
    payload = dump_state(method)
    latest = root / "latest.json"
    _atomic_write(latest, payload)
    method._last_checkpoint_sample = method._tot_sample_nums
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = find_latest_checkpoint(path)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    load_state(method, payload)
    method._last_checkpoint_sample = method._tot_sample_nums
    return checkpoint
