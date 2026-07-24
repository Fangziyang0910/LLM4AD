"""Atomic V5-only checkpoint save and restore."""

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
    ActionRelation,
    ExperienceKind,
    GlobalExperienceEntry,
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
            relation=ActionRelation(item["relation"]),
            evidence_edge_ids=tuple(int(x) for x in item["evidence_edge_ids"]),
            reference_evidence_edge_ids=tuple(
                int(x) for x in item["reference_evidence_edge_ids"]
            ),
            change=str(item["change"]),
            novel_difference=str(item.get("novel_difference", "")),
            anchor_role=str(item["anchor_role"]),
            primary_trajectory_id=int(item["primary_trajectory_id"]),
            root_lineage_id=int(item["root_lineage_id"]),
            reference_trajectory_id=item.get("reference_trajectory_id"),
            reference_program_id=item.get("reference_program_id"),
            delta_parent=item.get("delta_parent"),
            delta_route_best=item.get("delta_route_best"),
            delta_global_best=item.get("delta_global_best"),
            delta_loc=int(item.get("delta_loc", 0)),
            code_change_ratio=float(item.get("code_change_ratio", 0.0)),
            outcome=str(item.get("outcome", "unknown")),
            iteration=item.get("iteration"),
            new_global_best=bool(item.get("new_global_best", False)),
            global_best_update_reason=item.get("global_best_update_reason"),
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
        "prompt_window": memory.prompt_window,
        "next_id": memory._next_id,
        "trajectories": [asdict(route) for route in memory.trajectories()],
    }


def _memory_from_dict(payload: Mapping[str, Any]) -> TrajectoryMemory:
    memory = TrajectoryMemory(prompt_window=int(payload["prompt_window"]))
    memory._next_id = int(payload["next_id"])
    for item in payload["trajectories"]:
        value_payload = item.get("value")
        route = Trajectory(
            id=int(item["id"]),
            root_lineage_id=int(item["root_lineage_id"]),
            node_ids=tuple(int(x) for x in item["node_ids"]),
            edge_ids=tuple(int(x) for x in item["edge_ids"]),
            endpoint_id=int(item["endpoint_id"]),
            compact_best_id=int(item["compact_best_id"]),
            visit_count=int(item.get("visit_count", 0)),
            reference_use_count=int(item.get("reference_use_count", 0)),
            status=TrajectoryStatus(item["status"]),
            value=(
                None
                if value_payload is None
                else ValueVec(
                    quality=float(value_payload["quality"]),
                    trend=float(value_payload["trend"]),
                )
            ),
            scalar_value=item.get("scalar_value"),
        )
        if len(route.node_ids) != len(route.edge_ids) + 1:
            raise ValueError(f"inconsistent trajectory path: {route.id}")
        memory._trajectories[route.id] = route
    return memory


def dump_state(method) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_VERSION,
        "task_description": method._task_description_str,
        "template_program": method._evaluation.template_program,
        "function_name": method._function_to_evolve.name,
        "maximize": method._maximize,
        "search_configuration": _search_configuration(method),
        "initialization_complete": method._initialization_complete,
        "total_samples": method._tot_sample_nums,
        "next_attempt_id": method._next_attempt_id,
        "best_node_id": None if method._best_node is None else method._best_node.id,
        "best_trajectory_id": method._best_trajectory_id,
        "graph": _graph_to_dict(method._graph),
        "memory": _memory_to_dict(method._memory),
        "global_experience_entries": [
            asdict(entry) for entry in method._global_experience_entries
        ],
        "pending_experience_edge_ids": list(method._pending_experience_edge_ids),
        "experience_update_index": method._experience_update_index,
        "last_experience_validation": method._last_experience_validation,
        "rng_state": method._rng.getstate(),
    }


def _search_configuration(method) -> dict[str, Any]:
    return {
        "n_init": method._n_init,
        "actions_per_iteration": method._actions_per_iteration,
        "prompt_window": method._memory.prompt_window,
        "max_active_trajectories": method._max_active_trajectories,
        "management_threshold": method._management_threshold,
        "elite_count": method._elite_count,
        "diversity_count": method._diversity_count,
        "softmax_temperature": method._softmax_temperature,
        "value_weights": asdict(method._value_weights),
        "operators": [operator.name.value for operator in method._operators],
        "experience_batch_size": method._experience_batch_size,
        "max_context_tokens": method._max_context_tokens,
        "output_token_reserve": method._output_token_reserve,
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    if int(payload.get("format_version", 0)) != CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported TraceAAD v5 checkpoint format: "
            f"{payload.get('format_version')}"
        )
    if payload.get("task_description") != method._task_description_str:
        raise ValueError("checkpoint task description does not match the evaluation")
    if payload.get("template_program") != method._evaluation.template_program:
        raise ValueError("checkpoint template does not match the evaluation")
    if payload.get("function_name") != method._function_to_evolve.name:
        raise ValueError("checkpoint function does not match the evaluation template")
    if bool(payload.get("maximize")) != method._maximize:
        raise ValueError("checkpoint fitness direction does not match TraceAAD v5")
    if payload.get("search_configuration") != _search_configuration(method):
        raise ValueError("checkpoint search configuration does not match TraceAAD v5")
    graph = _graph_from_dict(payload["graph"])
    method._graph = graph
    method._memory = _memory_from_dict(payload["memory"])
    method._tot_sample_nums = int(payload["total_samples"])
    method._next_attempt_id = int(payload.get("next_attempt_id", 0))
    method._initialization_complete = bool(payload["initialization_complete"])
    best_id = payload.get("best_node_id")
    method._best_node = None if best_id is None else graph.get_node(int(best_id))
    best_route = payload.get("best_trajectory_id")
    method._best_trajectory_id = None if best_route is None else int(best_route)
    method._global_experience_entries = tuple(
        GlobalExperienceEntry(
            kind=ExperienceKind(item["kind"]),
            statement=str(item["statement"]),
            condition=str(item["condition"]),
            evidence_edge_ids=tuple(int(x) for x in item["evidence_edge_ids"]),
        )
        for item in payload.get("global_experience_entries", [])
    )
    method._pending_experience_edge_ids = list(
        int(x) for x in payload.get("pending_experience_edge_ids", [])
    )
    method._experience_update_index = int(payload.get("experience_update_index", 0))
    method._last_experience_validation = payload.get("last_experience_validation")
    method._rng.setstate(_as_tuple(payload["rng_state"]))
    _restore_profiler(method)


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


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
    for candidate in (
        source / "latest.json",
        source / "checkpoints" / "latest.json",
        source / "logs" / "checkpoints" / "latest.json",
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no TraceAAD v5 checkpoint found under {source}")


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else directory
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    _atomic_write(latest, dump_state(method))
    method._last_checkpoint_sample = method._tot_sample_nums
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = find_latest_checkpoint(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    method._last_checkpoint_sample = method._tot_sample_nums
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "dump_state",
    "find_latest_checkpoint",
    "load_checkpoint",
    "load_state",
    "save_checkpoint",
]
