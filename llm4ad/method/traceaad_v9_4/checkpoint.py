"""Checkpoint persistence for TraceAAD V9.4."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    EventStatus,
    FailureObservation,
    GenerationEvent,
    PROTOCOL_ID,
    ProgramNode,
    TrajectoryCreditUpdate,
    VirtualRoot,
)
from .tree import FactGraph

CHECKPOINT_VERSION = 5


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


def _graph_to_dict(graph: FactGraph) -> dict[str, Any]:
    return {
        "root": asdict(graph.root),
        "next_node_id": graph._next_node_id,
        "next_event_id": graph._next_event_id,
        "nodes": [asdict(node) for node in graph.nodes()],
        "events": [asdict(event) for event in graph.events()],
    }


def _graph_from_dict(payload: Mapping[str, Any]) -> FactGraph:
    graph = FactGraph()
    root = payload["root"]
    graph.root = VirtualRoot(
        id=int(root["id"]), child_ids=[int(item) for item in root["child_ids"]]
    )
    for item in payload["nodes"]:
        node = ProgramNode(
            id=int(item["id"]),
            code=str(item["code"]),
            idea=str(item["idea"]),
            fitness=float(item["fitness"]),
            directed_fitness=float(item["directed_fitness"]),
            program_loc=int(item["program_loc"]),
            code_hash=str(item["code_hash"]),
            parent_id=int(item["parent_id"]),
            incoming_event_id=(
                None
                if item["incoming_event_id"] is None
                else int(item["incoming_event_id"])
            ),
            child_ids=[int(child) for child in item["child_ids"]],
            depth=int(item["depth"]),
            creation_order=int(item["creation_order"]),
            budget_event_count=int(item["budget_event_count"]),
            trajectory_event_count=int(item["trajectory_event_count"]),
            trajectory_credit_sum=float(item["trajectory_credit_sum"]),
            last_budget_order=(
                None
                if item["last_budget_order"] is None
                else int(item["last_budget_order"])
            ),
        )
        graph._nodes[node.id] = node
    for item in payload["events"]:
        event = GenerationEvent(
            id=int(item["id"]),
            anchor_id=int(item["anchor_id"]),
            child_id=None if item["child_id"] is None else int(item["child_id"]),
            idea=str(item["idea"]),
            status=EventStatus(item["status"]),
            failure_kind=item["failure_kind"],
            error_type=item["error_type"],
            error_message=item["error_message"],
            result_fitness=(
                None
                if item["result_fitness"] is None
                else float(item["result_fitness"])
            ),
            anchor_credit=float(item["anchor_credit"]),
            credit_updates=tuple(
                TrajectoryCreditUpdate(
                    node_id=int(update["node_id"]),
                    distance=int(update["distance"]),
                    credit=float(update["credit"]),
                )
                for update in item["credit_updates"]
            ),
            outcome=str(item["outcome"]),
            delta_parent=(
                None if item["delta_parent"] is None else float(item["delta_parent"])
            ),
            delta_loc=None if item["delta_loc"] is None else int(item["delta_loc"]),
            code_change_ratio=(
                None
                if item["code_change_ratio"] is None
                else float(item["code_change_ratio"])
            ),
            new_global_best=bool(item["new_global_best"]),
            strict_breakthrough=bool(item["strict_breakthrough"]),
            global_best_update_reason=item["global_best_update_reason"],
            stage=str(item["stage"]),
            iteration=None if item["iteration"] is None else int(item["iteration"]),
            budget_order=int(item["budget_order"]),
        )
        graph._events[event.id] = event
    graph._next_node_id = int(payload["next_node_id"])
    graph._next_event_id = int(payload["next_event_id"])
    return graph


def dump_state(method) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "search_configuration": method.search_configuration(),
        "runtime_identity": method.runtime_identity(),
        "initialization_complete": method._initialization_complete,
        "initial_strategy_cards": list(method._initial_strategy_cards),
        "root_strategy_cards": {
            str(node_id): strategy
            for node_id, strategy in sorted(method._root_strategy_cards.items())
        },
        "strategy_planning_calls": method._strategy_planning_calls,
        "bootstrapped_root_ids": sorted(method._bootstrapped_root_ids),
        "initial_selected_anchor_ids": list(method._initial_selected_anchor_ids),
        "eligible_node_ids": sorted(method._eligible_node_ids),
        "failure_history": [
            asdict(observation) for observation in method._failure_history
        ],
        "total_budget_events": method._tot_sample_nums,
        "total_evaluations": method._evaluation_count,
        "next_iteration": method._next_iteration,
        "consecutive_sample_failures": method._consecutive_sample_failures,
        "search_aborted": method._search_aborted,
        "best_node_id": None if method._best_node is None else method._best_node.id,
        "best_node_sample_order": method._best_node_sample_order,
        "graph": _graph_to_dict(method._graph),
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported TraceAAD V9.4 checkpoint version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint protocol does not match TraceAAD V9.4")
    if payload.get("search_configuration") != method.search_configuration():
        raise ValueError("checkpoint search configuration does not match")
    # 各模型服务源（zhong / server1 / server3 / local）是同一模型，仅服务端与量化差异，
    # 跨源续跑时忽略 runtime_identity 中的 LLM 服务字段，其余身份字段仍严格比对。
    _identity_ignore = {"llm_model", "llm_base_url"}
    saved_identity = {
        key: value
        for key, value in (payload.get("runtime_identity") or {}).items()
        if key not in _identity_ignore
    }
    current_identity = {
        key: value
        for key, value in method.runtime_identity().items()
        if key not in _identity_ignore
    }
    if saved_identity != current_identity:
        raise ValueError("checkpoint runtime identity does not match")
    graph = _graph_from_dict(payload["graph"])
    best_id = payload["best_node_id"]
    method._graph = graph
    method._best_node = None if best_id is None else graph.get_node(int(best_id))
    method._best_node_sample_order = payload["best_node_sample_order"]
    method._tot_sample_nums = int(payload["total_budget_events"])
    method._evaluation_count = int(payload["total_evaluations"])
    method._next_iteration = int(payload["next_iteration"])
    # 中止标志与连续失败计数是运行时终止状态，不随 checkpoint 恢复：
    # resume 表示从中断处继续搜索，恢复时应从零重新计数，否则会立即再次中止。
    method._consecutive_sample_failures = 0
    method._search_aborted = False
    method._initialization_complete = bool(payload["initialization_complete"])
    method._initial_strategy_cards = tuple(payload["initial_strategy_cards"])
    method._root_strategy_cards = {
        int(node_id): str(strategy)
        for node_id, strategy in payload["root_strategy_cards"].items()
    }
    method._strategy_planning_calls = int(payload["strategy_planning_calls"])
    method._bootstrapped_root_ids = {
        int(node_id) for node_id in payload["bootstrapped_root_ids"]
    }
    method._initial_selected_anchor_ids = tuple(
        int(node_id) for node_id in payload["initial_selected_anchor_ids"]
    )
    method._eligible_node_ids = {
        int(node_id) for node_id in payload["eligible_node_ids"]
    }
    method._failure_history = [
        FailureObservation(
            failure_kind=str(item["failure_kind"]),
            error_type=item["error_type"],
            error_message=item["error_message"],
            budget_order=int(item["budget_order"]),
        )
        for item in payload["failure_history"]
    ]
    if method._artifacts is not None:
        method._artifacts.sync_after_resume(
            total_samples=method._tot_sample_nums,
            best_score=None if method._best_node is None else method._best_node.fitness,
            best_sample_order=method._best_node_sample_order,
        )


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    _atomic_write(latest, dump_state(method))
    method._last_checkpoint_budget = method._tot_sample_nums
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    method._last_checkpoint_budget = method._tot_sample_nums
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "dump_state",
    "load_checkpoint",
    "load_state",
    "save_checkpoint",
]
