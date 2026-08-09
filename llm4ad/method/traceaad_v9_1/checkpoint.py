"""Strict checkpoint persistence for trajectory-centred TraceAAD V9.1."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .complexity import code_hash, nonempty_loc
from .schema import ImprovementEdge, OperatorName, ProgramNode, PROTOCOL_ID, VirtualRoot
from .tree import SearchTree

CHECKPOINT_VERSION = 3


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


def _tree_to_dict(tree: SearchTree) -> dict[str, Any]:
    return {
        "root": asdict(tree.root),
        "next_node_id": tree._next_node_id,
        "next_edge_id": tree._next_edge_id,
        "nodes": [asdict(node) for node in tree.nodes()],
        "edges": [asdict(edge) for edge in tree.edges()],
    }


def _tree_from_dict(payload: Mapping[str, Any], *, maximize: bool) -> SearchTree:
    tree = SearchTree()
    root = payload["root"]
    tree.root = VirtualRoot(
        id=int(root["id"]),
        child_ids=[int(item) for item in root["child_ids"]],
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
            incoming_edge_id=(
                None
                if item["incoming_edge_id"] is None
                else int(item["incoming_edge_id"])
            ),
            child_ids=[int(child) for child in item["child_ids"]],
            depth=int(item["depth"]),
            creation_order=int(item["creation_order"]),
            batch_id=None if item["batch_id"] is None else int(item["batch_id"]),
            operator=str(item["operator"]),
            bootstrap_reference_node_ids=[
                int(node_id) for node_id in item["bootstrap_reference_node_ids"]
            ],
            trajectory_best_value=float(item["trajectory_best_value"]),
            trajectory_best_node_id=int(item["trajectory_best_node_id"]),
            verification_count=int(item["verification_count"]),
            valid_candidate_count=int(item["valid_candidate_count"]),
            route_advance_count=int(item["route_advance_count"]),
            global_advance_count=int(item["global_advance_count"]),
            recent_advances=[bool(value) for value in item["recent_advances"]],
            last_verification_batch_id=(
                None
                if item["last_verification_batch_id"] is None
                else int(item["last_verification_batch_id"])
            ),
        )
        if node.id in tree._nodes:
            raise ValueError(f"checkpoint contains duplicate node id {node.id}")
        tree._nodes[node.id] = node
    for item in payload["edges"]:
        edge = ImprovementEdge(
            id=int(item["id"]),
            parent_id=int(item["parent_id"]),
            child_id=int(item["child_id"]),
            operator=OperatorName(item["operator"]),
            implemented_idea=str(item["implemented_idea"]),
            reference_node_id=(
                None
                if item["reference_node_id"] is None
                else int(item["reference_node_id"])
            ),
            delta_parent=float(item["delta_parent"]),
            delta_global_best=(
                None
                if item["delta_global_best"] is None
                else float(item["delta_global_best"])
            ),
            trajectory_best_before=float(item["trajectory_best_before"]),
            advances_parent_trajectory=bool(item["advances_parent_trajectory"]),
            outcome=str(item["outcome"]),
            delta_loc=int(item["delta_loc"]),
            code_change_ratio=float(item["code_change_ratio"]),
            new_global_best=bool(item["new_global_best"]),
            global_best_update_reason=item["global_best_update_reason"],
            iteration=int(item["iteration"]),
            batch_id=int(item["batch_id"]),
            sibling_seq=int(item["sibling_seq"]),
            sample_order=int(item["sample_order"]),
        )
        if edge.id in tree._edges:
            raise ValueError(f"checkpoint contains duplicate edge id {edge.id}")
        tree._edges[edge.id] = edge
    tree._next_node_id = int(payload["next_node_id"])
    tree._next_edge_id = int(payload["next_edge_id"])
    validate_tree(tree, maximize=maximize)
    return tree


def validate_tree(tree: SearchTree, *, maximize: bool) -> None:
    if tree.root.id != -1:
        raise ValueError("checkpoint virtual root id must be -1")
    if len(set(tree.root.child_ids)) != len(tree.root.child_ids):
        raise ValueError("checkpoint root contains duplicate children")
    if tree._nodes and tree._next_node_id <= max(tree._nodes):
        raise ValueError("checkpoint next_node_id does not advance past nodes")
    if tree._edges and tree._next_edge_id <= max(tree._edges):
        raise ValueError("checkpoint next_edge_id does not advance past edges")
    creation_orders = [node.creation_order for node in tree.nodes()]
    if len(set(creation_orders)) != len(creation_orders):
        raise ValueError("checkpoint contains duplicate evaluator sample orders")

    incoming_children: set[int] = set()
    for edge in tree.edges():
        if edge.parent_id not in tree._nodes or edge.child_id not in tree._nodes:
            raise ValueError(f"checkpoint edge {edge.id} references an unknown node")
        if edge.child_id in incoming_children:
            raise ValueError(f"checkpoint node {edge.child_id} has multiple parents")
        incoming_children.add(edge.child_id)
        if (
            edge.reference_node_id is not None
            and edge.reference_node_id not in tree._nodes
        ):
            raise ValueError(f"checkpoint edge {edge.id} has unknown reference node")

    seen: set[int] = set()

    def visit(
        node_id: int,
        parent_id: int,
        depth: int,
        path_best_value: float | None,
        path_best_id: int | None,
    ) -> None:
        if node_id in seen:
            raise ValueError(
                "checkpoint tree contains a cycle or duplicate structural child"
            )
        if node_id not in tree._nodes:
            raise ValueError(f"checkpoint references unknown node {node_id}")
        seen.add(node_id)
        node = tree.get_node(node_id)
        if node.parent_id != parent_id or node.depth != depth:
            raise ValueError(f"checkpoint node {node.id} has invalid parent or depth")
        expected_directed = node.fitness if maximize else -node.fitness
        if (
            not math.isfinite(node.fitness)
            or node.directed_fitness != expected_directed
        ):
            raise ValueError(f"checkpoint node {node.id} has invalid fitness")
        if node.code_hash != code_hash(node.code) or node.program_loc != nonempty_loc(
            node.code
        ):
            raise ValueError(f"checkpoint node {node.id} has invalid code metadata")
        if len(set(node.child_ids)) != len(node.child_ids):
            raise ValueError(f"checkpoint node {node.id} contains duplicate children")
        expected_value = (
            node.directed_fitness if path_best_value is None else path_best_value
        )
        expected_id = node.id if path_best_id is None else path_best_id
        if (
            path_best_value is not None
            and node.directed_fitness > path_best_value + 1e-6
        ):
            expected_value, expected_id = node.directed_fitness, node.id
        if (
            node.trajectory_best_value != expected_value
            or node.trajectory_best_node_id != expected_id
        ):
            raise ValueError(f"checkpoint node {node.id} has invalid trajectory best")
        counts = (
            node.verification_count,
            node.valid_candidate_count,
            node.route_advance_count,
            node.global_advance_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError(f"checkpoint node {node.id} has negative evidence counts")
        if (
            node.route_advance_count > node.verification_count
            or node.global_advance_count > node.verification_count
        ):
            raise ValueError(
                f"checkpoint node {node.id} has inconsistent advance counts"
            )
        if node.valid_candidate_count < node.route_advance_count:
            raise ValueError(
                f"checkpoint node {node.id} has inconsistent valid-candidate count"
            )
        if len(node.recent_advances) > node.verification_count:
            raise ValueError(f"checkpoint node {node.id} has excess recent evidence")
        if (node.verification_count == 0) != (node.last_verification_batch_id is None):
            raise ValueError(f"checkpoint node {node.id} has inconsistent last batch")
        if (
            node.last_verification_batch_id is not None
            and node.last_verification_batch_id <= 0
        ):
            raise ValueError(f"checkpoint node {node.id} has invalid last batch")
        if parent_id == tree.root.id:
            if node.incoming_edge_id is not None or node.batch_id is not None:
                raise ValueError(
                    "checkpoint initial trajectory has expansion provenance"
                )
            for reference_id in node.bootstrap_reference_node_ids:
                if reference_id not in tree.root.child_ids or reference_id >= node.id:
                    raise ValueError(
                        f"checkpoint node {node.id} has invalid bootstrap history"
                    )
        else:
            if node.bootstrap_reference_node_ids:
                raise ValueError(f"checkpoint child {node.id} has bootstrap provenance")
            if (
                node.incoming_edge_id is None
                or node.incoming_edge_id not in tree._edges
            ):
                raise ValueError(
                    f"checkpoint node {node.id} has no valid incoming edge"
                )
            edge = tree.get_edge(node.incoming_edge_id)
            if (
                edge.parent_id != parent_id
                or edge.child_id != node.id
                or node.batch_id != edge.batch_id
            ):
                raise ValueError(
                    f"checkpoint node {node.id} has misaligned edge provenance"
                )
            parent = tree.get_node(parent_id)
            if edge.trajectory_best_before != parent.trajectory_best_value:
                raise ValueError(
                    f"checkpoint edge {edge.id} has invalid trajectory baseline"
                )
            expected_advance = (
                node.directed_fitness > parent.trajectory_best_value + 1e-6
            )
            if edge.advances_parent_trajectory != expected_advance:
                raise ValueError(
                    f"checkpoint edge {edge.id} has invalid trajectory outcome"
                )
        for child_id in node.child_ids:
            visit(child_id, node.id, depth + 1, expected_value, expected_id)

    for root_child_id in tree.root.child_ids:
        visit(root_child_id, tree.root.id, 1, None, None)
    if seen != set(tree._nodes):
        raise ValueError("checkpoint contains disconnected program nodes")
    if len(incoming_children) != max(0, len(tree._nodes) - len(tree.root.child_ids)):
        raise ValueError("checkpoint structural edge count is inconsistent")

    dual = {OperatorName.SYNTHESIZE, OperatorName.TRANSFER}
    for edge in tree.edges():
        has_reference = edge.reference_node_id is not None
        if (edge.operator in dual) != has_reference:
            raise ValueError(
                f"checkpoint edge {edge.id} has invalid reference provenance"
            )
        if has_reference:
            parent = tree.get_node(edge.parent_id)
            reference = tree.get_node(edge.reference_node_id)  # type: ignore[arg-type]
            if tree.same_lineage(parent.id, reference.id):
                raise ValueError(
                    f"checkpoint edge {edge.id} references the same lineage"
                )
            if parent.code_hash == reference.code_hash:
                raise ValueError(f"checkpoint edge {edge.id} references identical code")
            if reference.id >= edge.child_id:
                raise ValueError(
                    f"checkpoint edge {edge.id} references future evidence"
                )


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
        "tree": _tree_to_dict(method._tree),
        "rng_state": method._rng.getstate(),
        "search_configuration": method.search_configuration(),
        "runtime_identity": method.runtime_identity(),
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError("unsupported TraceAAD V9.1 checkpoint version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint protocol_id does not match TraceAAD V9.1")
    if payload.get("search_configuration") != method.search_configuration():
        raise ValueError("checkpoint search configuration does not match TraceAAD V9.1")
    if payload.get("runtime_identity") != method.runtime_identity():
        raise ValueError("checkpoint runtime identity does not match TraceAAD V9.1")
    tree = _tree_from_dict(payload["tree"], maximize=method._maximize)
    best_id = payload["best_node_id"]
    best = None if best_id is None else tree.get_node(int(best_id))
    expected_best = tree.best_node()
    if (best is None) != (expected_best is None) or (
        best is not None and expected_best is not None and best.id != expected_best.id
    ):
        raise ValueError("checkpoint global best is inconsistent with the trajectories")
    best_node_sample_order = (
        None
        if payload["best_node_sample_order"] is None
        else int(payload["best_node_sample_order"])
    )
    total_samples = int(payload["total_samples"])
    if total_samples < len(tree.nodes()):
        raise ValueError("checkpoint sample count is smaller than its valid-node count")
    if (
        best_node_sample_order is not None
        and not 1 <= best_node_sample_order <= total_samples
    ):
        raise ValueError("checkpoint best sample order is outside the evaluator budget")
    batch_count = int(payload["batch_count"])
    for node in tree.nodes():
        if (
            node.valid_candidate_count
            > node.verification_count * method._verification_batch_size
        ):
            raise ValueError(
                f"checkpoint node {node.id} exceeds verification batch capacity"
            )
        if (
            node.last_verification_batch_id is not None
            and node.last_verification_batch_id > batch_count
        ):
            raise ValueError(f"checkpoint node {node.id} references a future batch")
    if best is not None and best_node_sample_order != best.creation_order:
        raise ValueError("checkpoint best sample order does not identify the best node")
    method._tree = tree
    method._best_node = best
    method._best_node_sample_order = best_node_sample_order
    method._tot_sample_nums = total_samples
    method._next_attempt_id = int(payload["next_attempt_id"])
    method._batch_count = batch_count
    method._stalled_iterations = int(payload["stalled_iterations"])
    method._consecutive_sample_failures = int(payload["consecutive_sample_failures"])
    method._search_aborted = bool(payload["search_aborted"])
    method._initialization_complete = bool(payload["initialization_complete"])
    method._rng.setstate(_as_tuple(payload["rng_state"]))
    artifacts = method._artifacts
    if artifacts is not None and hasattr(artifacts, "sync_after_resume"):
        artifacts.sync_after_resume(
            total_samples=method._tot_sample_nums,
            best_score=None if best is None else best.fitness,
            best_sample_order=method._best_node_sample_order,
        )


def _as_tuple(value):
    return (
        tuple(_as_tuple(item) for item in value) if isinstance(value, list) else value
    )


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
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
    "validate_tree",
]
