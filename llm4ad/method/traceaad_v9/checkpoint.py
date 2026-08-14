"""Strict checkpoint persistence for TraceAAD V9."""

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
from .tree import SearchTree, is_node_better

CHECKPOINT_VERSION = 1


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
        visit_count=int(root["visit_count"]),
        subtree_value=(
            None if root["subtree_value"] is None else float(root["subtree_value"])
        ),
        subtree_best_node_id=(
            None
            if root["subtree_best_node_id"] is None
            else int(root["subtree_best_node_id"])
        ),
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
            visit_count=int(item["visit_count"]),
            expansion_count=int(item["expansion_count"]),
            subtree_value=float(item["subtree_value"]),
            subtree_best_node_id=int(item["subtree_best_node_id"]),
            creation_order=int(item["creation_order"]),
            batch_id=None if item["batch_id"] is None else int(item["batch_id"]),
            operator=str(item["operator"]),
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
            reference_root_branch_id=(
                None
                if item["reference_root_branch_id"] is None
                else int(item["reference_root_branch_id"])
            ),
            delta_parent=float(item["delta_parent"]),
            delta_global_best=(
                None
                if item["delta_global_best"] is None
                else float(item["delta_global_best"])
            ),
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
    if tree.root.visit_count < len(tree.root.child_ids):
        raise ValueError("checkpoint root visits are inconsistent with initialization")
    if len(set(tree.root.child_ids)) != len(tree.root.child_ids):
        raise ValueError("checkpoint root contains duplicate children")
    if tree._nodes and tree._next_node_id <= max(tree._nodes):
        raise ValueError("checkpoint next_node_id does not advance past nodes")
    if tree._edges and tree._next_edge_id <= max(tree._edges):
        raise ValueError("checkpoint next_edge_id does not advance past edges")

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
        if (
            edge.reference_root_branch_id is not None
            and edge.reference_root_branch_id not in tree.root.child_ids
        ):
            raise ValueError(f"checkpoint edge {edge.id} has unknown reference branch")

    seen: set[int] = set()

    def visit(node_id: int, parent_id: int, depth: int) -> ProgramNode:
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
        if node.visit_count < 1:
            raise ValueError(f"checkpoint node {node.id} has invalid visit count")
        if node.expansion_count < 0 or node.expansion_count > node.visit_count - 1:
            raise ValueError(f"checkpoint node {node.id} has invalid expansion count")
        successful_batches = {
            tree.get_node(child_id).batch_id for child_id in node.child_ids
        }
        if None in successful_batches or len(successful_batches) > node.expansion_count:
            raise ValueError(
                f"checkpoint node {node.id} has inconsistent expansion batches"
            )
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
        if parent_id == tree.root.id:
            if node.incoming_edge_id is not None:
                raise ValueError("checkpoint root child must not have an incoming edge")
        else:
            if (
                node.incoming_edge_id is None
                or node.incoming_edge_id not in tree._edges
            ):
                raise ValueError(
                    f"checkpoint node {node.id} has no valid incoming edge"
                )
            edge = tree.get_edge(node.incoming_edge_id)
            if edge.parent_id != parent_id or edge.child_id != node.id:
                raise ValueError(f"checkpoint node {node.id} has a misaligned edge")
            if node.batch_id != edge.batch_id:
                raise ValueError(
                    f"checkpoint node {node.id} has a misaligned expansion batch"
                )
        best = node
        for child_id in node.child_ids:
            child_best = visit(child_id, node.id, depth + 1)
            if is_node_better(child_best, best):
                best = child_best
        if (
            node.subtree_value != best.directed_fitness
            or node.subtree_best_node_id != best.id
        ):
            raise ValueError(f"checkpoint node {node.id} has invalid subtree backup")
        return best

    root_best: ProgramNode | None = None
    for root_child_id in tree.root.child_ids:
        candidate = visit(root_child_id, tree.root.id, 1)
        if is_node_better(candidate, root_best):
            root_best = candidate
    if seen != set(tree._nodes):
        raise ValueError("checkpoint contains disconnected program nodes")
    if len(incoming_children) != max(0, len(tree._nodes) - len(tree.root.child_ids)):
        raise ValueError("checkpoint structural edge count is inconsistent")
    expected_value = None if root_best is None else root_best.directed_fitness
    expected_id = None if root_best is None else root_best.id
    if (
        tree.root.subtree_value != expected_value
        or tree.root.subtree_best_node_id != expected_id
    ):
        raise ValueError("checkpoint virtual root has invalid subtree backup")
    dual = {OperatorName.SYNTHESIZE, OperatorName.TRANSFER}
    for edge in tree.edges():
        has_reference = edge.reference_node_id is not None
        if (edge.operator in dual) != has_reference:
            raise ValueError(
                f"checkpoint edge {edge.id} has invalid reference provenance"
            )
        if not has_reference:
            if edge.reference_root_branch_id is not None:
                raise ValueError(
                    f"checkpoint edge {edge.id} has a stray reference branch"
                )
            continue
        if edge.reference_root_branch_id is None:
            raise ValueError(f"checkpoint edge {edge.id} has no reference branch")
        reference = tree.get_node(edge.reference_node_id)
        parent = tree.get_node(edge.parent_id)
        if tree.root_branch_id(reference.id) != edge.reference_root_branch_id:
            raise ValueError(
                f"checkpoint edge {edge.id} has a misaligned reference branch"
            )
        if tree.root_branch_id(parent.id) == edge.reference_root_branch_id:
            raise ValueError(
                f"checkpoint edge {edge.id} references its own root branch"
            )
        if reference.code_hash == parent.code_hash:
            raise ValueError(f"checkpoint edge {edge.id} references identical code")


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
        raise ValueError("unsupported TraceAAD V9-Core checkpoint version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint protocol_id does not match TraceAAD V9-Core")
    if payload.get("search_configuration") != method.search_configuration():
        raise ValueError("checkpoint search configuration does not match TraceAAD V9-Core")
    if payload.get("runtime_identity") != method.runtime_identity():
        raise ValueError("checkpoint runtime identity does not match TraceAAD V9-Core")
    tree = _tree_from_dict(payload["tree"], maximize=method._maximize)
    best_id = payload["best_node_id"]
    best = None if best_id is None else tree.get_node(int(best_id))
    expected_best = (
        None
        if tree.root.subtree_best_node_id is None
        else tree.get_node(tree.root.subtree_best_node_id)
    )
    if (best is None) != (expected_best is None) or (
        best is not None and expected_best is not None and best.id != expected_best.id
    ):
        raise ValueError("checkpoint global best is inconsistent with the tree")
    method._tree = tree
    method._best_node = best
    method._best_node_sample_order = (
        None
        if payload["best_node_sample_order"] is None
        else int(payload["best_node_sample_order"])
    )
    method._tot_sample_nums = int(payload["total_samples"])
    if method._tot_sample_nums < len(tree.nodes()):
        raise ValueError("checkpoint sample count is smaller than its valid-node count")
    if (
        method._best_node_sample_order is not None
        and not 1 <= method._best_node_sample_order <= method._tot_sample_nums
    ):
        raise ValueError("checkpoint best sample order is outside the evaluator budget")
    method._next_attempt_id = int(payload["next_attempt_id"])
    method._batch_count = int(payload["batch_count"])
    method._stalled_iterations = int(payload["stalled_iterations"])
    method._consecutive_sample_failures = int(payload["consecutive_sample_failures"])
    method._search_aborted = bool(payload["search_aborted"])
    method._initialization_complete = bool(payload["initialization_complete"])
    method._rng.setstate(_as_tuple(payload["rng_state"]))
    artifacts = method._artifacts
    if artifacts is not None:
        artifacts.sync_after_resume(
            total_samples=method._tot_sample_nums,
            best_score=None if best is None else best.fitness,
            best_sample_order=method._best_node_sample_order,
        )


def _as_tuple(value):
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    return value


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
