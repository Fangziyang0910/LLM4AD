"""Exact state persistence for TraceAAD V9.17."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import (
    BlockKind,
    BlockState,
    GenerationState,
    Hypothesis,
    HypothesisStatus,
    Pending,
    Phase,
)
from .tree import Tree


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    method._assert_invariants()
    path = Path(target) / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "version": "v9_17",
        "config": {
            "budget": method._budget,
            "n_roots": method._n_roots,
            "max_history": method._max_history,
            "max_tokens": method._max_tokens,
            "maximize": method._tree.maximize,
            "seed": method._seed,
            "error_retries": method._error_retries,
            "error_handling": method._error_handling,
            "scheduler": (
                "adaptive_gain_continuation"
                if method._adaptive_sweeps
                else "fixed_cycle"
            ),
        },
        "tree": method._tree.to_dict(),
        "hypotheses": [asdict(item) for item in method._hypotheses.values()],
        "active_ids": method._active_ids,
        "reserve_ids": method._reserve_ids,
        "phase": method._phase.value,
        "generation": (
            None if method._generation is None else asdict(method._generation)
        ),
        "pending": None if method._pending is None else asdict(method._pending),
        "n_eval": method._n_eval,
        "n_llm_calls": method._n_llm_calls,
        "repair_llm_calls": method._repair_llm_calls,
        "n_calls": method._n_calls,
        "root_slots": method._root_slots,
        "refine_slots": method._refine_slots,
        "explore_slots": method._explore_slots,
        "next_hypothesis_id": method._next_hypothesis_id,
        "next_block_id": method._next_block_id,
        "initial_order": method._initial_order,
        "initial_cursor": method._initial_cursor,
        "bootstrap_deltas": method._bootstrap_deltas,
        "s_r": method._s_r,
        "s_r_frozen": method._s_r_frozen,
        "cycle": method._cycle,
        "sweep": method._sweep,
        "eligible_ids": method._eligible_ids,
        "sweep_order": method._sweep_order,
        "sweep_cursor": method._sweep_cursor,
        "successful_ids": method._successful_ids,
        "active_block": (
            None if method._active_block is None else asdict(method._active_block)
        ),
        "terminal_after_block": method._terminal_after_block,
        "discovery_attempted": method._discovery_attempted,
        "discovery_source_id": method._discovery_source_id,
        "discovery_candidate_hypothesis_id": (
            method._discovery_candidate_hypothesis_id
        ),
        "maturing_hypothesis_id": method._maturing_hypothesis_id,
        "discovery_attempts": method._discovery_attempts,
        "valid_discoveries": method._valid_discoveries,
        "block_counts": method._block_counts,
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return path


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("version") != "v9_17":
        raise ValueError("checkpoint is not a TraceAAD V9.17 checkpoint")
    expected = {
        "budget": method._budget,
        "n_roots": method._n_roots,
        "max_history": method._max_history,
        "max_tokens": method._max_tokens,
        "maximize": method._tree.maximize,
        "seed": method._seed,
        "error_retries": method._error_retries,
        "error_handling": method._error_handling,
    }
    actual = dict(state.get("config", {}))
    checkpoint_scheduler = actual.pop("scheduler", "adaptive_gain_continuation")
    expected_scheduler = (
        "adaptive_gain_continuation" if method._adaptive_sweeps else "fixed_cycle"
    )
    if actual != expected:
        raise ValueError("TraceAAD V9.17 checkpoint configuration mismatch")
    if checkpoint_scheduler != expected_scheduler:
        if not (
            method._allow_scheduler_fork
            and checkpoint_scheduler == "adaptive_gain_continuation"
            and expected_scheduler == "fixed_cycle"
            and _is_initialization_fork_state(state)
        ):
            raise ValueError("TraceAAD V9.17 checkpoint scheduler mismatch")

    method._tree = Tree.from_dict(state["tree"])
    method._hypotheses = {}
    for payload in state["hypotheses"]:
        payload["status"] = HypothesisStatus(payload["status"])
        hypothesis = Hypothesis(**payload)
        method._hypotheses[hypothesis.id] = hypothesis
    method._active_ids = [int(item) for item in state["active_ids"]]
    method._reserve_ids = [int(item) for item in state["reserve_ids"]]
    method._phase = Phase(state["phase"])
    generation = state.get("generation")
    method._generation = (
        None if generation is None else GenerationState(**generation)
    )
    pending = state.get("pending")
    method._pending = None if pending is None else Pending(**pending)
    method._n_eval = int(state["n_eval"])
    method._n_llm_calls = int(state["n_llm_calls"])
    method._repair_llm_calls = int(state["repair_llm_calls"])
    method._n_calls = int(state["n_calls"])
    method._root_slots = int(state["root_slots"])
    method._refine_slots = int(state["refine_slots"])
    method._explore_slots = int(state["explore_slots"])
    method._next_hypothesis_id = int(state["next_hypothesis_id"])
    method._next_block_id = int(state["next_block_id"])
    method._initial_order = [int(item) for item in state["initial_order"]]
    method._initial_cursor = int(state["initial_cursor"])
    method._bootstrap_deltas = [float(item) for item in state["bootstrap_deltas"]]
    method._s_r = float(state["s_r"])
    method._s_r_frozen = bool(state["s_r_frozen"])
    method._cycle = int(state["cycle"])
    method._sweep = int(state["sweep"])
    method._eligible_ids = [int(item) for item in state["eligible_ids"]]
    method._sweep_order = [int(item) for item in state["sweep_order"]]
    method._sweep_cursor = int(state["sweep_cursor"])
    method._successful_ids = [int(item) for item in state["successful_ids"]]
    active_block = state.get("active_block")
    if active_block is None:
        method._active_block = None
    else:
        active_block["kind"] = BlockKind(active_block["kind"])
        method._active_block = BlockState(**active_block)
    method._terminal_after_block = bool(state["terminal_after_block"])
    method._discovery_attempted = bool(state["discovery_attempted"])
    method._discovery_source_id = state.get("discovery_source_id")
    method._discovery_candidate_hypothesis_id = state.get(
        "discovery_candidate_hypothesis_id"
    )
    method._maturing_hypothesis_id = state.get("maturing_hypothesis_id")
    method._discovery_attempts = int(state["discovery_attempts"])
    method._valid_discoveries = int(state["valid_discoveries"])
    method._block_counts = {
        str(key): int(value) for key, value in state["block_counts"].items()
    }
    method._assert_invariants()
    return checkpoint


def _is_initialization_fork_state(state: dict[str, object]) -> bool:
    initial_order = state.get("initial_order")
    return bool(
        state.get("phase") == "development"
        and state.get("s_r_frozen") is True
        and state.get("cycle") == 1
        and state.get("sweep") == 1
        and isinstance(initial_order, list)
        and state.get("initial_cursor") == len(initial_order)
        and state.get("eligible_ids") == state.get("active_ids")
        and state.get("sweep_order") == []
        and state.get("sweep_cursor") == 0
        and state.get("successful_ids") == []
        and state.get("active_block") is None
        and state.get("generation") is None
        and state.get("pending") is None
        and state.get("discovery_attempted") is False
    )


__all__ = ["load_checkpoint", "save_checkpoint"]
