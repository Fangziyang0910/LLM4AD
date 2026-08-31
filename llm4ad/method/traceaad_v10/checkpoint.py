"""Atomic checkpoint persistence for TraceAAD V10."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .schema import AttemptRecord, Pending, ProgramNode, Thread

CHECKPOINT_VERSION = "v10"
CHECKPOINT_NAME = "latest.json"
VIEW_NAME = "view.json"


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    target.mkdir(parents=True, exist_ok=True)
    state = {
        "version": CHECKPOINT_VERSION,
        "task": method._task_key,
        "nodes": [asdict(node) for node in method._nodes.values()],
        "threads": [asdict(thread) for thread in method._threads.values()],
        "attempts": [asdict(item) for item in method._attempts],
        "global_memory": list(method._global_memory),
        "pending": None if method._pending is None else asdict(method._pending),
        "slot_best": list(method._slot_best),
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "round_index": method._round_index,
        "gen_llm_calls": method._gen_llm_calls,
        "critic_llm_calls": method._critic_llm_calls,
        "critic_invalid": method._critic_invalid,
        "repair_llm_calls": method._repair_llm_calls,
        "repair_eval_calls": method._repair_eval_calls,
        "draw_index": method._draw_index,
        "llm_tokens": dict(method._llm_tokens),
        "token_accounting_failures": method._token_accounting_failures,
        "best_node_id": method._best_node_id,
        "rng_state": method._rng.getstate(),
    }
    path = target / CHECKPOINT_NAME
    _atomic_write(path, json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    _atomic_write(target / VIEW_NAME, json.dumps(_view_state(method), ensure_ascii=False) + "\n")
    return path


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint is not a TraceAAD V10 checkpoint")
    if state.get("task") != method._task_key:
        raise ValueError("checkpoint task does not match the run")
    method._nodes = {
        int(item["id"]): ProgramNode(**item) for item in state.get("nodes", [])
    }
    method._threads = {
        int(item["id"]): Thread(**item) for item in state.get("threads", [])
    }
    method._attempts = [AttemptRecord(**item) for item in state.get("attempts", [])]
    pending = state.get("pending")
    method._pending = None if pending is None else Pending(**pending)
    method._global_memory = [int(item) for item in state.get("global_memory", [])]
    method._slot_best = list(state.get("slot_best", []))
    method._n_eval = int(state.get("n_eval", 0))
    method._n_calls = int(state.get("n_calls", 0))
    method._round_index = int(state.get("round_index", 0))
    method._gen_llm_calls = int(state.get("gen_llm_calls", 0))
    method._critic_llm_calls = int(state.get("critic_llm_calls", 0))
    method._critic_invalid = int(state.get("critic_invalid", 0))
    method._repair_llm_calls = int(state.get("repair_llm_calls", 0))
    method._repair_eval_calls = int(state.get("repair_eval_calls", 0))
    method._draw_index = int(state.get("draw_index", 0))
    stored_tokens = state.get("llm_tokens")
    if isinstance(stored_tokens, dict):
        method._llm_tokens = {str(key): int(value) for key, value in stored_tokens.items()}
    method._token_accounting_failures = int(state.get("token_accounting_failures", 0))
    best = state.get("best_node_id")
    method._best_node_id = None if best is None else int(best)
    rng_state = state.get("rng_state")
    if rng_state is not None:
        method._rng.setstate(_to_tuple(rng_state))
    if method._nodes and method._best_node_id not in method._nodes:
        raise ValueError("checkpoint best node is missing")
    return checkpoint


def _view_state(method) -> dict[str, object]:
    return {
        "version": CHECKPOINT_VERSION,
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "round_index": method._round_index,
        "critic_invalid": method._critic_invalid,
        "best_node_id": method._best_node_id,
        "nodes": [
            {
                "id": node.id,
                "fitness": node.fitness,
                "parent_id": node.parent_id,
                "thread_id": node.thread_id,
                "idea": node.idea,
                "slot": node.slot,
            }
            for node in method._nodes.values()
        ],
        "threads": [asdict(thread) for thread in method._threads.values()],
    }


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _to_tuple(value):
    if isinstance(value, list):
        return tuple(_to_tuple(item) for item in value)
    return value


__all__ = ["CHECKPOINT_NAME", "CHECKPOINT_VERSION", "VIEW_NAME", "load_checkpoint", "save_checkpoint"]
