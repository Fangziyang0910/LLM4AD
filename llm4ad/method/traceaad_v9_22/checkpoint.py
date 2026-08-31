"""Atomic checkpoint persistence for V9.22."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .schema import Attempt, Hypothesis, Pending, ProgramNode, Realization

CHECKPOINT_VERSION = "v9_22"
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
        "hypotheses": [asdict(h) for h in method._hypotheses.values()],
        "realizations": [asdict(item) for item in method._realizations],
        "attempts": [asdict(item) for item in method._attempts],
        "global_memory": list(method._global_memory),
        "pending": None if method._pending is None else asdict(method._pending),
        "batch_context": method._batch_context,
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "llm_calls": method._llm_calls,
        "repair_llm_calls": method._repair_llm_calls,
        "repair_eval_calls": method._repair_eval_calls,
        "batch_index": method._batch_index,
        "action_stats": method._action_stats,
        "quality_rank_count": method._quality_rank_count,
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
        raise ValueError("checkpoint is not a TraceAAD V9.22 checkpoint")
    if state.get("task") != method._task_key:
        raise ValueError("checkpoint task does not match the run")
    method._nodes = {
        int(item["id"]): ProgramNode(**item) for item in state.get("nodes", [])
    }
    method._hypotheses = {
        int(item["id"]): Hypothesis(**item)
        for item in state.get("hypotheses", [])
    }
    method._realizations = [Realization(**item) for item in state.get("realizations", [])]
    method._attempts = [Attempt(**item) for item in state.get("attempts", [])]
    pending = state.get("pending")
    method._pending = None if pending is None else Pending(**pending)
    method._batch_context = state.get("batch_context")
    method._global_memory = [int(item) for item in state.get("global_memory", [])]
    method._n_eval = int(state.get("n_eval", 0))
    method._n_calls = int(state.get("n_calls", 0))
    method._repair_llm_calls = int(state.get("repair_llm_calls", 0))
    method._repair_eval_calls = int(state.get("repair_eval_calls", 0))
    method._batch_index = int(state.get("batch_index", 0))
    stored_action_stats = state.get("action_stats")
    if isinstance(stored_action_stats, dict):
        method._action_stats = {
            str(action): {str(key): float(value) for key, value in values.items()}
            for action, values in stored_action_stats.items()
            if isinstance(values, dict)
        }
    method._quality_rank_count = int(state.get("quality_rank_count", 0))
    best = state.get("best_node_id")
    method._best_node_id = None if best is None else int(best)
    method._llm_calls = int(state.get("llm_calls", method._n_calls))
    rng_state = state.get("rng_state")
    if rng_state is not None:
        method._rng.setstate(_to_tuple(rng_state))
    if method._nodes and method._best_node_id not in method._nodes:
        raise ValueError("checkpoint best node is missing")
    return checkpoint


def _view_state(method) -> dict[str, object]:
    nodes = []
    for node in method._nodes.values():
        nodes.append(
            {
                "id": node.id,
                "fitness": node.fitness,
                "parent_id": node.parent_id,
                "hypothesis_id": node.hypothesis_id,
                "idea": node.idea,
                "role": node.role,
                "slot": node.slot,
            }
        )
    hypotheses = []
    for hypothesis in method._hypotheses.values():
        hypotheses.append(
            {
                "id": hypothesis.id,
                "entry_idea": hypothesis.entry_idea,
                "scaffold_node_id": hypothesis.scaffold_node_id,
                "working_node_id": hypothesis.working_node_id,
                "trials": hypothesis.trials,
                "response_working_mean": hypothesis.response_working_mean,
                "response_scaffold_mean": hypothesis.response_scaffold_mean,
                "usable_trials": hypothesis.usable_trials,
                "working_improvements": hypothesis.working_improvements,
                "scaffold_improvements": hypothesis.scaffold_improvements,
                "last_batch": hypothesis.last_batch,
            }
        )
    return {
        "version": CHECKPOINT_VERSION,
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "best_node_id": method._best_node_id,
        "nodes": nodes,
        "hypotheses": hypotheses,
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
