"""Versioned checkpoint persistence for TraceAAD V9.18-R0."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .schema import Intent, Pending
from .selection import Decision
from .tree import Tree

CHECKPOINT_VERSION = "v9_18_r0"


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    path = Path(target) / "latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "allocation_mode": method._allocation_mode,
        "explore_context": method._explore_context,
        "tree": method._tree.to_dict(),
        "pending": None if method._pending is None else asdict(method._pending),
        "decision": _decision_payload(method._decision),
        "n_eval": method._n_eval,
        "n_calls": method._n_calls,
        "n_stag": method._n_stag,
        "attempt": method._attempt_number,
        "attempt_kind": method._attempt_kind,
        "ordinary_decisions": method._n_ordinary_decisions,
        "next_entry_id": method._next_entry_id,
        "sigma_q": method._sigma_q,
        "last_best_eval": method._last_best_eval,
        "recent_outcomes": list(method._recent_outcomes),
    }
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return path


def load_checkpoint(
    method, path: str | Path, *, allow_protocol_mismatch: bool = False
) -> Path:
    checkpoint = Path(path)
    state = json.loads(checkpoint.read_text(encoding="utf-8"))
    if state.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            "checkpoint is not a TraceAAD V9.18-R0 checkpoint; "
            "V9.16 landing checkpoints are not compatible"
        )
    if not allow_protocol_mismatch and state.get("allocation_mode") != method._allocation_mode:
        raise ValueError("checkpoint allocation mode does not match the run")
    if not allow_protocol_mismatch and state.get("explore_context") != method._explore_context:
        raise ValueError("checkpoint explore context does not match the run")
    method._tree = Tree.from_dict(state["tree"])
    method._decision = _restore_decision(method._tree, state.get("decision"))
    pending = state.get("pending")
    method._pending = None if pending is None else Pending(**pending)
    method._n_eval = int(state["n_eval"])
    method._n_calls = int(state.get("n_calls", method._n_eval))
    method._n_stag = int(state["n_stag"])
    method._attempt_number = int(state.get("attempt", 1))
    method._attempt_kind = str(state.get("attempt_kind", "initial"))
    method._n_ordinary_decisions = int(state.get("ordinary_decisions", 0))
    method._next_entry_id = int(state.get("next_entry_id", 1))
    sigma = state.get("sigma_q")
    method._sigma_q = None if sigma is None else float(sigma)
    method._last_best_eval = int(state.get("last_best_eval", 0))
    method._recent_outcomes.clear()
    method._recent_outcomes.extend(str(item) for item in state.get("recent_outcomes", []))
    return checkpoint


def _decision_payload(decision: Decision | None) -> dict[str, object] | None:
    if decision is None:
        return None
    return {
        "intent": decision.intent.value,
        "parent_id": decision.parent.id,
        "p_explore": decision.p_explore,
        "beta": decision.beta,
        "ess": decision.ess,
        "n_valid": decision.n_valid,
        "parent_q": decision.parent_q,
        "sigma_q": decision.sigma_q,
        "allocation_mode": decision.allocation_mode,
        "selected_score": decision.selected_score,
        "opportunity": decision.opportunity,
        "decision_index": decision.decision_index,
        "operator_draw": decision.operator_draw,
        "selection_scores": [list(item) for item in decision.selection_scores],
        "selection_snapshot": [list(item) for item in decision.selection_snapshot],
    }


def _restore_decision(tree: Tree, payload: object) -> Decision | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("checkpoint decision is malformed")
    parent = tree.get_algorithm(int(payload["parent_id"]))
    selection_scores = tuple(
        (int(item[0]), float(item[1])) for item in payload["selection_scores"]
    )
    selection_snapshot = tuple(
        (int(item[0]), float(item[1]), float(item[2]), int(item[3]), float(item[4]))
        for item in payload.get("selection_snapshot", [])
    )
    return Decision(
        intent=Intent(str(payload["intent"])),
        parent=parent,
        p_explore=float(payload["p_explore"]),
        beta=float(payload["beta"]),
        ess=float(payload["ess"]),
        n_valid=int(payload["n_valid"]),
        parent_q=float(payload["parent_q"]),
        sigma_q=float(payload["sigma_q"]),
        allocation_mode=str(payload["allocation_mode"]),
        selected_score=float(payload["selected_score"]),
        opportunity=float(payload["opportunity"]),
        decision_index=int(payload["decision_index"]),
        operator_draw=float(payload["operator_draw"]),
        selection_scores=selection_scores,
        selection_snapshot=selection_snapshot,
    )


__all__ = ["CHECKPOINT_VERSION", "load_checkpoint", "save_checkpoint"]
