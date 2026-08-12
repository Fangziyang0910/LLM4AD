"""Single-schema checkpoint persistence for TraceAAD V9.6."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from .forest import SearchForest
from .schema import (
    PROTOCOL_ID,
    AnchorState,
    AttemptKind,
    AttemptRecord,
    DiffStatistics,
    DirectOutcome,
    PendingAttempt,
    PendingStage,
    ProgramArtifact,
)

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
        os.unlink(temporary)
        raise


def _forest_to_dict(forest: SearchForest) -> dict[str, Any]:
    return {
        "evaluator_contract_hash": forest.evaluator_contract_hash,
        "maximize": forest.maximize,
        "next_artifact_id": forest._next_artifact_id,
        "next_state_id": forest._next_state_id,
        "next_attempt_id": forest._next_attempt_id,
        "root_state_ids": list(forest.root_state_ids),
        "artifacts": [asdict(item) for item in forest.artifacts()],
        "states": [asdict(item) for item in forest.states()],
        "attempts": [asdict(item) for item in forest.attempts()],
    }


def _forest_from_dict(payload: Mapping[str, Any]) -> SearchForest:
    forest = SearchForest(
        str(payload["evaluator_contract_hash"]), maximize=bool(payload["maximize"])
    )
    for item in payload["artifacts"]:
        artifact = ProgramArtifact(
            artifact_id=int(item["artifact_id"]),
            evaluator_contract_hash=str(item["evaluator_contract_hash"]),
            evaluator_input_hash=str(item["evaluator_input_hash"]),
            evaluator_input_code=str(item["evaluator_input_code"]),
            fitness=float(item["fitness"]),
            directed_fitness=float(item["directed_fitness"]),
            code_length=int(item["code_length"]),
            program_loc=int(item["program_loc"]),
            first_discovery_order=int(item["first_discovery_order"]),
        )
        forest._artifacts[artifact.artifact_id] = artifact
        forest._artifact_keys[
            (artifact.evaluator_contract_hash, artifact.evaluator_input_hash)
        ] = artifact.artifact_id
    for item in payload["states"]:
        state = AnchorState(
            state_id=int(item["state_id"]),
            artifact_id=int(item["artifact_id"]),
            parent_state_id=(
                None
                if item["parent_state_id"] is None
                else int(item["parent_state_id"])
            ),
            incoming_attempt_id=(
                None
                if item["incoming_attempt_id"] is None
                else int(item["incoming_attempt_id"])
            ),
            depth=int(item["depth"]),
            creation_order=int(item["creation_order"]),
            generation_count_n=int(item["generation_count_n"]),
        )
        forest._states[state.state_id] = state
        if state.parent_state_id is not None:
            forest._relations.add((state.parent_state_id, state.artifact_id))
    for item in payload["attempts"]:
        stats = item["diff_statistics"]
        attempt = AttemptRecord(
            attempt_id=int(item["attempt_id"]),
            status="finalized",
            anchor_state_id=(
                None
                if item["anchor_state_id"] is None
                else int(item["anchor_state_id"])
            ),
            child_state_id=(
                None if item["child_state_id"] is None else int(item["child_state_id"])
            ),
            artifact_id=(
                None if item["artifact_id"] is None else int(item["artifact_id"])
            ),
            declared_idea=item["declared_idea"],
            raw_code_hash=item["raw_code_hash"],
            evaluator_input_hash=item["evaluator_input_hash"],
            actual_diff=item["actual_diff"],
            diff_statistics=(
                None
                if stats is None
                else DiffStatistics(
                    added_lines=int(stats["added_lines"]),
                    removed_lines=int(stats["removed_lines"]),
                    changed_lines=int(stats["changed_lines"]),
                )
            ),
            parent_fitness=(
                None
                if item["parent_fitness"] is None
                else float(item["parent_fitness"])
            ),
            child_fitness=(
                None if item["child_fitness"] is None else float(item["child_fitness"])
            ),
            directed_delta=(
                None
                if item["directed_delta"] is None
                else float(item["directed_delta"])
            ),
            direct_outcome=(
                None
                if item["direct_outcome"] is None
                else DirectOutcome(item["direct_outcome"])
            ),
            attempt_kind=AttemptKind(item["attempt_kind"]),
            failure_category=item["failure_category"],
            failure_feedback=item["failure_feedback"],
            evaluator_called=bool(item["evaluator_called"]),
            candidate_order=int(item["candidate_order"]),
            creation_time=str(item["creation_time"]),
            stage=str(item["stage"]),
            iteration=(None if item["iteration"] is None else int(item["iteration"])),
        )
        forest._attempts[attempt.attempt_id] = attempt
    forest.root_state_ids = [int(item) for item in payload["root_state_ids"]]
    forest._next_artifact_id = int(payload["next_artifact_id"])
    forest._next_state_id = int(payload["next_state_id"])
    forest._next_attempt_id = int(payload["next_attempt_id"])
    return forest


def _pending_from_dict(item: Mapping[str, Any] | None) -> PendingAttempt | None:
    if item is None:
        return None
    stats = item["diff_statistics"]
    return PendingAttempt(
        attempt_id=int(item["attempt_id"]),
        anchor_state_id=(
            None if item["anchor_state_id"] is None else int(item["anchor_state_id"])
        ),
        stage_name=str(item["stage_name"]),
        iteration=None if item["iteration"] is None else int(item["iteration"]),
        candidate_order=int(item["candidate_order"]),
        response=str(item["response"]),
        prompt=str(item["prompt"]),
        prompt_tokens=int(item["prompt_tokens"]),
        response_tokens=int(item["response_tokens"]),
        sample_time=float(item["sample_time"]),
        generation_seed=(
            None if item["generation_seed"] is None else int(item["generation_seed"])
        ),
        processing_stage=PendingStage(item["processing_stage"]),
        declared_idea=item["declared_idea"],
        raw_code=item["raw_code"],
        raw_code_hash=item["raw_code_hash"],
        evaluator_input_code=item["evaluator_input_code"],
        evaluator_input_hash=item["evaluator_input_hash"],
        actual_diff=item["actual_diff"],
        diff_statistics=(
            None
            if stats is None
            else DiffStatistics(
                added_lines=int(stats["added_lines"]),
                removed_lines=int(stats["removed_lines"]),
                changed_lines=int(stats["changed_lines"]),
            )
        ),
        evaluator_called=bool(item["evaluator_called"]),
        evaluated_fitness=(
            None
            if item["evaluated_fitness"] is None
            else float(item["evaluated_fitness"])
        ),
        failure_category=item["failure_category"],
        failure_feedback=item["failure_feedback"],
        evaluate_time=(
            None if item["evaluate_time"] is None else float(item["evaluate_time"])
        ),
    )


def dump_state(method) -> dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "search_configuration": method.search_configuration(),
        "runtime_identity": method.runtime_identity(),
        "forest": _forest_to_dict(method._forest),
        "pending_attempt": (
            None if method._pending_attempt is None else asdict(method._pending_attempt)
        ),
        "candidate_count": method._candidate_count,
        "llm_request_count": method._llm_request_count,
        "evaluation_count": method._evaluation_count,
        "transport_failure_count": method._transport_failure_count,
        "next_iteration": method._next_iteration,
        "initialization_complete": method._initialization_complete,
        "bootstrapped_root_ids": sorted(method._bootstrapped_root_ids),
        "bootstrap_deltas": list(method._bootstrap_deltas),
        "optimism_scale": method._optimism_scale,
        "best_artifact_id": method._best_artifact_id,
        "best_artifact_sample_order": method._best_artifact_sample_order,
        "outcome_counts": dict(method._outcome_counts),
    }


def load_state(method, payload: Mapping[str, Any]) -> None:
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported TraceAAD V9.6 checkpoint version")
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint protocol does not match TraceAAD V9.6")
    if payload.get("search_configuration") != method.search_configuration():
        raise ValueError("checkpoint search configuration does not match")
    if payload.get("runtime_identity") != method.runtime_identity():
        raise ValueError("checkpoint runtime identity does not match")
    method._forest = _forest_from_dict(payload["forest"])
    method._pending_attempt = _pending_from_dict(payload["pending_attempt"])
    method._candidate_count = int(payload["candidate_count"])
    method._llm_request_count = int(payload["llm_request_count"])
    method._evaluation_count = int(payload["evaluation_count"])
    method._transport_failure_count = int(payload["transport_failure_count"])
    method._next_iteration = int(payload["next_iteration"])
    method._initialization_complete = bool(payload["initialization_complete"])
    method._bootstrapped_root_ids = {
        int(item) for item in payload["bootstrapped_root_ids"]
    }
    method._bootstrap_deltas = [float(item) for item in payload["bootstrap_deltas"]]
    method._optimism_scale = (
        None if payload["optimism_scale"] is None else float(payload["optimism_scale"])
    )
    method._best_artifact_id = (
        None
        if payload["best_artifact_id"] is None
        else int(payload["best_artifact_id"])
    )
    method._best_artifact_sample_order = payload["best_artifact_sample_order"]
    method._outcome_counts = {
        str(key): int(value) for key, value in payload["outcome_counts"].items()
    }
    if method._artifacts is not None:
        best = (
            None
            if method._best_artifact_id is None
            else method._forest.get_artifact(method._best_artifact_id)
        )
        method._artifacts.sync_after_resume(
            total_samples=method._candidate_count,
            best_score=None if best is None else best.fitness,
            best_sample_order=method._best_artifact_sample_order,
        )


def save_checkpoint(method, directory: str | Path | None = None) -> Path | None:
    target = method._checkpoint_dir if directory is None else Path(directory)
    if target is None:
        return None
    latest = Path(target) / "latest.json"
    _atomic_write(latest, dump_state(method))
    return latest


def load_checkpoint(method, path: str | Path) -> Path:
    checkpoint = Path(path)
    load_state(method, json.loads(checkpoint.read_text(encoding="utf-8")))
    return checkpoint


__all__ = [
    "CHECKPOINT_VERSION",
    "dump_state",
    "load_checkpoint",
    "load_state",
    "save_checkpoint",
]
