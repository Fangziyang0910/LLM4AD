"""Streaming V9.8 P3 forced-continuation probe.

The probe starts from valid ``parent_path_explore`` children produced by P1/P2.
Each child is cloned into a child-chain protocol and a hypothesis-level protocol.
Both receive five atomic Refine responses; each response is evaluated and appended
before the next one is generated.  Horizons 0/1/3/5 are nested reads of this prefix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9.complexity import code_change_ratio
from llm4ad.method.traceaad_v9_8.prompt import (
    build_generation_prompt,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_8.schema import Intent
from llm4ad.method.traceaad_v9_8.source import code_diff, code_hash

from .._common import BACKENDS, build_llm_client, build_task, resolve_backend
from .v98_mechanism_probe import (
    LOGICAL_MODEL_NAME,
    OUTPUT_TOKENS,
    TOTAL_CONTEXT_TOKENS,
    TRANSPORT_RETRIES,
    _append_jsonl,
    _read_json,
    _read_jsonl,
    _text_hash,
    _write_json,
    _write_jsonl,
    status_p12,
)

PROTOCOL_ID = "traceaad-v9.8-stage-p3-streaming-v1"
PROTOCOLS = ("child_chain", "hypothesis_level")
HORIZONS = (0, 1, 3, 5)
MAX_STEPS = 5
DESIGN_SEED = 980503


def _sampling_seed_base(seed: int, block_order: int) -> int:
    return seed * 1000 + block_order * 10


@dataclass(slots=True)
class Node:
    id: int
    code: str
    code_hash: str
    fitness: float
    q: float
    history_text: str
    parent_id: int | None
    order: int
    n_refine: int = 0


def _all_p12_results(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _select_p3_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    eligible = sorted(
        (
            row
            for row in rows
            if row["condition"] == "parent_path_explore"
            and row.get("valid") is True
            and row.get("no_op") is not True
        ),
        key=lambda row: (row["anchor_id"], int(row["replicate"]), row["trial_id"]),
    )
    first_by_anchor: dict[str, dict[str, Any]] = {}
    for row in eligible:
        first_by_anchor.setdefault(row["anchor_id"], row)
    return list(first_by_anchor.values()), len(eligible)


def _history_event(
    *,
    operator: str,
    hypothesis: str,
    idea: str | None,
    diff: str,
    added: int,
    removed: int,
    outcome: str,
    parent_fitness: float,
    child_fitness: float,
) -> str:
    removed_examples = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---") and line[1:].strip()
    ]
    added_examples = [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
    ]
    removed_text = " | ".join(f"`{line}`" for line in removed_examples[:2]) or "none"
    added_text = " | ".join(f"`{line}`" for line in added_examples[:2]) or "none"
    return "\n".join(
        [
            "[History {index}] Formation step",
            f"Operator: {operator}",
            f"Hypothesis: {hypothesis}",
            f"Idea: {idea or 'unavailable'}",
            f"Change: +{added}/-{removed} lines; removed: {removed_text}; added: {added_text}",
            f"Result: {outcome}",
            f"Fitness: {parent_fitness:.6g} -> {child_fitness:.6g}",
        ]
    )


def _append_formation(history: str, event: str) -> str:
    blocks = history.split("\n\n")
    body = blocks[1:] if blocks and blocks[0].startswith("[Recent") else blocks
    body.append(event)
    body = body[-8:]
    rendered = ["[Recent Algorithm Formation Path]"]
    for index, block in enumerate(body, start=1):
        renamed = block.replace("{index}", str(index), 1)
        renamed = re.sub(r"\[History \d+\]", f"[History {index}]", renamed, count=1)
        rendered.append(renamed)
    return "\n\n".join(rendered)


def prepare_p3(
    p12_dir: Path,
    run_dir: Path,
    *,
    seed: int = DESIGN_SEED,
    limit: int | None = None,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    p12_status = status_p12(p12_dir)
    if p12_status["completed_trials"] != p12_status["total_trials"]:
        raise ValueError("P1/P2 must be complete before preparing P3")
    anchors = {row["anchor_id"]: row for row in _read_jsonl(p12_dir / "anchors.jsonl")}
    # P1/P2 replicates are repeated observations nested within a fixed source
    # anchor.  P3 keeps at most one valid child per source anchor so its paired
    # continuation unit is not silently inflated by those repeated samples.
    explore_rows, eligible_count = _select_p3_rows(_all_p12_results(p12_dir))
    if limit is not None:
        explore_rows = explore_rows[:limit]
    units: list[dict[str, Any]] = []
    for index, row in enumerate(explore_rows):
        anchor = anchors[row["anchor_id"]]
        outcome = "improve" if row["delta_q"] > 0 else "regress" if row["delta_q"] < 0 else "plateau"
        explore_event = _history_event(
            operator="Explore",
            hypothesis="create H1 from H0",
            idea=row.get("idea"),
            diff=row.get("diff", ""),
            added=int(row.get("added", 0)),
            removed=int(row.get("removed", 0)),
            outcome=outcome,
            parent_fitness=float(row["parent_fitness"]),
            child_fitness=float(row["child_fitness"]),
        )
        units.append(
            {
                "unit_id": f"explore_{index:03d}",
                "source_trial_id": row["trial_id"],
                "task": row["task"],
                "stratum": row["stratum"],
                "replicate": row["replicate"],
                "anchor_id": row["anchor_id"],
                "parent_code": anchor["code"],
                "parent_fitness": row["parent_fitness"],
                "parent_q": row["parent_q"],
                "entry_code": row["candidate_code"],
                "entry_code_hash": row["candidate_code_hash"],
                "entry_fitness": row["child_fitness"],
                "entry_q": row["child_q"],
                "entry_idea": row.get("idea"),
                "entry_delta_q": row["delta_q"],
                "source_s0": row["source_s0"],
                "entry_history_text": _append_formation(
                    anchor["history_text"], explore_event
                ),
            }
        )
    schedule: list[dict[str, Any]] = []
    for block_order, unit in enumerate(units):
        for protocol_order, protocol in enumerate(PROTOCOLS):
            schedule.append(
                {
                    "continuation_id": f"{unit['unit_id']}_{protocol}",
                    "unit_id": unit["unit_id"],
                    "block_order": block_order,
                    "protocol_order": protocol_order,
                    "protocol": protocol,
                    "task": unit["task"],
                    "sampling_seed_base": _sampling_seed_base(seed, block_order),
                }
            )
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "units.jsonl", units)
    _write_jsonl(run_dir / "schedule.jsonl", schedule)
    _write_json(
        run_dir / "probe_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "p12_source": str(p12_dir.resolve()),
            "logical_model_name": LOGICAL_MODEL_NAME,
            "protocols": list(PROTOCOLS),
            "horizons": list(HORIZONS),
            "max_steps": MAX_STEPS,
            "unit_count": len(units),
            "continuation_count": len(schedule),
            "response_count": len(schedule) * MAX_STEPS,
            "eligible_parent_path_explore_children": eligible_count,
            "selection_rule": (
                "one child per source anchor; first valid non-noop replicate in "
                "ascending replicate order"
            ),
            "design_seed": seed,
            "generation_evaluation_pipeline": "same_process_immediate_per_response",
            "independent_unit": "source anchor represented by one Explore child",
            "repeated_measure": "nested horizon prefix within child and protocol",
        },
    )


def _restore_state(
    unit: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[dict[int, Node], int, set[tuple[int, str]]]:
    nodes = {
        0: Node(
            id=0,
            code=unit["entry_code"],
            code_hash=unit["entry_code_hash"],
            fitness=float(unit["entry_fitness"]),
            q=float(unit["entry_q"]),
            history_text=unit["entry_history_text"],
            parent_id=None,
            order=0,
        )
    }
    tip_id = 0
    relations: set[tuple[int, str]] = set()
    for row in sorted(rows, key=lambda item: int(item["step"])):
        selected_id = int(row["selected_node_id"])
        nodes[selected_id].n_refine += 1
        child_id = row.get("child_node_id")
        if child_id is not None:
            child = Node(
                id=int(child_id),
                code=row["candidate_code"],
                code_hash=row["candidate_code_hash"],
                fitness=float(row["child_fitness"]),
                q=float(row["child_q"]),
                history_text=row["child_history_text"],
                parent_id=selected_id,
                order=int(row["step"]),
            )
            nodes[child.id] = child
            relations.add((selected_id, child.code_hash))
        tip_id = int(row["tip_after"])
    return nodes, tip_id, relations


def _ancestor_hashes(nodes: dict[int, Node], node_id: int) -> set[str]:
    hashes: set[str] = set()
    current = nodes[node_id].parent_id
    while current is not None:
        hashes.add(nodes[current].code_hash)
        current = nodes[current].parent_id
    return hashes


def _select_node(protocol: str, nodes: dict[int, Node], tip_id: int, s0: float) -> Node:
    if protocol == "child_chain":
        return nodes[tip_id]
    return max(
        nodes.values(),
        key=lambda node: (
            node.q + s0 / math.sqrt(node.n_refine + 1),
            -node.n_refine,
            -len(node.code),
            -node.order,
            -node.id,
        ),
    )


def _draw_response(*, llm, prompt: str, sampling_seed: int) -> dict[str, Any]:
    started = time.time()
    response = ""
    for attempt in range(1, TRANSPORT_RETRIES + 1):
        try:
            response = llm.draw_sample(
                prompt, max_tokens=OUTPUT_TOKENS, seed=sampling_seed
            )
            break
        except Exception:
            if attempt == TRANSPORT_RETRIES:
                raise
    return {
        "prompt": prompt,
        "prompt_hash": _text_hash(prompt),
        "prompt_tokens": int(llm.count_prompt_tokens(prompt)),
        "response": response,
        "response_hash": _text_hash(response),
        "response_tokens": int(llm.count_tokens(response)),
        "sampling_seed": sampling_seed,
        "sample_seconds": time.time() - started,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _evaluate_response(
    *,
    continuation: dict[str, Any],
    unit: dict[str, Any],
    step: int,
    selected: Node,
    tip_id: int,
    nodes: dict[int, Node],
    relations: set[tuple[int, str]],
    call: dict[str, Any],
    evaluation,
    evaluator: SecureEvaluator,
) -> dict[str, Any]:
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    assert template is not None and len(template.functions) == 1
    base = {
        **continuation,
        "source_trial_id": unit["source_trial_id"],
        "step": step,
        "selected_node_id": selected.id,
        "selected_fitness": selected.fitness,
        "selected_q": selected.q,
        "selected_n_refine_before": selected.n_refine,
        "frontier_before": max(node.q for node in nodes.values()),
        "prompt_hash": call["prompt_hash"],
        "prompt_tokens": call["prompt_tokens"],
        "response_hash": call["response_hash"],
        "response_tokens": call["response_tokens"],
        "sampling_seed": call["sampling_seed"],
        "sample_seconds": call["sample_seconds"],
    }
    parsed = parse_program_response(call["response"])
    candidate_code = parsed.code
    candidate_hash = code_hash(candidate_code)
    diff, added, removed = code_diff(selected.code, candidate_code)
    common = {
        **base,
        "idea": parsed.declared_idea,
        "candidate_code": candidate_code,
        "candidate_code_hash": candidate_hash,
        "diff": diff,
        "added": added,
        "removed": removed,
        "change_ratio": code_change_ratio(selected.code, candidate_code),
    }
    existing = next((node for node in nodes.values() if node.code_hash == candidate_hash), None)
    if existing is not None:
        if existing.id == selected.id:
            kind = "no_op"
        elif candidate_hash in _ancestor_hashes(nodes, selected.id):
            kind = "ancestral_return"
        elif (selected.id, candidate_hash) in relations:
            kind = "repeated_duplicate"
        else:
            kind = "cached"
        if kind != "cached":
            return {
                **common,
                "status": "valid",
                "valid": True,
                "kind": kind,
                "failure_kind": None,
                "evaluator_called": False,
                "child_node_id": None,
                "child_fitness": existing.fitness,
                "child_q": existing.q,
                "delta_q": existing.q - selected.q,
                "tip_after": tip_id,
                "frontier_after": base["frontier_before"],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        child_fitness = existing.fitness
        child_q = existing.q
        evaluator_called = False
        evaluate_seconds = 0.0
    else:
        outcome, evaluate_seconds = evaluator.evaluate_program_record_time_with_details(
            candidate_code
        )
        if outcome.failure_kind == "prepare_error":
            raise RuntimeError(
                f"evaluator infrastructure failure for {continuation['continuation_id']} "
                f"step {step}: {outcome.error}"
            )
        score = getattr(outcome.result, "fitness", outcome.result)
        try:
            child_fitness = float(score)
        except (TypeError, ValueError, OverflowError):
            child_fitness = math.nan
        if not math.isfinite(child_fitness):
            return {
                **common,
                "status": "invalid",
                "valid": False,
                "failure_kind": outcome.failure_kind or "invalid_result",
                "failure_error": outcome.error,
                "evaluator_called": True,
                "evaluate_seconds": evaluate_seconds,
                "child_node_id": None,
                "tip_after": tip_id,
                "frontier_after": base["frontier_before"],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        child_q = child_fitness
        evaluator_called = True
    delta_q = child_q - selected.q
    direct_outcome = "improve" if delta_q > 0 else "regress" if delta_q < 0 else "plateau"
    event = _history_event(
        operator="Refine",
        hypothesis="inherit H1",
        idea=parsed.declared_idea,
        diff=diff,
        added=added,
        removed=removed,
        outcome=direct_outcome,
        parent_fitness=selected.fitness,
        child_fitness=child_fitness,
    )
    child_id = max(nodes) + 1
    child_history = _append_formation(selected.history_text, event)
    frontier_after = max(base["frontier_before"], child_q)
    return {
        **common,
        "status": "valid",
        "valid": True,
        "kind": "cached" if existing is not None else "new",
        "failure_kind": None,
        "evaluator_called": evaluator_called,
        "evaluate_seconds": evaluate_seconds,
        "child_node_id": child_id,
        "child_fitness": child_fitness,
        "child_q": child_q,
        "delta_q": delta_q,
        "child_history_text": child_history,
        "tip_after": child_id if continuation["protocol"] == "child_chain" else tip_id,
        "frontier_after": frontier_after,
        "internal_gain_h": frontier_after - float(unit["entry_q"]),
        "parent_recovery_h": frontier_after - float(unit["parent_q"]),
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_p3_shard(
    run_dir: Path,
    *,
    backend: str,
    shard_index: int,
    num_shards: int,
    eval_workers: int,
) -> None:
    config = _read_json(run_dir / "probe_config.json")
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("P3 protocol mismatch")
    units = {row["unit_id"]: row for row in _read_jsonl(run_dir / "units.jsonl")}
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["block_order"]) % num_shards == shard_index
    ]
    result_path = run_dir / "results" / f"shard_{shard_index:02d}.jsonl"
    call_path = run_dir / "llm_calls" / f"shard_{shard_index:02d}.jsonl"
    all_results = _read_jsonl(result_path)
    all_calls = {
        (row["continuation_id"], int(row["step"])): row
        for row in _read_jsonl(call_path)
    }
    profile = resolve_backend(backend, None, None, None)
    llm = build_llm_client(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=OUTPUT_TOKENS,
        temperature=1.0,
    )
    resources: dict[str, tuple[Any, SecureEvaluator]] = {}
    try:
        for continuation in schedule:
            unit = units[continuation["unit_id"]]
            rows = [
                row
                for row in all_results
                if row["continuation_id"] == continuation["continuation_id"]
            ]
            nodes, tip_id, relations = _restore_state(unit, rows)
            task = continuation["task"]
            if task not in resources:
                evaluation, _ = build_task(task, eval_workers=eval_workers)
                resources[task] = evaluation, SecureEvaluator(evaluation)
            evaluation, evaluator = resources[task]
            for step in range(len(rows) + 1, MAX_STEPS + 1):
                selected = _select_node(
                    continuation["protocol"], nodes, tip_id, float(unit["source_s0"])
                )
                prompt = build_generation_prompt(
                    task_description=evaluation.task_description,
                    code=selected.code,
                    fitness=selected.fitness,
                    history_text=selected.history_text,
                    intent=Intent.REFINE,
                    maximize=True,
                )
                if int(llm.count_prompt_tokens(prompt)) + OUTPUT_TOKENS > TOTAL_CONTEXT_TOKENS:
                    raise ValueError("P3 prompt exceeds total context")
                call_key = continuation["continuation_id"], step
                call = all_calls.get(call_key)
                if call is None:
                    call = {
                        "continuation_id": continuation["continuation_id"],
                        "unit_id": continuation["unit_id"],
                        "protocol": continuation["protocol"],
                        "step": step,
                        **_draw_response(
                            llm=llm,
                            prompt=prompt,
                            sampling_seed=int(continuation["sampling_seed_base"]) + step,
                        ),
                    }
                    _append_jsonl(call_path, call)
                    all_calls[call_key] = call
                result = _evaluate_response(
                    continuation=continuation,
                    unit=unit,
                    step=step,
                    selected=selected,
                    tip_id=tip_id,
                    nodes=nodes,
                    relations=relations,
                    call=call,
                    evaluation=evaluation,
                    evaluator=evaluator,
                )
                _append_jsonl(result_path, result)
                all_results.append(result)
                nodes, tip_id, relations = _restore_state(
                    unit,
                    [
                        row
                        for row in all_results
                        if row["continuation_id"] == continuation["continuation_id"]
                    ],
                )
                print(
                    f"p3 shard={shard_index} continuation={continuation['continuation_id']} "
                    f"step={step}/{MAX_STEPS} status={result['status']}",
                    flush=True,
                )
            completed = sum(
                len(
                    {
                        int(row["step"])
                        for row in all_results
                        if row["continuation_id"] == item["continuation_id"]
                    }
                )
                == MAX_STEPS
                for item in schedule
            )
            _write_json(
                run_dir / "shards" / f"shard_{shard_index:02d}" / "summary.json",
                {
                    "status": "finished" if completed == len(schedule) else "running",
                    "completed_continuations": completed,
                    "assigned_continuations": len(schedule),
                    "completed_responses": len(all_results),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
    finally:
        llm.close()


def status_p3(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    results: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        results.extend(_read_jsonl(path))
    by_continuation: dict[str, set[int]] = {}
    for row in results:
        by_continuation.setdefault(row["continuation_id"], set()).add(int(row["step"]))
    return {
        "protocol_id": config["protocol_id"],
        "unit_count": config["unit_count"],
        "continuation_count": config["continuation_count"],
        "completed_continuations": sum(len(steps) == MAX_STEPS for steps in by_continuation.values()),
        "completed_responses": len(results),
        "total_responses": config["response_count"],
        "valid_responses": sum(row.get("valid") is True for row in results),
        "evaluator_calls": sum(row.get("evaluator_called") is True for row in results),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--p12-dir", type=Path, required=True)
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED)
    prepare.add_argument("--limit", type=int)
    run = subparsers.add_parser("run")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--eval-workers", type=int, default=1)
    inspect = subparsers.add_parser("status")
    inspect.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_p3(args.p12_dir, args.run_dir, seed=args.seed, limit=args.limit)
    elif args.command == "run":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index must be in [0, num_shards)")
        run_p3_shard(
            args.run_dir,
            backend=args.backend,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            eval_workers=args.eval_workers,
        )
    else:
        print(json.dumps(status_p3(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
