"""Fixed-anchor A/B probe for deterministic compact actual-change evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9.complexity import code_change_ratio
from llm4ad.method.traceaad_v9_5.prompt import (
    ProgramResponseError,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_5.source import nonempty_loc, text_hash

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    build_llm_client,
    build_task,
    resolve_backend,
)
from .generation_probe import (
    DESIGN_SEED,
    LOGICAL_MODEL_NAME,
    OUTPUT_TOKENS,
    TOTAL_CONTEXT_TOKENS,
    TRANSPORT_RETRIES,
    _append_jsonl,
    _read_json,
    _read_jsonl,
    _response_tokens,
    _write_json,
    _write_jsonl,
    build_prompt_pair,
)


PROTOCOL_ID = "traceaad-v95-concise-history-compact-change-probe-v1"
BASE_PROBE = EXPERIMENTS_ROOT / "generation_probe" / "20260812_v95_concise_history_probe"
CONDITIONS = ("concise_history", "compact_actual_change")
DESIGN_SEED_V2 = DESIGN_SEED + 2
COMPACT_CHANGE_MAX_CHARS = 520
CHANGE_EXAMPLES_PER_SIDE = 2


@dataclass(frozen=True, slots=True)
class CompactPromptPair:
    concise_history: str
    compact_actual_change: str
    canonical_concise_history: str
    compact_change_block: str


def _one_line(value: Any, limit: int) -> str:
    rendered = " ".join(str(value or "").split()) or "none"
    return rendered if len(rendered) <= limit else rendered[: limit - 3].rstrip() + "..."


def _changed_code_lines(actual_diff: str, prefix: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for line in actual_diff.splitlines():
        if not line.startswith(prefix) or line.startswith(prefix * 3):
            continue
        code = line[1:].strip()
        if not code or code.startswith(("#", '"""', "'''")):
            continue
        normalized = _one_line(code, 150)
        if normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _representative_lines(lines: list[str], count: int) -> list[str]:
    if len(lines) <= count:
        return lines
    if count == 1:
        return [lines[0]]
    indices = [round(index * (len(lines) - 1) / (count - 1)) for index in range(count)]
    return [lines[index] for index in indices]


def compact_actual_change(candidate: dict[str, Any]) -> str:
    actual_diff = str(candidate.get("actual_diff") or "")
    statistics = candidate.get("diff_statistics") or {}
    added_count = int(statistics.get("added_lines") or 0)
    removed_count = int(statistics.get("removed_lines") or 0)
    if not actual_diff:
        return "No executable code change was recorded."
    removed = _representative_lines(
        _changed_code_lines(actual_diff, "-"), CHANGE_EXAMPLES_PER_SIDE
    )
    added = _representative_lines(
        _changed_code_lines(actual_diff, "+"), CHANGE_EXAMPLES_PER_SIDE
    )
    summary = (
        f"Diff size: +{added_count}/-{removed_count} lines. "
        f"Removed examples: {' | '.join(f'`{line}`' for line in removed) or 'none'}. "
        f"Added examples: {' | '.join(f'`{line}`' for line in added) or 'none'}."
    )
    return _one_line(summary, COMPACT_CHANGE_MAX_CHARS)


def render_compact_change_block(anchor: dict[str, Any]) -> str:
    lines = ["[Compact Actual Changes]"]
    for index, item in enumerate(anchor["history"], start=1):
        lines.append(f"Event {index}: {item['compact_actual_change']}")
    return "\n".join(lines)


def build_compact_prompt_pair(
    anchor: dict[str, Any], task_description: str
) -> CompactPromptPair:
    base = build_prompt_pair(anchor, task_description)
    concise = base.concise_history
    compact_block = render_compact_change_block(anchor)
    marker = "\n\n[Instruction]"
    if concise.count(marker) != 1:
        raise AssertionError("concise prompt has an unexpected instruction boundary")
    treatment = concise.replace(marker, f"\n\n{compact_block}{marker}", 1)
    if treatment.replace(f"\n\n{compact_block}", "", 1) != concise:
        raise AssertionError("prompt pair differs outside compact actual changes")
    return CompactPromptPair(
        concise_history=concise,
        compact_actual_change=treatment,
        canonical_concise_history=concise,
        compact_change_block=compact_block,
    )


def _candidate_map(anchor: dict[str, Any]) -> dict[int, dict[str, Any]]:
    path = (
        EXPERIMENTS_ROOT
        / anchor["task"]
        / "traceaad_v9_5"
        / anchor["source_run"]
        / "artifacts"
        / "candidates.jsonl"
    )
    return {int(row["attempt_id"]): row for row in _read_jsonl(path)}


def load_enriched_anchors(base_probe: Path) -> list[dict[str, Any]]:
    anchors = _read_jsonl(base_probe / "anchors.jsonl")
    maps: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    enriched: list[dict[str, Any]] = []
    for anchor in anchors:
        key = (anchor["task"], anchor["source_run"])
        if key not in maps:
            maps[key] = _candidate_map(anchor)
        candidates = maps[key]
        row = dict(anchor)
        row["history"] = []
        for item in anchor["history"]:
            candidate = candidates[int(item["attempt_id"])]
            event = dict(item)
            event["compact_actual_change"] = compact_actual_change(candidate)
            event["actual_diff_available"] = bool(candidate.get("actual_diff"))
            row["history"].append(event)
        enriched.append(row)
    return enriched


def build_schedule(
    anchors: list[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 7919)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        for replicate in range(1, replicates + 1):
            grouped[(anchor["task"], anchor["stratum"])].append(
                {"anchor": anchor, "replicate": replicate}
            )
    blocks: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        rng.shuffle(group)
        if len(group) % 2:
            raise ValueError(f"AB/BA balance requires an even block count for {key}")
        half = len(group) // 2
        for index, item in enumerate(group):
            item["condition_order"] = (
                list(CONDITIONS) if index < half else list(reversed(CONDITIONS))
            )
            blocks.append(item)
    rng.shuffle(blocks)
    schedule: list[dict[str, Any]] = []
    for pair_order, block in enumerate(blocks):
        anchor = block["anchor"]
        pair_id = f"{anchor['anchor_id']}_rep{block['replicate']}"
        sampling_seed = seed * 1000 + pair_order + 1
        for within_pair_order, condition in enumerate(block["condition_order"]):
            schedule.append(
                {
                    "trial_id": f"{pair_id}_{condition}",
                    "pair_id": pair_id,
                    "pair_order": pair_order,
                    "within_pair_order": within_pair_order,
                    "anchor_id": anchor["anchor_id"],
                    "task": anchor["task"],
                    "stratum": anchor["stratum"],
                    "replicate": block["replicate"],
                    "condition": condition,
                    "sampling_seed": sampling_seed,
                }
            )
    return schedule


def prepare_probe(
    run_dir: Path,
    *,
    base_probe: Path = BASE_PROBE,
    replicates: int = 3,
    seed: int = DESIGN_SEED_V2,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    anchors = load_enriched_anchors(base_probe)
    schedule = build_schedule(anchors, replicates=replicates, seed=seed)
    audits = []
    for anchor in anchors:
        evaluation, _ = build_task(anchor["task"], eval_workers=None)
        pair = build_compact_prompt_pair(anchor, evaluation.task_description)
        audits.append(
            {
                "anchor_id": anchor["anchor_id"],
                "canonical_concise_history_hash": text_hash(
                    pair.canonical_concise_history
                ),
                "concise_history_prompt_hash": text_hash(pair.concise_history),
                "compact_actual_change_prompt_hash": text_hash(
                    pair.compact_actual_change
                ),
                "compact_change_block_hash": text_hash(pair.compact_change_block),
            }
        )
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "anchors.jsonl", anchors)
    _write_jsonl(run_dir / "schedule.jsonl", schedule)
    _write_jsonl(run_dir / "prompt_audit.jsonl", audits)
    _write_json(
        run_dir / "probe_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_probe": str(base_probe),
            "logical_model_name": LOGICAL_MODEL_NAME,
            "conditions": list(CONDITIONS),
            "control_condition": CONDITIONS[0],
            "treatment_condition": CONDITIONS[1],
            "anchor_count": len(anchors),
            "replicates_per_anchor_condition": replicates,
            "trial_count": len(schedule),
            "design_seed": seed,
            "temperature": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "max_total_context": TOTAL_CONTEXT_TOKENS,
            "compact_change_policy": {
                "source": "recorded actual unified diff",
                "max_chars_per_event": COMPACT_CHANGE_MAX_CHARS,
                "examples_per_side": CHANGE_EXAMPLES_PER_SIDE,
                "selection": "first and last unique non-comment changed code lines",
                "llm_summary": False,
            },
            "generation_evaluation_phases_separated": True,
            "evaluation_execution": "one local sequential evaluator process using native task workers",
            "independent_unit": "same 72 anchor snapshots as the concise-history probe",
            "within_anchor_repeated_sampling": replicates,
            "delta_q": "q_child - q_parent",
        },
    )


def _generate_trial(
    *, trial: dict[str, Any], anchor: dict[str, Any], llm, evaluation
) -> dict[str, Any]:
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    if template is None or len(template.functions) != 1:
        raise ValueError("probe requires one evolvable template function")
    pair = build_compact_prompt_pair(anchor, evaluation.task_description)
    prompt = (
        pair.concise_history
        if trial["condition"] == "concise_history"
        else pair.compact_actual_change
    )
    prompt_tokens = int(llm.count_prompt_tokens(prompt))
    if prompt_tokens + OUTPUT_TOKENS > TOTAL_CONTEXT_TOKENS:
        raise ValueError(
            f"trial {trial['trial_id']} exceeds total context: "
            f"{prompt_tokens}+{OUTPUT_TOKENS}>{TOTAL_CONTEXT_TOKENS}"
        )
    start = time.time()
    response = ""
    for transport_attempt in range(1, TRANSPORT_RETRIES + 1):
        try:
            response = llm.draw_sample(
                prompt,
                max_tokens=OUTPUT_TOKENS,
                seed=int(trial["sampling_seed"]),
            )
            break
        except Exception:
            if transport_attempt == TRANSPORT_RETRIES:
                raise
    base = {
        **trial,
        "protocol_id": PROTOCOL_ID,
        "snapshot_id": anchor["snapshot_id"],
        "source_run": anchor["source_run"],
        "source_repeat": anchor["source_repeat"],
        "parent_fitness": anchor["fitness"],
        "parent_q": anchor["q"],
        "parent_code_hash": anchor["code_hash"],
        "history_event_count": len(anchor["history"]),
        "actual_diff_available_count": sum(
            bool(item["actual_diff_available"]) for item in anchor["history"]
        ),
        "canonical_concise_history_hash": text_hash(
            pair.canonical_concise_history
        ),
        "compact_change_block_hash": text_hash(pair.compact_change_block),
        "prompt_hash": text_hash(prompt),
        "prompt_tokens": prompt_tokens,
        "response_hash": text_hash(response),
        "response": response,
        "response_tokens": _response_tokens(llm, response),
        "sample_seconds": time.time() - start,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        parsed = parse_program_response(response, template, template.functions[0].name)
    except ProgramResponseError as exc:
        return {
            **base,
            "generation_status": "parse_failed",
            "failure_kind": "parse",
            "failure_error": str(exc),
            "idea": exc.declared_idea,
        }
    code = str(parsed.program)
    return {
        **base,
        "generation_status": "generated",
        "failure_kind": None,
        "idea": parsed.declared_idea,
        "candidate_code_hash": text_hash(code),
        "candidate_code": code,
        "change_ratio": code_change_ratio(anchor["code"], code),
        "loc_delta": nonempty_loc(code) - nonempty_loc(anchor["code"]),
    }


def generate_shard(
    run_dir: Path,
    *,
    backend: str,
    shard_index: int,
    num_shards: int,
) -> None:
    config = _read_json(run_dir / "probe_config.json")
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("probe configuration protocol mismatch")
    anchors = {
        row["anchor_id"]: row for row in _read_jsonl(run_dir / "anchors.jsonl")
    }
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["pair_order"]) % num_shards == shard_index
    ]
    result_path = run_dir / "generations" / f"shard_{shard_index:02d}.jsonl"
    completed = (
        {row["trial_id"] for row in _read_jsonl(result_path)}
        if result_path.is_file()
        else set()
    )
    profile = resolve_backend(backend, None, None, None)
    llm = build_llm_client(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=OUTPUT_TOKENS,
        temperature=1.0,
    )
    evaluations: dict[str, Any] = {}
    shard_dir = run_dir / "shards" / f"shard_{shard_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        shard_dir / "shard_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "phase": "generation",
            "logical_model_name": LOGICAL_MODEL_NAME,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "trial_count": len(schedule),
        },
    )
    try:
        for trial in schedule:
            if trial["trial_id"] in completed:
                continue
            if trial["task"] not in evaluations:
                evaluations[trial["task"]], _ = build_task(
                    trial["task"], eval_workers=None
                )
            result = _generate_trial(
                trial=trial,
                anchor=anchors[trial["anchor_id"]],
                llm=llm,
                evaluation=evaluations[trial["task"]],
            )
            _append_jsonl(result_path, result)
            completed.add(trial["trial_id"])
            _write_json(
                shard_dir / "summary.json",
                {
                    "status": (
                        "finished" if len(completed) == len(schedule) else "running"
                    ),
                    "phase": "generation",
                    "completed_trials": len(completed),
                    "assigned_trials": len(schedule),
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(
                f"generation shard={shard_index} trial={trial['trial_id']} "
                f"status={result['generation_status']} "
                f"completed={len(completed)}/{len(schedule)}",
                flush=True,
            )
    finally:
        llm.close()


def evaluate_generated(run_dir: Path) -> None:
    schedule = _read_jsonl(run_dir / "schedule.jsonl")
    generated_rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "generations").glob("shard_*.jsonl")):
        generated_rows.extend(_read_jsonl(path))
    generated = {row["trial_id"]: row for row in generated_rows}
    if len(generated_rows) != len(generated) or len(generated) != len(schedule):
        raise ValueError("generation phase must be complete and duplicate-free")
    result_path = run_dir / "results" / "results.jsonl"
    completed = (
        {row["trial_id"] for row in _read_jsonl(result_path)}
        if result_path.is_file()
        else set()
    )
    resources: dict[str, tuple[Any, SecureEvaluator]] = {}
    for trial in schedule:
        if trial["trial_id"] in completed:
            continue
        generation = generated[trial["trial_id"]]
        if generation["generation_status"] == "parse_failed":
            result = {
                **generation,
                "status": "invalid",
                "valid": False,
                "evaluator_called": False,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        else:
            task = trial["task"]
            if task not in resources:
                evaluation, _ = build_task(task, eval_workers=None)
                resources[task] = (evaluation, SecureEvaluator(evaluation))
            _, evaluator = resources[task]
            if generation["candidate_code_hash"] == generation["parent_code_hash"]:
                result = {
                    **generation,
                    "status": "valid",
                    "valid": True,
                    "failure_kind": None,
                    "child_fitness": generation["parent_fitness"],
                    "child_q": generation["parent_q"],
                    "delta_q": 0.0,
                    "improved": False,
                    "evaluator_called": False,
                    "no_op": True,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
            else:
                outcome, seconds = evaluator.evaluate_program_record_time_with_details(
                    generation["candidate_code"]
                )
                if outcome.failure_kind == "prepare_error":
                    raise RuntimeError(
                        f"evaluator infrastructure failure for {trial['trial_id']}: "
                        f"{outcome.error}"
                    )
                score = getattr(outcome.result, "fitness", outcome.result)
                try:
                    child_fitness = float(score)
                except (TypeError, ValueError, OverflowError):
                    child_fitness = math.nan
                if math.isfinite(child_fitness):
                    delta_q = child_fitness - float(generation["parent_q"])
                    result = {
                        **generation,
                        "status": "valid",
                        "valid": True,
                        "failure_kind": None,
                        "child_fitness": child_fitness,
                        "child_q": child_fitness,
                        "delta_q": delta_q,
                        "improved": delta_q > 0.0,
                        "evaluator_called": True,
                        "evaluate_seconds": seconds,
                        "no_op": False,
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                    }
                else:
                    result = {
                        **generation,
                        "status": "invalid",
                        "valid": False,
                        "failure_kind": outcome.failure_kind or "invalid_result",
                        "failure_error": outcome.error,
                        "evaluator_called": True,
                        "evaluate_seconds": seconds,
                        "completed_at": datetime.now().isoformat(timespec="seconds"),
                    }
        _append_jsonl(result_path, result)
        completed.add(trial["trial_id"])
        _write_json(
            run_dir / "evaluation_summary.json",
            {
                "status": (
                    "finished" if len(completed) == len(schedule) else "running"
                ),
                "completed_trials": len(completed),
                "total_trials": len(schedule),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(
            f"evaluation trial={trial['trial_id']} status={result['status']} "
            f"completed={len(completed)}/{len(schedule)}",
            flush=True,
        )


def smoke_evaluate_generated(run_dir: Path, *, limit: int) -> None:
    generated: list[dict[str, Any]] = []
    for path in sorted((run_dir / "generations").glob("shard_*.jsonl")):
        generated.extend(_read_jsonl(path))
    selected = [
        row for row in generated if row["generation_status"] == "generated"
    ][:limit]
    if len(selected) != limit:
        raise ValueError(f"only {len(selected)} parsed generations are available")
    resources: dict[str, SecureEvaluator] = {}
    for row in selected:
        if row["task"] not in resources:
            evaluation, _ = build_task(row["task"], eval_workers=None)
            resources[row["task"]] = SecureEvaluator(evaluation)
        outcome, seconds = resources[
            row["task"]
        ].evaluate_program_record_time_with_details(row["candidate_code"])
        print(
            json.dumps(
                {
                    "trial_id": row["trial_id"],
                    "task": row["task"],
                    "failure_kind": outcome.failure_kind,
                    "fitness": getattr(outcome.result, "fitness", outcome.result),
                    "evaluate_seconds": seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def status(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    generated: list[dict[str, Any]] = []
    for path in sorted((run_dir / "generations").glob("shard_*.jsonl")):
        generated.extend(_read_jsonl(path))
    result_path = run_dir / "results" / "results.jsonl"
    results = _read_jsonl(result_path) if result_path.is_file() else []
    return {
        "protocol_id": config["protocol_id"],
        "total_trials": int(config["trial_count"]),
        "generated_unique_trials": len({row["trial_id"] for row in generated}),
        "parse_failed": sum(
            row.get("generation_status") == "parse_failed" for row in generated
        ),
        "evaluated_unique_trials": len({row["trial_id"] for row in results}),
        "valid_trials": sum(row.get("valid") is True for row in results),
        "invalid_trials": sum(row.get("valid") is False for row in results),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", type=Path, required=True)
    prepare.add_argument("--base-probe", type=Path, default=BASE_PROBE)
    prepare.add_argument("--replicates", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED_V2)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-dir", type=Path, required=True)
    generate.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    smoke = subparsers.add_parser("smoke-evaluate")
    smoke.add_argument("--run-dir", type=Path, required=True)
    smoke.add_argument("--limit", type=int, default=2)
    inspect = subparsers.add_parser("status")
    inspect.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_probe(
            args.run_dir,
            base_probe=args.base_probe,
            replicates=args.replicates,
            seed=args.seed,
        )
    elif args.command == "generate":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index must be in [0, num_shards)")
        generate_shard(
            args.run_dir,
            backend=args.backend,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.command == "evaluate":
        evaluate_generated(args.run_dir)
    elif args.command == "smoke-evaluate":
        smoke_evaluate_generated(args.run_dir, limit=args.limit)
    else:
        print(json.dumps(status(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
