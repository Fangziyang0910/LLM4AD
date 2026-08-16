"""Streaming V9.8 Stage-P mechanism probes.

P1/P2 use a paired 2x2 fixed-anchor design:
``code_only/parent_path`` x ``refine/explore``.  Every model response is parsed
and evaluated immediately by the same shard process, then appended atomically.
P3 continuation support is prepared from the completed Explore trials by the
companion commands in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9.complexity import code_change_ratio
from llm4ad.method.traceaad_v9_7.forest import Forest as V97Forest
from llm4ad.method.traceaad_v9_7.history import _compact_change
from llm4ad.method.traceaad_v9_7.schema import Outcome as V97Outcome
from llm4ad.method.traceaad_v9_8.prompt import (
    ProgramResponseError,
    build_generation_prompt,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_8.schema import Intent
from llm4ad.method.traceaad_v9_8.source import code_diff, code_hash

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    build_llm_client,
    build_task,
    resolve_backend,
)

PROTOCOL_ID = "traceaad-v9.8-stage-p1-p2-streaming-v1"
SOURCE_BATCH = "20260814_150927"
LOGICAL_MODEL_NAME = "Qwen3.6-27B"
STRATA = ("low", "middle", "high")
HISTORY_MODES = ("code_only", "parent_path")
INTENTS = (Intent.REFINE.value, Intent.EXPLORE.value)
CONDITIONS = tuple(f"{history}_{intent}" for history in HISTORY_MODES for intent in INTENTS)
CONDITION_SEQUENCES = (
    CONDITIONS,
    (CONDITIONS[1], CONDITIONS[2], CONDITIONS[3], CONDITIONS[0]),
    (CONDITIONS[2], CONDITIONS[3], CONDITIONS[0], CONDITIONS[1]),
    (CONDITIONS[3], CONDITIONS[0], CONDITIONS[1], CONDITIONS[2]),
)
ANCHORS_PER_STRATUM = 6
REPLICATES = 3
DESIGN_SEED = 980501
OUTPUT_TOKENS = 8192
TOTAL_CONTEXT_TOKENS = 32768
TRANSPORT_RETRIES = 3


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _formal_run_dirs(task: str, batch: str) -> list[Path]:
    root = EXPERIMENTS_ROOT / task / "traceaad_v9_7"
    runs = sorted(root.glob(f"v9_7_{batch}_{task}_rep*"))
    return [
        run
        for run in runs
        if run.is_dir()
        and (run / "logs" / "summary.json").is_file()
        and _read_json(run / "logs" / "summary.json").get("status") == "finished"
    ]


def _render_v98_path(forest: V97Forest, anchor_id: int, *, max_events: int = 8) -> str:
    path_anchor_ids = forest.ancestor_ids(anchor_id)
    current_hypothesis = 0
    rendered: list[tuple[Any, int, int]] = []
    for child_anchor_id in path_anchor_ids[1:]:
        child_anchor = forest.get_anchor(child_anchor_id)
        if child_anchor.attempt_id is None:
            continue
        attempt = forest.get_attempt(child_anchor.attempt_id)
        if attempt.intent not in {Intent.REFINE.value, Intent.EXPLORE.value}:
            raise ValueError("source path contains a formation event without an intent")
        source_hypothesis = current_hypothesis
        if attempt.intent == Intent.EXPLORE.value:
            current_hypothesis += 1
        rendered.append((attempt, source_hypothesis, current_hypothesis))
    rendered = rendered[-max_events:]
    lines = ["[Recent Algorithm Formation Path]"]
    for index, (attempt, source_hypothesis, child_hypothesis) in enumerate(
        rendered, start=1
    ):
        boundary = (
            f"inherit H{source_hypothesis}"
            if attempt.intent == Intent.REFINE.value
            else f"create H{child_hypothesis} from H{source_hypothesis}"
        )
        lines.extend(
            [
                "",
                f"[History {index}] Formation step",
                f"Operator: {attempt.intent.capitalize()}",
                f"Hypothesis: {boundary}",
                f"Idea: {attempt.idea or 'unavailable'}",
                f"Change: {_compact_change(attempt)}",
                f"Result: {attempt.outcome.value}",
                f"Fitness: {attempt.parent_fitness:.6g} -> {attempt.child_fitness:.6g}",
            ]
        )
    return "\n".join(lines)


def extract_snapshot_pool(task: str, batch: str) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for run in _formal_run_dirs(task, batch):
        config = _read_json(run / "run_config.json")
        checkpoint = _read_json(run / "checkpoints" / "latest.json")
        forest = V97Forest.from_dict(checkpoint["forest"])
        s0 = float(checkpoint["s"] or 0.0)
        for anchor in forest.anchors():
            path_ids = forest.parent_path_ids(anchor.id)
            if len(path_ids) < 2:
                continue
            attempts = [forest.get_attempt(attempt_id) for attempt_id in path_ids]
            if any(
                attempt.outcome in {None, V97Outcome.INVALID}
                or attempt.intent not in {Intent.REFINE.value, Intent.EXPLORE.value}
                for attempt in attempts
            ):
                continue
            program = forest.get_program(anchor.program_id)
            snapshots.append(
                {
                    "snapshot_id": f"{task}:{run.name}:anchor_{anchor.id}",
                    "task": task,
                    "source_run": run.name,
                    "source_repeat": int(config["repeat"]),
                    "anchor_state_id": anchor.id,
                    "program_id": program.id,
                    "code_hash": code_hash(program.code),
                    "code": program.code,
                    "fitness": program.fitness,
                    "q": program.q,
                    "depth": len(path_ids),
                    "history_event_count": min(8, len(path_ids)),
                    "history_text": _render_v98_path(forest, anchor.id),
                    "source_s0": s0,
                }
            )
    return snapshots


def _deduplicate_snapshots(
    snapshots: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in snapshots:
        by_code[item["code_hash"]].append(item)
    chosen: list[dict[str, Any]] = []
    for code_hash_value in sorted(by_code):
        candidates = sorted(by_code[code_hash_value], key=lambda row: row["snapshot_id"])
        chosen.append(candidates[rng.randrange(len(candidates))])
    return chosen


def _rank_tertiles(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (row["q"], row["snapshot_id"]))
    base, extra = divmod(len(ordered), 3)
    sizes = [base + (1 if index < extra else 0) for index in range(3)]
    result: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name, size in zip(STRATA, sizes, strict=True):
        result[name] = ordered[offset : offset + size]
        offset += size
    return result


def _balanced_sample(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repeat[int(row["source_repeat"])].append(row)
    repeats = sorted(by_repeat)
    if not repeats:
        raise ValueError("eligible stratum has no completed snapshots")
    for candidates in by_repeat.values():
        rng.shuffle(candidates)
    base, extra = divmod(count, len(repeats))
    quota = {
        repeat: base + (1 if index < extra else 0)
        for index, repeat in enumerate(repeats)
    }
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for repeat in repeats:
        candidates = by_repeat[repeat]
        take = min(quota[repeat], len(candidates))
        selected.extend(candidates[:take])
        remaining.extend(candidates[take:])
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    if len(selected) != count:
        raise ValueError(f"eligible stratum has fewer than {count} unique programs")
    return selected


def select_anchors(
    *, batch: str, anchors_per_stratum: int, seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        rng = random.Random(seed + 1009 * (task_index + 1))
        pool = _deduplicate_snapshots(extract_snapshot_pool(task, batch), rng)
        strata = _rank_tertiles(pool)
        for stratum in STRATA:
            for row in _balanced_sample(strata[stratum], anchors_per_stratum, rng):
                selected.append({**row, "stratum": stratum})
    selected.sort(
        key=lambda row: (row["task"], STRATA.index(row["stratum"]), row["snapshot_id"])
    )
    for index, row in enumerate(selected):
        row["anchor_id"] = f"anchor_{index:03d}"
    return selected


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
        for index, item in enumerate(group):
            item["condition_order"] = list(
                CONDITION_SEQUENCES[index % len(CONDITION_SEQUENCES)]
            )
            blocks.append(item)
    rng.shuffle(blocks)
    schedule: list[dict[str, Any]] = []
    for block_order, block in enumerate(blocks):
        anchor = block["anchor"]
        block_id = f"{anchor['anchor_id']}_rep{block['replicate']}"
        sampling_seed = seed * 1000 + block_order + 1
        for within_block_order, condition in enumerate(block["condition_order"]):
            history_mode, intent = condition.rsplit("_", 1)
            schedule.append(
                {
                    "trial_id": f"{block_id}_{condition}",
                    "block_id": block_id,
                    "block_order": block_order,
                    "within_block_order": within_block_order,
                    "anchor_id": anchor["anchor_id"],
                    "task": anchor["task"],
                    "stratum": anchor["stratum"],
                    "replicate": block["replicate"],
                    "condition": condition,
                    "history_mode": history_mode,
                    "intent": intent,
                    "sampling_seed": sampling_seed,
                }
            )
    return schedule


def _prompt(anchor: dict[str, Any], task_description: str, condition: str) -> str:
    history_mode, intent_text = condition.rsplit("_", 1)
    history_text = anchor["history_text"] if history_mode == "parent_path" else ""
    return build_generation_prompt(
        task_description=task_description,
        code=anchor["code"],
        fitness=float(anchor["fitness"]),
        history_text=history_text,
        intent=Intent(intent_text),
        maximize=True,
    )


def prepare_p12(
    run_dir: Path,
    *,
    batch: str = SOURCE_BATCH,
    anchors_per_stratum: int = ANCHORS_PER_STRATUM,
    replicates: int = REPLICATES,
    seed: int = DESIGN_SEED,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    anchors = select_anchors(
        batch=batch, anchors_per_stratum=anchors_per_stratum, seed=seed
    )
    schedule = build_schedule(anchors, replicates=replicates, seed=seed)
    audit: list[dict[str, Any]] = []
    for anchor in anchors:
        evaluation, _ = build_task(anchor["task"], eval_workers=1)
        prompts = {
            condition: _prompt(anchor, evaluation.task_description, condition)
            for condition in CONDITIONS
        }
        for intent in INTENTS:
            without = prompts[f"code_only_{intent}"]
            with_path = prompts[f"parent_path_{intent}"]
            if with_path.replace(anchor["history_text"], "", 1) != without:
                raise AssertionError("history condition changes prompt outside the path")
        audit.append(
            {
                "anchor_id": anchor["anchor_id"],
                "history_hash": _text_hash(anchor["history_text"]),
                "prompt_hashes": {key: _text_hash(value) for key, value in prompts.items()},
            }
        )
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "anchors.jsonl", anchors)
    _write_jsonl(run_dir / "schedule.jsonl", schedule)
    _write_jsonl(run_dir / "prompt_audit.jsonl", audit)
    _write_json(
        run_dir / "probe_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_batch": batch,
            "source_method": "traceaad_v9_7",
            "logical_model_name": LOGICAL_MODEL_NAME,
            "conditions": list(CONDITIONS),
            "tasks": list(TASKS),
            "strata": list(STRATA),
            "anchors_per_stratum": anchors_per_stratum,
            "anchor_count": len(anchors),
            "replicates_per_anchor_condition": replicates,
            "trial_count": len(schedule),
            "design_seed": seed,
            "temperature": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "max_total_context": TOTAL_CONTEXT_TOKENS,
            "generation_evaluation_pipeline": "same_process_immediate_per_trial",
            "independent_unit": "fixed anchor snapshot",
            "within_anchor_repeated_sampling": replicates,
            "blocking": "task x quality stratum; paired seed within anchor replicate",
        },
    )


def _draw_call(*, trial: dict[str, Any], prompt: str, llm) -> dict[str, Any]:
    started = time.time()
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
    return {
        "trial_id": trial["trial_id"],
        "prompt": prompt,
        "prompt_hash": _text_hash(prompt),
        "prompt_tokens": int(llm.count_prompt_tokens(prompt)),
        "response": response,
        "response_hash": _text_hash(response),
        "response_tokens": int(llm.count_tokens(response)),
        "sampling_seed": trial["sampling_seed"],
        "sample_seconds": time.time() - started,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _evaluate_call(
    *,
    trial: dict[str, Any],
    anchor: dict[str, Any],
    call: dict[str, Any],
    evaluation,
    evaluator: SecureEvaluator,
) -> dict[str, Any]:
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    if template is None or len(template.functions) != 1:
        raise ValueError("probe requires one evolvable template function")
    base = {
        **trial,
        "protocol_id": PROTOCOL_ID,
        "snapshot_id": anchor["snapshot_id"],
        "source_run": anchor["source_run"],
        "source_repeat": anchor["source_repeat"],
        "parent_fitness": anchor["fitness"],
        "parent_q": anchor["q"],
        "parent_code_hash": anchor["code_hash"],
        "history_event_count": anchor["history_event_count"],
        "history_hash": _text_hash(anchor["history_text"]),
        "source_s0": anchor["source_s0"],
        "prompt_hash": call["prompt_hash"],
        "prompt_tokens": call["prompt_tokens"],
        "response_hash": call["response_hash"],
        "response_tokens": call["response_tokens"],
        "sample_seconds": call["sample_seconds"],
    }
    try:
        parsed = parse_program_response(call["response"], template, template.functions[0].name)
    except ProgramResponseError as exc:
        return {
            **base,
            "status": "invalid",
            "valid": False,
            "failure_kind": "parse",
            "failure_error": str(exc),
            "idea": exc.declared_idea,
            "evaluator_called": False,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
    candidate_code = str(parsed.program)
    candidate_hash = code_hash(candidate_code)
    diff, added, removed = code_diff(anchor["code"], candidate_code)
    candidate_base = {
        **base,
        "idea": parsed.declared_idea,
        "candidate_code_hash": candidate_hash,
        "candidate_code": candidate_code,
        "diff": diff,
        "added": added,
        "removed": removed,
        "change_ratio": code_change_ratio(anchor["code"], candidate_code),
    }
    if candidate_hash == anchor["code_hash"]:
        return {
            **candidate_base,
            "status": "valid",
            "valid": True,
            "failure_kind": None,
            "child_fitness": anchor["fitness"],
            "child_q": anchor["q"],
            "delta_q": 0.0,
            "improved": False,
            "evaluator_called": False,
            "no_op": True,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
    outcome, evaluate_seconds = evaluator.evaluate_program_record_time_with_details(
        candidate_code
    )
    if outcome.failure_kind == "prepare_error":
        raise RuntimeError(
            f"evaluator infrastructure failure for {trial['trial_id']}: {outcome.error}"
        )
    score = getattr(outcome.result, "fitness", outcome.result)
    try:
        child_fitness = float(score)
    except (TypeError, ValueError, OverflowError):
        child_fitness = math.nan
    if not math.isfinite(child_fitness):
        return {
            **candidate_base,
            "status": "invalid",
            "valid": False,
            "failure_kind": outcome.failure_kind or "invalid_result",
            "failure_error": outcome.error,
            "evaluator_called": True,
            "evaluate_seconds": evaluate_seconds,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
    child_q = child_fitness
    delta_q = child_q - float(anchor["q"])
    return {
        **candidate_base,
        "status": "valid",
        "valid": True,
        "failure_kind": None,
        "child_fitness": child_fitness,
        "child_q": child_q,
        "delta_q": delta_q,
        "improved": delta_q > 0.0,
        "evaluator_called": True,
        "evaluate_seconds": evaluate_seconds,
        "no_op": False,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_p12_shard(
    run_dir: Path,
    *,
    backend: str,
    shard_index: int,
    num_shards: int,
    eval_workers: int,
) -> None:
    config = _read_json(run_dir / "probe_config.json")
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("probe configuration protocol mismatch")
    anchors = {row["anchor_id"]: row for row in _read_jsonl(run_dir / "anchors.jsonl")}
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["block_order"]) % num_shards == shard_index
    ]
    result_path = run_dir / "results" / f"shard_{shard_index:02d}.jsonl"
    call_path = run_dir / "llm_calls" / f"shard_{shard_index:02d}.jsonl"
    completed = {row["trial_id"] for row in _read_jsonl(result_path)}
    calls = {row["trial_id"]: row for row in _read_jsonl(call_path)}
    profile = resolve_backend(backend, None, None, None)
    llm = build_llm_client(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=OUTPUT_TOKENS,
        temperature=1.0,
    )
    resources: dict[str, tuple[Any, SecureEvaluator]] = {}
    shard_dir = run_dir / "shards" / f"shard_{shard_index:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        shard_dir / "shard_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "logical_model_name": LOGICAL_MODEL_NAME,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "trial_count": len(schedule),
            "eval_workers": eval_workers,
            "pipeline": "generate_then_immediately_evaluate_each_trial",
        },
    )
    try:
        for trial in schedule:
            if trial["trial_id"] in completed:
                continue
            task = trial["task"]
            if task not in resources:
                evaluation, _ = build_task(task, eval_workers=eval_workers)
                resources[task] = evaluation, SecureEvaluator(evaluation)
            evaluation, evaluator = resources[task]
            anchor = anchors[trial["anchor_id"]]
            prompt = _prompt(anchor, evaluation.task_description, trial["condition"])
            prompt_tokens = int(llm.count_prompt_tokens(prompt))
            if prompt_tokens + OUTPUT_TOKENS > TOTAL_CONTEXT_TOKENS:
                raise ValueError(f"trial {trial['trial_id']} exceeds total context")
            call = calls.get(trial["trial_id"])
            if call is None:
                call = _draw_call(trial=trial, prompt=prompt, llm=llm)
                _append_jsonl(call_path, call)
                calls[trial["trial_id"]] = call
            result = _evaluate_call(
                trial=trial,
                anchor=anchor,
                call=call,
                evaluation=evaluation,
                evaluator=evaluator,
            )
            _append_jsonl(result_path, result)
            completed.add(trial["trial_id"])
            _write_json(
                shard_dir / "summary.json",
                {
                    "status": "finished" if len(completed) == len(schedule) else "running",
                    "completed_trials": len(completed),
                    "assigned_trials": len(schedule),
                    "valid_trials": sum(
                        row.get("valid") is True for row in _read_jsonl(result_path)
                    ),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(
                f"p12 shard={shard_index} trial={trial['trial_id']} "
                f"status={result['status']} completed={len(completed)}/{len(schedule)}",
                flush=True,
            )
    finally:
        llm.close()


def status_p12(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    results: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        results.extend(_read_jsonl(path))
    return {
        "protocol_id": config["protocol_id"],
        "completed_trials": len({row["trial_id"] for row in results}),
        "total_trials": int(config["trial_count"]),
        "valid_trials": sum(row.get("valid") is True for row in results),
        "invalid_trials": sum(row.get("valid") is False for row in results),
        "evaluator_calls": sum(row.get("evaluator_called") is True for row in results),
        "by_condition": {
            condition: sum(row["condition"] == condition for row in results)
            for condition in CONDITIONS
        },
    }


def _default_run_dir() -> Path:
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_v98_p1_p2"
    return EXPERIMENTS_ROOT / "generation_probe" / name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-p12")
    prepare.add_argument("--run-dir", type=Path)
    prepare.add_argument("--source-batch", default=SOURCE_BATCH)
    prepare.add_argument("--anchors-per-stratum", type=int, default=ANCHORS_PER_STRATUM)
    prepare.add_argument("--replicates", type=int, default=REPLICATES)
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED)
    run = subparsers.add_parser("run-p12")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--eval-workers", type=int, default=1)
    inspect = subparsers.add_parser("status-p12")
    inspect.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-p12":
        run_dir = args.run_dir or _default_run_dir()
        prepare_p12(
            run_dir,
            batch=args.source_batch,
            anchors_per_stratum=args.anchors_per_stratum,
            replicates=args.replicates,
            seed=args.seed,
        )
        print(run_dir.resolve())
    elif args.command == "run-p12":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index must be in [0, num_shards)")
        run_p12_shard(
            args.run_dir,
            backend=args.backend,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            eval_workers=args.eval_workers,
        )
    else:
        print(json.dumps(status_p12(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
