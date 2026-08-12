"""Fixed-anchor A/B probe for concise TraceAAD history.

The probe is deliberately independent from the V9.5 search implementation.  It
reuses immutable facts from recorded V9.5 snapshots, then asks the same model to
modify the same executable anchor with or without a concise history block.
"""

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
from typing import Any, Iterable

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9.complexity import code_change_ratio
from llm4ad.method.traceaad_v9_5.prompt import (
    ProgramResponseError,
    _output_contract,
    fitness_direction_hint,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_5.source import nonempty_loc, text_hash

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    TASK_SHORT,
    build_llm_client,
    build_task,
    resolve_backend,
)


FORMAL_BATCH = "20260811_171029"
PROTOCOL_ID = "traceaad-v95-fixed-anchor-concise-history-probe-v1"
LOGICAL_MODEL_NAME = "Qwen3.6-27B"
CONDITIONS = ("no_history", "concise_history")
STRATA = ("low", "middle", "high")
ANCHORS_PER_STRATUM = 6
REPLICATES = 3
DESIGN_SEED = 950501
MAX_EVIDENCE_ITEMS = 8
IDEA_RENDER_CHARS = 300
OUTPUT_TOKENS = 8192
TOTAL_CONTEXT_TOKENS = 32768
TRANSPORT_RETRIES = 3


@dataclass(frozen=True, slots=True)
class PromptPair:
    no_history: str
    concise_history: str
    canonical_without_history: str
    history_block: str


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _one_line(value: Any, limit: int = IDEA_RENDER_CHARS) -> str:
    text = " ".join(str(value or "").split()) or "not recorded"
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _format_fitness(value: Any) -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):.12g}"


def render_concise_history(anchor: dict[str, Any]) -> str:
    lines = ["[Concise Search History]"]
    for index, item in enumerate(anchor["history"], start=1):
        lines.extend(
            [
                f"Event {index}",
                f"Source: {item['source']}",
                f"Idea: {_one_line(item.get('idea'))}",
                f"Result: {item['result']}",
                (
                    "Fitness: "
                    f"{_format_fitness(item.get('parent_fitness'))} -> "
                    f"{_format_fitness(item.get('child_fitness'))}"
                ),
            ]
        )
    return "\n".join(lines)


def build_prompt_pair(anchor: dict[str, Any], task_description: str) -> PromptPair:
    prefix = "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(True),
            "",
            "[Current Fitness]",
            _format_fitness(anchor["fitness"]),
            "",
            "[Current Code]",
            "```python",
            str(anchor["code"]).rstrip(),
            "```",
            "",
        ]
    )
    suffix = "\n".join(
        [
            "[Instruction]",
            (
                "Improve the current algorithm. Propose one coherent modification "
                "that may improve fitness."
            ),
            (
                "If search history is provided, treat it as evidence rather than a "
                "strict prohibition; a previously unsuccessful idea may be revisited "
                "with a materially different implementation."
            ),
            "Keep the target function signature and contract unchanged.",
            "Return one complete, self-contained implementation.",
            _output_contract(),
        ]
    )
    history = render_concise_history(anchor)
    no_history = prefix + suffix
    with_history = prefix + history + "\n\n" + suffix
    canonical = prefix + suffix
    if with_history.replace(history + "\n\n", "", 1) != no_history:
        raise AssertionError("A/B prompts differ outside the concise history block")
    return PromptPair(
        no_history=no_history,
        concise_history=with_history,
        canonical_without_history=canonical,
        history_block=history,
    )


def _formal_run_dirs(task: str, batch: str) -> list[Path]:
    root = EXPERIMENTS_ROOT / task / "traceaad_v9_5"
    short = TASK_SHORT[task]
    runs = sorted(root.glob(f"v9_5_{batch}_{short}_rep*"))
    return [
        run
        for run in runs
        if run.is_dir()
        and (run / "logs" / "summary.json").is_file()
        and _read_json(run / "logs" / "summary.json").get("status") == "finished"
    ]


def _history_items(
    attempts: dict[int, dict[str, Any]], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for source, key in (
        ("Formation", "selected_formation_ids"),
        ("Direct", "selected_direct_ids"),
    ):
        for attempt_id in evidence[key]:
            attempt = attempts[int(attempt_id)]
            items.append(
                {
                    "attempt_id": int(attempt_id),
                    "source": source,
                    "idea": attempt.get("declared_idea"),
                    "result": attempt.get("direct_outcome") or "invalid",
                    "parent_fitness": attempt.get("parent_fitness"),
                    "child_fitness": attempt.get("child_fitness"),
                }
            )
    return items


def extract_snapshot_pool(task: str, batch: str) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for run in _formal_run_dirs(task, batch):
        config = _read_json(run / "run_config.json")
        checkpoint = _read_json(run / "checkpoints" / "latest.json")
        forest = checkpoint["forest"]
        states = {int(item["state_id"]): item for item in forest["states"]}
        artifacts = {int(item["artifact_id"]): item for item in forest["artifacts"]}
        attempts = {int(item["attempt_id"]): item for item in forest["attempts"]}
        pending_selection: dict[str, Any] | None = None
        for decision_order, event in enumerate(
            _read_jsonl(run / "artifacts" / "decisions.jsonl")
        ):
            if event.get("event") == "anchor_selected":
                pending_selection = event
                continue
            if event.get("event") != "evidence_built" or pending_selection is None:
                continue
            state_id = int(event["anchor_state_id"])
            if state_id != int(pending_selection["selected_state_id"]):
                continue
            history = _history_items(attempts, event)
            if len(history) >= 2:
                state = states[state_id]
                artifact = artifacts[int(state["artifact_id"])]
                snapshots.append(
                    {
                        "snapshot_id": (
                            f"{task}:{run.name}:iteration_"
                            f"{int(pending_selection['iteration'])}:state_{state_id}"
                        ),
                        "task": task,
                        "source_run": run.name,
                        "source_repeat": int(config["repeat"]),
                        "source_iteration": int(pending_selection["iteration"]),
                        "state_id": state_id,
                        "artifact_id": int(artifact["artifact_id"]),
                        "code_hash": str(artifact["evaluator_input_hash"]),
                        "code": str(artifact["evaluator_input_code"]),
                        "fitness": float(artifact["fitness"]),
                        "q": float(artifact["directed_fitness"]),
                        "depth": int(state["depth"]),
                        "history": history[:MAX_EVIDENCE_ITEMS],
                        "selected_formation_ids": list(event["selected_formation_ids"]),
                        "selected_direct_ids": list(event["selected_direct_ids"]),
                        "source_decision_order": decision_order,
                    }
                )
            pending_selection = None
    return snapshots


def _deduplicate_snapshots(
    snapshots: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in snapshots:
        by_code[item["code_hash"]].append(item)
    chosen = []
    for code_hash in sorted(by_code):
        candidates = sorted(by_code[code_hash], key=lambda row: row["snapshot_id"])
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
        raise ValueError("eligible stratum has no completed formal-run snapshots")
    for candidates in by_repeat.values():
        rng.shuffle(candidates)
    order = repeats[:]
    rng.shuffle(order)
    base, extra = divmod(count, len(repeats))
    quota = {
        repeat: base + (1 if index < extra else 0)
        for index, repeat in enumerate(order)
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
        raise ValueError(
            f"eligible stratum has {len(rows)} snapshots, fewer than required {count}"
        )
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
            chosen = _balanced_sample(strata[stratum], anchors_per_stratum, rng)
            for row in chosen:
                row = dict(row)
                row["stratum"] = stratum
                selected.append(row)
    selected.sort(key=lambda row: (row["task"], STRATA.index(row["stratum"]), row["snapshot_id"]))
    for index, row in enumerate(selected):
        row["anchor_id"] = f"anchor_{index:03d}"
    return selected


def build_schedule(
    anchors: list[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 7919)
    blocks: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        for replicate in range(1, replicates + 1):
            grouped[(anchor["task"], anchor["stratum"])].append(
                {"anchor": anchor, "replicate": replicate}
            )
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
    rows: list[dict[str, Any]] = []
    for pair_order, block in enumerate(blocks):
        anchor = block["anchor"]
        pair_id = f"{anchor['anchor_id']}_rep{block['replicate']}"
        sampling_seed = seed * 1000 + pair_order + 1
        for within_pair_order, condition in enumerate(block["condition_order"]):
            rows.append(
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
    return rows


def prepare_probe(
    run_dir: Path,
    *,
    batch: str = FORMAL_BATCH,
    anchors_per_stratum: int = ANCHORS_PER_STRATUM,
    replicates: int = REPLICATES,
    seed: int = DESIGN_SEED,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    anchors = select_anchors(
        batch=batch, anchors_per_stratum=anchors_per_stratum, seed=seed
    )
    schedule = build_schedule(anchors, replicates=replicates, seed=seed)
    prompt_audit = []
    for anchor in anchors:
        evaluation, _ = build_task(anchor["task"], eval_workers=1)
        pair = build_prompt_pair(anchor, evaluation.task_description)
        if text_hash(pair.no_history) != text_hash(pair.canonical_without_history):
            raise AssertionError("No-History prompt does not equal canonical prompt")
        if pair.concise_history.replace(pair.history_block + "\n\n", "", 1) != pair.no_history:
            raise AssertionError("prompt pair differs outside history block")
        prompt_audit.append(
            {
                "anchor_id": anchor["anchor_id"],
                "canonical_without_history_hash": text_hash(
                    pair.canonical_without_history
                ),
                "no_history_prompt_hash": text_hash(pair.no_history),
                "concise_history_prompt_hash": text_hash(pair.concise_history),
                "history_block_hash": text_hash(pair.history_block),
            }
        )
    _write_jsonl(run_dir / "anchors.jsonl", anchors)
    _write_jsonl(run_dir / "schedule.jsonl", schedule)
    _write_jsonl(run_dir / "prompt_audit.jsonl", prompt_audit)
    _write_json(
        run_dir / "probe_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_batch": batch,
            "logical_model_name": LOGICAL_MODEL_NAME,
            "conditions": list(CONDITIONS),
            "tasks": list(TASKS),
            "strata": list(STRATA),
            "stratum_definition": "equal-count rank tertiles of directed q within each task's eligible deduplicated snapshot pool",
            "eligibility": "recorded anchor snapshot with at least two selected evidence events",
            "anchors_per_stratum": anchors_per_stratum,
            "anchor_count": len(anchors),
            "replicates_per_anchor_condition": replicates,
            "trial_count": len(schedule),
            "design_seed": seed,
            "max_evidence_items": MAX_EVIDENCE_ITEMS,
            "idea_render_chars": IDEA_RENDER_CHARS,
            "temperature": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "max_total_context": TOTAL_CONTEXT_TOKENS,
            "independent_unit": "anchor snapshot",
            "within_anchor_repeated_sampling": replicates,
            "delta_q": "q_child - q_parent",
        },
    )


def _response_tokens(llm, response: str) -> int | None:
    try:
        return int(llm.count_tokens(response))
    except Exception:
        return None


def _run_trial(
    *,
    trial: dict[str, Any],
    anchor: dict[str, Any],
    llm,
    evaluation,
    evaluator: SecureEvaluator,
) -> dict[str, Any]:
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    if template is None or len(template.functions) != 1:
        raise ValueError("probe requires one evolvable template function")
    pair = build_prompt_pair(anchor, evaluation.task_description)
    prompt = (
        pair.no_history
        if trial["condition"] == "no_history"
        else pair.concise_history
    )
    canonical_hash = text_hash(pair.canonical_without_history)
    if text_hash(pair.no_history) != canonical_hash:
        raise AssertionError("A prompt failed canonical integrity check")
    if pair.concise_history.replace(pair.history_block + "\n\n", "", 1) != pair.no_history:
        raise AssertionError("B prompt failed canonical integrity check")
    prompt_tokens = int(llm.count_prompt_tokens(prompt))
    if prompt_tokens + OUTPUT_TOKENS > TOTAL_CONTEXT_TOKENS:
        raise ValueError(
            f"trial {trial['trial_id']} exceeds total context: "
            f"{prompt_tokens}+{OUTPUT_TOKENS}>{TOTAL_CONTEXT_TOKENS}"
        )
    response = ""
    start = time.time()
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
    sample_seconds = time.time() - start
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
        "canonical_without_history_hash": canonical_hash,
        "prompt_hash": text_hash(prompt),
        "history_block_hash": text_hash(pair.history_block),
        "prompt_tokens": prompt_tokens,
        "response_hash": text_hash(response),
        "response": response,
        "response_tokens": _response_tokens(llm, response),
        "sample_seconds": sample_seconds,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        parsed = parse_program_response(
            response, template, template.functions[0].name
        )
    except ProgramResponseError as exc:
        return {
            **base,
            "status": "invalid",
            "valid": False,
            "failure_kind": "parse",
            "failure_error": str(exc),
            "idea": exc.declared_idea,
            "evaluator_called": False,
        }
    candidate_code = str(parsed.program)
    candidate_hash = text_hash(candidate_code)
    change_ratio = code_change_ratio(anchor["code"], candidate_code)
    loc_delta = nonempty_loc(candidate_code) - nonempty_loc(anchor["code"])
    if candidate_hash == anchor["code_hash"]:
        child_fitness = float(anchor["fitness"])
        child_q = float(anchor["q"])
        return {
            **base,
            "status": "valid",
            "valid": True,
            "failure_kind": None,
            "idea": parsed.declared_idea,
            "candidate_code_hash": candidate_hash,
            "candidate_code": candidate_code,
            "child_fitness": child_fitness,
            "child_q": child_q,
            "delta_q": 0.0,
            "improved": False,
            "change_ratio": change_ratio,
            "loc_delta": loc_delta,
            "evaluator_called": False,
            "no_op": True,
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
            **base,
            "status": "invalid",
            "valid": False,
            "failure_kind": outcome.failure_kind or "invalid_result",
            "failure_error": outcome.error,
            "idea": parsed.declared_idea,
            "candidate_code_hash": candidate_hash,
            "candidate_code": candidate_code,
            "change_ratio": change_ratio,
            "loc_delta": loc_delta,
            "evaluator_called": True,
            "evaluate_seconds": evaluate_seconds,
        }
    child_q = child_fitness
    delta_q = child_q - float(anchor["q"])
    return {
        **base,
        "status": "valid",
        "valid": True,
        "failure_kind": None,
        "idea": parsed.declared_idea,
        "candidate_code_hash": candidate_hash,
        "candidate_code": candidate_code,
        "child_fitness": child_fitness,
        "child_q": child_q,
        "delta_q": delta_q,
        "improved": delta_q > 0.0,
        "change_ratio": change_ratio,
        "loc_delta": loc_delta,
        "evaluator_called": True,
        "evaluate_seconds": evaluate_seconds,
        "no_op": False,
    }



def run_shard(
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
    anchors = {
        row["anchor_id"]: row for row in _read_jsonl(run_dir / "anchors.jsonl")
    }
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["pair_order"]) % num_shards == shard_index
    ]
    result_path = run_dir / "results" / f"shard_{shard_index:02d}.jsonl"
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
    task_resources: dict[str, tuple[Any, SecureEvaluator]] = {}
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
        },
    )
    try:
        for trial in schedule:
            if trial["trial_id"] in completed:
                continue
            task = trial["task"]
            if task not in task_resources:
                evaluation, _ = build_task(task, eval_workers=eval_workers)
                task_resources[task] = (evaluation, SecureEvaluator(evaluation))
            evaluation, evaluator = task_resources[task]
            result = _run_trial(
                trial=trial,
                anchor=anchors[trial["anchor_id"]],
                llm=llm,
                evaluation=evaluation,
                evaluator=evaluator,
            )
            _append_jsonl(result_path, result)
            completed.add(trial["trial_id"])
            _write_json(
                shard_dir / "summary.json",
                {
                    "status": (
                        "finished" if len(completed) == len(schedule) else "running"
                    ),
                    "completed_trials": len(completed),
                    "assigned_trials": len(schedule),
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(
                f"shard={shard_index} trial={trial['trial_id']} "
                f"status={result['status']} completed={len(completed)}/{len(schedule)}",
                flush=True,
            )
    finally:
        llm.close()


def status(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    results = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        results.extend(_read_jsonl(path))
    summary = {
        "protocol_id": config["protocol_id"],
        "completed_trials": len({row["trial_id"] for row in results}),
        "total_trials": int(config["trial_count"]),
        "valid_trials": sum(bool(row.get("valid")) for row in results),
        "invalid_trials": sum(row.get("valid") is False for row in results),
        "by_task_condition": {},
    }
    for task in TASKS:
        for condition in CONDITIONS:
            matching = [
                row
                for row in results
                if row["task"] == task and row["condition"] == condition
            ]
            summary["by_task_condition"][f"{task}:{condition}"] = {
                "completed": len(matching),
                "valid": sum(bool(row.get("valid")) for row in matching),
            }
    return summary


def _default_run_dir() -> Path:
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_v95_history_probe"
    return EXPERIMENTS_ROOT / "generation_probe" / name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-dir", type=Path, default=None)
    prepare.add_argument("--source-batch", default=FORMAL_BATCH)
    prepare.add_argument("--anchors-per-stratum", type=int, default=ANCHORS_PER_STRATUM)
    prepare.add_argument("--replicates", type=int, default=REPLICATES)
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED)

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
        run_dir = args.run_dir or _default_run_dir()
        prepare_probe(
            run_dir,
            batch=args.source_batch,
            anchors_per_stratum=args.anchors_per_stratum,
            replicates=args.replicates,
            seed=args.seed,
        )
        print(run_dir.resolve())
    elif args.command == "run":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index must be in [0, num_shards)")
        run_shard(
            args.run_dir,
            backend=args.backend,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            eval_workers=args.eval_workers,
        )
    else:
        print(json.dumps(status(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
