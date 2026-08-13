"""Fixed-anchor three-arm probe: code only vs parent path vs parent path + child attempts.

Anchors are recorded V9.6 generation moments whose shown history contains at
least two formation steps and at least one direct attempt.  History blocks are
re-rendered with the V9.6 renderer from the recorded selection, and each full
block is verified verbatim against the prompt the model actually received.
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
from typing import Any

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9.complexity import code_change_ratio
from llm4ad.method.traceaad_v9_6.checkpoint import _forest_from_dict
from llm4ad.method.traceaad_v9_6.history import (
    HistorySelection,
    format_fitness,
    render_history,
)
from llm4ad.method.traceaad_v9_6.prompt import (
    ProgramResponseError,
    _output_contract,
    fitness_direction_hint,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_6.source import nonempty_loc, text_hash

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    TASK_SHORT,
    build_llm_client,
    build_task,
    resolve_backend,
)
from .generation_probe import (
    ANCHORS_PER_STRATUM,
    DESIGN_SEED,
    LOGICAL_MODEL_NAME,
    OUTPUT_TOKENS,
    REPLICATES,
    STRATA,
    TOTAL_CONTEXT_TOKENS,
    TRANSPORT_RETRIES,
    _append_jsonl,
    _balanced_sample,
    _deduplicate_snapshots,
    _rank_tertiles,
    _read_json,
    _read_jsonl,
    _response_tokens,
    _write_json,
    _write_jsonl,
)

PROTOCOL_ID = "traceaad-v96-parent-path-child-probe-v1"
SOURCE_BATCH = "20260812_191011"
CONDITIONS = ("code_only", "parent_path", "parent_path_child")
CONDITION_ROTATIONS = (
    ("code_only", "parent_path", "parent_path_child"),
    ("parent_path", "parent_path_child", "code_only"),
    ("parent_path_child", "code_only", "parent_path"),
)
DESIGN_SEED_V3 = DESIGN_SEED + 3
MIN_FORMATION_EVENTS = 2
MIN_DIRECT_EVENTS = 1


@dataclass(frozen=True, slots=True)
class PromptTriple:
    code_only: str
    parent_path: str
    parent_path_child: str


def build_prompt_triple(anchor: dict[str, Any], task_description: str) -> PromptTriple:
    prefix = "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(True),
            "",
            "[Current Algorithm]",
            f"Fitness: {format_fitness(float(anchor['fitness']))}",
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
                "If history is provided, treat it as evidence rather than a strict "
                "prohibition; a previously unsuccessful idea may be revisited with "
                "a materially different implementation."
            ),
            "Keep the target function signature and contract unchanged.",
            "Return one complete, self-contained implementation.",
            _output_contract(),
        ]
    )
    formation_block = str(anchor["formation_block"])
    full_block = str(anchor["full_block"])
    if not full_block.startswith(formation_block):
        raise AssertionError("full history block is not an extension of the parent path")
    triple = PromptTriple(
        code_only=prefix + suffix,
        parent_path=prefix + formation_block + "\n\n" + suffix,
        parent_path_child=prefix + full_block + "\n\n" + suffix,
    )
    if triple.parent_path.replace(formation_block + "\n\n", "", 1) != triple.code_only:
        raise AssertionError("parent_path differs outside the history block")
    if triple.parent_path_child.replace(full_block + "\n\n", "", 1) != triple.code_only:
        raise AssertionError("parent_path_child differs outside the history block")
    child_extension = full_block[len(formation_block) :]
    if triple.parent_path_child.replace(child_extension, "", 1) != triple.parent_path:
        raise AssertionError("child attempts are not a pure extension of the parent path")
    return triple


def _prompt_for_condition(triple: PromptTriple, condition: str) -> str:
    return getattr(triple, condition)


def _formal_run_dirs(task: str, batch: str) -> list[Path]:
    root = EXPERIMENTS_ROOT / task / "traceaad_v9_6"
    runs = sorted(root.glob(f"v9_6_{batch}_{TASK_SHORT[task]}_rep*"))
    return [
        run
        for run in runs
        if run.is_dir()
        and (run / "logs" / "summary.json").is_file()
        and _read_json(run / "logs" / "summary.json").get("status") == "finished"
    ]


def _chronological(forest, attempt_ids) -> tuple[int, ...]:
    return tuple(
        sorted(
            (int(item) for item in attempt_ids),
            key=lambda attempt_id: forest.get_attempt(attempt_id).candidate_order,
        )
    )


def extract_snapshot_pool(task: str, batch: str) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for run in _formal_run_dirs(task, batch):
        config = _read_json(run / "run_config.json")
        checkpoint = _read_json(run / "checkpoints" / "latest.json")
        forest = _forest_from_dict(checkpoint["forest"])
        pending_selection: dict[str, Any] | None = None
        for decision_order, event in enumerate(
            _read_jsonl(run / "artifacts" / "decisions.jsonl")
        ):
            if event.get("event") == "anchor_selected":
                pending_selection = event
                continue
            if event.get("event") != "history_built" or pending_selection is None:
                continue
            state_id = int(event["anchor_state_id"])
            if state_id != int(pending_selection["selected_state_id"]):
                continue
            selection_moment = pending_selection
            pending_selection = None
            formation_ids = _chronological(forest, event["selected_formation_ids"])
            direct_ids = _chronological(forest, event["selected_direct_ids"])
            if (
                len(formation_ids) < MIN_FORMATION_EVENTS
                or len(direct_ids) < MIN_DIRECT_EVENTS
                or int(event.get("dropped_for_context", 0)) != 0
            ):
                continue
            shown_ids = _chronological(forest, event["shown_event_ids"])
            if shown_ids != _chronological(forest, (*formation_ids, *direct_ids)):
                raise AssertionError(
                    f"shown events do not match the recorded selection at {run.name}"
                )
            formation_pool = _chronological(forest, event["formation_pool_ids"])
            direct_pool = _chronological(forest, event["direct_pool_ids"])
            full_block = render_history(
                forest,
                HistorySelection(
                    event_ids=shown_ids,
                    formation_event_ids=formation_ids,
                    direct_event_ids=direct_ids,
                    formation_pool_ids=formation_pool,
                    direct_pool_ids=direct_pool,
                ),
            )
            formation_block = render_history(
                forest,
                HistorySelection(
                    event_ids=formation_ids,
                    formation_event_ids=formation_ids,
                    direct_event_ids=(),
                    formation_pool_ids=formation_pool,
                    direct_pool_ids=direct_pool,
                ),
            )
            if not full_block.startswith(formation_block):
                raise AssertionError(
                    f"direct events interleave with formation at {run.name}"
                )
            state = forest.get_state(state_id)
            artifact = forest.get_artifact(state.artifact_id)
            snapshots.append(
                {
                    "snapshot_id": (
                        f"{task}:{run.name}:iteration_"
                        f"{int(selection_moment['iteration'])}:state_{state_id}"
                    ),
                    "task": task,
                    "source_run": run.name,
                    "source_repeat": int(config["repeat"]),
                    "source_iteration": int(selection_moment["iteration"]),
                    "state_id": state_id,
                    "artifact_id": int(artifact.artifact_id),
                    "code_hash": artifact.evaluator_input_hash,
                    "code": artifact.evaluator_input_code,
                    "fitness": float(artifact.fitness),
                    "q": float(artifact.directed_fitness),
                    "depth": int(state.depth),
                    "formation_event_count": len(formation_ids),
                    "direct_event_count": len(direct_ids),
                    "formation_pool_size": len(formation_pool),
                    "direct_pool_size": len(direct_pool),
                    "formation_block": formation_block,
                    "full_block": full_block,
                    "source_decision_order": decision_order,
                }
            )
    return snapshots


def _audit_against_recorded_prompts(anchors: list[dict[str, Any]]) -> None:
    """Assert every sampled full history block was shown verbatim to the model."""
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        by_run[(anchor["task"], anchor["source_run"])].append(anchor)
    for (task, run_name), grouped in sorted(by_run.items()):
        run = EXPERIMENTS_ROOT / task / "traceaad_v9_6" / run_name
        recorded = "\n\x00\n".join(
            str(row["prompt"])
            for row in _read_jsonl(run / "artifacts" / "llm_calls.jsonl")
        )
        for anchor in grouped:
            if anchor["full_block"] not in recorded:
                raise AssertionError(
                    f"re-rendered history block for {anchor['snapshot_id']} was "
                    "never shown to the model in the recorded formal run"
                )


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
                row = dict(row)
                row["stratum"] = stratum
                selected.append(row)
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
        if len(group) % len(CONDITION_ROTATIONS):
            raise ValueError(f"rotation balance requires blocks divisible by 3 for {key}")
        for index, item in enumerate(group):
            item["condition_order"] = list(
                CONDITION_ROTATIONS[index % len(CONDITION_ROTATIONS)]
            )
            blocks.append(item)
    rng.shuffle(blocks)
    schedule: list[dict[str, Any]] = []
    for block_order, block in enumerate(blocks):
        anchor = block["anchor"]
        block_id = f"{anchor['anchor_id']}_rep{block['replicate']}"
        sampling_seed = seed * 1000 + block_order + 1
        for within_block_order, condition in enumerate(block["condition_order"]):
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
                    "sampling_seed": sampling_seed,
                }
            )
    return schedule


def prepare_probe(
    run_dir: Path,
    *,
    batch: str = SOURCE_BATCH,
    anchors_per_stratum: int = ANCHORS_PER_STRATUM,
    replicates: int = REPLICATES,
    seed: int = DESIGN_SEED_V3,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    anchors = select_anchors(
        batch=batch, anchors_per_stratum=anchors_per_stratum, seed=seed
    )
    _audit_against_recorded_prompts(anchors)
    schedule = build_schedule(anchors, replicates=replicates, seed=seed)
    audits = []
    for anchor in anchors:
        evaluation, _ = build_task(anchor["task"], eval_workers=None)
        triple = build_prompt_triple(anchor, evaluation.task_description)
        audits.append(
            {
                "anchor_id": anchor["anchor_id"],
                "code_only_prompt_hash": text_hash(triple.code_only),
                "parent_path_prompt_hash": text_hash(triple.parent_path),
                "parent_path_child_prompt_hash": text_hash(triple.parent_path_child),
                "formation_block_hash": text_hash(anchor["formation_block"]),
                "full_block_hash": text_hash(anchor["full_block"]),
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
            "source_batch": batch,
            "source_method": "traceaad_v9_6",
            "logical_model_name": LOGICAL_MODEL_NAME,
            "conditions": list(CONDITIONS),
            "tasks": list(TASKS),
            "strata": list(STRATA),
            "stratum_definition": (
                "equal-count rank tertiles of directed q within each task's "
                "eligible deduplicated snapshot pool"
            ),
            "eligibility": (
                "recorded V9.6 generation moment with at least "
                f"{MIN_FORMATION_EVENTS} formation steps, at least "
                f"{MIN_DIRECT_EVENTS} direct attempt shown, and no context drop"
            ),
            "history_fidelity": (
                "each full history block re-rendered by the V9.6 renderer and "
                "verified verbatim inside a recorded formal-run prompt"
            ),
            "anchors_per_stratum": anchors_per_stratum,
            "anchor_count": len(anchors),
            "replicates_per_anchor_condition": replicates,
            "trial_count": len(schedule),
            "design_seed": seed,
            "temperature": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "max_total_context": TOTAL_CONTEXT_TOKENS,
            "generation_evaluation_phases_separated": True,
            "evaluation_execution": (
                "one local sequential evaluator process using native task workers"
            ),
            "independent_unit": "anchor snapshot",
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
    triple = build_prompt_triple(anchor, evaluation.task_description)
    prompt = _prompt_for_condition(triple, trial["condition"])
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
        "formation_event_count": anchor["formation_event_count"],
        "direct_event_count": anchor["direct_event_count"],
        "formation_block_hash": text_hash(anchor["formation_block"]),
        "full_block_hash": text_hash(anchor["full_block"]),
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
    anchors = {row["anchor_id"]: row for row in _read_jsonl(run_dir / "anchors.jsonl")}
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["block_order"]) % num_shards == shard_index
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
                "status": "finished" if len(completed) == len(schedule) else "running",
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
    selected = [row for row in generated if row["generation_status"] == "generated"][
        :limit
    ]
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
    prepare.add_argument("--source-batch", default=SOURCE_BATCH)
    prepare.add_argument("--anchors-per-stratum", type=int, default=ANCHORS_PER_STRATUM)
    prepare.add_argument("--replicates", type=int, default=REPLICATES)
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED_V3)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--run-dir", type=Path, required=True)
    generate.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    generate.add_argument("--shard-index", type=int, required=True)
    generate.add_argument("--num-shards", type=int, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    smoke = subparsers.add_parser("smoke-evaluate")
    smoke.add_argument("--run-dir", type=Path, required=True)
    smoke.add_argument("--limit", type=int, default=3)
    inspect = subparsers.add_parser("status")
    inspect.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_probe(
            args.run_dir,
            batch=args.source_batch,
            anchors_per_stratum=args.anchors_per_stratum,
            replicates=args.replicates,
            seed=args.seed,
        )
        print(args.run_dir.resolve())
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
