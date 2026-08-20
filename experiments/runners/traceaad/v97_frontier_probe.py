"""V9.7 fixed-anchor region-frontier context probe.

Paired single-step identification of search-global generation information.
Anchors are drawn from the official V9.7 batch restricted to anchors created
after the static macro-family vocabulary is saturated (creation iteration
>= 200; measured last new-family entry is 125).  For every anchor the global
state at creation time is replayed: visited mechanism regions with their
frontier fitness, and the best program from a different region.

Conditions share task, current code, V9.7 parent path, intent instruction and
sampling seed within an anchor-replicate block; they differ only in appended
context sections:

- ``pp``    V9.7 baseline prompt (parent path only).
- ``fp``    + [Searched Mechanism Regions] frontier table.
- ``fc``    + frontier table + [Reference] best cross-region program.

Primary contrasts: ``fp_explore - pp_explore``, ``fp_refine - pp_refine``,
``fc_explore - fp_explore``; mechanism outcomes are sub-frontier re-entry
rate, destination family distribution, validity and edit size.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.analysis.analyze_v97_search_geometry import (
    macro_family,
    mechanism_tags,
)
from llm4ad.base import SecureEvaluator
from llm4ad.method.traceaad_v9_7.forest import Forest as V97Forest
from llm4ad.method.traceaad_v9_7.history import parent_path, render_path
from llm4ad.method.traceaad_v9_7.prompt import build_generation_prompt
from llm4ad.method.traceaad_v9_7.schema import Intent, Outcome
from llm4ad.method.traceaad_v9_8.source import code_hash

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    build_llm_client,
    build_task,
    resolve_backend,
)
from .v98_mechanism_probe import (
    LOGICAL_MODEL_NAME,
    OUTPUT_TOKENS,
    REPLICATES,
    SOURCE_BATCH,
    STRATA,
    TOTAL_CONTEXT_TOKENS,
    _balanced_sample,
    _deduplicate_snapshots,
    _draw_call,
    _evaluate_call,
    _formal_run_dirs,
    _rank_tertiles,
    _read_json,
    _read_jsonl,
    _append_jsonl,
    _text_hash,
    _write_json,
    _write_jsonl,
)

PROTOCOL_ID = "traceaad-v9.7-region-frontier-probe-v1"
PREFIXES = ("pp", "fp", "fc")
CONDITIONS = (
    "pp_refine",
    "pp_explore",
    "fp_refine",
    "fp_explore",
    "fc_explore",
)
CONDITION_SEQUENCES = tuple(
    tuple(CONDITIONS[(shift + index) % len(CONDITIONS)] for index in range(len(CONDITIONS)))
    for shift in range(len(CONDITIONS))
)
ANCHORS_PER_STRATUM = 6
DESIGN_SEED = 971101
MIN_CREATED_ITERATION = 200
MAX_FRONTIER_ROWS = 12


def _read_checkpoint(run: Path) -> dict[str, Any]:
    return _read_json(run / "checkpoints" / "latest.json")


def _eligible_anchor(forest: V97Forest, anchor_id: int) -> bool:
    path_attempt_ids = forest.parent_path_ids(anchor_id)
    if len(path_attempt_ids) < 2:
        return False
    for attempt_id in path_attempt_ids:
        attempt = forest.get_attempt(attempt_id)
        if attempt.outcome in {None, Outcome.INVALID} or attempt.intent not in {
            Intent.REFINE.value,
            Intent.EXPLORE.value,
        }:
            return False
    return True


def _frontier_text(rows: list[dict[str, Any]], global_best: float) -> str:
    lines = [
        "[Searched Mechanism Regions]",
        (
            "Earlier in this search the following mechanism regions were already "
            "implemented and evaluated. Best fitness is the highest fitness any "
            "program using these mechanisms achieved (higher is better):"
        ),
    ]
    for row in rows:
        mechanisms = ", ".join(sorted(row["tags"]))
        lines.append(
            f"- mechanisms: {mechanisms} | best fitness: {row['best_fitness']:.6g} "
            f"| programs tried: {row['count']}"
        )
    lines.append(f"Current global best fitness across all regions: {global_best:.6g}.")
    return "\n".join(lines)


def _reference_text(program: dict[str, Any]) -> str:
    mechanisms = ", ".join(sorted(program["tags"]))
    return "\n".join(
        [
            "[Reference: Best Program From A Different Mechanism Region]",
            (
                f"The following program from another region achieved fitness "
                f"{program['fitness']:.6g} using mechanisms: {mechanisms}."
            ),
            "```python",
            program["code"].rstrip(),
            "```",
        ]
    )


def _snapshot_global_state(
    family_stats: dict[str, dict[str, Any]],
    best_by_family: dict[str, dict[str, Any]],
    anchor_family: str,
) -> dict[str, Any] | None:
    """Freeze the searched-region view at one anchor-creation moment."""

    if not any(family != anchor_family for family in family_stats):
        return None
    rows = sorted(
        (
            {
                "family": family,
                "tags": sorted(row["tags"]),
                "best_fitness": float(best_by_family[family]["fitness"]),
                "best_q": float(best_by_family[family]["q"]),
                "count": row["count"],
            }
            for family, row in family_stats.items()
        ),
        key=lambda row: (-row["best_q"], row["family"]),
    )
    if len(rows) > MAX_FRONTIER_ROWS:
        raise AssertionError(f"frontier table exceeds {MAX_FRONTIER_ROWS} rows")
    cross = max(
        (record for family, record in best_by_family.items() if family != anchor_family),
        key=lambda record: (float(record["q"]), -int(record["order"])),
    )
    return {
        "anchor_family": anchor_family,
        "frontier_rows": rows,
        "frontier_text": _frontier_text(rows, max(row["best_q"] for row in rows)),
        "cross_top": {
            "family": cross["family"],
            "tags": cross["tags"],
            "fitness": cross["fitness"],
            "q": cross["q"],
            "code_hash": cross["code_hash"],
            "code": cross["code"],
        },
        "cross_text": _reference_text(cross),
        "visited_families": {row["family"]: row["best_q"] for row in rows},
    }


def extract_snapshot_pool(task: str, batch: str, min_created_iteration: int) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for run in _formal_run_dirs(task, batch):
        config = _read_json(run / "run_config.json")
        checkpoint = _read_checkpoint(run)
        forest = V97Forest.from_dict(checkpoint["forest"])
        programs = {int(row["id"]): row for row in checkpoint["forest"]["programs"]}
        attempts = sorted(checkpoint["forest"]["attempts"], key=lambda row: int(row["order"]))
        eligible: dict[int, Any] = {}
        for anchor in forest.anchors():
            if anchor.attempt_id is not None and _eligible_anchor(forest, anchor.id):
                eligible[int(anchor.attempt_id)] = anchor
        family_stats: dict[str, dict[str, Any]] = {}
        best_by_family: dict[str, dict[str, Any]] = {}
        created: set[int] = set()
        for attempt in attempts:
            program_id = attempt.get("program_id")
            if not isinstance(program_id, int) or program_id in created:
                continue
            created.add(program_id)
            record = programs[program_id]
            tags = mechanism_tags(task, record["code"])
            family = macro_family(task, tags)
            row = family_stats.setdefault(
                family, {"tags": set(), "best_q": float("-inf"), "count": 0}
            )
            row["tags"] |= set(tags)
            row["count"] += 1
            q = float(record["q"])
            if q > row["best_q"]:
                row["best_q"] = q
            if family not in best_by_family or q > float(best_by_family[family]["q"]):
                best_by_family[family] = {
                    "family": family,
                    "q": q,
                    "fitness": float(record["fitness"]),
                    "code": record["code"],
                    "tags": sorted(tags),
                    "code_hash": code_hash(record["code"]),
                    "order": int(attempt["order"]),
                }
            anchor = eligible.get(int(attempt["id"]))
            if anchor is None:
                continue
            iteration = attempt.get("iteration")
            if iteration is None or int(iteration) < min_created_iteration:
                continue
            state = _snapshot_global_state(family_stats, best_by_family, family)
            if state is None:
                continue
            program = forest.get_program(anchor.program_id)
            path_ids = parent_path(forest, anchor.id)
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
                    "depth": len(forest.parent_path_ids(anchor.id)),
                    "history_event_count": len(path_ids),
                    "history_text": render_path(forest, path_ids),
                    "source_s0": float(checkpoint["s"] or 0.0),
                    "created_iteration": int(iteration),
                    **state,
                }
            )
    return snapshots


def select_anchors(
    *,
    batch: str,
    tasks: tuple[str, ...],
    anchors_per_stratum: int,
    seed: int,
    min_created_iteration: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        rng = random.Random(seed + 1009 * (task_index + 1))
        pool = _deduplicate_snapshots(
            extract_snapshot_pool(task, batch, min_created_iteration), rng
        )
        strata = _rank_tertiles(pool)
        for stratum in STRATA:
            for row in _balanced_sample(strata[stratum], anchors_per_stratum, rng):
                selected.append({**row, "stratum": stratum})
    selected.sort(key=lambda row: (row["task"], STRATA.index(row["stratum"]), row["snapshot_id"]))
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
            prefix, intent = condition.split("_", 1)
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
                    "context_prefix": prefix,
                    "intent": intent,
                    "sampling_seed": sampling_seed,
                }
            )
    return schedule


def _history_text(anchor: dict[str, Any], prefix: str) -> str:
    history_text = anchor["history_text"]
    if prefix in {"fp", "fc"}:
        history_text = history_text + "\n\n" + anchor["frontier_text"]
    if prefix == "fc":
        history_text = history_text + "\n\n" + anchor["cross_text"]
    return history_text


def _prompt(anchor: dict[str, Any], task_description: str, condition: str) -> str:
    prefix, intent_text = condition.split("_", 1)
    return build_generation_prompt(
        task_description=task_description,
        code=anchor["code"],
        fitness=float(anchor["fitness"]),
        history_text=_history_text(anchor, prefix),
        intent=Intent(intent_text),
        maximize=True,
    )


def prepare_probe(
    run_dir: Path,
    *,
    batch: str,
    tasks: tuple[str, ...],
    anchors_per_stratum: int,
    replicates: int,
    seed: int,
    min_created_iteration: int,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    anchors = select_anchors(
        batch=batch,
        tasks=tasks,
        anchors_per_stratum=anchors_per_stratum,
        seed=seed,
        min_created_iteration=min_created_iteration,
    )
    schedule = build_schedule(anchors, replicates=replicates, seed=seed)
    audit: list[dict[str, Any]] = []
    for anchor in anchors:
        evaluation, _ = build_task(anchor["task"], eval_workers=1)
        prompts = {
            condition: _prompt(anchor, evaluation.task_description, condition)
            for condition in CONDITIONS
        }
        pp_refine_without = prompts["pp_refine"].replace(
            _history_text(anchor, "pp"), "", 1
        ).strip()
        pp_explore_without = prompts["pp_explore"].replace(
            _history_text(anchor, "pp"), "", 1
        ).strip()
        for condition, prompt in prompts.items():
            prefix, intent = condition.split("_", 1)
            without_history = prompt.replace(_history_text(anchor, prefix), "", 1).strip()
            baseline = pp_refine_without if intent == "refine" else pp_explore_without
            if without_history != baseline:
                raise AssertionError(
                    f"condition {condition} changes prompt outside the history slot"
                )
        audit.append(
            {
                "anchor_id": anchor["anchor_id"],
                "history_hash": _text_hash(anchor["history_text"]),
                "frontier_hash": _text_hash(anchor["frontier_text"]),
                "cross_hash": _text_hash(anchor["cross_text"]),
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
            "tasks": list(tasks),
            "strata": list(STRATA),
            "anchors_per_stratum": anchors_per_stratum,
            "anchor_count": len(anchors),
            "replicates_per_anchor_condition": replicates,
            "trial_count": len(schedule),
            "design_seed": seed,
            "min_created_iteration": min_created_iteration,
            "anchor_eligibility": (
                "valid formation path with search intents; created at iteration >= "
                f"{min_created_iteration} (static macro-family entries end by 125)"
            ),
            "primary_contrasts": [
                "fp_explore - pp_explore",
                "fp_refine - pp_refine",
                "fc_explore - fp_explore",
            ],
            "mechanism_outcomes": [
                "sub_frontier_reentry_rate",
                "destination_family_distribution",
                "validity_rate",
                "change_ratio",
                "prompt_tokens",
            ],
            "temperature": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "max_total_context": TOTAL_CONTEXT_TOKENS,
            "generation_evaluation_pipeline": "same_process_immediate_per_trial",
            "independent_unit": "fixed anchor snapshot",
            "within_anchor_repeated_sampling": replicates,
            "blocking": "task x quality stratum; paired seed within anchor replicate",
        },
    )


def run_probe_shard(
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
                f"frontier shard={shard_index} trial={trial['trial_id']} "
                f"status={result['status']} completed={len(completed)}/{len(schedule)}",
                flush=True,
            )
    finally:
        llm.close()


def status_probe(run_dir: Path) -> dict[str, Any]:
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
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_v97_region_frontier"
    return EXPERIMENTS_ROOT / "generation_probe" / name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-probe")
    prepare.add_argument("--run-dir", type=Path)
    prepare.add_argument("--source-batch", default=SOURCE_BATCH)
    prepare.add_argument("--tasks", default=",".join(TASKS))
    prepare.add_argument("--anchors-per-stratum", type=int, default=ANCHORS_PER_STRATUM)
    prepare.add_argument("--replicates", type=int, default=REPLICATES)
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED)
    prepare.add_argument("--min-created-iteration", type=int, default=MIN_CREATED_ITERATION)
    run = subparsers.add_parser("run-probe")
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--num-shards", type=int, required=True)
    run.add_argument("--eval-workers", type=int, default=1)
    inspect = subparsers.add_parser("status-probe")
    inspect.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-probe":
        run_dir = args.run_dir or _default_run_dir()
        prepare_probe(
            run_dir,
            batch=args.source_batch,
            tasks=tuple(args.tasks.split(",")),
            anchors_per_stratum=args.anchors_per_stratum,
            replicates=args.replicates,
            seed=args.seed,
            min_created_iteration=args.min_created_iteration,
        )
        print(run_dir.resolve())
    elif args.command == "run-probe":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index must be in [0, num_shards)")
        run_probe_shard(
            args.run_dir,
            backend=args.backend,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            eval_workers=args.eval_workers,
        )
    else:
        print(json.dumps(status_probe(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
