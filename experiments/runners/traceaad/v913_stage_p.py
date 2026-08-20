"""V9.13 Stage P: fixed real-decision snapshot experiment.

Paired single-step identification of the proxy-region frontier context on
snapshots replayed from the official V9.7 batch ``20260814_150927``.  A
snapshot is one *real* Explore decision: the route/anchor actually selected,
the evaluator calls completed before that decision (``b_t``), the forest
trimmed to that moment, the parent-path events the run actually showed, and
the proxy-region frontier computed only from programs whose real evaluation
had completed (``e(p) <= b_t``).

Sampling (frozen): per task, 3 source runs x 3 evaluator intervals
([200,466], [467,733], [734,999]) x 2 snapshots (one from each half of the
unit's selected-anchor quality) = 18 snapshots per task, 72 in total.

Conditions share task, current code, the replayed parent path, the V9.7
intent instruction and one sampling seed within a snapshot-replicate block;
they differ only in the appended global-facts section:

- ``pp``    V9.7 baseline prompt (parent path only).
- ``fp``    + searched proxy-region floor table (floor semantics, own
  region first, every region's frontier with mechanism tags and quality).

Primary contrasts (design section 7.2): ``fp_explore - pp_explore`` (gates
the candidate) and ``fp_refine - pp_refine`` for operator dependence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from llm4ad.method.traceaad_v9_13.prompt import (
    ProgramResponseError,
    build_generation_prompt,
    parse_program_response,
)
from llm4ad.method.traceaad_v9_13.regions import (
    macro_family,
    mechanism_tags,
    render_frontier_table,
)
from llm4ad.method.traceaad_v9_13.schema import Intent, Outcome, Attempt
from llm4ad.method.traceaad_v9_13.forest import Forest
from llm4ad.method.traceaad_v9_13.history import render_path
from llm4ad.method.traceaad_v9_13.selection import select
from llm4ad.method.traceaad_v9_13.source import code_diff, code_hash

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    build_llm_client,
    build_task,
    resolve_backend,
)

PROTOCOL_ID = "traceaad-v9.13-stage-p-decision-snapshot-v3"
SOURCE_BATCH = "20260814_150927"
SOURCE_METHOD_DIR = "traceaad_v9_7"
LOGICAL_MODEL_NAME = "Qwen3.6-27B"
OUTPUT_TOKENS = 8192
TOTAL_CONTEXT_TOKENS = 32768
REPLICATES = 3
TRANSPORT_RETRIES = 3
MIN_EVALS_BEFORE_DECISION = 200
EVAL_INTERVALS = ((200, 466), (467, 733), (734, 999))
SNAPSHOTS_PER_UNIT = 2
DESIGN_SEED = 913301
PREFIXES = ("pp", "fp")
CONDITIONS = (
    "pp_refine",
    "pp_explore",
    "fp_refine",
    "fp_explore",
)
CONDITION_SEQUENCES = tuple(
    tuple(CONDITIONS[(shift + index) % len(CONDITIONS)] for index in range(len(CONDITIONS)))
    for shift in range(len(CONDITIONS))
)
SUB_FRONTIER_MARGIN_S = 1.0


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


def _source_run_dirs(task: str, batch: str) -> list[Path]:
    """Completed source runs of the V9.7 batch (data provenance only)."""

    root = EXPERIMENTS_ROOT / task / SOURCE_METHOD_DIR
    runs = sorted(root.glob(f"v9_7_{batch}_{task}_rep*"))
    return [
        run
        for run in runs
        if run.is_dir()
        and (run / "logs" / "summary.json").is_file()
        and _read_json(run / "logs" / "summary.json").get("status") == "finished"
    ]


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


# mechanism_tags is a pure function of (task, code); extraction touches every
# evaluated program at every decision, so memoize on the exact code string.
_TAG_CACHE: dict[tuple[str, str], tuple[frozenset[str], str]] = {}


def _tags_and_family(task: str, code: str) -> tuple[frozenset[str], str]:
    key = (task, code)
    cached = _TAG_CACHE.get(key)
    if cached is None:
        tags = mechanism_tags(task, code)
        cached = (tags, macro_family(task, tags))
        _TAG_CACHE[key] = cached
    return cached


def _read_candidates(run: Path) -> list[dict[str, Any]]:
    return _read_jsonl(run / "artifacts" / "candidates.jsonl")


def _pair_search_decisions(run: Path, n_roots: int) -> list[dict[str, Any]]:
    """Pair each search response with the history events its decision built.

    Returns one record per search response with the decision-time evaluator
    count ``b_t`` (real evaluator calls completed before the response), the
    parent-path event ids the run actually showed, and the route, anchor, and
    score the real V9.7 selection produced for that decision.
    """

    candidates = _read_candidates(run)
    decision_rows = _read_jsonl(run / "artifacts" / "decisions.jsonl")
    history = [row for row in decision_rows if row.get("event") == "history_built"]
    selected = sorted(
        (row for row in decision_rows if row.get("event") == "anchor_selected"),
        key=lambda row: int(row["iteration"]),
    )
    search_rows = [row for row in candidates if row.get("stage") == "search"]
    bootstrap_count = len(history) - len(search_rows)
    if bootstrap_count != n_roots:
        raise ValueError(
            f"{run.name}: expected {n_roots} bootstrap history events, "
            f"found {bootstrap_count}"
        )
    if len(selected) != len(search_rows):
        raise ValueError(
            f"{run.name}: {len(selected)} anchor selections for "
            f"{len(search_rows)} search responses"
        )
    search_history = history[bootstrap_count:]

    evaluator_calls = 0
    completed_by_order: dict[int, int] = {}
    eval_index_of_program: dict[int, int] = {}
    for row in sorted(candidates, key=lambda item: int(item["order"])):
        if bool(row.get("evaluator_called")):
            evaluator_calls += 1
            if row.get("kind") in {"new", "root_new"} and row.get("program_id") is not None:
                eval_index_of_program[int(row["program_id"])] = evaluator_calls
        completed_by_order[int(row["order"])] = evaluator_calls

    decisions: list[dict[str, Any]] = []
    for row, event, choice in zip(search_rows, search_history, selected, strict=True):
        iteration = int(row["iteration"])
        if int(choice["iteration"]) != iteration:
            raise ValueError(f"{run.name}: anchor selection iteration mismatch at {iteration}")
        if row.get("intent") != event.get("intent") or int(row["anchor_id"]) != int(
            event["anchor_id"]
        ):
            raise ValueError(f"{run.name}: decision pairing mismatch at iteration {iteration}")
        if int(choice["selected_state_id"]) != int(row["anchor_id"]):
            raise ValueError(f"{run.name}: selected anchor mismatch at iteration {iteration}")
        order = int(row["order"])
        decisions.append(
            {
                "iteration": iteration,
                "order": order,
                "anchor_id": int(row["anchor_id"]),
                "route_id": int(choice["route_id"]),
                "selected_score": float(choice["selected_score"]),
                "intent": str(row["intent"]),
                "b_t": completed_by_order[order - 1] if order > 1 else 0,
                "shown_event_ids": [int(item) for item in event["shown_event_ids"]],
                "selected_event_ids": [int(item) for item in event["selected_event_ids"]],
                "dropped_for_context": int(event["dropped_for_context"]),
            }
        )
    if evaluator_calls != 1000:
        raise ValueError(f"{run.name}: replayed {evaluator_calls} evaluator calls, expected 1000")
    return decisions, eval_index_of_program


def trim_forest(forest_payload: dict[str, Any], order_cut: int) -> Forest:
    """Rebuild the forest as it stood before response ``order_cut``."""

    forest = Forest(maximize=bool(forest_payload["maximize"]))
    for item in sorted(forest_payload["programs"], key=lambda row: int(row["id"])):
        if int(item["order"]) < order_cut:
            forest.add_program(
                code=item["code"],
                fitness=float(item["fitness"]),
                order=int(item["order"]),
            )
    for item in sorted(forest_payload["anchors"], key=lambda row: int(row["id"])):
        if int(item["order"]) >= order_cut:
            continue
        if item["parent_id"] is None:
            forest.add_root(program_id=int(item["program_id"]), order=int(item["order"]))
        else:
            forest.add_child(
                parent_id=int(item["parent_id"]),
                program_id=int(item["program_id"]),
                attempt_id=int(item["attempt_id"]),
                order=int(item["order"]),
            )
    for item in sorted(forest_payload["attempts"], key=lambda row: int(row["id"])):
        if int(item["order"]) >= order_cut:
            continue
        if item["outcome"] is not None:
            item = {**item, "outcome": Outcome(item["outcome"])}
        forest.add_attempt(Attempt(**item))
    for anchor in forest.anchors():
        anchor.n = sum(
            1
            for attempt in forest.attempts()
            if attempt.anchor_id == anchor.id
        )
    return forest


def extract_decision_snapshots(task: str, batch: str) -> list[dict[str, Any]]:
    """Replay each run once and freeze eligible Explore decisions.

    One sweep advances through response order, maintaining exactly the state
    a decision may see: the archive of evaluated program hashes, the global
    best q, and the per-family frontier (programs enter in real evaluation
    order, which equals program-id order because every program is created by
    its own evaluator call).  A decision snapshot is emitted only when all
    design-section-7.1 eligibility conditions hold.
    """

    snapshots: list[dict[str, Any]] = []
    for run in _source_run_dirs(task, batch):
        config = _read_json(run / "run_config.json")
        checkpoint = _read_json(run / "checkpoints" / "latest.json")
        n_roots = int(config["method_params"]["n_roots"])
        s0 = float(checkpoint["s"] or 0.0)
        if s0 <= 0:
            raise ValueError(f"{run.name}: source run has s=0; Stage P requires s>0")
        decisions, _eval_index = _pair_search_decisions(run, n_roots)
        forest_payload = checkpoint["forest"]
        anchors_by_id = {int(row["id"]): row for row in forest_payload["anchors"]}
        programs_by_order = sorted(
            forest_payload["programs"], key=lambda row: (int(row["order"]), int(row["id"]))
        )
        forest = Forest.from_dict(forest_payload)

        archive: set[str] = set()
        global_best_q: float | None = None
        frontier: dict[str, dict[str, Any]] = {}
        arrival = 0
        pointer = 0
        seen_states: set[tuple[Any, ...]] = set()
        for decision in decisions:
            order_cut = int(decision["order"])
            while pointer < len(programs_by_order) and int(
                programs_by_order[pointer]["order"]
            ) < order_cut:
                row = programs_by_order[pointer]
                pointer += 1
                archive.add(code_hash(row["code"]))
                q = float(row["q"])
                global_best_q = q if global_best_q is None else max(global_best_q, q)
                arrival += 1
                tags, family = _tags_and_family(task, row["code"])
                candidate = {
                    "family": family,
                    "tags": sorted(tags),
                    "q": q,
                    "program_id": int(row["id"]),
                    "code_hash": code_hash(row["code"]),
                    "code": row["code"],
                    "length": int(row["length"]),
                    "key": (q, -int(row["length"]), -arrival),
                }
                current = frontier.get(family)
                if current is None or candidate["key"] > current["key"]:
                    frontier[family] = candidate

            if decision["intent"] != Intent.EXPLORE.value:
                continue
            if decision["b_t"] < MIN_EVALS_BEFORE_DECISION:
                continue
            if len(frontier) < 2:
                continue
            anchor_row = anchors_by_id[int(decision["anchor_id"])]
            program = forest.get_program(int(anchor_row["program_id"]))
            anchor_tags, anchor_family = _tags_and_family(task, program.code)
            rows = sorted(
                frontier.values(), key=lambda row: (-row["q"], row["family"])
            )
            shown = tuple(decision["shown_event_ids"])
            if not _valid_parent_path(forest, int(anchor_row["id"]), shown):
                continue
            visited = {row["family"]: row["q"] for row in rows}
            fingerprint = (
                run.name,
                int(anchor_row["id"]),
                decision["b_t"],
                _text_hash(json.dumps(visited, sort_keys=True)),
            )
            if fingerprint in seen_states:
                continue
            seen_states.add(fingerprint)
            clean_rows = [
                {key: row[key] for key in ("family", "tags", "q", "program_id")}
                for row in rows
            ]
            snapshots.append(
                {
                    "snapshot_id": f"{task}:{run.name}:it{decision['iteration']}",
                    "task": task,
                    "source_run": run.name,
                    "source_repeat": int(config["repeat"]),
                    "decision": decision,
                    "anchor_state_id": int(anchor_row["id"]),
                    "anchor_family": anchor_family,
                    "anchor_tags": sorted(anchor_tags),
                    "program_id": program.id,
                    "code_hash": code_hash(program.code),
                    "code": program.code,
                    "fitness": program.fitness,
                    "q": program.q,
                    "history_event_ids": list(shown),
                    "history_text": render_path(forest, shown),
                    "source_s0": s0,
                    "maximize": bool(forest_payload["maximize"]),
                    "archive_code_hashes": sorted(archive),
                    "global_best_q": global_best_q,
                    "frontier_rows": clean_rows,
                    "visited_families": visited,
                }
            )
    return snapshots


def _valid_parent_path(forest: Forest, anchor_id: int, shown: tuple[int, ...]) -> bool:
    expected = forest.parent_path_ids(anchor_id)[-len(shown):] if shown else ()
    if tuple(shown) != tuple(expected):
        return False
    for attempt_id in shown:
        attempt = forest.get_attempt(attempt_id)
        if attempt.outcome in {None, Outcome.INVALID}:
            return False
        if attempt.intent not in {Intent.REFINE.value, Intent.EXPLORE.value}:
            return False
    return True


def _interval_of(b_t: int) -> int | None:
    for index, (low, high) in enumerate(EVAL_INTERVALS):
        if low <= b_t <= high:
            return index
    return None


def select_snapshots(
    *, batch: str, tasks: tuple[str, ...], seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        rng = random.Random(seed + 1009 * (task_index + 1))
        pool = extract_decision_snapshots(task, batch)
        units: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for snapshot in pool:
            interval = _interval_of(snapshot["decision"]["b_t"])
            if interval is None:
                continue
            units[(snapshot["source_run"], interval)].append(snapshot)
        for source_run in sorted({run for run, _ in units}):
            for interval in range(len(EVAL_INTERVALS)):
                unit = units.get((source_run, interval), [])
                if len(unit) < SNAPSHOTS_PER_UNIT:
                    raise ValueError(
                        f"Stage P unit {task}/{source_run}/interval{interval} has "
                        f"{len(unit)} eligible snapshots, needs {SNAPSHOTS_PER_UNIT}; "
                        "revise the sampling protocol before generating data"
                    )
                ordered = sorted(unit, key=lambda row: (row["q"], row["snapshot_id"]))
                lower = ordered[: len(ordered) // 2]
                upper = ordered[len(ordered) // 2 :]
                for half in (lower, upper):
                    chosen = sorted(half, key=lambda row: row["snapshot_id"])[
                        rng.randrange(len(half))
                    ]
                    selected.append({**chosen, "eval_interval": interval})
    selected.sort(
        key=lambda row: (
            row["task"],
            row["source_run"],
            row["eval_interval"],
            row["snapshot_id"],
        )
    )
    for index, row in enumerate(selected):
        row["snapshot_index"] = f"s{index:03d}"
    return selected


def build_schedule(
    snapshots: list[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    """Blocks are snapshot x replicate; one sampling seed is shared by the
    five conditions inside a block and condition order is rotated."""

    rng = random.Random(seed + 7919)
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for snapshot in snapshots:
        for replicate in range(1, replicates + 1):
            grouped[
                (snapshot["task"], snapshot["source_run"], snapshot["eval_interval"])
            ].append({"snapshot": snapshot, "replicate": replicate})
    schedule: list[dict[str, Any]] = []
    for group_order, key in enumerate(sorted(grouped)):
        group = grouped[key]
        rng.shuffle(group)
        for index, item in enumerate(group):
            snapshot = item["snapshot"]
            block_id = f"{snapshot['snapshot_index']}_rep{item['replicate']}"
            sampling_seed = seed * 1000 + group_order * 100 + index + 1
            for within_block_order, condition in enumerate(
                CONDITION_SEQUENCES[index % len(CONDITION_SEQUENCES)]
            ):
                prefix, intent = condition.split("_", 1)
                schedule.append(
                    {
                        "trial_id": f"{block_id}_{condition}",
                        "block_id": block_id,
                        "group_order": group_order,
                        "group_key": list(key),
                        "within_block_order": within_block_order,
                        "snapshot_index": snapshot["snapshot_index"],
                        "snapshot_id": snapshot["snapshot_id"],
                        "task": snapshot["task"],
                        "source_run": snapshot["source_run"],
                        "eval_interval": snapshot["eval_interval"],
                        "replicate": item["replicate"],
                        "condition": condition,
                        "context_prefix": prefix,
                        "intent": intent,
                        "sampling_seed": sampling_seed,
                    }
                )
    return schedule


def _frontier_text(snapshot: dict[str, Any]) -> str:
    return render_frontier_table(
        snapshot["frontier_rows"],
        snapshot["anchor_family"],
        snapshot["global_best_q"],
    )


def _history_text(snapshot: dict[str, Any], prefix: str) -> str:
    if prefix != "fp":
        return snapshot["history_text"]
    return snapshot["history_text"] + "\n\n" + _frontier_text(snapshot)


def build_trial_prompt(snapshot: dict[str, Any], task_description: str, condition: str) -> str:
    prefix, intent_text = condition.split("_", 1)
    return build_generation_prompt(
        task_description=task_description,
        code=snapshot["code"],
        fitness=float(snapshot["fitness"]),
        history_text=_history_text(snapshot, prefix),
        intent=Intent(intent_text),
        maximize=True,
    )


def prepare_probe(run_dir: Path, *, batch: str, tasks: tuple[str, ...], seed: int) -> None:
    if run_dir.exists():
        raise FileExistsError(f"probe directory already exists: {run_dir}")
    snapshots = select_snapshots(batch=batch, tasks=tasks, seed=seed)
    schedule = build_schedule(snapshots, replicates=REPLICATES, seed=seed)
    audit: list[dict[str, Any]] = []

    forest_payloads: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        evaluation, _ = build_task(snapshot["task"], eval_workers=1)
        # Replay audit: the trimmed forest must reproduce the real decision.
        if snapshot["source_run"] not in forest_payloads:
            forest_payloads[snapshot["source_run"]] = _forest_cache_entry(
                snapshot["task"], snapshot["source_run"], batch
            )["forest"]
        forest = trim_forest(
            forest_payloads[snapshot["source_run"]], snapshot["decision"]["order"]
        )
        choice = select(forest, float(snapshot["source_s0"]))
        if (
            choice.anchor_id != snapshot["anchor_state_id"]
            or choice.route_id != snapshot["decision"]["route_id"]
        ):
            raise AssertionError(
                f"replayed selection diverges from the real decision for "
                f"{snapshot['snapshot_id']}"
            )
        prompts = {
            condition: build_trial_prompt(
                snapshot, evaluation.task_description, condition
            )
            for condition in CONDITIONS
        }
        # PP is exactly the builder without the global-facts slot; the
        # byte-identity of that shape with the V9.7 protocol prompt is
        # enforced by the V9.13 test suite.
        plain = {
            intent: build_generation_prompt(
                task_description=evaluation.task_description,
                code=snapshot["code"],
                fitness=float(snapshot["fitness"]),
                history_text=snapshot["history_text"],
                intent=Intent(intent),
                maximize=True,
            )
            for intent in ("refine", "explore")
        }
        for intent, expected in plain.items():
            if prompts[f"pp_{intent}"] != expected:
                raise AssertionError(
                    f"pp_{intent} prompt differs from the no-context build for "
                    f"{snapshot['snapshot_id']}"
                )
        # Conditions differ only inside the history/global-facts slot;
        # refine and explore baselines differ by the intent instruction.
        baselines = {
            intent: prompts[f"pp_{intent}"]
            .replace(snapshot["history_text"], "", 1)
            .strip()
            for intent in ("refine", "explore")
        }
        for condition, prompt in prompts.items():
            prefix = condition.split("_", 1)[0]
            intent = condition.split("_", 1)[1]
            without = prompt.replace(_history_text(snapshot, prefix), "", 1).strip()
            if without != baselines[intent]:
                raise AssertionError(
                    f"condition {condition} changes the prompt outside the "
                    "history/global-facts slot"
                )
        visited = {row["family"] for row in snapshot["frontier_rows"]}
        if snapshot["anchor_family"] not in visited:
            raise AssertionError(
                f"anchor family missing from frontier rows for {snapshot['snapshot_id']}"
            )
        # Floor-table shape: own region block carries tags; other regions do not.
        table = _frontier_text(snapshot)
        own_tags = next(
            row["tags"]
            for row in snapshot["frontier_rows"]
            if row["family"] == snapshot["anchor_family"]
        )
        if own_tags:
            expected = f"Observed tags of frontier program: {', '.join(sorted(own_tags))}"
            if expected not in table:
                raise AssertionError(
                    f"own-region tags missing from floor table for {snapshot['snapshot_id']}"
                )
        others_section = table.split("[Other Searched Regions]", 1)[1]
        others_section = others_section.split("Global best", 1)[0]
        level_lines = [line for line in others_section.splitlines() if line.strip()]
        tag_pattern = re.compile(r"^Region \d+ observed tags of frontier program: .+$")
        quality_pattern = re.compile(r"^Region \d+ directed quality: -?[\d.e+-]+$")
        if level_lines:
            tag_lines = [line for line in level_lines if "observed tags" in line]
            quality_lines = [line for line in level_lines if "directed quality" in line]
            if len(tag_lines) != len(quality_lines) or 2 * len(tag_lines) != len(
                level_lines
            ):
                raise AssertionError(
                    f"other-region section is not tag+quality pairs for "
                    f"{snapshot['snapshot_id']}"
                )
            for line in tag_lines:
                if tag_pattern.match(line) is None:
                    raise AssertionError(
                        f"malformed other-region tag line for "
                        f"{snapshot['snapshot_id']}: {line!r}"
                    )
            for line in quality_lines:
                if quality_pattern.match(line) is None:
                    raise AssertionError(
                        f"malformed other-region quality line for "
                        f"{snapshot['snapshot_id']}: {line!r}"
                    )
        audit.append(
            {
                "snapshot_index": snapshot["snapshot_index"],
                "history_hash": _text_hash(snapshot["history_text"]),
                "frontier_hash": _text_hash(_frontier_text(snapshot)),
                "prompt_hashes": {
                    key: _text_hash(value) for key, value in prompts.items()
                },
            }
        )
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "snapshots.jsonl", snapshots)
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
            "eval_intervals": [list(item) for item in EVAL_INTERVALS],
            "snapshots_per_unit": SNAPSHOTS_PER_UNIT,
            "snapshot_count": len(snapshots),
            "replicates_per_snapshot_condition": REPLICATES,
            "trial_count": len(schedule),
            "design_seed": seed,
            "min_evals_before_decision": MIN_EVALS_BEFORE_DECISION,
            "snapshot_unit": "real V9.7 Explore decision replayed at decision time",
            "availability_rule": "programs with real evaluator order e(p) <= b_t",
            "primary_contrasts": [
                "fp_explore - pp_explore",
                "fp_refine - pp_refine",
            ],
            "temperature": 1.0,
            "max_new_tokens": OUTPUT_TOKENS,
            "max_total_context": TOTAL_CONTEXT_TOKENS,
            "generation_evaluation_pipeline": "same_process_immediate_per_trial",
            "independent_unit": "real decision snapshot",
            "within_snapshot_repeated_sampling": REPLICATES,
            "blocking": (
                "task x source run x eval interval; one shared sampling seed "
                "for the five conditions within each snapshot replicate block"
            ),
        },
    )
    print(f"prepared {len(snapshots)} snapshots / {len(schedule)} trials in {run_dir}")


def _forest_cache_entry(task: str, run_name: str, batch: str) -> dict[str, Any]:
    runs = [run for run in _source_run_dirs(task, batch) if run.name == run_name]
    if len(runs) != 1:
        raise ValueError(f"cannot locate source run {run_name} for task {task}")
    return _read_json(runs[0] / "checkpoints" / "latest.json")


def run_probe_shard(
    run_dir: Path,
    *,
    backend: str,
    shard_index: int,
    num_shards: int,
    eval_workers: int,
    batch: str = SOURCE_BATCH,
) -> None:
    config = _read_json(run_dir / "probe_config.json")
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("probe configuration protocol mismatch")
    snapshots = {row["snapshot_index"]: row for row in _read_jsonl(run_dir / "snapshots.jsonl")}
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["group_order"]) % num_shards == shard_index
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
    forests: dict[str, dict[str, Any]] = {}
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
            snapshot = snapshots[trial["snapshot_index"]]
            if trial["source_run"] not in forests:
                forests[trial["source_run"]] = _forest_cache_entry(
                    task, trial["source_run"], batch
                )["forest"]
            prompt = build_trial_prompt(snapshot, evaluation.task_description, trial["condition"])
            prompt_tokens = int(llm.count_prompt_tokens(prompt))
            if prompt_tokens + OUTPUT_TOKENS > TOTAL_CONTEXT_TOKENS:
                raise ValueError(f"trial {trial['trial_id']} exceeds total context")
            call = calls.get(trial["trial_id"])
            if call is None:
                call = _draw_call(trial=trial, prompt=prompt, llm=llm)
                _append_jsonl(call_path, call)
                calls[trial["trial_id"]] = call
            result = _evaluate_trial(
                trial=trial,
                snapshot=snapshot,
                call=call,
                evaluation=evaluation,
                evaluator=evaluator,
                forest_payload=forests[trial["source_run"]],
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
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(
                f"stage-p shard={shard_index} trial={trial['trial_id']} "
                f"status={result['status']} completed={len(completed)}/{len(schedule)}",
                flush=True,
            )
    finally:
        llm.close()


def _evaluate_trial(
    *,
    trial: dict[str, Any],
    snapshot: dict[str, Any],
    call: dict[str, Any],
    evaluation,
    evaluator: SecureEvaluator,
    forest_payload: dict[str, Any],
) -> dict[str, Any]:
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    if template is None or len(template.functions) != 1:
        raise ValueError("probe requires one evolvable template function")
    s0 = float(snapshot["source_s0"])
    visited_frontier: dict[str, float] = snapshot["visited_families"]
    base = {
        **{key: trial[key] for key in (
            "trial_id", "block_id", "snapshot_index", "snapshot_id", "task",
            "source_run", "eval_interval", "replicate", "condition",
            "context_prefix", "intent", "sampling_seed",
        )},
        "protocol_id": PROTOCOL_ID,
        "b_t": snapshot["decision"]["b_t"],
        "iteration": snapshot["decision"]["iteration"],
        "anchor_state_id": snapshot["anchor_state_id"],
        "anchor_family": snapshot["anchor_family"],
        "parent_fitness": snapshot["fitness"],
        "parent_q": snapshot["q"],
        "parent_code_hash": snapshot["code_hash"],
        "history_hash": _text_hash(snapshot["history_text"]),
        "frontier_hash": _text_hash(_frontier_text(snapshot)),
        "source_s0": s0,
        "global_best_q": snapshot["global_best_q"],
        "prompt_hash": call["prompt_hash"],
        "prompt_tokens": call["prompt_tokens"],
        "response_hash": call["response_hash"],
        "response_tokens": call["response_tokens"],
        "sample_seconds": call["sample_seconds"],
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        parsed = parse_program_response(
            call["response"], template, template.functions[0].name
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
            "no_op": False,
            "archive_duplicate": False,
            "code_novel": False,
            "next_selection": False,
        }
    candidate_code = str(parsed.program)
    candidate_hash = code_hash(candidate_code)
    diff, added, removed = code_diff(snapshot["code"], candidate_code)
    tags = mechanism_tags(snapshot["task"], candidate_code)
    family = macro_family(snapshot["task"], tags)
    no_op = candidate_hash == snapshot["code_hash"]
    archive_duplicate = candidate_hash in set(snapshot["archive_code_hashes"])
    destination = (
        "current_region"
        if family == snapshot["anchor_family"]
        else "other_visited"
        if family in visited_frontier
        else "new_region"
    )
    candidate_base = {
        **base,
        "idea": parsed.declared_idea,
        "candidate_code_hash": candidate_hash,
        "candidate_code": candidate_code,
        "candidate_tags": sorted(tags),
        "candidate_family": family,
        "destination": destination,
        "diff": diff,
        "added": added,
        "removed": removed,
        "no_op": no_op,
        "archive_duplicate": archive_duplicate,
        "code_novel": not archive_duplicate,
    }
    if no_op:
        return {
            **candidate_base,
            "status": "valid",
            "valid": True,
            "failure_kind": None,
            "child_fitness": snapshot["fitness"],
            "child_q": snapshot["q"],
            "delta_q": 0.0,
            "evaluator_called": False,
            "next_selection": False,
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
        child_fitness = float("nan")
    if not math.isfinite(child_fitness):
        return {
            **candidate_base,
            "status": "invalid",
            "valid": False,
            "failure_kind": outcome.failure_kind or "invalid_result",
            "failure_error": outcome.error,
            "evaluator_called": True,
            "evaluate_seconds": evaluate_seconds,
            "next_selection": False,
        }
    child_q = child_fitness if bool(snapshot["maximize"]) else -child_fitness
    delta_q = child_q - float(snapshot["q"])
    next_selection = _next_selection(
        forest_payload,
        snapshot,
        candidate_code,
        candidate_hash,
        child_fitness,
    )
    frontier_q = visited_frontier.get(family)
    result = {
        **candidate_base,
        "status": "valid",
        "valid": True,
        "failure_kind": None,
        "child_fitness": child_fitness,
        "child_q": child_q,
        "delta_q": delta_q,
        "delta_q_over_s": delta_q / s0,
        "global_gap_over_s": (child_q - float(snapshot["global_best_q"])) / s0,
        "parent_improvement": delta_q > 0.0,
        "evaluator_called": True,
        "evaluate_seconds": evaluate_seconds,
        "next_selection": next_selection,
    }
    if frontier_q is not None:
        result["frontier_gap_over_s"] = (child_q - frontier_q) / s0
        result["sub_frontier"] = child_q < frontier_q - SUB_FRONTIER_MARGIN_S * s0
        result["advances_frontier"] = child_q > frontier_q
    else:
        result["frontier_gap_over_s"] = None
        result["sub_frontier"] = False
        result["advances_frontier"] = None
    return result


def _next_selection(
    forest_payload: dict[str, Any],
    snapshot: dict[str, Any],
    candidate_code: str,
    candidate_hash: str,
    child_fitness: float,
) -> bool:
    """Apply the V9.7 update rules to the decision snapshot and report
    whether the new child becomes the anchor of the immediate next
    selection."""

    forest = trim_forest(forest_payload, snapshot["decision"]["order"])
    anchor_id = snapshot["anchor_state_id"]
    anchor = forest.get_anchor(anchor_id)
    anchor.n += 1
    duplicate = forest.program_for_code(candidate_code)
    child = None
    if candidate_hash == snapshot["code_hash"]:
        child = None  # no-op response creates no state
    elif duplicate is not None:
        parent_program_id = anchor.program_id
        if (
            duplicate.id != parent_program_id
            and duplicate.id not in forest.ancestor_program_ids(anchor_id)
            and not forest.relation_exists(anchor_id, duplicate.id)
        ):
            child = forest.add_child(
                parent_id=anchor_id,
                program_id=duplicate.id,
                attempt_id=forest.next_attempt_id(),
                order=10**9,
            )
    else:
        program = forest.add_program(
            code=candidate_code, fitness=child_fitness, order=10**9
        )
        child = forest.add_child(
            parent_id=anchor_id,
            program_id=program.id,
            attempt_id=forest.next_attempt_id(),
            order=10**9,
        )
    if child is None:
        return False
    choice = select(forest, float(snapshot["source_s0"]))
    return choice.anchor_id == child.id


def status_probe(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    results: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        results.extend(_read_jsonl(path))
    by_condition = {
        condition: {
            "trials": sum(row["condition"] == condition for row in results),
            "valid": sum(
                row["condition"] == condition and row.get("valid") is True
                for row in results
            ),
        }
        for condition in CONDITIONS
    }
    return {
        "protocol_id": config["protocol_id"],
        "completed_trials": len({row["trial_id"] for row in results}),
        "total_trials": int(config["trial_count"]),
        "valid_trials": sum(row.get("valid") is True for row in results),
        "invalid_trials": sum(row.get("valid") is False for row in results),
        "evaluator_calls": sum(row.get("evaluator_called") is True for row in results),
        "by_condition": by_condition,
    }


def _default_run_dir() -> Path:
    name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_v913_stage_p"
    return EXPERIMENTS_ROOT / "generation_probe" / name


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-probe")
    prepare.add_argument("--run-dir", type=Path)
    prepare.add_argument("--source-batch", default=SOURCE_BATCH)
    prepare.add_argument("--tasks", default=",".join(TASKS))
    prepare.add_argument("--seed", type=int, default=DESIGN_SEED)
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
            seed=args.seed,
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
