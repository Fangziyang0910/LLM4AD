"""Analyze the paired V9.17 Adaptive versus FixedCycle scheduler ablation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.runners._common import EXPERIMENTS_ROOT, TASKS

BUDGETS = (100, 250, 500, 750, 1000)


def analyze_run(run_dir: Path) -> dict[str, Any]:
    rows = _read_rows(run_dir / "evaluations.csv")
    primary = [row for row in rows if row["attempt_kind"] == "initial"]
    valid = [
        (int(row["eval_count"]), float(row["fitness"]))
        for row in rows
        if row["status"] == "ok" and _finite(row["fitness"])
    ]
    best_at_budget = {
        str(budget): max(
            (fitness for slot, fitness in valid if slot <= budget), default=None
        )
        for budget in BUDGETS
    }
    intent_counts = Counter(row["intent"] or "root" for row in primary)
    post_root = intent_counts["refine"] + intent_counts["explore"]
    discovery_slots = [
        int(row["eval_count"]) for row in primary if row["mode"] == "discovery"
    ]
    initial_slots = max(
        (
            int(row["eval_count"])
            for row in primary
            if row["mode"] in {"root", "initial_maturation"}
        ),
        default=0,
    )
    discovery_distances = [
        slot - (initial_slots if index == 0 else discovery_slots[index - 1])
        for index, slot in enumerate(discovery_slots)
    ]

    block_events = _block_finish_events(run_dir / "mechanism_events.jsonl")
    development = [
        event for event in block_events if event["block_kind"] == "development"
    ]
    cycle_sweeps: dict[int, int] = defaultdict(int)
    by_cycle_hypothesis: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in development:
        cycle = int(event["cycle"])
        sweep = int(event["sweep"])
        hypothesis = int(event["hypothesis_id"])
        cycle_sweeps[cycle] = max(cycle_sweeps[cycle], sweep)
        by_cycle_hypothesis[(cycle, hypothesis)].append(event)

    positive_with_next = 0
    positive_next_success = 0
    for events in by_cycle_hypothesis.values():
        ordered = sorted(events, key=lambda event: int(event["sweep"]))
        for current, following in zip(ordered, ordered[1:]):
            if float(current["gain"]) <= 0:
                continue
            if int(following["sweep"]) != int(current["sweep"]) + 1:
                continue
            positive_with_next += 1
            positive_next_success += float(following["gain"]) > 0

    extra_blocks = [event for event in development if int(event["sweep"]) > 1]
    global_advance = _global_frontier_advances(rows, extra_blocks)
    refine_allocations = Counter(
        int(row["hypothesis_id"])
        for row in primary
        if row["intent"] == "refine" and row["hypothesis_id"]
    )
    allocation_total = sum(refine_allocations.values())
    shares = (
        [count / allocation_total for count in refine_allocations.values()]
        if allocation_total
        else []
    )
    summary_path = run_dir / "logs" / "summary.json"
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    return {
        "run_dir": str(run_dir),
        "complete": (
            summary.get("status") == "finished"
            and summary.get("budget_slots") == 1000
        ),
        "current_primary_slots": max(
            (int(row["eval_count"]) for row in primary), default=0
        ),
        "intent_slots": dict(intent_counts),
        "explore_fraction_post_root": (
            intent_counts["explore"] / post_root if post_root else None
        ),
        "refine_fraction_post_root": (
            intent_counts["refine"] / post_root if post_root else None
        ),
        "discovery_slots": discovery_slots,
        "discovery_distances": discovery_distances,
        "mean_discovery_distance": _mean(discovery_distances),
        "sweeps_per_cycle": dict(sorted(cycle_sweeps.items())),
        "mean_sweeps_per_cycle": _mean(list(cycle_sweeps.values())),
        "max_sweeps_per_cycle": max(cycle_sweeps.values(), default=None),
        "positive_blocks_with_observed_next": positive_with_next,
        "positive_next_successes": positive_next_success,
        "positive_next_success_rate": (
            positive_next_success / positive_with_next
            if positive_with_next
            else None
        ),
        "adaptive_extra_blocks": len(extra_blocks),
        "extra_blocks_advancing_hypothesis_frontier": sum(
            float(event["gain"]) > 0 for event in extra_blocks
        ),
        "extra_blocks_advancing_global_frontier": global_advance,
        "refine_allocation_max_share": max(shares, default=None),
        "refine_allocation_hhi": sum(share * share for share in shares),
        "best_at_budget": best_at_budget,
        "search_best": max((fitness for _, fitness in valid), default=None),
    }


def analyze_pairs(adaptive_batch: str, fixed_batch: str) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for repeat in range(1, 4):
        for task in TASKS:
            adaptive = (
                EXPERIMENTS_ROOT
                / task
                / "traceaad_v9_17"
                / f"v9_17_{adaptive_batch}_{task}_rep{repeat}"
            )
            fixed = (
                EXPERIMENTS_ROOT
                / task
                / "traceaad_v9_17_fixed_cycle"
                / f"v9_17_fixed_cycle_{fixed_batch}_{task}_rep{repeat}"
            )
            adaptive_metrics = analyze_run(adaptive) if adaptive.is_dir() else None
            fixed_metrics = analyze_run(fixed) if fixed.is_dir() else None
            pairs.append(
                {
                    "task": task,
                    "repeat": repeat,
                    "adaptive": adaptive_metrics,
                    "fixed_cycle": fixed_metrics,
                    "adaptive_minus_fixed": _paired_differences(
                        adaptive_metrics, fixed_metrics
                    ),
                }
            )
    return {
        "adaptive_batch": adaptive_batch,
        "fixed_batch": fixed_batch,
        "budgets": list(BUDGETS),
        "pairs": pairs,
    }


def _paired_differences(
    adaptive: dict[str, Any] | None, fixed: dict[str, Any] | None
) -> dict[str, float | None] | None:
    if adaptive is None or fixed is None:
        return None
    differences: dict[str, float | None] = {}
    for budget in BUDGETS:
        key = str(budget)
        left = adaptive["best_at_budget"][key]
        right = fixed["best_at_budget"][key]
        differences[f"best_at_{budget}"] = (
            left - right if left is not None and right is not None else None
        )
    left = adaptive["search_best"]
    right = fixed["search_best"]
    differences["search_best"] = (
        left - right if left is not None and right is not None else None
    )
    return differences


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _block_finish_events(path: Path) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event") == "block_finish":
            by_id[int(event["block_id"])] = event
    return [by_id[key] for key in sorted(by_id)]


def _global_frontier_advances(
    rows: list[dict[str, str]], blocks: list[dict[str, Any]]
) -> int:
    valid = [
        row for row in rows if row["status"] == "ok" and _finite(row["fitness"])
    ]
    advances = 0
    for event in blocks:
        block_id = str(event["block_id"])
        block_rows = [row for row in valid if row["block_id"] == block_id]
        if not block_rows:
            continue
        first_slot = min(int(row["eval_count"]) for row in block_rows)
        before = max(
            (
                float(row["fitness"])
                for row in valid
                if int(row["eval_count"]) < first_slot
            ),
            default=-math.inf,
        )
        advances += max(float(row["fitness"]) for row in block_rows) > before
    return advances


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-batch", required=True)
    parser.add_argument("--fixed-batch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze_pairs(args.adaptive_batch, args.fixed_batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
