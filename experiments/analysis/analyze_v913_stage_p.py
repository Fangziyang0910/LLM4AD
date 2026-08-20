#!/usr/bin/env python3
"""Analyze the V9.13 Stage P decision-snapshot probe.

Aggregation follows design sections 7.3-7.4: the three responses of each
snapshot-condition are technical replicates and are averaged first; contrasts
are snapshot-level paired differences; uncertainty is described with
direction counts and hierarchical (source run -> snapshot) bootstrap
intervals.  Section 8's frozen candidate-selection rule is evaluated and
reported, never applied silently.

Usage:

    uv run python experiments/analysis/analyze_v913_stage_p.py \
        --probe-dir experiments/generation_probe/<...>_v913_stage_p
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 913901

CONTRASTS = (
    ("fp_explore", "pp_explore"),
    ("fp_refine", "pp_refine"),
)

# Metric name -> (per-response key or predicate, conditional-on-valid).
# "conditional" metrics average over valid responses only; a
# snapshot-condition with no valid response yields None and drops out of
# that metric's paired analysis.
RESPONSE_METRICS: dict[str, tuple[Any, bool]] = {
    "valid_rate": (lambda row: bool(row.get("valid")), False),
    "evaluable_novel_rate": (
        lambda row: bool(row.get("valid")) and bool(row.get("code_novel")),
        False,
    ),
    "conditional_delta_q_over_s": ("delta_q_over_s", True),
    "parent_improvement_rate": ("parent_improvement", True),
    "global_gap_over_s": ("global_gap_over_s", True),
    "archive_duplicate_rate": (lambda row: bool(row.get("archive_duplicate")), False),
    "no_op_rate": (lambda row: bool(row.get("no_op")), False),
    "next_selection_rate": (lambda row: bool(row.get("next_selection")), False),
    "sub_frontier_response_rate": (
        lambda row: (
            bool(row.get("valid"))
            and bool(row.get("code_novel"))
            and row.get("destination") in {"current_region", "other_visited"}
            and bool(row.get("sub_frontier"))
        ),
        False,
    ),
    "proxy_region_switch_rate": (
        lambda row: (
            bool(row.get("valid"))
            and bool(row.get("code_novel"))
            and row.get("destination") != "current_region"
        ),
        False,
    ),
    "frontier_advance_rate": (
        lambda row: (
            bool(row.get("valid"))
            and bool(row.get("code_novel"))
            and bool(row.get("advances_frontier"))
        ),
        False,
    ),
    "prompt_tokens": (lambda row: float(row.get("prompt_tokens") or 0.0), False),
    "response_tokens": (lambda row: float(row.get("response_tokens") or 0.0), False),
}


def _iter_results(probe_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((probe_dir / "results").glob("shard_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _metric_value(row: dict[str, Any], spec: tuple[Any, bool]) -> float | None:
    key, conditional = spec
    if conditional:
        if not row.get("valid"):
            return None
        value = row.get(key)
        return None if value is None else float(value)
    if callable(key):
        return 1.0 if key(row) else 0.0
    value = row.get(key)
    return None if value is None else float(value)


def _condition_mean(rows: Sequence[dict[str, Any]], metric: str) -> float | None:
    spec = RESPONSE_METRICS[metric]
    if spec[1]:
        values = [
            _metric_value(row, spec) for row in rows if row.get("valid")
        ]
    else:
        values = [_metric_value(row, spec) for row in rows]
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    if not spec[1]:
        return statistics.mean(kept)
    return statistics.mean(kept)


def snapshot_condition_table(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, float | None]]:
    """(task, snapshot, condition) -> metric means over the replicates."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["task"], row["snapshot_index"], row["condition"])
        ].append(row)
    table: dict[tuple[str, str, str], dict[str, float | None]] = {}
    for key, block in sorted(grouped.items()):
        table[key] = {
            metric: _condition_mean(block, metric)
            for metric in RESPONSE_METRICS
        }
    return table


def _hierarchical_bootstrap(
    diffs: Sequence[float],
    runs: Sequence[str],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> list[float]:
    by_run: dict[str, list[float]] = defaultdict(list)
    for value, run in zip(diffs, runs):
        by_run[run].append(value)
    run_names = sorted(by_run)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(draws):
        pooled: list[float] = []
        for _ in range(len(run_names)):
            chosen_run = run_names[rng.randrange(len(run_names))]
            values = by_run[chosen_run]
            pooled.append(values[rng.randrange(len(values))])
        samples.append(statistics.mean(pooled))
    samples.sort()
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def analyze(probe_dir: Path) -> dict[str, Any]:
    config = json.loads((probe_dir / "probe_config.json").read_text(encoding="utf-8"))
    rows = _iter_results(probe_dir)
    expected = int(config["trial_count"])
    completed = {row["trial_id"] for row in rows}
    schedule = _read_jsonl(probe_dir / "schedule.jsonl")
    missing = [row["trial_id"] for row in schedule if row["trial_id"] not in completed]

    table = snapshot_condition_table(rows)
    snapshot_meta = {
        row["snapshot_index"]: (row["task"], row["source_run"])
        for row in schedule
    }

    contrasts: dict[str, Any] = {}
    for treatment, control in CONTRASTS:
        per_task: dict[str, Any] = {}
        for task in sorted({key[0] for key in table}):
            diffs_by_metric: dict[str, dict[str, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for (row_task, snapshot, condition), metrics in table.items():
                if row_task != task or condition != treatment:
                    continue
                for metric, value in metrics.items():
                    control_value = table.get((row_task, snapshot, control), {}).get(
                        metric
                    )
                    if value is None or control_value is None:
                        continue
                    run = snapshot_meta[snapshot][1]
                    diffs_by_metric[metric][run].append(float(value) - float(control_value))
            task_report: dict[str, Any] = {}
            for metric, by_run in sorted(diffs_by_metric.items()):
                flat = [v for values in by_run.values() for v in values]
                if not flat:
                    continue
                ordered_runs = [run for run in sorted(by_run)]
                all_diffs = [v for run in ordered_runs for v in by_run[run]]
                run_means = {
                    run: statistics.mean(by_run[run]) for run in ordered_runs
                }
                ci = _hierarchical_bootstrap(
                    all_diffs, [run for run in ordered_runs for _ in by_run[run]]
                )
                task_report[metric] = {
                    "snapshot_pairs": len(flat),
                    "paired_mean_difference": statistics.mean(flat),
                    "positive_snapshots": sum(value > 0 for value in flat),
                    "negative_snapshots": sum(value < 0 for value in flat),
                    "source_run_means": run_means,
                    "bootstrap_95ci": ci,
                }
            per_task[task] = task_report
        contrasts[f"{treatment} - {control}"] = per_task

    destination = _destination_distribution(rows)
    decision_rule = evaluate_decision_rule(contrasts)
    return {
        "analysis": "traceaad_v913_stage_p",
        "probe_dir": str(probe_dir),
        "protocol_id": config["protocol_id"],
        "trial_accounting": {
            "expected": expected,
            "completed": len(completed),
            "missing": missing,
        },
        "contrasts": contrasts,
        "destination_distribution": destination,
        "candidate_selection_rule": decision_rule,
        "boundary": (
            "Three source runs per task: intervals describe uncertainty and "
            "do not support high-precision significance claims; thresholds in "
            "the decision rule are development rules, not statistical tests."
        ),
    }


def _destination_distribution(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["condition"])].append(row)
    for (task, condition), block in sorted(grouped.items()):
        novel = [
            row for row in block if row.get("valid") and row.get("code_novel")
        ]
        counts = {"current_region": 0, "other_visited": 0, "new_region": 0}
        for row in novel:
            counts[row["destination"]] = counts.get(row["destination"], 0) + 1
        output[f"{task}:{condition}"] = {
            "code_novel_valid": len(novel),
            "destination_counts": counts,
        }
    return output


# Design section 8: frozen candidate-selection thresholds.
RULE_THRESHOLD_DELTA_Q_DROP = 0.5
RULE_THRESHOLD_VALID_DROP = 0.05
RULE_THRESHOLD_DUPLICATE_RISE = 0.05
RULE_TASKS_REQUIRED = 3


def evaluate_decision_rule(contrasts: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen section-8 rule; report everything, choose once."""

    def task_means(metric: str) -> dict[str, float]:
        block = contrasts.get("fp_explore - pp_explore", {})
        return {
            task: report[metric]["paired_mean_difference"]
            for task, report in block.items()
            if metric in report
        }

    delta = task_means("conditional_delta_q_over_s")
    valid = task_means("valid_rate")
    duplicate = task_means("archive_duplicate_rate")
    nxt = task_means("next_selection_rate")
    sub = task_means("sub_frontier_response_rate")
    tasks = sorted(set(delta) & set(valid) & set(duplicate) & set(nxt) & set(sub))
    checks = {
        "tasks_evaluated": tasks,
        "delta_q_improves_on_k_tasks": sum(value > 0 for value in delta.values()),
        "max_delta_q_drop": max([0.0] + [-value for value in delta.values()]),
        "max_valid_rate_drop": max([0.0] + [-value for value in valid.values()]),
        "max_duplicate_rate_rise": max([0.0] + list(duplicate.values())),
        "next_selection_improves_on_k_tasks": sum(value > 0 for value in nxt.values()),
        "sub_frontier_not_worse_on_k_tasks": sum(value <= 0 for value in sub.values()),
        "thresholds": {
            "tasks_required": RULE_TASKS_REQUIRED,
            "max_delta_q_drop": RULE_THRESHOLD_DELTA_Q_DROP,
            "max_valid_rate_drop": RULE_THRESHOLD_VALID_DROP,
            "max_duplicate_rate_rise": RULE_THRESHOLD_DUPLICATE_RISE,
        },
    }
    passed = bool(tasks) and (
        checks["delta_q_improves_on_k_tasks"] >= RULE_TASKS_REQUIRED
        and checks["max_delta_q_drop"] <= RULE_THRESHOLD_DELTA_Q_DROP
        and checks["max_valid_rate_drop"] <= RULE_THRESHOLD_VALID_DROP
        and checks["max_duplicate_rate_rise"] <= RULE_THRESHOLD_DUPLICATE_RISE
        and checks["next_selection_improves_on_k_tasks"] >= RULE_TASKS_REQUIRED
        and checks["sub_frontier_not_worse_on_k_tasks"] >= RULE_TASKS_REQUIRED
    )
    if passed:
        reason = "FP passed the frozen gate; freeze FP and proceed to Stage A"
    else:
        reason = "FP failed the frozen gate; V9.13 stops at identification"
    return {
        "fp_gate": {"passed": passed, "checks": checks},
        "outcome": {"selected_treatment": "fp" if passed else "none", "reason": reason},
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir", type=Path, required=True, help="Stage P probe directory"
    )
    args = parser.parse_args()
    result = analyze(args.probe_dir)
    out_dir = args.probe_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path}")
    rule = result["candidate_selection_rule"]
    print(json.dumps(rule.get("outcome"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
