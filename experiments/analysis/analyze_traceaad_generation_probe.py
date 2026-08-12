#!/usr/bin/env python3
"""Analyze the fixed-anchor concise-history probe at the anchor level."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


DEFAULT_CONTROL = "no_history"
DEFAULT_TREATMENT = "concise_history"
DEFAULT_STRATA = ("low", "middle", "high")
QUALITY_METRICS = (
    "valid_rate",
    "conditional_delta_q",
    "conditional_improvement_rate",
    "conditional_change_ratio",
    "conditional_abs_loc_delta",
)
DIAGNOSTIC_METRICS = (
    "exact_parent_no_op_rate",
    "mean_prompt_tokens",
    "mean_response_tokens",
)
METRICS = QUALITY_METRICS + DIAGNOSTIC_METRICS
BOOTSTRAP_SEED = 950502
BOOTSTRAP_SAMPLES = 10_000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def aggregate_anchor_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]
    return {
        "task": rows[0]["task"],
        "stratum": rows[0]["stratum"],
        "anchor_id": rows[0]["anchor_id"],
        "condition": rows[0]["condition"],
        "replicate_count": len(rows),
        "valid_n": len(valid),
        "valid_rate": len(valid) / len(rows),
        "exact_parent_no_op_rate": mean(
            [float(row.get("no_op") is True) for row in rows]
        ),
        "mean_prompt_tokens": mean([float(row["prompt_tokens"]) for row in rows]),
        "mean_response_tokens": mean(
            [float(row["response_tokens"]) for row in rows]
        ),
        "conditional_delta_q": mean([float(row["delta_q"]) for row in valid]),
        "conditional_improvement_rate": mean(
            [float(bool(row["improved"])) for row in valid]
        ),
        "conditional_change_ratio": mean(
            [float(row["change_ratio"]) for row in valid]
        ),
        "conditional_abs_loc_delta": mean(
            [abs(float(row["loc_delta"])) for row in valid]
        ),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_ci(values: list[float], *, seed: int) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(statistics.fmean(sample))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def paired_summary(
    anchors: list[dict[str, Any]],
    *,
    selector: Callable[[dict[str, Any]], bool],
    seed_offset: int,
    control: str,
    treatment: str,
) -> dict[str, Any]:
    selected = [row for row in anchors if selector(row)]
    result: dict[str, Any] = {"anchor_count": len(selected), "metrics": {}}
    for metric_index, metric in enumerate(METRICS):
        differences = []
        a_values = []
        b_values = []
        for row in selected:
            a_row = row["conditions"].get(control)
            b_row = row["conditions"].get(treatment)
            if a_row is None or b_row is None:
                continue
            a = a_row.get(metric)
            b = b_row.get(metric)
            if a is None or b is None:
                continue
            a_values.append(float(a))
            b_values.append(float(b))
            differences.append(float(b) - float(a))
        result["metrics"][metric] = {
            "paired_anchor_count": len(differences),
            f"{control}_mean": mean(a_values),
            f"{treatment}_mean": mean(b_values),
            "paired_mean_difference_b_minus_a": mean(differences),
            "paired_median_difference_b_minus_a": (
                None if not differences else statistics.median(differences)
            ),
            "paired_mean_difference_bootstrap_95ci": bootstrap_ci(
                differences, seed=BOOTSTRAP_SEED + seed_offset + metric_index
            ),
            "anchors_better_b": sum(value > 0 for value in differences),
            "anchors_tied": sum(value == 0 for value in differences),
            "anchors_better_a": sum(value < 0 for value in differences),
        }
    return result


def strata_order(values: set[str]) -> list[str]:
    known = [item for item in DEFAULT_STRATA if item in values]
    return known + sorted(values - set(known))


def analyze(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "probe_config.json")
    control = str(config.get("control_condition", DEFAULT_CONTROL))
    treatment = str(config.get("treatment_condition", DEFAULT_TREATMENT))
    condition_names = (control, treatment)
    schedule = read_jsonl(run_dir / "schedule.jsonl")
    expected = {row["trial_id"] for row in schedule}
    result_rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("*.jsonl")):
        result_rows.extend(read_jsonl(path))
    by_trial = {row["trial_id"]: row for row in result_rows}
    duplicates = len(result_rows) - len(by_trial)
    missing = sorted(expected - set(by_trial))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_trial.values():
        grouped[(row["anchor_id"], row["condition"])].append(row)
    anchor_conditions = {
        key: aggregate_anchor_condition(rows) for key, rows in grouped.items()
    }
    anchors = []
    for anchor_id in sorted({key[0] for key in anchor_conditions}):
        conditions = {
            condition: anchor_conditions.get((anchor_id, condition))
            for condition in condition_names
        }
        available = next((item for item in conditions.values() if item), None)
        if available is None:
            continue
        anchors.append(
            {
                "anchor_id": anchor_id,
                "task": available["task"],
                "stratum": available["stratum"],
                "conditions": conditions,
            }
        )

    tasks = config.get("tasks") or sorted({anchor["task"] for anchor in anchors})
    strata = config.get("strata") or strata_order(
        {anchor["stratum"] for anchor in anchors}
    )

    summaries: dict[str, Any] = {}
    for task_index, task in enumerate(tasks):
        summaries[f"task:{task}"] = paired_summary(
            anchors,
            selector=lambda row, task=task: row["task"] == task,
            seed_offset=100 * task_index,
            control=control,
            treatment=treatment,
        )
        for stratum_index, stratum in enumerate(strata):
            summaries[f"task:{task}:stratum:{stratum}"] = paired_summary(
                anchors,
                selector=lambda row, task=task, stratum=stratum: (
                    row["task"] == task and row["stratum"] == stratum
                ),
                seed_offset=100 * task_index + 10 * (stratum_index + 1),
                control=control,
                treatment=treatment,
            )

    overall_signs: dict[str, Any] = {}
    for metric in QUALITY_METRICS:
        differences = []
        for anchor in anchors:
            a_row = anchor["conditions"].get(control)
            b_row = anchor["conditions"].get(treatment)
            if a_row is None or b_row is None:
                continue
            a = a_row.get(metric)
            b = b_row.get(metric)
            if a is not None and b is not None:
                differences.append(float(b) - float(a))
        overall_signs[metric] = {
            "paired_anchor_count": len(differences),
            "anchors_better_b": sum(value > 0 for value in differences),
            "anchors_tied": sum(value == 0 for value in differences),
            "anchors_better_a": sum(value < 0 for value in differences),
            "note": "Direction count only; raw delta-q magnitudes are not pooled across tasks.",
        }

    return {
        "protocol_id": config["protocol_id"],
        "conditions": [control, treatment],
        "control_condition": control,
        "treatment_condition": treatment,
        "complete": not missing and duplicates == 0,
        "expected_trials": len(expected),
        "completed_unique_trials": len(by_trial),
        "missing_trial_count": len(missing),
        "missing_trial_ids": missing,
        "duplicate_result_rows": duplicates,
        "independent_unit": "anchor snapshot",
        "anchor_level_rows": anchors,
        "overall_paired_direction": overall_signs,
        "paired_summaries": summaries,
        "interpretation_boundary": (
            "This probe identifies one-step generation effects for fixed anchors. "
            "It does not estimate full-search or held-out performance. Delta-q "
            "summaries are task-specific."
        ),
    }


def write_anchor_csv(path: Path, analysis: dict[str, Any]) -> None:
    fields = [
        "task",
        "stratum",
        "anchor_id",
        "condition",
        "replicate_count",
        "valid_n",
        *METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for anchor in analysis["anchor_level_rows"]:
            for condition in analysis["conditions"]:
                row = anchor["conditions"].get(condition)
                if row is not None:
                    writer.writerow({key: row.get(key) for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze(args.run_dir)
    output = args.run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_anchor_csv(output / "anchor_condition_metrics.csv", analysis)
    print(json.dumps({key: analysis[key] for key in ("complete", "expected_trials", "completed_unique_trials", "missing_trial_count", "duplicate_result_rows")}, indent=2))


if __name__ == "__main__":
    main()
