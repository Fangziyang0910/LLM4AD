#!/usr/bin/env python3
"""Analyze the three-arm parent-path probe at the anchor level.

Contrasts: parent_path - code_only (value of the improvement path) and
parent_path_child - parent_path (added value of direct child attempts).
Metric definitions and aggregation are identical to the two earlier probes.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from analyze_traceaad_generation_probe import (  # noqa: E402
    METRICS,
    QUALITY_METRICS,
    aggregate_anchor_condition,
    paired_summary,
    read_json,
    read_jsonl,
)

CONDITIONS = ("code_only", "parent_path", "parent_path_child")
CONTRASTS = (
    ("code_only", "parent_path"),
    ("parent_path", "parent_path_child"),
    ("code_only", "parent_path_child"),
)


def analyze(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "probe_config.json")
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
            for condition in CONDITIONS
        }
        available = next(item for item in conditions.values() if item)
        anchors.append(
            {
                "anchor_id": anchor_id,
                "task": available["task"],
                "stratum": available["stratum"],
                "conditions": conditions,
            }
        )

    tasks = config["tasks"]
    strata = config["strata"]
    contrasts: dict[str, Any] = {}
    for contrast_index, (control, treatment) in enumerate(CONTRASTS):
        summaries: dict[str, Any] = {}
        for task_index, task in enumerate(tasks):
            summaries[f"task:{task}"] = paired_summary(
                anchors,
                selector=lambda row, task=task: row["task"] == task,
                seed_offset=10_000 * contrast_index + 100 * task_index,
                control=control,
                treatment=treatment,
            )
            for stratum_index, stratum in enumerate(strata):
                summaries[f"task:{task}:stratum:{stratum}"] = paired_summary(
                    anchors,
                    selector=lambda row, task=task, stratum=stratum: (
                        row["task"] == task and row["stratum"] == stratum
                    ),
                    seed_offset=(
                        10_000 * contrast_index
                        + 100 * task_index
                        + 10 * (stratum_index + 1)
                    ),
                    control=control,
                    treatment=treatment,
                )
        directions: dict[str, Any] = {}
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
            directions[metric] = {
                "paired_anchor_count": len(differences),
                "anchors_better_treatment": sum(v > 0 for v in differences),
                "anchors_tied": sum(v == 0 for v in differences),
                "anchors_better_control": sum(v < 0 for v in differences),
                "note": (
                    "Direction count only; raw delta-q magnitudes are not "
                    "pooled across tasks."
                ),
            }
        contrasts[f"{treatment}_minus_{control}"] = {
            "control": control,
            "treatment": treatment,
            "overall_paired_direction": directions,
            "paired_summaries": summaries,
        }

    return {
        "protocol_id": config["protocol_id"],
        "conditions": list(CONDITIONS),
        "complete": not missing and duplicates == 0,
        "expected_trials": len(expected),
        "completed_unique_trials": len(by_trial),
        "missing_trial_count": len(missing),
        "duplicate_result_rows": duplicates,
        "independent_unit": "anchor snapshot",
        "anchor_level_rows": anchors,
        "contrasts": contrasts,
        "interpretation_boundary": (
            "This probe identifies one-step generation effects for fixed V9.6 "
            "anchors whose shown history had >=2 formation steps and >=1 direct "
            "attempt. It does not estimate full-search or held-out performance. "
            "Delta-q summaries are task-specific."
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


def print_compact(analysis: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else f"{value:.{digits}f}"

    for name, contrast in analysis["contrasts"].items():
        control_key = f"{contrast['control']}_mean"
        treatment_key = f"{contrast['treatment']}_mean"
        print(f"\n=== {name} ===")
        for key, summary in contrast["paired_summaries"].items():
            if ":stratum:" in key:
                continue
            task = key.split(":", 1)[1]
            m = summary["metrics"]
            dq = m["conditional_delta_q"]
            vr = m["valid_rate"]
            ir = m["conditional_improvement_rate"]
            cr = m["conditional_change_ratio"]
            ci = dq["paired_mean_difference_bootstrap_95ci"]
            ci_text = (
                "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            )
            print(
                f"{task:22s} dq {fmt(dq[control_key])} -> "
                f"{fmt(dq[treatment_key])} "
                f"diff {fmt(dq['paired_mean_difference_b_minus_a'])} {ci_text} "
                f"| valid {fmt(vr[control_key], 3)} -> "
                f"{fmt(vr[treatment_key], 3)} "
                f"| improve {fmt(ir[control_key], 3)} -> "
                f"{fmt(ir[treatment_key], 3)} "
                f"| change {fmt(cr[control_key], 3)} -> "
                f"{fmt(cr[treatment_key], 3)}"
            )
        dq_dir = contrast["overall_paired_direction"]["conditional_delta_q"]
        print(
            "overall dq direction: treatment better "
            f"{dq_dir['anchors_better_treatment']}, control better "
            f"{dq_dir['anchors_better_control']}, tied {dq_dir['anchors_tied']}"
        )


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
    print(
        json.dumps(
            {
                key: analysis[key]
                for key in (
                    "complete",
                    "expected_trials",
                    "completed_unique_trials",
                    "missing_trial_count",
                    "duplicate_result_rows",
                )
            },
            indent=2,
        )
    )
    print_compact(analysis)


if __name__ == "__main__":
    main()
