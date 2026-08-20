#!/usr/bin/env python3
"""Paired analysis of the V9.7 region-frontier fixed-anchor probe.

Contrasts: fp - pp (frontier table effect, per intent) and fc - fp (cross-region
reference program effect on top of the table).  Mechanism outcomes classify
every valid child against the anchor's frozen global state: destination family
(own / other visited / new) and sub-frontier re-entry (below that family's
frontier by more than one source-scale s0).

Usage:

    uv run python experiments/analysis/analyze_v97_frontier_probe.py <run_dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))
from analyze_traceaad_generation_probe import bootstrap_ci, read_json, read_jsonl  # noqa: E402
from analyze_v97_search_geometry import macro_family, mechanism_tags  # noqa: E402

CONDITIONS = ("pp_refine", "pp_explore", "fp_refine", "fp_explore", "fc_explore")
CONTRASTS = (
    ("pp_explore", "fp_explore"),
    ("pp_refine", "fp_refine"),
    ("fp_explore", "fc_explore"),
)
METRICS = (
    "conditional_delta_q",
    "conditional_improvement_rate",
    "valid_rate",
    "conditional_change_ratio",
    "sub_frontier_rate",
    "new_region_rate",
    "family_switch_rate",
)
BOOTSTRAP_SEED = 971139


def enrich(row: dict[str, Any], anchors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    anchor = anchors[row["anchor_id"]]
    enriched = {**row, "anchor_family": anchor["anchor_family"]}
    if row.get("valid") is not True or not row.get("candidate_code"):
        return {**enriched, "child_family": None, "destination": None}
    tags = mechanism_tags(row["task"], row["candidate_code"])
    child_family = macro_family(row["task"], tags)
    visited = anchor["visited_families"]
    s0 = float(anchor["source_s0"])
    if child_family == anchor["anchor_family"]:
        destination = "own"
    elif child_family in visited:
        destination = "visited"
    else:
        destination = "new"
    child_q = float(row["child_q"])
    sub_frontier = destination in {"own", "visited"} and (
        child_q < float(visited[child_family]) - s0
    )
    return {
        **enriched,
        "child_family": child_family,
        "destination": destination,
        "sub_frontier": bool(sub_frontier),
        "family_switch": child_family != anchor["anchor_family"],
        "new_region": destination == "new",
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("valid") is True]

    def mean(key: str, population: list[dict[str, Any]]) -> float | None:
        values = [float(row[key]) for row in population if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    return {
        "task": rows[0]["task"],
        "stratum": rows[0]["stratum"],
        "replicate_count": len(rows),
        "valid_rate": mean("valid_numeric", rows),
        "conditional_delta_q": mean("delta_q", valid),
        "conditional_improvement_rate": mean("improved_numeric", valid),
        "conditional_change_ratio": mean("change_ratio", valid),
        "sub_frontier_rate": mean("sub_frontier_numeric", valid),
        "new_region_rate": mean("new_region_numeric", valid),
        "family_switch_rate": mean("family_switch_numeric", valid),
        "mean_prompt_tokens": mean("prompt_tokens", rows),
        "destinations": dict(Counter(row["destination"] for row in valid)),
    }


def numerics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "valid_numeric": float(row.get("valid") is True),
        "improved_numeric": float(bool(row.get("improved"))),
        "sub_frontier_numeric": float(bool(row.get("sub_frontier"))),
        "new_region_numeric": float(bool(row.get("new_region"))),
        "family_switch_numeric": float(bool(row.get("family_switch"))),
    }


def paired(
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
        differences: list[float] = []
        control_values: list[float] = []
        treatment_values: list[float] = []
        for row in selected:
            a_row = row["conditions"].get(control)
            b_row = row["conditions"].get(treatment)
            if a_row is None or b_row is None:
                continue
            a, b = a_row.get(metric), b_row.get(metric)
            if a is None or b is None:
                continue
            control_values.append(float(a))
            treatment_values.append(float(b))
            differences.append(float(b) - float(a))
        result["metrics"][metric] = {
            "paired_anchor_count": len(differences),
            f"{control}_mean": statistics.fmean(control_values) if control_values else None,
            f"{treatment}_mean": statistics.fmean(treatment_values)
            if treatment_values
            else None,
            "paired_mean_difference_b_minus_a": (
                statistics.fmean(differences) if differences else None
            ),
            "paired_mean_difference_bootstrap_95ci": bootstrap_ci(
                differences, seed=BOOTSTRAP_SEED + seed_offset + metric_index
            ),
            "anchors_better_b": sum(value > 0 for value in differences),
            "anchors_tied": sum(value == 0 for value in differences),
            "anchors_better_a": sum(value < 0 for value in differences),
        }
    return result


def analyze(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "probe_config.json")
    anchors_raw = {row["anchor_id"]: row for row in read_jsonl(run_dir / "anchors.jsonl")}
    schedule = read_jsonl(run_dir / "schedule.jsonl")
    expected = {row["trial_id"] for row in schedule}
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    enriched = [numerics(enrich(row, anchors_raw)) for row in rows]
    by_trial = {row["trial_id"]: row for row in enriched}

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_trial.values():
        grouped[(row["anchor_id"], row["condition"])].append(row)
    anchors: list[dict[str, Any]] = []
    for anchor_id in sorted({key[0] for key in grouped}):
        conditions = {
            condition: aggregate(grouped[(anchor_id, condition)])
            for condition in CONDITIONS
            if (anchor_id, condition) in grouped
        }
        exemplar = next(iter(conditions.values()))
        anchors.append(
            {
                "anchor_id": anchor_id,
                "task": exemplar["task"],
                "stratum": exemplar["stratum"],
                "conditions": conditions,
            }
        )

    contrasts: dict[str, Any] = {}
    for contrast_index, (control, treatment) in enumerate(CONTRASTS):
        summaries: dict[str, Any] = {}
        for task_index, task in enumerate(sorted({row["task"] for row in enriched})):
            summaries[f"task:{task}"] = paired(
                anchors,
                selector=lambda row, task=task: row["task"] == task,
                seed_offset=10_000 * contrast_index + 100 * task_index,
                control=control,
                treatment=treatment,
            )
        contrasts[f"{treatment}_minus_{control}"] = {
            "control": control,
            "treatment": treatment,
            "paired_summaries": summaries,
        }

    destinations: dict[str, Any] = {}
    for task in sorted({row["task"] for row in enriched}):
        for condition in CONDITIONS:
            cell = [
                row
                for row in enriched
                if row["task"] == task and row["condition"] == condition and row["valid"] is True
            ]
            if cell:
                destinations[f"{task}:{condition}"] = dict(
                    Counter(row["destination"] for row in cell)
                )

    return {
        "protocol_id": config["protocol_id"],
        "run_dir": str(run_dir),
        "expected_trials": len(expected),
        "completed_unique_trials": len(by_trial),
        "missing_trial_count": len(expected - set(by_trial)),
        "anchor_level_rows": anchors,
        "contrasts": contrasts,
        "destination_counts": destinations,
        "interpretation_boundary": (
            "Single-step identification on fixed anchors; no claim about "
            "complete-search or held-out outcomes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    analysis = analyze(args.run_dir)
    out_dir = args.run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(analysis, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'summary.json'}\n")
    print(
        f"{'contrast':<30}{'task':<8}{'dq diff':>12}{'95% CI':>24}"
        f"{'subfrt diff':>12}{'newreg diff':>12}"
    )
    for name, block in analysis["contrasts"].items():
        for key, summary in block["paired_summaries"].items():
            metrics = summary["metrics"]
            dq = metrics["conditional_delta_q"]
            sub = metrics["sub_frontier_rate"]
            new = metrics["new_region_rate"]

            def fmt(entry: dict[str, Any]) -> str:
                value = entry["paired_mean_difference_b_minus_a"]
                ci = entry["paired_mean_difference_bootstrap_95ci"]
                if value is None:
                    return "n/a"
                ci_text = "n/a" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
                return f"{value:+.4f} {ci_text}"

            print(
                f"{name:<30}{key.split(':')[1]:<8}"
                f"{fmt(dq):<37}"
                f"{(sub['paired_mean_difference_b_minus_a'] if sub['paired_mean_difference_b_minus_a'] is not None else float('nan')):>+12.1%}"
                f"{(new['paired_mean_difference_b_minus_a'] if new['paired_mean_difference_b_minus_a'] is not None else float('nan')):>+12.1%}"
            )


if __name__ == "__main__":
    main()
