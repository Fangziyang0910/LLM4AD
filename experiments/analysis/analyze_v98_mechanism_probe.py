#!/usr/bin/env python3
"""Analyze V9.8 streaming P1/P2 or P3 mechanism probes."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.analysis.analyze_v97_search_geometry import (
    jaccard_distance,
    macro_family,
    mechanism_tags,
)
from experiments.runners.traceaad.v98_continuation_probe import HORIZONS
from experiments.runners.traceaad.v98_mechanism_probe import CONDITIONS


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _results(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((run_dir / "results").glob("shard_*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _mean(values: Iterable[float]) -> float | None:
    kept = list(values)
    return statistics.mean(kept) if kept else None


def _sd(values: Iterable[float]) -> float | None:
    kept = list(values)
    return statistics.stdev(kept) if len(kept) >= 2 else None


def _metric(values: Iterable[float]) -> dict[str, Any]:
    kept = list(values)
    return {"n": len(kept), "mean": _mean(kept), "sample_sd": _sd(kept)}


def _contrast_metric(values: Iterable[float]) -> dict[str, Any]:
    kept = list(values)
    return {
        **_metric(kept),
        "positive": sum(value > 0 for value in kept),
        "negative": sum(value < 0 for value in kept),
        "tie": sum(value == 0 for value in kept),
    }


def _anchor_means(
    rows: Iterable[dict[str, Any]], key: str, *, require_value: bool = True
) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None and require_value:
            continue
        grouped[row["anchor_id"]].append(float(value or 0.0))
    return [statistics.mean(values) for values in grouped.values()]


def analyze_p12(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    rows = _results(run_dir)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        parent_tags = mechanism_tags(row["task"], row.get("candidate_code", ""))
        source_tags = mechanism_tags(row["task"], _anchor_code(run_dir, row["anchor_id"]))
        enriched.append(
            {
                **row,
                "parent_family": macro_family(row["task"], source_tags),
                "child_family": (
                    macro_family(row["task"], parent_tags)
                    if row.get("valid") and row.get("candidate_code")
                    else None
                ),
                "family_switch": (
                    macro_family(row["task"], source_tags)
                    != macro_family(row["task"], parent_tags)
                    if row.get("valid") and row.get("candidate_code")
                    else None
                ),
                "tag_distance": (
                    jaccard_distance(source_tags, parent_tags)
                    if row.get("valid") and row.get("candidate_code")
                    else None
                ),
            }
        )
    by_cell: dict[str, Any] = {}
    for task in sorted({row["task"] for row in rows}):
        for condition in CONDITIONS:
            cell = [
                row for row in enriched if row["task"] == task and row["condition"] == condition
            ]
            valid = [row for row in cell if row.get("valid") is True]
            anchor_ids = {row["anchor_id"] for row in cell}
            validity_rows = [
                {**row, "valid_numeric": float(row.get("valid") is True)} for row in cell
            ]
            by_cell[f"{task}:{condition}"] = {
                "repeated_observations": len(cell),
                "independent_anchors": len(anchor_ids),
                "valid_observations": len(valid),
                "valid_rate_by_anchor": _metric(
                    _anchor_means(validity_rows, "valid_numeric")
                ),
                "improve_rate_valid_by_anchor": _metric(
                    _anchor_means(
                        [
                            {**row, "improved_numeric": float(row.get("improved") is True)}
                            for row in valid
                        ],
                        "improved_numeric",
                    )
                ),
                "delta_q_by_anchor": _metric(_anchor_means(valid, "delta_q")),
                "change_ratio_by_anchor": _metric(
                    _anchor_means(valid, "change_ratio")
                ),
                "family_switch_rate_by_anchor": _metric(
                    _anchor_means(
                        [
                            {**row, "switch_numeric": float(row["family_switch"] is True)}
                            for row in valid
                        ],
                        "switch_numeric",
                    )
                ),
                "tag_distance_by_anchor": _metric(
                    _anchor_means(
                        [row for row in valid if row["tag_distance"] is not None],
                        "tag_distance",
                    )
                ),
            }
    by_block: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in enriched:
        if row.get("valid") is True:
            by_block[row["block_id"]][row["condition"]] = row
    contrast_rows: list[dict[str, Any]] = []
    for block_id, cells in by_block.items():
        if set(cells) != set(CONDITIONS):
            continue
        q = {condition: float(row["delta_q"]) for condition, row in cells.items()}
        exemplar = next(iter(cells.values()))
        contrast_rows.append(
            {
                "block_id": block_id,
                "anchor_id": exemplar["anchor_id"],
                "task": exemplar["task"],
                "intent_code_only": q["code_only_explore"] - q["code_only_refine"],
                "intent_parent_path": q["parent_path_explore"] - q["parent_path_refine"],
                "history_refine": q["parent_path_refine"] - q["code_only_refine"],
                "history_explore": q["parent_path_explore"] - q["code_only_explore"],
                "history_intent_interaction": (
                    q["parent_path_explore"] - q["code_only_explore"]
                )
                - (q["parent_path_refine"] - q["code_only_refine"]),
            }
        )
    contrast_names = (
        "intent_code_only",
        "intent_parent_path",
        "history_refine",
        "history_explore",
        "history_intent_interaction",
    )
    contrast_summary: dict[str, Any] = {}
    for task in ("all", *sorted({row["task"] for row in contrast_rows})):
        task_rows = contrast_rows if task == "all" else [
            row for row in contrast_rows if row["task"] == task
        ]
        contrast_summary[task] = {
            name: _contrast_metric(_anchor_means(task_rows, name))
            for name in contrast_names
        }
    summary = {
        "protocol_id": config["protocol_id"],
        "completed_trials": len(rows),
        "expected_trials": config["trial_count"],
        "complete": len(rows) == int(config["trial_count"]),
        "by_task_condition": by_cell,
        "complete_valid_blocks": sum(set(cells) == set(CONDITIONS) for cells in by_block.values()),
        "anchors_with_complete_valid_block": len(
            {row["anchor_id"] for row in contrast_rows}
        ),
        "paired_delta_q_contrasts_by_anchor": contrast_summary,
        "interpretation_boundary": (
            "Fixed-anchor repeated observations identify proposal-kernel and context "
            "effects only; blocks and trials are not independent full-search repeats."
        ),
    }
    _write_analysis(run_dir, summary, by_cell)
    return summary


def _anchor_code(run_dir: Path, anchor_id: str) -> str:
    if not hasattr(_anchor_code, "cache"):
        setattr(
            _anchor_code,
            "cache",
            {row["anchor_id"]: row["code"] for row in _read_jsonl(run_dir / "anchors.jsonl")},
        )
    return getattr(_anchor_code, "cache")[anchor_id]


def analyze_p3(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    units = {row["unit_id"]: row for row in _read_jsonl(run_dir / "units.jsonl")}
    rows = _results(run_dir)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["continuation_id"]].append(row)
    observations: list[dict[str, Any]] = []
    for continuation_id, group in grouped.items():
        group.sort(key=lambda row: int(row["step"]))
        unit = units[group[0]["unit_id"]]
        for horizon in HORIZONS:
            prefix = [row for row in group if int(row["step"]) <= horizon]
            frontier = max(
                [float(unit["entry_q"])]
                + [float(row["child_q"]) for row in prefix if row.get("child_q") is not None]
            )
            observations.append(
                {
                    "continuation_id": continuation_id,
                    "unit_id": unit["unit_id"],
                    "task": unit["task"],
                    "protocol": group[0]["protocol"],
                    "horizon": horizon,
                    "observed_steps": len(prefix),
                    "frontier_q": frontier,
                    "internal_gain": frontier - float(unit["entry_q"]),
                    "parent_recovery": frontier - float(unit["parent_q"]),
                    "recovered_parent": frontier >= float(unit["parent_q"]),
                    "valid_responses": sum(row.get("valid") is True for row in prefix),
                }
            )
    by_cell: dict[str, Any] = {}
    for task in sorted({row["task"] for row in observations}):
        for protocol in ("child_chain", "hypothesis_level"):
            for horizon in HORIZONS:
                cell = [
                    row
                    for row in observations
                    if row["task"] == task
                    and row["protocol"] == protocol
                    and row["horizon"] == horizon
                ]
                by_cell[f"{task}:{protocol}:H{horizon}"] = {
                    "n": len(cell),
                    "internal_gain": _metric(row["internal_gain"] for row in cell),
                    "parent_recovery": _metric(row["parent_recovery"] for row in cell),
                    "parent_recovery_rate": (
                        sum(row["recovered_parent"] for row in cell) / len(cell)
                        if cell
                        else None
                    ),
                }
    paired_protocol: dict[str, Any] = {}
    paired = {
        (row["unit_id"], row["horizon"]): row for row in observations
        if row["protocol"] == "child_chain"
    }
    hypothesis_rows = [row for row in observations if row["protocol"] == "hypothesis_level"]
    for task in sorted({row["task"] for row in observations}):
        for horizon in HORIZONS:
            differences = []
            for row in hypothesis_rows:
                if row["task"] != task or row["horizon"] != horizon:
                    continue
                chain = paired.get((row["unit_id"], horizon))
                if chain is None:
                    continue
                differences.append(
                    {
                        "internal_gain": row["internal_gain"] - chain["internal_gain"],
                        "parent_recovery": row["parent_recovery"] - chain["parent_recovery"],
                    }
                )
            paired_protocol[f"{task}:H{horizon}"] = {
                "n": len(differences),
                "hypothesis_minus_chain_internal_gain": _metric(
                    row["internal_gain"] for row in differences
                ),
                "hypothesis_minus_chain_parent_recovery": _metric(
                    row["parent_recovery"] for row in differences
                ),
            }
    summary = {
        "protocol_id": config["protocol_id"],
        "completed_responses": len(rows),
        "expected_responses": config["response_count"],
        "complete": len(rows) == int(config["response_count"]),
        "by_task_protocol_horizon": by_cell,
        "paired_protocol_contrasts": paired_protocol,
        "interpretation_boundary": (
            "Horizons are nested repeated measurements of the same Explore child; "
            "they are not independent replicates and do not estimate online C effects."
        ),
    }
    _write_analysis(run_dir, summary, by_cell)
    return summary


def _write_analysis(run_dir: Path, summary: dict[str, Any], cells: dict[str, Any]) -> None:
    out = run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (out / "cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cell", "metrics_json"])
        for name, metrics in sorted(cells.items()):
            writer.writerow([name, json.dumps(metrics, sort_keys=True)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("p12", "p3"), required=True)
    args = parser.parse_args()
    summary = analyze_p12(args.run_dir) if args.stage == "p12" else analyze_p3(args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
