"""Replay a running TraceAAD V9.12 batch from local checkpoints.

The report is deliberately descriptive.  Search scores are compared at the
smallest completed evaluator count shared by the three V9.12 repeats of each
task.  Mechanism statistics come from the recorded decisions and checkpoint
forest; they do not identify a causal effect on held-out quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


TASKS = ("tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _sample_sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _unit_interval(seed: int | None, iteration: int) -> float:
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(f"v9.12:operator:{token}:{iteration}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _run_dir(root: Path, task: str, version: str, batch: str, repeat: int) -> Path:
    return root / task / f"traceaad_{version}" / f"{batch}_{task}_rep{repeat}"


def _curve_value(run_dir: Path, horizon: int) -> float:
    curve_path = run_dir / "best_curve.csv"
    if not curve_path.exists():
        return _legacy_curve(run_dir, horizon)[0]
    selected: float | None = None
    with curve_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["eval_count"]) <= horizon:
                selected = float(row["best_fitness"])
            else:
                break
    if selected is None:
        raise ValueError(f"no best score at eval {horizon}: {run_dir}")
    return selected


def _last_best_eval(run_dir: Path, horizon: int) -> int:
    curve_path = run_dir / "best_curve.csv"
    if not curve_path.exists():
        return _legacy_curve(run_dir, horizon)[1]
    selected = 0
    with curve_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = int(row["eval_count"])
            if value <= horizon:
                selected = value
            else:
                break
    return selected


def _legacy_curve(run_dir: Path, horizon: int) -> tuple[float, int]:
    """Read the pre-V9.11 candidate stream on the evaluator-call axis."""
    best: float | None = None
    best_eval = 0
    eval_count = 0
    for row in _read_jsonl(run_dir / "artifacts" / "candidates.jsonl"):
        if row.get("evaluator_called"):
            eval_count += 1
        if eval_count > horizon:
            break
        fitness = row.get("child_fitness")
        if fitness is not None and math.isfinite(float(fitness)):
            value = float(fitness)
            if best is None or value > best:
                best = value
                best_eval = eval_count
    if best is None:
        raise ValueError(f"no best score at eval {horizon}: {run_dir}")
    return best, best_eval


def _anchor_depth(anchor_id: int, anchors: dict[int, dict[str, Any]]) -> int:
    depth = 0
    current = anchors[anchor_id]
    while current["parent_id"] is not None:
        depth += 1
        current = anchors[int(current["parent_id"])]
    return depth


def _run_snapshot(run_dir: Path) -> dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / "latest.json"
    checkpoint = _read_json(checkpoint_path)
    config = _read_json(run_dir / "run_config.json")
    summary_path = run_dir / "logs" / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else None
    events = _read_jsonl(run_dir / "logs" / "events.jsonl")

    forest = checkpoint["forest"]
    programs = {int(item["id"]): item for item in forest["programs"]}
    anchors = {int(item["id"]): item for item in forest["anchors"]}
    attempts = [item for item in forest["attempts"] if item["stage"] == "search"]
    attempts_by_iteration = {
        int(item["iteration"]): item
        for item in attempts
        if item["iteration"] is not None
    }
    selected = [item for item in events if item.get("event") == "regime_selected"]
    completed = [item for item in events if item.get("event") == "regime_completed"]
    history_events = [item for item in events if item.get("event") == "history_built"]
    probability_decisions = [
        item for item in selected if item.get("operator_probability_applied") is True
    ]
    seed = config["method_params"]["seed"]

    probability_counts: Counter[str] = Counter()
    elevated = 0
    changed_vs_min = 0
    avoided_vs_fixed_max = 0
    actual_explore = 0
    expected_explore = 0.0
    probability_formula_mismatches = 0
    for item in probability_decisions:
        probability = float(item["explore_probability"])
        failure_evidence = float(item["refine_failure_evidence"])
        probability_formula_mismatches += not math.isclose(
            probability, 0.10 + 0.20 * failure_evidence, abs_tol=1e-12
        )
        probability_counts[f"{probability:.3f}"] += 1
        elevated += probability > 0.1000000001
        actual_explore += item["sampled_intent"] == "explore"
        expected_explore += probability
        draw = _unit_interval(seed, int(item["iteration"]))
        changed_vs_min += 0.10 <= draw < probability
        avoided_vs_fixed_max += probability <= draw < 0.30

    route_counts = Counter(int(item["selected_root_state_id"]) for item in selected)
    top_route_share = _ratio(max(route_counts.values(), default=0), len(selected))

    followup_events = [item for item in completed if item.get("followup") is True]
    followup_attempts: list[dict[str, Any]] = []
    followup_improved_child = 0
    followup_recovered_source = 0
    followup_global_breakthrough = 0
    for event in followup_events:
        attempt = attempts_by_iteration.get(int(event["response_order"]) - 1)
        if attempt is None:
            continue
        followup_attempts.append(attempt)
        explore_anchor = anchors[int(attempt["anchor_id"])]
        source_id = explore_anchor["parent_id"]
        if attempt["child_id"] is None or source_id is None:
            continue
        child_program = programs[int(anchors[int(attempt["child_id"])]["program_id"])]
        child_q = float(child_program["q"])
        explore_q = float(programs[explore_anchor["program_id"]]["q"])
        source_q = float(programs[anchors[int(source_id)]["program_id"]]["q"])
        followup_improved_child += child_q > explore_q
        followup_recovered_source += child_q > source_q
        child_order = int(child_program["order"])
        prior_best = max(
            float(program["q"])
            for program in programs.values()
            if int(program["order"]) < child_order
        )
        followup_global_breakthrough += child_q > prior_best

    explore_attempts = [item for item in attempts if item["intent"] == "explore"]
    ordinary_refine_attempts = [
        item
        for item in attempts
        if item["intent"] == "refine" and item not in followup_attempts
    ]
    valid_explore = sum(item["child_id"] is not None for item in explore_attempts)
    explore_improve = sum(
        item["dq"] is not None and float(item["dq"]) > 0 for item in explore_attempts
    )
    refine_improve = sum(
        item["dq"] is not None and float(item["dq"]) > 0
        for item in ordinary_refine_attempts
    )
    pending_followup = checkpoint["exploration_followup_anchor_id"] is not None
    phase_operator: dict[str, dict[str, float | int | None]] = {}
    max_iteration = max((int(item["iteration"]) for item in selected), default=0)
    for phase_index, phase_name in enumerate(("early", "middle", "late")):
        phase_rows = [
            item
            for item in probability_decisions
            if min(2, (3 * int(item["iteration"])) // max(1, max_iteration + 1))
            == phase_index
        ]
        phase_operator[phase_name] = {
            "n": len(phase_rows),
            "mean_explore_probability": _mean(
                [float(item["explore_probability"]) for item in phase_rows]
            ),
            "actual_explore_fraction": _ratio(
                sum(item["sampled_intent"] == "explore" for item in phase_rows),
                len(phase_rows),
            ),
        }

    best_program = max(
        programs.values(),
        key=lambda item: (float(item["q"]), -int(item["length"]), -int(item["order"])),
    )
    best_anchor = next(
        anchor
        for anchor in anchors.values()
        if int(anchor["program_id"]) == int(best_program["id"])
    )
    n_eval = int(checkpoint["n_eval"])
    return {
        "run_dir": str(run_dir),
        "repeat": int(config["repeat"]),
        "checkpoint_mtime": datetime.fromtimestamp(checkpoint_path.stat().st_mtime)
        .astimezone()
        .isoformat(timespec="seconds"),
        "status": summary["status"] if summary is not None else "running",
        "n_eval": n_eval,
        "protocol_id": checkpoint["protocol_id"],
        "n_roots": len(forest["root_ids"]),
        "n_bootstrapped": len(checkpoint["bootstrapped"]),
        "bootstrap_scale": checkpoint["s"],
        "best_q": float(best_program["q"]),
        "best_depth": _anchor_depth(int(best_anchor["id"]), anchors),
        "max_depth": max(_anchor_depth(anchor_id, anchors) for anchor_id in anchors),
        "last_best_eval": _last_best_eval(run_dir, n_eval),
        "n_programs": len(programs),
        "n_anchors": len(anchors),
        "never_selected_anchor_fraction": _ratio(
            sum(int(anchor["n"]) == 0 for anchor in anchors.values()), len(anchors)
        ),
        "top_route_share": top_route_share,
        "n_probability_decisions": len(probability_decisions),
        "probability_formula_mismatches": probability_formula_mismatches,
        "budget_suppressed_explore": sum(
            item.get("explore_suppressed_for_budget") is True for item in selected
        ),
        "probability_counts": dict(sorted(probability_counts.items())),
        "phase_operator": phase_operator,
        "mean_explore_probability": _mean(
            [float(item["explore_probability"]) for item in probability_decisions]
        ),
        "elevated_probability_decisions": elevated,
        "elevated_probability_fraction": _ratio(elevated, len(probability_decisions)),
        "actual_normal_explore": actual_explore,
        "actual_normal_explore_fraction": _ratio(
            actual_explore, len(probability_decisions)
        ),
        "expected_normal_explore": expected_explore,
        "selection_changes_vs_fixed_0_10": changed_vs_min,
        "explores_avoided_vs_fixed_0_30": avoided_vs_fixed_max,
        "n_history_prompts": len(history_events),
        "history_prompt_nonempty_fraction": _ratio(
            sum(bool(item.get("shown_event_ids")) for item in history_events),
            len(history_events),
        ),
        "history_context_drop_count": sum(
            int(item.get("dropped_for_context", 0)) for item in history_events
        ),
        "n_explore_attempts": len(explore_attempts),
        "valid_explore_children": valid_explore,
        "explore_child_valid_fraction": _ratio(valid_explore, len(explore_attempts)),
        "explore_improve_parent": explore_improve,
        "explore_improve_parent_fraction": _ratio(
            explore_improve, len(explore_attempts)
        ),
        "n_ordinary_refine_attempts": len(ordinary_refine_attempts),
        "ordinary_refine_improve": refine_improve,
        "ordinary_refine_improve_fraction": _ratio(
            refine_improve, len(ordinary_refine_attempts)
        ),
        "n_followup": len(followup_attempts),
        "pending_followup": pending_followup,
        "unconsumed_valid_explore_children": (
            valid_explore - len(followup_attempts) - int(pending_followup)
        ),
        "followup_valid_fraction": _ratio(
            sum(item["child_id"] is not None for item in followup_attempts),
            len(followup_attempts),
        ),
        "followup_improved_explore_child": followup_improved_child,
        "followup_improved_explore_child_fraction": _ratio(
            followup_improved_child, len(followup_attempts)
        ),
        "followup_recovered_source_parent": followup_recovered_source,
        "followup_recovered_source_parent_fraction": _ratio(
            followup_recovered_source, len(followup_attempts)
        ),
        "followup_global_breakthrough": followup_global_breakthrough,
        "followup_global_breakthrough_fraction": _ratio(
            followup_global_breakthrough, len(followup_attempts)
        ),
    }


def _pool_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    sum_keys = (
        "n_probability_decisions",
        "probability_formula_mismatches",
        "budget_suppressed_explore",
        "elevated_probability_decisions",
        "actual_normal_explore",
        "selection_changes_vs_fixed_0_10",
        "explores_avoided_vs_fixed_0_30",
        "n_history_prompts",
        "history_context_drop_count",
        "n_explore_attempts",
        "valid_explore_children",
        "explore_improve_parent",
        "n_ordinary_refine_attempts",
        "ordinary_refine_improve",
        "n_followup",
        "followup_improved_explore_child",
        "followup_recovered_source_parent",
        "followup_global_breakthrough",
        "unconsumed_valid_explore_children",
    )
    totals = {key: sum(int(run[key]) for run in runs) for key in sum_keys}
    probability_counts: Counter[str] = Counter()
    for run in runs:
        probability_counts.update(run["probability_counts"])
    pooled_phases: dict[str, dict[str, float | int | None]] = {}
    for phase_name in ("early", "middle", "late"):
        n = sum(int(run["phase_operator"][phase_name]["n"]) for run in runs)
        probability_sum = sum(
            float(run["phase_operator"][phase_name]["mean_explore_probability"])
            * int(run["phase_operator"][phase_name]["n"])
            for run in runs
            if run["phase_operator"][phase_name]["mean_explore_probability"] is not None
        )
        explore_sum = sum(
            float(run["phase_operator"][phase_name]["actual_explore_fraction"])
            * int(run["phase_operator"][phase_name]["n"])
            for run in runs
            if run["phase_operator"][phase_name]["actual_explore_fraction"] is not None
        )
        pooled_phases[phase_name] = {
            "n": n,
            "mean_explore_probability": probability_sum / n if n else None,
            "actual_explore_fraction": explore_sum / n if n else None,
        }
    return {
        **totals,
        "probability_counts": dict(sorted(probability_counts.items())),
        "phase_operator": pooled_phases,
        "mean_explore_probability": _mean(
            [
                float(level)
                for level, count in probability_counts.items()
                for _ in range(count)
            ]
        ),
        "elevated_probability_fraction": _ratio(
            totals["elevated_probability_decisions"], totals["n_probability_decisions"]
        ),
        "actual_normal_explore_fraction": _ratio(
            totals["actual_normal_explore"], totals["n_probability_decisions"]
        ),
        "selection_change_fraction_vs_fixed_0_10": _ratio(
            totals["selection_changes_vs_fixed_0_10"], totals["n_probability_decisions"]
        ),
        "avoidance_fraction_vs_fixed_0_30": _ratio(
            totals["explores_avoided_vs_fixed_0_30"], totals["n_probability_decisions"]
        ),
        "explore_child_valid_fraction": _ratio(
            totals["valid_explore_children"], totals["n_explore_attempts"]
        ),
        "explore_improve_parent_fraction": _ratio(
            totals["explore_improve_parent"], totals["n_explore_attempts"]
        ),
        "ordinary_refine_improve_fraction": _ratio(
            totals["ordinary_refine_improve"], totals["n_ordinary_refine_attempts"]
        ),
        "followup_improved_explore_child_fraction": _ratio(
            totals["followup_improved_explore_child"], totals["n_followup"]
        ),
        "followup_recovered_source_parent_fraction": _ratio(
            totals["followup_recovered_source_parent"], totals["n_followup"]
        ),
        "followup_global_breakthrough_fraction": _ratio(
            totals["followup_global_breakthrough"], totals["n_followup"]
        ),
        "top_route_share_range": [
            min(float(run["top_route_share"]) for run in runs),
            max(float(run["top_route_share"]) for run in runs),
        ],
        "never_selected_anchor_fraction_range": [
            min(float(run["never_selected_anchor_fraction"]) for run in runs),
            max(float(run["never_selected_anchor_fraction"]) for run in runs),
        ],
        "best_depth_range": [
            min(int(run["best_depth"]) for run in runs),
            max(int(run["best_depth"]) for run in runs),
        ],
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = args.experiments_root.resolve()
    result: dict[str, Any] = {
        "snapshot_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch": args.batch,
        "baselines": {"v9_7": args.v97_batch, "v9_11": args.v911_batch},
        "tasks": {},
    }
    for task in TASKS:
        runs = [
            _run_snapshot(_run_dir(root, task, "v9_12", args.batch, repeat))
            for repeat in range(1, 4)
        ]
        horizon = min(int(run["n_eval"]) for run in runs)
        comparison: dict[str, Any] = {}
        for version, batch in (
            ("v9_12", args.batch),
            ("v9_11", args.v911_batch),
            ("v9_7", args.v97_batch),
        ):
            values = [
                _curve_value(_run_dir(root, task, version, batch, repeat), horizon)
                for repeat in range(1, 4)
            ]
            comparison[version] = {
                "q_by_repeat": values,
                "mean_q": _mean(values),
                "sample_sd_q": _sample_sd(values),
            }
        current = comparison["v9_12"]
        for baseline in ("v9_11", "v9_7"):
            base = comparison[baseline]
            deltas = [
                current_value - base_value
                for current_value, base_value in zip(
                    current["q_by_repeat"], base["q_by_repeat"], strict=True
                )
            ]
            comparison[f"v9_12_minus_{baseline}"] = {
                "mean_q_difference": float(current["mean_q"]) - float(base["mean_q"]),
                "repeat_differences": deltas,
                "repeat_direction_better_equal_worse": [
                    sum(delta > 0 for delta in deltas),
                    sum(math.isclose(delta, 0.0, abs_tol=1e-12) for delta in deltas),
                    sum(delta < 0 for delta in deltas),
                ],
            }
        result["tasks"][task] = {
            "matched_eval_horizon": horizon,
            "runs": runs,
            "pooled_mechanism": _pool_runs(runs),
            "matched_eval_search": comparison,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="v9_12_20260819_130358")
    parser.add_argument("--v911-batch", default="v9_11_20260819_022200")
    parser.add_argument("--v97-batch", default="v9_7_20260814_150927")
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(args)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
