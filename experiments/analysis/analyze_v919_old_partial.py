"""Audit the stopped, original TraceAAD V9.19 search campaign.

The campaign used the pre-BGFT V9.19 controller:
Q + L + U + R_traj allocation, region failure-rate operator scheduling,
four operators, and the behavesim_v3_combined_panel protocol.

This script reads existing run artifacts only. It keeps interrupted search
quality, completed held-out evaluation, and process activation separate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
OLD_CAMPAIGN_ROOT = (
    EXPERIMENTS_ROOT / "其他实验" / "V9.19-旧机制中断-20260828"
)
TASKS = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
)
OPERATORS = (
    "local_refine",
    "mechanism_refine",
    "structural_explore",
    "behavior_crossover",
)
WEIGHTS = {"q": 0.50, "L": 0.25, "U": 0.10, "r_traj": 0.15}
LOWER_IS_BETTER = {
    "tsp_construct",
    "cvrp_aco",
    "online_bin_packing",
    "vrptw_construct",
}
V916_HELDOUT = {
    "tsp_construct": (
        "tsp_construct/traceaad_v9_16/"
        "eval_best_20260823_v916_complete/results.json"
    ),
    "cvrp_aco": (
        "cvrp_aco/traceaad_v9_16/"
        "eval_best_20260823_v916_complete/results.json"
    ),
    "op_aco": (
        "op_aco/traceaad_v9_16/"
        "eval_best_20260823_v916_complete/results.json"
    ),
    "online_bin_packing": (
        "online_bin_packing/traceaad_v9_16/"
        "eval_best_20260823_v916_complete/results.json"
    ),
    "vrptw_construct": (
        "vrptw_construct/traceaad_v9_16/"
        "eval_best_20260824_v916_multiscale/results.json"
    ),
}
V917_HELDOUT = {
    task: f"{task}/traceaad_v9_17/"
    "eval_best_20260824_v917_adaptive_complete/results.json"
    for task in TASKS
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.fmean(finite) if finite else None


def _sample_std(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return statistics.stdev(finite) if len(finite) >= 2 else None


def _mean_std(values: list[float]) -> dict[str, float | None]:
    return {"mean": _mean(values), "sample_std": _sample_std(values)}


def _corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _run_dir(task: str, repeat: int) -> Path:
    return (
        OLD_CAMPAIGN_ROOT
        / task
        / "traceaad_v9_19"
        / f"v9_19_{task}_rep{repeat}"
    )


def _v916_run_dir(task: str, repeat: int) -> Path:
    return (
        EXPERIMENTS_ROOT
        / task
        / "traceaad_v9_16"
        / f"v9_16_20260822_151700_{task}_rep{repeat}"
    )


def _v917_run_dir(task: str, repeat: int) -> Path:
    return (
        EXPERIMENTS_ROOT
        / task
        / "traceaad_v9_17"
        / f"v9_17_20260823_adaptive_{task}_rep{repeat}"
    )


def _root_of(node_id: int, algorithms: dict[int, dict[str, Any]]) -> int:
    current = node_id
    seen: set[int] = set()
    while algorithms[current].get("parent_id") not in (None, 0):
        if current in seen:
            raise ValueError(f"parent cycle at node {current}")
        seen.add(current)
        current = int(algorithms[current]["parent_id"])
    return current


def _depth_of(node_id: int, algorithms: dict[int, dict[str, Any]]) -> int:
    depth = 0
    current = node_id
    seen: set[int] = set()
    while current and algorithms[current].get("parent_id") not in (None, 0):
        if current in seen:
            raise ValueError(f"parent cycle at node {current}")
        seen.add(current)
        depth += 1
        current = int(algorithms[current]["parent_id"])
    return depth


def _best_at(run_dir: Path, budget: int) -> float:
    rows = _read_rows(run_dir / "evaluations.csv")
    scores = [
        float(row["fitness"])
        for row in rows
        if row.get("fitness")
        and int(row["eval_count"]) <= budget
        and math.isfinite(float(row["fitness"]))
    ]
    if not scores:
        raise ValueError(f"no score by budget {budget}: {run_dir}")
    return max(scores)


def _operator_geometry(
    checkpoint: dict[str, Any],
    run_dir: Path,
    rows: list[dict[str, str]],
) -> dict[str, dict[str, list[float]]]:
    algorithms = {
        int(item["id"]): item
        for item in checkpoint.get("tree", {}).get("algorithms", [])
    }
    with np.load(run_dir / "checkpoints" / "behave.npz") as payload:
        ids = [int(value) for value in payload["ids"].tolist()]
        matrix = np.asarray(payload["matrix"], dtype=float)
    index = {node_id: position for position, node_id in enumerate(ids)}
    result: dict[str, dict[str, list[float]]] = {
        operator: {"parent_distance": [], "nearest_prior_distance": []}
        for operator in OPERATORS
    }
    result["behavior_crossover"].update(
        {
            "parent_reference_distance": [],
            "child_reference_distance": [],
            "child_closer_to_reference": [],
            "reference_better_than_parent": [],
        }
    )
    ordered = sorted(
        (
            item
            for item in algorithms.values()
            if int(item["id"]) in index and int(item["id"]) != 0
        ),
        key=lambda item: (int(item["created_slot"]), int(item["id"])),
    )
    prior: list[int] = []
    for item in ordered:
        node_id = int(item["id"])
        parent_id = item.get("parent_id")
        operator = item.get("operator")
        if operator in result and parent_id and int(parent_id) in index:
            result[operator]["parent_distance"].append(
                float(matrix[index[node_id], index[int(parent_id)]])
            )
        if operator in result and prior:
            prior_indices = [index[prior_id] for prior_id in prior]
            result[operator]["nearest_prior_distance"].append(
                float(np.min(matrix[index[node_id], prior_indices]))
            )
        prior.append(node_id)
    for row in rows:
        if row.get("operator") != "behavior_crossover":
            continue
        if not row.get("child_id") or not row.get("reference_id"):
            continue
        child_id = int(row["child_id"])
        parent_id = int(row["parent_id"])
        reference_id = int(row["reference_id"])
        if not all(
            node_id in index and node_id in algorithms
            for node_id in (child_id, parent_id, reference_id)
        ):
            continue
        child_parent = float(matrix[index[child_id], index[parent_id]])
        child_reference = float(matrix[index[child_id], index[reference_id]])
        result["behavior_crossover"]["parent_reference_distance"].append(
            float(matrix[index[parent_id], index[reference_id]])
        )
        result["behavior_crossover"]["child_reference_distance"].append(
            child_reference
        )
        result["behavior_crossover"]["child_closer_to_reference"].append(
            float(child_reference < child_parent)
        )
        result["behavior_crossover"]["reference_better_than_parent"].append(
            float(
                float(algorithms[reference_id]["fitness"])
                > float(algorithms[parent_id]["fitness"])
            )
        )
    return result


def analyze_run(task: str, repeat: int) -> dict[str, Any]:
    run_dir = _run_dir(task, repeat)
    config = _read_json(run_dir / "run_config.json")
    behave_protocol = config["method_params"]["behave_protocol"]
    if behave_protocol != "behavesim_v3_combined_panel":
        raise ValueError(
            f"expected stopped V9.19 v3 protocol, found {behave_protocol}: "
            f"{run_dir}"
        )
    summary = _read_json(run_dir / "logs" / "summary.json")
    checkpoint = _read_json(run_dir / "checkpoints" / "latest.json")
    rows = _read_rows(run_dir / "evaluations.csv")
    events = _read_events(run_dir / "mechanism_events.jsonl")
    algorithms = {
        int(item["id"]): item
        for item in checkpoint.get("tree", {}).get("algorithms", [])
    }

    final_rows: dict[int, dict[str, str]] = {}
    for row in rows:
        final_rows[int(row["slot"])] = row
    ordinary = [
        row
        for _, row in sorted(final_rows.items())
        if row.get("operator") in OPERATORS
    ]

    operator_outcomes: dict[str, Counter[str]] = {
        operator: Counter() for operator in OPERATORS
    }
    parent_counts: Counter[int] = Counter()
    root_counts: Counter[int] = Counter()
    selected_depths: list[int] = []
    for row in ordinary:
        operator = row["operator"]
        operator_outcomes[operator][row.get("outcome") or "missing"] += 1
        parent_id = int(row["parent_id"])
        parent_counts[parent_id] += 1
        root_counts[_root_of(parent_id, algorithms)] += 1
        selected_depths.append(_depth_of(parent_id, algorithms))

    operator_events = {
        int(event["decision_index"]): event
        for event in events
        if event.get("event") == "operator_decision"
    }
    allocation: dict[str, list[float]] = defaultdict(list)
    for event in events:
        if event.get("event") != "pre_decision":
            continue
        snapshot = event["snapshot"]
        parent_id = int(event["parent_id"])
        selected = next(
            item for item in snapshot if int(item["id"]) == parent_id
        )
        arrays = {
            "q": np.asarray([float(item["q"]) for item in snapshot]),
            "L": np.asarray([float(item["L"]) for item in snapshot]),
            "U": np.asarray([float(item["U"]) for item in snapshot]),
            "r_traj": np.asarray(
                [float(item["r_traj"]) for item in snapshot]
            ),
            "S": np.asarray([float(item["S"]) for item in snapshot]),
        }
        for key in ("q", "L", "U", "r_traj", "S"):
            allocation[f"selected_{key}"].append(float(selected[key]))
        allocation["pool_size"].append(float(len(snapshot)))
        allocation["relative_ess"].append(float(event["ess"]) / len(snapshot))
        allocation["beta"].append(float(event["beta"]))
        q_l = _corr(arrays["q"], arrays["L"])
        q_u = _corr(arrays["q"], arrays["U"])
        q_r = _corr(arrays["q"], arrays["r_traj"])
        if q_l is not None:
            allocation["corr_q_L"].append(q_l)
        if q_u is not None:
            allocation["corr_q_U"].append(q_u)
        if q_r is not None:
            allocation["corr_q_r_traj"].append(q_r)
        allocation["top_S_differs_top_q"].append(
            float(np.argmax(arrays["S"]) != np.argmax(arrays["q"]))
        )
        for key in ("q", "L", "U", "r_traj"):
            weighted = WEIGHTS[key] * arrays[key]
            allocation[f"weighted_std_{key}"].append(float(np.std(weighted)))
        operator_event = operator_events[int(event["decision_index"])]
        allocation["G"].append(float(operator_event["gain"]))
        allocation["p_explore"].append(float(operator_event["p_explore"]))
        allocation["explore_draw"].append(
            float(operator_event["family"] == "explore")
        )
        allocation["window_attempts"].append(
            float(operator_event["window_attempts"])
        )

    geometry = _operator_geometry(checkpoint, run_dir, ordinary)
    operator_records: dict[str, dict[str, Any]] = {}
    for operator in OPERATORS:
        counts = operator_outcomes[operator]
        total = sum(counts.values())
        nodes = sum(counts[name] for name in ("improve", "plateau", "regress"))
        operator_records[operator] = {
            "primary_decisions": total,
            "outcomes": dict(counts),
            "node_creation_rate": nodes / total if total else None,
            "strict_improve_rate": counts["improve"] / total if total else None,
            "strict_improve_given_node": (
                counts["improve"] / nodes if nodes else None
            ),
            "mean_parent_child_distance": _mean(
                geometry[operator]["parent_distance"]
            ),
            "mean_nearest_prior_distance": _mean(
                geometry[operator]["nearest_prior_distance"]
            ),
        }
        if operator == "behavior_crossover":
            operator_records[operator]["crossover_geometry"] = {
                key: _mean(geometry[operator][key])
                for key in (
                    "parent_reference_distance",
                    "child_reference_distance",
                    "child_closer_to_reference",
                    "reference_better_than_parent",
                )
            }

    best_id = int(summary["best_algorithm_id"])
    best_node = algorithms[best_id]
    best_path_operators: list[str] = []
    path_node_id = best_id
    while path_node_id and path_node_id in algorithms:
        path_node = algorithms[path_node_id]
        if path_node.get("operator") in OPERATORS:
            best_path_operators.append(str(path_node["operator"]))
        parent_id = path_node.get("parent_id")
        if not parent_id:
            break
        path_node_id = int(parent_id)
    best_path_operators.reverse()
    decisions = len(ordinary)
    parent_values = list(parent_counts.values())
    root_values = list(root_counts.values())
    return {
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "task": task,
        "repeat": repeat,
        "summary_status": summary.get("status"),
        "budget_slots": int(summary["budget_slots"]),
        "budget_fraction": float(summary["budget_slots"]) / 1000.0,
        "best_score": float(summary["best_score"]),
        "best_algorithm_id": best_id,
        "best_created_slot": int(best_node["created_slot"]),
        "best_operator": best_node.get("operator"),
        "best_path_length": len(best_path_operators),
        "best_path_operator_counts": dict(Counter(best_path_operators)),
        "n_algorithms": int(summary["n_algorithms"]),
        "ordinary_decisions": int(summary["ordinary_decisions"]),
        "behave_protocol": behave_protocol,
        "profiling": {
            "profiled_nodes": int(summary["profiled_nodes"]),
            "probe_executions": int(summary["probe_executions"]),
            "wall_time_seconds": float(summary["profiling_wall_time"]),
            "seconds_per_profiled_node": (
                float(summary["profiling_wall_time"])
                / int(summary["profiled_nodes"])
            ),
        },
        "primary_outcomes": dict(
            Counter(row.get("outcome") or "missing" for row in ordinary)
        ),
        "operator": operator_records,
        "allocation": {
            key: _mean(values)
            for key, values in sorted(allocation.items())
        }
        | {
            "corr_G_selected_r_traj": _corr(
                np.asarray(allocation["G"]),
                np.asarray(allocation["selected_r_traj"]),
            ),
            "p_explore_min": min(allocation["p_explore"]),
            "p_explore_max": max(allocation["p_explore"]),
            "window_10_fraction": (
                sum(value == 10 for value in allocation["window_attempts"])
                / len(allocation["window_attempts"])
            ),
        },
        "selection": {
            "unique_parents": len(parent_counts),
            "unique_parent_fraction": (
                len(parent_counts) / decisions if decisions else None
            ),
            "max_parent_share": (
                max(parent_values) / decisions if parent_values else None
            ),
            "effective_parent_count": (
                decisions * decisions / sum(value * value for value in parent_values)
                if parent_values
                else None
            ),
            "max_root_share": (
                max(root_values) / decisions if root_values else None
            ),
            "effective_root_count": (
                decisions * decisions / sum(value * value for value in root_values)
                if root_values
                else None
            ),
            "mean_selected_depth": _mean([float(value) for value in selected_depths]),
            "max_tree_depth": max(
                _depth_of(node_id, algorithms)
                for node_id in algorithms
                if node_id != 0
            ),
        },
        "v916_matched_budget_best": _best_at(
            _v916_run_dir(task, repeat), int(summary["budget_slots"])
        ),
        "v916_final_best": float(
            _read_json(
                _v916_run_dir(task, repeat) / "logs" / "summary.json"
            )["best_score"]
        ),
        "v917_matched_budget_best": _best_at(
            _v917_run_dir(task, repeat), int(summary["budget_slots"])
        ),
        "v917_final_best": float(
            _read_json(
                _v917_run_dir(task, repeat) / "logs" / "summary.json"
            )["best_score"]
        ),
    }


def _sum_counters(records: list[dict[str, Any]], path: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        current: Any = record
        for key in path.split("."):
            current = current[key]
        counter.update(current)
    return dict(counter)


def _heldout_container(payload: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "eval_results_by_size",
        "results_by_size",
        "results_by_split",
        "eval_results_by_scale",
    ):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    raise ValueError("held-out results contain no recognized unit container")


def _objective_values(block: dict[str, Any]) -> list[float]:
    rows = sorted(
        block.get("results", []),
        key=lambda row: (row.get("run_name", ""), row.get("program_path", "")),
    )
    values: list[float] = []
    for row in rows:
        for key in ("eval_objective", "objective", "bins_used_mean"):
            if row.get(key) is not None:
                values.append(float(row[key]))
                break
    return values


def _signed_gain(task: str, current: float, previous: float) -> float:
    if task in LOWER_IS_BETTER:
        return previous - current
    return current - previous


def _direction_counts(gains: list[float]) -> dict[str, int]:
    tolerance = 1e-12
    return {
        "better": sum(gain > tolerance for gain in gains),
        "tie": sum(abs(gain) <= tolerance for gain in gains),
        "worse": sum(gain < -tolerance for gain in gains),
    }


def _heldout_comparison(task: str) -> dict[str, Any] | None:
    current_path = (
        OLD_CAMPAIGN_ROOT
        / task
        / "traceaad_v9_19"
        / "eval_best_20260828_v919_stopped_partial"
        / "results.json"
    )
    if not current_path.is_file():
        return None
    current = _read_json(current_path)
    v916 = _read_json(EXPERIMENTS_ROOT / V916_HELDOUT[task])
    v917 = _read_json(EXPERIMENTS_ROOT / V917_HELDOUT[task])
    current_container = _heldout_container(current)
    v916_container = _heldout_container(v916)
    v917_container = _heldout_container(v917)
    units: dict[str, Any] = {}
    for unit in sorted(
        set(current_container) & set(v916_container) & set(v917_container)
    ):
        current_summary = current_container[unit]["summary"]
        v916_summary = v916_container[unit]["summary"]
        v917_summary = v917_container[unit]["summary"]
        current_objective = float(current_summary["mean_eval_objective"])
        v916_objective = float(v916_summary["mean_eval_objective"])
        v917_objective = float(v917_summary["mean_eval_objective"])
        current_values = _objective_values(current_container[unit])
        v916_values = _objective_values(v916_container[unit])
        v917_values = _objective_values(v917_container[unit])
        if not (len(current_values) == len(v916_values) == len(v917_values)):
            raise ValueError(f"repeat count mismatch: {task}/{unit}")
        gains_v916 = [
            _signed_gain(task, current_value, previous_value)
            for current_value, previous_value in zip(
                current_values, v916_values, strict=True
            )
        ]
        gains_v917 = [
            _signed_gain(task, current_value, previous_value)
            for current_value, previous_value in zip(
                current_values, v917_values, strict=True
            )
        ]
        units[unit] = {
            "v919_partial_objective": current_objective,
            "v919_partial_sample_std": current_summary[
                "sample_std_eval_objective"
            ],
            "v919_partial_objectives_by_repeat": current_values,
            "v919_successful_runs": current_summary[
                "num_successful_eval_runs"
            ],
            "v916_final_objective": v916_objective,
            "v916_final_sample_std": v916_summary[
                "sample_std_eval_objective"
            ],
            "signed_gain_over_v916": _signed_gain(
                task, current_objective, v916_objective
            ),
            "signed_gains_over_v916_by_repeat": gains_v916,
            "directions_vs_v916": _direction_counts(gains_v916),
            "v917_final_objective": v917_objective,
            "v917_final_sample_std": v917_summary[
                "sample_std_eval_objective"
            ],
            "signed_gain_over_v917": _signed_gain(
                task, current_objective, v917_objective
            ),
            "signed_gains_over_v917_by_repeat": gains_v917,
            "directions_vs_v917": _direction_counts(gains_v917),
        }
    return {
        "v919_results": str(current_path.relative_to(REPO_ROOT)),
        "v916_results": V916_HELDOUT[task],
        "v917_results": V917_HELDOUT[task],
        "allow_incomplete_search_runs": current.get(
            "allow_incomplete_search_runs"
        ),
        "units": units,
    }


def _aggregate_task(task: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    current_scores = [record["best_score"] for record in records]
    matched_scores = [
        record["v916_matched_budget_best"] for record in records
    ]
    final_scores = [record["v916_final_best"] for record in records]
    v917_matched_scores = [
        record["v917_matched_budget_best"] for record in records
    ]
    v917_final_scores = [record["v917_final_best"] for record in records]
    gains_v916_matched = [
        current - previous
        for current, previous in zip(
            current_scores, matched_scores, strict=True
        )
    ]
    gains_v917_matched = [
        current - previous
        for current, previous in zip(
            current_scores, v917_matched_scores, strict=True
        )
    ]
    operators: dict[str, Any] = {}
    for operator in OPERATORS:
        counts = _sum_counters(records, f"operator.{operator}.outcomes")
        total = sum(counts.values())
        nodes = sum(counts.get(name, 0) for name in ("improve", "plateau", "regress"))
        parent_distances = [
            record["operator"][operator]["mean_parent_child_distance"]
            for record in records
            if record["operator"][operator]["mean_parent_child_distance"] is not None
        ]
        prior_distances = [
            record["operator"][operator]["mean_nearest_prior_distance"]
            for record in records
            if record["operator"][operator]["mean_nearest_prior_distance"] is not None
        ]
        operators[operator] = {
            "primary_decisions": total,
            "outcomes": counts,
            "node_creation_rate": nodes / total if total else None,
            "strict_improve_rate": counts.get("improve", 0) / total if total else None,
            "mean_parent_child_distance_across_runs": _mean(parent_distances),
            "mean_nearest_prior_distance_across_runs": _mean(prior_distances),
        }
        if operator == "behavior_crossover":
            operators[operator]["crossover_geometry_across_runs"] = {
                key: _mean(
                    [
                        record["operator"][operator]["crossover_geometry"][key]
                        for record in records
                        if record["operator"][operator]["crossover_geometry"][key]
                        is not None
                    ]
                )
                for key in (
                    "parent_reference_distance",
                    "child_reference_distance",
                    "child_closer_to_reference",
                    "reference_better_than_parent",
                )
            }
    allocation_keys = records[0]["allocation"].keys()
    selection_keys = records[0]["selection"].keys()
    return {
        "task": task,
        "run_count": len(records),
        "budget_slots": [record["budget_slots"] for record in records],
        "complete_search_runs": sum(
            record["summary_status"] == "finished"
            and record["budget_slots"] == 1000
            for record in records
        ),
        "search": {
            "v919_partial": _mean_std(current_scores),
            "v916_matched_budget": _mean_std(matched_scores),
            "v916_final": _mean_std(final_scores),
            "mean_score_delta_vs_v916_matched": (
                statistics.fmean(current_scores)
                - statistics.fmean(matched_scores)
            ),
            "score_deltas_vs_v916_matched_by_repeat": gains_v916_matched,
            "directions_vs_v916_matched": _direction_counts(
                gains_v916_matched
            ),
            "mean_score_delta_vs_v916_final": (
                statistics.fmean(current_scores)
                - statistics.fmean(final_scores)
            ),
            "v917_matched_budget": _mean_std(v917_matched_scores),
            "v917_final": _mean_std(v917_final_scores),
            "mean_score_delta_vs_v917_matched": (
                statistics.fmean(current_scores)
                - statistics.fmean(v917_matched_scores)
            ),
            "score_deltas_vs_v917_matched_by_repeat": gains_v917_matched,
            "directions_vs_v917_matched": _direction_counts(
                gains_v917_matched
            ),
            "mean_score_delta_vs_v917_final": (
                statistics.fmean(current_scores)
                - statistics.fmean(v917_final_scores)
            ),
        },
        "primary_outcomes": _sum_counters(records, "primary_outcomes"),
        "best_paths": {
            "mean_best_created_budget_fraction": _mean(
                [
                    record["best_created_slot"] / record["budget_slots"]
                    for record in records
                ]
            ),
            "mean_length": _mean(
                [float(record["best_path_length"]) for record in records]
            ),
            "paths_containing_operator": {
                operator: sum(
                    record["best_path_operator_counts"].get(operator, 0) > 0
                    for record in records
                )
                for operator in OPERATORS
            },
            "operator_counts": {
                operator: sum(
                    record["best_path_operator_counts"].get(operator, 0)
                    for record in records
                )
                for operator in OPERATORS
            },
        },
        "operator": operators,
        "allocation": {
            key: _mean(
                [
                    float(record["allocation"][key])
                    for record in records
                    if record["allocation"][key] is not None
                ]
            )
            for key in allocation_keys
        },
        "selection": {
            key: _mean(
                [
                    float(record["selection"][key])
                    for record in records
                    if record["selection"][key] is not None
                ]
            )
            for key in selection_keys
        },
        "profiling": {
            "wall_time_seconds": sum(
                record["profiling"]["wall_time_seconds"] for record in records
            ),
            "mean_seconds_per_profiled_node": _mean(
                [
                    record["profiling"]["seconds_per_profiled_node"]
                    for record in records
                ]
            ),
        },
        "heldout": _heldout_comparison(task),
    }


def build_report() -> dict[str, Any]:
    runs = [
        analyze_run(task, repeat)
        for task in TASKS
        for repeat in range(1, 4)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in runs:
        grouped[record["task"]].append(record)
    return {
        "schema": "traceaad_v919_old_stopped_partial_v1",
        "mechanism": "pre_bgft_v919",
        "protocol": "behavesim_v3_combined_panel",
        "campaign_root": str(OLD_CAMPAIGN_ROOT.relative_to(REPO_ROOT)),
        "evidence_status": (
            "descriptive partial-run process evidence; two of fifteen searches "
            "completed 1000 slots"
        ),
        "run_count": len(runs),
        "completed_1000_slot_runs": sum(
            record["summary_status"] == "finished"
            and record["budget_slots"] == 1000
            for record in runs
        ),
        "runs": runs,
        "tasks": {
            task: _aggregate_task(task, grouped[task]) for task in TASKS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report()
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    output = args.output
    if not output.is_absolute():
        output = REPO_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
