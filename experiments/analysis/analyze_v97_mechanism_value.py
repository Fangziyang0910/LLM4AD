#!/usr/bin/env python3
"""Process analysis of TraceAAD V9.7: search dynamics, lineage, and generation.

This script measures existing 1000-eval runs. It does not estimate a
counterfactual search that would have followed from a different earlier
choice, and it does not attribute held-out gaps to a single mechanism.

Usage:

    uv run python experiments/analysis/analyze_v97_mechanism_value.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from analyze_v97_allocation import analyze_batch  # noqa: E402

OUT = REPO / "docs" / "analysis" / "traceaad_v97_mechanism_value"
BUDGET = 1000
MILESTONES = (100, 250, 500, 750, 1000)
TASKS = ("tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing")
TASK_LABEL = {
    "tsp_construct": "TSP",
    "cvrp_aco": "CVRP",
    "op_aco": "OP",
    "online_bin_packing": "OBP",
}
OFFICIAL_BATCH = "20260814_150927"
BATCH_CONTEXT = "20260813_184519"
V96_BATCH = "20260812_191011"

METHODS = {
    "V8": {
        "tsp_construct": ["traceaad_v8/20260804_173300_tspc_v8_rep*"],
        "cvrp_aco": ["traceaad_v8/20260804_173300_cvrp_v8_rep*"],
        "op_aco": ["traceaad_v8/20260804_173300_opaco_v8_rep*"],
        "online_bin_packing": ["traceaad_v8/20260804_173300_obp_v8_rep*"],
    },
    "V9": {
        "tsp_construct": ["traceaad_v9/version9/*_v9_tsp_rep*"],
        "cvrp_aco": ["traceaad_v9/version9/*_v9_cvrp_rep*"],
        "op_aco": ["traceaad_v9/version9/*_v9_op_rep*"],
        "online_bin_packing": ["traceaad_v9/version9/*_v9_obp_rep*"],
    },
    "V9.6": {
        "tsp_construct": [f"traceaad_v9_6/v9_6_{V96_BATCH}_tsp_rep*"],
        "cvrp_aco": [f"traceaad_v9_6/v9_6_{V96_BATCH}_cvrp_rep*"],
        "op_aco": [f"traceaad_v9_6/v9_6_{V96_BATCH}_op_rep*"],
        "online_bin_packing": [f"traceaad_v9_6/v9_6_{V96_BATCH}_obp_rep*"],
    },
    "V9.7-batch": {
        "tsp_construct": [f"traceaad_v9_7/v9_7_{BATCH_CONTEXT}_tsp_rep*"],
        "cvrp_aco": [f"traceaad_v9_7/v9_7_{BATCH_CONTEXT}_cvrp_rep*"],
        "op_aco": [f"traceaad_v9_7/v9_7_{BATCH_CONTEXT}_op_rep*"],
        "online_bin_packing": [f"traceaad_v9_7/v9_7_{BATCH_CONTEXT}_obp_rep*"],
    },
    "V9.7": {
        "tsp_construct": [f"traceaad_v9_7/v9_7_{OFFICIAL_BATCH}_tsp_construct_rep*"],
        "cvrp_aco": [f"traceaad_v9_7/v9_7_{OFFICIAL_BATCH}_cvrp_aco_rep*"],
        "op_aco": [f"traceaad_v9_7/v9_7_{OFFICIAL_BATCH}_op_aco_rep*"],
        "online_bin_packing": [
            f"traceaad_v9_7/v9_7_{OFFICIAL_BATCH}_online_bin_packing_rep*"
        ],
    },
    "MCTS-AHD": {
        "tsp_construct": ["mcts_ahd/20260709_2135*"],
        "cvrp_aco": [
            "mcts_ahd/20260812_113425_cvrp_local_rep1",
            "mcts_ahd/20260812_113425_cvrp_local_rep2",
            "mcts_ahd/20260812_113425_cvrp_local_rep3",
        ],
        "op_aco": ["mcts_ahd/*mctsahd_rep*"],
        "online_bin_packing": ["mcts_ahd/*mctsahd_rep*"],
    },
}

COLORS = {
    "V8": "#7B2D8E",
    "V9": "#D62728",
    "V9.6": "#2CA02C",
    "V9.7-batch": "#17BECF",
    "V9.7": "#FF7F0E",
    "MCTS-AHD": "#247BA0",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return rows


def run_dirs(task: str, method: str) -> list[Path]:
    task_dir = REPO / "experiments" / task
    found: list[Path] = []
    for pattern in METHODS[method][task]:
        if "*" in pattern:
            found.extend(sorted(task_dir.glob(pattern)))
        else:
            path = task_dir / pattern
            if path.exists():
                found.append(path)
    return [path for path in found if path.is_dir() and (path / "logs").exists()]


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def candidate_score(row: dict[str, Any]) -> float | None:
    return finite(row.get("child_fitness", row.get("score")))


def candidate_order(row: dict[str, Any]) -> int | None:
    value = row.get("sample_order", row.get("order"))
    return int(value) if isinstance(value, int) else None


def eval_points(run: Path) -> list[tuple[int, float | None]]:
    """Return (eval_index, score_or_None) in evaluator-call order, 1-indexed."""
    path = run / "artifacts" / "candidates.jsonl"
    if not path.is_file():
        return _generic_eval_points(run)
    rows = load_jsonl(path)
    has_flag = any("evaluator_called" in row for row in rows[:30])
    points: list[tuple[int, float | None]] = []
    if has_flag:
        for row in rows:
            if not row.get("evaluator_called"):
                continue
            score = candidate_score(row) if row.get("status") == "ok" else None
            points.append((len(points) + 1, score))
    else:
        for row in rows:
            score = candidate_score(row)
            order = candidate_order(row)
            if (
                order is None
                or score is None
                or row.get("status") not in (None, "ok")
            ):
                continue
            points.append((order, score))
        points.sort(key=lambda item: item[0])
    return [(index, score) for index, score in points if 1 <= index <= BUDGET]


def _generic_eval_points(run: Path) -> list[tuple[int, float | None]]:
    points: list[tuple[int, float]] = []
    samples_dir = run / "logs" / "samples"
    if not samples_dir.is_dir():
        return []
    for path in sorted(samples_dir.glob("samples_*.json")):
        if path.name == "samples_best.json":
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            order, score = row.get("sample_order"), finite(row.get("score"))
            if isinstance(order, int) and score is not None:
                points.append((order, score))
    points.sort(key=lambda item: item[0])
    remapped: list[tuple[int, float | None]] = []
    for _, score in points:
        remapped.append((len(remapped) + 1, score))
    return [(index, score) for index, score in remapped if index <= BUDGET]


def best_so_far(points: list[tuple[int, float | None]]) -> np.ndarray:
    raw = np.full(BUDGET, -np.inf)
    for index, score in points:
        if 1 <= index <= BUDGET and score is not None:
            raw[index - 1] = max(raw[index - 1], score)
    curve = np.maximum.accumulate(raw)
    finite_mask = np.isfinite(curve)
    if not finite_mask.any():
        return np.full(BUDGET, np.nan)
    first = int(np.argmax(finite_mask))
    curve[:first] = curve[first]
    return curve


def search_stats(points: list[tuple[int, float | None]]) -> dict[str, Any]:
    curve = best_so_far(points)
    scored = [(index, score) for index, score in points if score is not None]
    if not scored:
        raise ValueError("run has no scored evaluator calls")
    best = float(np.nanmax(curve))
    first_hit = next(
        index for index, score in scored if math.isclose(score, best, rel_tol=0, abs_tol=1e-12)
        or score >= best - 1e-12
    )
    last_refresh = first_hit
    n_break = 0
    incumbent = -np.inf
    for index, score in scored:
        if score > incumbent + 1e-12:
            n_break += 1
            incumbent = score
            last_refresh = index
    start = float(curve[0])
    gain = best - start
    def frac_at(eval_n: int) -> float | None:
        if not math.isfinite(gain) or abs(gain) < 1e-12:
            return None
        return float((curve[eval_n - 1] - start) / gain)

    def first_frac(target: float) -> int | None:
        ratio = frac_at
        for eval_n in range(1, BUDGET + 1):
            value = ratio(eval_n)
            if value is not None and value >= target:
                return eval_n
        return None

    milestones = {str(n): float(curve[n - 1]) for n in MILESTONES}
    return {
        "n_eval_logged": len(points),
        "final_best_q": best,
        "start_q": start,
        "total_gain": gain,
        "first_hit_final_best": first_hit,
        "last_refresh": last_refresh,
        "n_breakthroughs": n_break,
        "frac_gain_at": {str(n): frac_at(n) for n in MILESTONES},
        "eval_to_50pct_gain": first_frac(0.5),
        "eval_to_90pct_gain": first_frac(0.9),
        "milestones": milestones,
        "late_refresh_after_500": last_refresh > 500,
        "late_refresh_after_750": last_refresh > 750,
        "curve": curve.tolist(),
    }


def idea_key(text: str | None) -> str:
    if not text:
        return ""
    tokens = text.lower().replace("`", " ").split()
    return " ".join(tokens[:12])


def v97_generation(run: Path) -> dict[str, Any]:
    rows = load_jsonl(run / "artifacts" / "candidates.jsonl")
    search = [row for row in rows if row.get("stage") == "search"]

    def outcome_of(row: dict[str, Any]) -> str | None:
        if row.get("outcome"):
            return str(row["outcome"])
        if row.get("direct_outcome"):
            return str(row["direct_outcome"])
        if row.get("status") not in (None, "ok"):
            return "invalid"
        return None

    def change_of(row: dict[str, Any]) -> int | None:
        if row.get("added") is not None or row.get("removed") is not None:
            return int(row.get("added") or 0) + int(row.get("removed") or 0)
        stats = row.get("diff_statistics") or {}
        if "changed_lines" in stats:
            return int(stats["changed_lines"])
        return None

    def dq_of(row: dict[str, Any]) -> float | None:
        value = finite(row.get("dq"))
        if value is not None:
            return value
        child = candidate_score(row)
        parent = finite(row.get("parent_fitness"))
        if child is None or parent is None:
            return None
        return child - parent

    by_intent: dict[str, dict[str, Any]] = {}
    for intent in ("refine", "explore"):
        subset = [
            row
            for row in search
            if row.get("intent") == intent or row.get("operator") == intent
        ]
        outcomes = Counter(outcome_of(row) for row in subset)
        valid = [
            row
            for row in subset
            if outcome_of(row) in ("improve", "plateau", "regress")
        ]
        improves = [row for row in valid if outcome_of(row) == "improve"]
        changes = [change_of(row) for row in valid]
        changes_f = [value for value in changes if value is not None]
        dqs_f = [value for value in (dq_of(row) for row in valid) if value is not None]
        by_intent[intent] = {
            "n": len(subset),
            "outcomes": dict(outcomes),
            "improve_rate": len(improves) / len(subset) if subset else None,
            "valid_rate": len(valid) / len(subset) if subset else None,
            "invalid_rate": outcomes.get("invalid", 0) / len(subset) if subset else None,
            "median_change_lines": median(changes_f) if changes_f else None,
            "median_dq": median(dqs_f) if dqs_f else None,
            "mean_dq_improve": mean(
                [dq_of(row) or 0.0 for row in improves]
            )
            if improves
            else None,
        }
    ideas = [idea_key(row.get("idea")) for row in search if row.get("idea")]
    unique = len(set(ideas))
    repeats = sum(
        1 for left, right in zip(ideas, ideas[1:]) if left and left == right
    )
    all_outcomes = Counter(outcome_of(row) for row in search)
    valid = [row for row in search if outcome_of(row) in ("improve", "plateau", "regress")]
    changes_f = [value for value in (change_of(row) for row in valid) if value is not None]
    return {
        "n_search": len(search),
        "outcomes": dict(all_outcomes),
        "improve_rate": all_outcomes.get("improve", 0) / len(search) if search else None,
        "invalid_rate": all_outcomes.get("invalid", 0) / len(search) if search else None,
        "median_change_lines": median(changes_f) if changes_f else None,
        "idea_unique_rate": unique / len(ideas) if ideas else None,
        "consecutive_identical_idea_rate": repeats / max(len(ideas) - 1, 1),
        "by_intent": by_intent,
    }


def v97_lineage(run: Path) -> dict[str, Any]:
    summary = load_json(run / "logs" / "summary.json")
    rows = load_jsonl(run / "artifacts" / "candidates.jsonl")
    parent_of: dict[int, int | None] = {}
    creator: dict[int, dict[str, Any]] = {}
    eval_index_of_order: dict[int, int] = {}
    eval_i = 0
    for row in rows:
        child = row.get("child_id", row.get("child_state_id"))
        parent = row.get("anchor_id", row.get("parent_node_id"))
        if isinstance(child, int):
            parent_of[child] = parent if isinstance(parent, int) else None
            creator[child] = row
        if row.get("evaluator_called"):
            eval_i += 1
            order = candidate_order(row)
            if order is not None:
                eval_index_of_order[order] = eval_i
    best_id = summary.get("best_program_id", summary.get("best_artifact_id"))
    if best_id is None:
        raise ValueError(f"no best program id in {run}")
    best_id = int(best_id)
    created = [
        row
        for row in rows
        if row.get("program_id", row.get("artifact_id")) == best_id
        and isinstance(row.get("child_id", row.get("child_state_id")), int)
    ]
    if not created:
        raise ValueError(f"best program {best_id} not found in {run}")
    birth = created[0]
    node = int(birth.get("child_id", birth.get("child_state_id")))
    path: list[dict[str, Any]] = []
    seen: set[int] = set()
    while node is not None and node not in seen:
        seen.add(node)
        row = creator.get(node, {})
        outcome = row.get("outcome") or row.get("direct_outcome")
        path.append(
            {
                "anchor_id": node,
                "outcome": outcome,
                "intent": row.get("intent") or row.get("operator"),
                "stage": row.get("stage"),
                "dq": finite(row.get("dq")),
                "change_lines": int(row.get("added") or 0) + int(row.get("removed") or 0),
                "order": candidate_order(row),
            }
        )
        parent = parent_of.get(node)
        node = parent if isinstance(parent, int) else None
    path.reverse()
    trailing = 0
    for item in reversed(path):
        if item["outcome"] == "improve":
            trailing += 1
        elif item["stage"] == "root_generation":
            continue
        else:
            break
    n_improve = sum(item["outcome"] == "improve" for item in path)
    n_regress = sum(item["outcome"] == "regress" for item in path)
    decisions = load_jsonl(run / "artifacts" / "decisions.jsonl")
    routes = [row for row in decisions if row.get("event") == "route_selected"]
    selected = [int(row["selected_root_state_id"]) for row in routes]
    counts = Counter(selected)
    top_id, top_n = counts.most_common(1)[0] if counts else (None, 0)
    last_other = None
    for index, route_id in enumerate(selected, start=1):
        if top_id is not None and route_id != top_id:
            last_other = index
    shown = []
    for row in decisions:
        if row.get("event") != "history_built":
            continue
        ids = row.get("shown_event_ids") or row.get("selected_formation_ids") or []
        shown.append(len(ids))
    birth_eval = eval_index_of_order.get(int(summary["best_sample_order"]))
    return {
        "best_program_id": best_id,
        "best_sample_order": int(summary["best_sample_order"]),
        "best_eval_index": birth_eval,
        "lineage_depth": max(len(path) - 1, 0),
        "lineage_nodes": len(path),
        "path_improves": n_improve,
        "path_regresses": n_regress,
        "trailing_improves": trailing,
        "path_intents": dict(Counter(item["intent"] for item in path if item["intent"])),
        "n_anchors": int(summary.get("n_anchors") or 0),
        "n_programs": int(summary.get("n_programs") or 0),
        "s": finite(summary.get("s", summary.get("optimism_scale"))),
        "max_route_share": top_n / len(selected) if selected else None,
        "n_routes_selected": len(counts),
        "last_non_top_route_decision": last_other,
        "mean_shown_history": mean(shown) if shown else None,
        "median_shown_history": median(shown) if shown else None,
        "path_outcomes": [item["outcome"] for item in path],
    }


def v96_generation(run: Path) -> dict[str, Any]:
    rows = load_jsonl(run / "artifacts" / "candidates.jsonl")
    search = [row for row in rows if row.get("stage") == "search"]
    outcomes = Counter(row.get("direct_outcome") or row.get("status") for row in search)
    n = len(search)
    invalid = sum(
        1
        for row in search
        if row.get("status") not in (None, "ok") or row.get("direct_outcome") == "invalid"
    )
    improve = sum(1 for row in search if row.get("direct_outcome") == "improve")
    changes = []
    for row in search:
        stats = row.get("diff_statistics") or {}
        if "changed_lines" in stats:
            changes.append(int(stats["changed_lines"]))
    ideas = [idea_key(row.get("idea")) for row in search if row.get("idea")]
    unique = len(set(ideas))
    return {
        "n_search": n,
        "outcomes": dict(outcomes),
        "improve_rate": improve / n if n else None,
        "invalid_rate": invalid / n if n else None,
        "median_change_lines": median(changes) if changes else None,
        "idea_unique_rate": unique / len(ideas) if ideas else None,
    }


def load_probe_behavior() -> dict[str, Any]:
    summary = load_json(
        REPO
        / "experiments"
        / "generation_probe"
        / "20260813_v96_parent_path_probe"
        / "analysis"
        / "summary.json"
    )
    contrast = summary["contrasts"]["parent_path_minus_code_only"]["paired_summaries"]
    out: dict[str, Any] = {}
    for task in TASKS:
        block = contrast[f"task:{task}"]["metrics"]
        out[TASK_LABEL[task]] = {
            key: {
                "control_mean": block[key].get("code_only_mean"),
                "treatment_mean": block[key].get("parent_path_mean"),
                "paired_mean_diff": block[key]["paired_mean_difference_b_minus_a"],
                "ci95": block[key]["paired_mean_difference_bootstrap_95ci"],
                "anchors_better_treatment": block[key].get("anchors_better_b"),
                "anchors_better_control": block[key].get("anchors_better_a"),
            }
            for key in (
                "conditional_delta_q",
                "conditional_improvement_rate",
                "conditional_change_ratio",
                "conditional_abs_loc_delta",
                "valid_rate",
            )
        }
    return out


def summarize_runs(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {"n": 0}
    numeric = [float(value) for value in values]
    return {
        "n": len(numeric),
        "mean": mean(numeric),
        "median": median(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "std": float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0,
        "values": numeric,
    }


def analyze() -> dict[str, Any]:
    search: dict[str, Any] = {}
    generation: dict[str, Any] = {}
    lineage: dict[str, Any] = {}
    for task in TASKS:
        search[task] = {}
        generation[task] = {}
        lineage[task] = {}
        for method in METHODS:
            dirs = run_dirs(task, method)
            run_rows = []
            gen_rows = []
            lin_rows = []
            for run in dirs:
                points = eval_points(run)
                if not points:
                    continue
                stats = search_stats(points)
                stats["run"] = run.name
                run_rows.append(stats)
                if method in ("V9.7", "V9.7-batch"):
                    gen_rows.append(v97_generation(run))
                    lin_rows.append(v97_lineage(run))
                elif method == "V9.6":
                    gen_rows.append(v96_generation(run))
            if not run_rows:
                continue
            search[task][method] = {
                "n_runs": len(run_rows),
                "runs": [
                    {k: v for k, v in row.items() if k != "curve"} for row in run_rows
                ],
                "mean_curve": np.mean(
                    np.vstack([np.array(row["curve"]) for row in run_rows]), axis=0
                ).tolist(),
                "min_curve": np.min(
                    np.vstack([np.array(row["curve"]) for row in run_rows]), axis=0
                ).tolist(),
                "max_curve": np.max(
                    np.vstack([np.array(row["curve"]) for row in run_rows]), axis=0
                ).tolist(),
                "aggregate": {
                    key: summarize_runs(run_rows, key)
                    for key in (
                        "final_best_q",
                        "first_hit_final_best",
                        "last_refresh",
                        "n_breakthroughs",
                        "eval_to_50pct_gain",
                        "eval_to_90pct_gain",
                    )
                },
                "milestone_means": {
                    str(n): mean(row["milestones"][str(n)] for row in run_rows)
                    for n in MILESTONES
                },
                "frac_gain_means": {
                    str(n): mean(
                        row["frac_gain_at"][str(n)]
                        for row in run_rows
                        if row["frac_gain_at"][str(n)] is not None
                    )
                    for n in MILESTONES
                },
            }
            if gen_rows:
                generation[task][method] = {
                    "n_runs": len(gen_rows),
                    "improve_rate": summarize_runs(gen_rows, "improve_rate"),
                    "invalid_rate": summarize_runs(gen_rows, "invalid_rate"),
                    "median_change_lines": summarize_runs(
                        gen_rows, "median_change_lines"
                    ),
                    "idea_unique_rate": summarize_runs(gen_rows, "idea_unique_rate"),
                    "runs": gen_rows,
                }
            if lin_rows:
                lineage[task][method] = {
                    "n_runs": len(lin_rows),
                    "lineage_depth": summarize_runs(lin_rows, "lineage_depth"),
                    "path_improves": summarize_runs(lin_rows, "path_improves"),
                    "path_regresses": summarize_runs(lin_rows, "path_regresses"),
                    "trailing_improves": summarize_runs(lin_rows, "trailing_improves"),
                    "max_route_share": summarize_runs(lin_rows, "max_route_share"),
                    "last_non_top_route_decision": summarize_runs(
                        lin_rows, "last_non_top_route_decision"
                    ),
                    "mean_shown_history": summarize_runs(lin_rows, "mean_shown_history"),
                    "runs": lin_rows,
                }

    allocation_official = analyze_batch(OFFICIAL_BATCH)
    allocation_batch = analyze_batch(BATCH_CONTEXT)
    return {
        "analysis": "traceaad_v97_mechanism_value",
        "official_batch": OFFICIAL_BATCH,
        "batch_context_batch": BATCH_CONTEXT,
        "evidence_boundary": (
            "Search-process comparisons evaluate joint protocols. "
            "V9.7 vs V9.7-batch isolates history composition under the same "
            "allocation and intent mix. V9.6 vs V9.7-batch jointly changes "
            "allocation and intent. The parent-path probe isolates lineage "
            "at a fixed anchor and does not estimate full-search value."
        ),
        "search": search,
        "generation": generation,
        "lineage": lineage,
        "probe_parent_path_minus_code_only": load_probe_behavior(),
        "allocation": {
            "official": {
                "overall": allocation_official["aggregate"]["overall"],
                "tasks": allocation_official["aggregate"]["tasks"],
            },
            "batch_context": {
                "overall": allocation_batch["aggregate"]["overall"],
                "tasks": allocation_batch["aggregate"]["tasks"],
            },
        },
    }


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_curves(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    methods = ["V8", "V9", "V9.6", "V9.7-batch", "V9.7", "MCTS-AHD"]
    x = np.arange(1, BUDGET + 1)
    for ax, task in zip(axes.ravel(), TASKS):
        for method in methods:
            block = result["search"].get(task, {}).get(method)
            if not block:
                continue
            y = np.array(block["mean_curve"])
            lo = np.array(block["min_curve"])
            hi = np.array(block["max_curve"])
            ax.fill_between(x, lo, hi, color=COLORS[method], alpha=0.12, linewidth=0)
            ax.plot(x, y, color=COLORS[method], lw=1.8, label=method)
        ax.set_title(TASK_LABEL[task])
        ax.set_xlim(1, BUDGET)
    axes[1, 0].set_xlabel("Evaluator calls")
    axes[1, 1].set_xlabel("Evaluator calls")
    axes[0, 0].set_ylabel("Best-so-far q")
    axes[1, 0].set_ylabel("Best-so-far q")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig1_best_so_far.png", dpi=200)
    plt.close(fig)


def plot_efficiency(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    methods = ["V8", "V9", "V9.6", "V9.7-batch", "V9.7"]
    x = np.arange(len(TASKS))
    width = 0.15
    for i, method in enumerate(methods):
        last = [
            result["search"][task][method]["aggregate"]["last_refresh"]["mean"]
            if method in result["search"][task]
            else np.nan
            for task in TASKS
        ]
        hit = [
            result["search"][task][method]["aggregate"]["first_hit_final_best"]["mean"]
            if method in result["search"][task]
            else np.nan
            for task in TASKS
        ]
        axes[0].bar(x + (i - 2) * width, last, width, color=COLORS[method], label=method)
        axes[1].bar(x + (i - 2) * width, hit, width, color=COLORS[method], label=method)
    for ax, title in zip(axes, ("Mean last best refresh", "Mean first hit of final best")):
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
        ax.set_ylabel("Evaluator calls")
        ax.set_title(title)
        ax.set_ylim(0, 1050)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_refresh_timing.png", dpi=200)
    plt.close(fig)


def plot_generation(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.3))
    methods = ["V9.6", "V9.7-batch", "V9.7"]
    x = np.arange(len(TASKS))
    width = 0.24
    metrics = [
        ("improve_rate", "Search-step improve rate", axes[0], 1.0),
        ("invalid_rate", "Search-step invalid rate", axes[1], 0.25),
        ("median_change_lines", "Median changed lines", axes[2], None),
    ]
    for key, title, ax, ymax in metrics:
        for i, method in enumerate(methods):
            vals = []
            for task in TASKS:
                block = result["generation"].get(task, {}).get(method)
                vals.append(
                    block[key]["mean"] if block and key in block else np.nan
                )
            ax.bar(x + (i - 1) * width, vals, width, color=COLORS[method], label=method)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
        ax.set_title(title)
        if ymax is not None:
            ax.set_ylim(0, ymax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_generation_stats.png", dpi=200)
    plt.close(fig)


def plot_lineage(result: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3))
    methods = ["V9.7-batch", "V9.7"]
    x = np.arange(len(TASKS))
    width = 0.32
    for i, method in enumerate(methods):
        depth = [
            result["lineage"][task][method]["lineage_depth"]["mean"]
            for task in TASKS
        ]
        share = [
            100.0 * result["lineage"][task][method]["max_route_share"]["mean"]
            for task in TASKS
        ]
        axes[0].bar(x + (i - 0.5) * width, depth, width, color=COLORS[method], label=method)
        axes[1].bar(x + (i - 0.5) * width, share, width, color=COLORS[method], label=method)
    axes[0].set_title("Mean lineage depth of final best")
    axes[0].set_ylabel("Formation steps")
    axes[1].set_title("Top-route share of decisions")
    axes[1].set_ylabel("Percent")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABEL[t] for t in TASKS])
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_lineage_and_routes.png", dpi=200)
    plt.close(fig)


def plot_probe(result: dict[str, Any]) -> None:
    probe = result["probe_parent_path_minus_code_only"]
    tasks = list(probe)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    x = np.arange(len(tasks))
    width = 0.36
    for ax, metric, title, ylabel in (
        (
            axes[0],
            "conditional_change_ratio",
            "Modification magnitude",
            "Line-change ratio",
        ),
        (
            axes[1],
            "conditional_improvement_rate",
            "Parent-improvement rate",
            "Rate",
        ),
    ):
        control = [probe[task][metric]["control_mean"] for task in tasks]
        treat = [probe[task][metric]["treatment_mean"] for task in tasks]
        ax.bar(x - width / 2, control, width, color="#9AA0A6", label="code only")
        ax.bar(x + width / 2, treat, width, color="#FF7F0E", label="parent path")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_probe_behavior.png", dpi=200)
    plt.close(fig)


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """Drop per-run curves already stored as aggregates."""
    payload = json.loads(json.dumps(result))
    for task in payload["search"].values():
        for method in task.values():
            method.pop("mean_curve", None)
            method.pop("min_curve", None)
            method.pop("max_curve", None)
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = analyze()
    (OUT / "summary.json").write_text(
        json.dumps(compact(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (OUT / "curves.json").write_text(
        json.dumps(
            {
                task: {
                    method: {
                        "mean": block["mean_curve"],
                        "min": block["min_curve"],
                        "max": block["max_curve"],
                    }
                    for method, block in methods.items()
                }
                for task, methods in result["search"].items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    style()
    plot_curves(result)
    plot_efficiency(result)
    plot_generation(result)
    plot_lineage(result)
    plot_probe(result)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
