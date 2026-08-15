#!/usr/bin/env python3
"""Diagnose TraceAAD V9.7 allocation and proposal geometry.

The analysis is descriptive and post hoc.  It reconstructs the official
``20260814_150927`` runs, but it does not fabricate counterfactual continuations
for routes that the live allocator stopped selecting.  Static mechanism tags
are an auditable proxy for algorithm-family movement, not semantic ground truth.

Usage:

    uv run python experiments/analysis/analyze_v97_search_geometry.py
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import tokenize
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "analysis" / "traceaad_v97_search_geometry"
BATCH = "20260814_150927"
TASKS = ("tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing")
TASK_LABELS = {
    "tsp_construct": "TSP",
    "cvrp_aco": "CVRP",
    "op_aco": "OP",
    "online_bin_packing": "OBP",
}
RUN_PATTERNS = {
    "tsp_construct": f"traceaad_v9_7/v9_7_{BATCH}_tsp_construct_rep*",
    "cvrp_aco": f"traceaad_v9_7/v9_7_{BATCH}_cvrp_aco_rep*",
    "op_aco": f"traceaad_v9_7/v9_7_{BATCH}_op_aco_rep*",
    "online_bin_packing": (f"traceaad_v9_7/v9_7_{BATCH}_online_bin_packing_rep*"),
}
EPS = 1e-12


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Append-only logs may have a truncated final line.  Completed
                # official runs should not, but tolerating only EOF keeps the
                # reader safe for an interrupted diagnostic rerun.
                if handle.read(1):
                    raise ValueError(f"invalid JSONL at {path}:{line_number}")
                break


def run_dirs(task: str) -> list[Path]:
    task_dir = REPO / "experiments" / task
    found = sorted(path for path in task_dir.glob(RUN_PATTERNS[task]) if path.is_dir())
    complete: list[Path] = []
    for path in found:
        checkpoint = path / "checkpoints" / "latest.json"
        summary = path / "logs" / "summary.json"
        if not checkpoint.is_file() or not summary.is_file():
            continue
        row = read_json(summary)
        if row.get("status") == "finished" and int(row["evaluator_call_count"]) == 1000:
            complete.append(path)
    return complete


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.mean(kept) if kept else None


def median_or_none(values: Iterable[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    return statistics.median(kept) if kept else None


def quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float | int | None]) -> dict[str, Any]:
    kept = [float(value) for value in values if value is not None]
    return {
        "n": len(kept),
        "mean": statistics.mean(kept) if kept else None,
        "median": statistics.median(kept) if kept else None,
        "min": min(kept) if kept else None,
        "max": max(kept) if kept else None,
        "q25": quantile(kept, 0.25),
        "q75": quantile(kept, 0.75),
        "values": kept,
    }


def code_tokens(code: str) -> tuple[str, Counter[str]]:
    """Return code without comments/strings plus identifier counts.

    Generated comments and Idea text can claim a mechanism that the executable
    program does not contain.  Static tags therefore inspect lexical code tokens
    after dropping comments and all string literals, including docstrings.
    """

    kept: list[str] = []
    names: Counter[str] = Counter()
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in stream:
            if token.type in {
                tokenize.COMMENT,
                tokenize.STRING,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            }:
                continue
            value = token.string.lower()
            kept.append(value)
            if token.type == tokenize.NAME:
                names[value] += 1
    except (IndentationError, tokenize.TokenError):
        # Every program in the checkpoint was executable when evaluated.  This
        # fallback is only for tokenizer edge cases in generated source.
        lowered = code.lower()
        kept = re.findall(r"[a-z_][a-z0-9_]*|\S", lowered)
        names.update(re.findall(r"[a-z_][a-z0-9_]*", lowered))
    return " ".join(kept), names


def has_any(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def mechanism_tags(task: str, code: str) -> frozenset[str]:
    text, names = code_tokens(code)
    tags: set[str] = set()

    if task == "tsp_construct":
        if has_any(text, r"lru_cache", r"remaining_mask", r"bitmask"):
            tags.add("exact_dp")
        if has_any(text, r"beam_search", r"beam_width", r"\bbeam\b"):
            tags.add("beam_search")
        if has_any(
            text,
            r"completion_cost",
            r"temp_remaining",
            r"sampled_nn",
            r"nearest_neighbor.*completion",
            r"simulate.*tour",
        ):
            tags.add("tour_rollout")
        if has_any(text, r"2_opt", r"2opt", r"cost_original"):
            tags.add("two_opt")
        if has_any(text, r"argpartition", r"top_k", r"candidates_to_check"):
            tags.add("candidate_screening")
        if has_any(
            text,
            r"dist_unvisited",
            r"forward_cost",
            r"connectivity",
            r"isolation",
            r"future_cost",
        ):
            tags.add("one_step_lookahead")
        if has_any(text, r"remaining_count", r"progress", r"stage_factor"):
            tags.add("stage_adaptive")
        if names["destination_node"] > 1:
            tags.add("destination_aware")

    elif task == "cvrp_aco":
        if has_any(text, r"cluster_ids", r"sorted_customer_indices"):
            tags.add("sweep_partition")
        if has_any(text, r"clarke", r"saving", r"savings"):
            tags.add("clarke_wright_savings")
        if has_any(text, r"arctan2", r"angle_diff", r"angular"):
            tags.add("angular_geometry")
        if has_any(text, r"radial", r"dist_to_depot", r"depot_distance"):
            tags.add("depot_radial")
        if names["demands"] > 1 or names["capacity"] > 1:
            tags.add("demand_capacity")
        if has_any(text, r"feasible_mask", r"infeasible_mask", r"cap_factor"):
            tags.add("pair_feasibility")
        if has_any(text, r"neighbor", r"cluster_bonus", r"density"):
            tags.add("neighborhood_density")

    elif task == "op_aco":
        if names["prize"] > 1 and names["distance"] > 1:
            tags.add("prize_distance")
        if has_any(
            text,
            r"round_trip",
            r"return.*depot",
            r"feasible_mask",
            r"dist_j_to_depot",
        ):
            tags.add("return_feasibility")
        if has_any(
            text,
            r"remaining_budget",
            r"budget_factor",
            r"cost_pressure",
            r"budget_tightness",
        ):
            tags.add("budget_pressure")
        if has_any(text, r"top_k_indices", r"target_scores", r"directional_factor"):
            tags.add("directional_targets")
        if has_any(
            text,
            r"cluster_prize",
            r"near_prizes",
            r"forward_factor",
            r"normalized_potential",
        ):
            tags.add("neighborhood_potential")
        if has_any(text, r"softmax", r"exp_scores", r"np \. exp"):
            tags.add("nonlinear_emphasis")

    elif task == "online_bin_packing":
        if has_any(text, r"global\s+_", r"item_history", r"history"):
            tags.add("stateful_history")
        if has_any(text, r"np \. mean", r"np \. std", r"percentile", r"pdf_score"):
            tags.add("distribution_model")
        if has_any(text, r"probab", r"likelihood", r"future", r"fit_score"):
            tags.add("future_fit")
        if has_any(
            text,
            r"fragment",
            r"awkward",
            r"closure_bonus",
            r"gap_penalty",
            r"tiny_threshold",
        ):
            tags.add("fragmentation_control")
        if has_any(text, r"diversity", r"spread", r"balance"):
            tags.add("balance_spread")
        if has_any(text, r"remaining_after", r"leftover", r"remainder"):
            tags.add("best_fit")
        if has_any(text, r"threshold", r"np \. where"):
            tags.add("piecewise_bands")
    else:
        raise ValueError(f"unknown task: {task}")
    return frozenset(tags)


def macro_family(task: str, tags: frozenset[str]) -> str:
    if task == "tsp_construct":
        if tags & {"exact_dp", "beam_search"}:
            return "explicit_search"
        if tags & {"tour_rollout", "two_opt"}:
            return "completion_rollout"
        if "one_step_lookahead" in tags:
            return "lookahead_score"
        return "local_score"
    if task == "cvrp_aco":
        if "sweep_partition" in tags:
            return "sweep_partition"
        if "clarke_wright_savings" in tags:
            return "savings_geometry"
        if "angular_geometry" in tags and "neighborhood_density" in tags:
            return "spatial_cluster"
        if "angular_geometry" in tags:
            return "angular_radial"
        if "demand_capacity" in tags:
            return "capacity_distance"
        return "distance_only"
    if task == "op_aco":
        if "directional_targets" in tags:
            return "target_direction"
        if "neighborhood_potential" in tags:
            return "neighborhood_potential"
        if {"return_feasibility", "budget_pressure"} <= tags:
            return "budget_feasible_density"
        if "return_feasibility" in tags:
            return "feasibility_density"
        return "prize_distance"
    if task == "online_bin_packing":
        if {"stateful_history", "distribution_model"} <= tags:
            return "online_distribution"
        if "distribution_model" in tags:
            return "distribution_fit"
        if "future_fit" in tags:
            return "future_gap_model"
        if "fragmentation_control" in tags:
            return "fragmentation_best_fit"
        if "balance_spread" in tags:
            return "balance_fit"
        return "best_fit"
    raise ValueError(f"unknown task: {task}")


def jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def pairwise_distances(items: Sequence[frozenset[str]]) -> list[float]:
    return [
        jaccard_distance(items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


def route_geometry(run: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    forest = checkpoint["forest"]
    programs = {int(row["id"]): row for row in forest["programs"]}
    anchors = {int(row["id"]): row for row in forest["anchors"]}
    task = read_json(run / "run_config.json")["task"]
    root_ids = [int(value) for value in forest["root_ids"]]
    root_tags = {
        root_id: mechanism_tags(
            task, programs[int(anchors[root_id]["program_id"])]["code"]
        )
        for root_id in root_ids
    }
    root_families = {
        root_id: macro_family(task, tags) for root_id, tags in root_tags.items()
    }

    route_events: list[dict[str, Any]] = []
    for row in iter_jsonl(run / "artifacts" / "decisions.jsonl"):
        if row.get("event") == "route_selected":
            route_events.append(row)
    route_events.sort(key=lambda row: int(row["iteration"]))
    if not route_events:
        raise ValueError(f"no route decisions: {run}")

    selections = [int(row["selected_root_state_id"]) for row in route_events]
    counts = Counter(selections)
    top_route, top_count = counts.most_common(1)[0]
    stable_start = 1
    for index, selected in enumerate(selections, start=1):
        if selected != top_route:
            stable_start = index + 1

    viable_counts: list[int] = []
    viable_ids: list[list[int]] = []
    for event in route_events:
        routes = event["routes"]
        current_frontier = max(float(route["best_q"]) for route in routes)
        live = [
            int(route["root_state_id"])
            for route in routes
            if float(route["best_q"]) + float(route["optimism"])
            >= current_frontier - EPS
        ]
        viable_counts.append(len(live))
        viable_ids.append(live)
    if any(right > left for left, right in zip(viable_counts, viable_counts[1:])):
        raise AssertionError(f"viable route set grew in {run}")
    collapse = next(
        (index for index, count in enumerate(viable_counts, start=1) if count == 1),
        None,
    )
    if collapse is not None:
        surviving = set(viable_ids[collapse - 1])
        if len(surviving) != 1 or any(
            selected not in surviving for selected in selections[collapse - 1 :]
        ):
            raise AssertionError(f"irreversible route collapse mismatch in {run}")

    first_routes = route_events[0]["routes"]
    initial_q = {
        int(row["root_state_id"]): float(row["best_q"]) for row in first_routes
    }
    ordered_initial = sorted(
        root_ids,
        key=lambda root_id: (initial_q[root_id], -root_id),
        reverse=True,
    )
    first_gap = initial_q[ordered_initial[0]] - initial_q[ordered_initial[1]]
    scale = float(checkpoint["s"])
    initial_values = list(initial_q.values())
    initial_best_tie_count = sum(
        math.isclose(value, initial_q[ordered_initial[0]], rel_tol=0, abs_tol=EPS)
        for value in initial_values
    )

    search_attempts = [
        row for row in forest["attempts"] if row.get("stage") == "search"
    ]
    route_visits: Counter[int] = Counter()
    for attempt in search_attempts:
        parent = attempt.get("anchor_id")
        if isinstance(parent, int):
            route_visits[int(anchors[parent]["root_id"])] += 1
    if route_visits != counts:
        raise AssertionError(f"attempt and decision route counts differ in {run}")

    observed_route_best: dict[int, float] = {}
    for root_id in root_ids:
        observed_route_best[root_id] = max(
            float(programs[int(anchor["program_id"])]["q"])
            for anchor in anchors.values()
            if int(anchor["root_id"]) == root_id
        )

    best_program_id = int(checkpoint["best_id"])
    best_anchors = [
        anchor
        for anchor in anchors.values()
        if int(anchor["program_id"]) == best_program_id
    ]
    best_anchor = min(best_anchors, key=lambda row: int(row["order"]))
    final_route = int(best_anchor["root_id"])
    final_tags = mechanism_tags(task, programs[best_program_id]["code"])
    final_family = macro_family(task, final_tags)

    family_counts = Counter(root_families.values())
    signature_counts = Counter(tuple(sorted(tags)) for tags in root_tags.values())
    audit = {
        "root_families": {str(key): value for key, value in root_families.items()},
        "root_tags": {str(key): sorted(value) for key, value in root_tags.items()},
        "final_route": final_route,
        "final_family": final_family,
        "final_tags": sorted(final_tags),
    }
    return {
        "n_decisions": len(route_events),
        "top_route": top_route,
        "final_best_route": final_route,
        "top_route_is_final_best_route": top_route == final_route,
        "top_route_share": top_count / len(selections),
        "routes_selected": len(counts),
        "stable_top_route_start": stable_start,
        "irreversible_single_viable_start": collapse,
        "viable_routes_at_start": viable_counts[0],
        "dead_routes_at_start": len(root_ids) - viable_counts[0],
        "initial_best_route": ordered_initial[0],
        "initial_best_is_top_route": ordered_initial[0] == top_route,
        "initial_rank_of_top_route": ordered_initial.index(top_route) + 1,
        "initial_rank_of_final_best_route": ordered_initial.index(final_route) + 1,
        "initial_top_second_gap": first_gap,
        "initial_top_second_gap_over_s": first_gap / scale if scale > 0 else None,
        "initial_q_range_over_s": (
            (max(initial_values) - min(initial_values)) / scale if scale > 0 else None
        ),
        "initial_q_sd_over_s": (
            statistics.pstdev(initial_values) / scale if scale > 0 else None
        ),
        "initial_best_tie_count": initial_best_tie_count,
        "s": scale,
        "route_visits": {str(root): route_visits[root] for root in root_ids},
        "routes_with_zero_search_visits": sum(
            route_visits[root] == 0 for root in root_ids
        ),
        "routes_with_at_most_five_visits": sum(
            route_visits[root] <= 5 for root in root_ids
        ),
        "routes_with_positive_observed_gain": sum(
            observed_route_best[root] > initial_q[root] + EPS for root in root_ids
        ),
        "observed_route_best_range_over_s": (
            (max(observed_route_best.values()) - min(observed_route_best.values()))
            / scale
            if scale > 0
            else None
        ),
        "root_macro_family_count": len(family_counts),
        "root_dominant_macro_share": max(family_counts.values()) / len(root_ids),
        "root_signature_count": len(signature_counts),
        "root_mean_pairwise_tag_distance": mean_or_none(
            pairwise_distances(list(root_tags.values()))
        ),
        "root_to_final_macro_changed": root_families[final_route] != final_family,
        "root_to_final_tag_distance": jaccard_distance(
            root_tags[final_route], final_tags
        ),
        "audit": audit,
    }


def descendant_summaries(
    anchors: dict[int, dict[str, Any]], programs: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    children: dict[int, list[int]] = defaultdict(list)
    for anchor in anchors.values():
        parent = anchor.get("parent_id")
        if isinstance(parent, int):
            children[parent].append(int(anchor["id"]))

    memo: dict[int, dict[str, Any]] = {}

    def visit(anchor_id: int) -> dict[str, Any]:
        if anchor_id in memo:
            return memo[anchor_id]
        child_rows = [visit(child) for child in children.get(anchor_id, [])]
        own_q = float(programs[int(anchors[anchor_id]["program_id"])]["q"])
        descendant_best = max(
            (row["subtree_best_q"] for row in child_rows), default=None
        )
        subtree_best = max(
            [own_q] + [float(row["subtree_best_q"]) for row in child_rows]
        )
        descendant_count = sum(1 + int(row["descendant_count"]) for row in child_rows)
        memo[anchor_id] = {
            "own_q": own_q,
            "descendant_best_q": descendant_best,
            "subtree_best_q": subtree_best,
            "descendant_count": descendant_count,
        }
        return memo[anchor_id]

    for anchor_id in anchors:
        visit(anchor_id)
    return memo


def final_best_path(
    checkpoint: dict[str, Any], anchors: dict[int, dict[str, Any]]
) -> list[int]:
    best_program_id = int(checkpoint["best_id"])
    candidates = [
        anchor
        for anchor in anchors.values()
        if int(anchor["program_id"]) == best_program_id
    ]
    node = int(min(candidates, key=lambda row: int(row["order"]))["id"])
    path: list[int] = []
    seen: set[int] = set()
    while node not in seen:
        seen.add(node)
        path.append(node)
        parent = anchors[node].get("parent_id")
        if not isinstance(parent, int):
            break
        node = parent
    path.reverse()
    return path


def proposal_geometry(run: Path, checkpoint: dict[str, Any]) -> dict[str, Any]:
    task = read_json(run / "run_config.json")["task"]
    forest = checkpoint["forest"]
    programs = {int(row["id"]): row for row in forest["programs"]}
    anchors = {int(row["id"]): row for row in forest["anchors"]}
    attempts = sorted(forest["attempts"], key=lambda row: int(row["order"]))
    subtrees = descendant_summaries(anchors, programs)
    best_path_ordered = final_best_path(checkpoint, anchors)
    best_path = set(best_path_ordered)
    attempts_by_child = {
        int(attempt["child_id"]): attempt
        for attempt in attempts
        if isinstance(attempt.get("child_id"), int)
    }

    incumbent = -math.inf
    route_frontier: dict[int, float] = defaultdict(lambda: -math.inf)
    scale = float(checkpoint["s"])
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        child_q = attempt.get("child_fitness")
        child_q_f = float(child_q) if isinstance(child_q, (int, float)) else None
        breakthrough = child_q_f is not None and child_q_f > incumbent + EPS
        increment = (
            child_q_f - incumbent if breakthrough and math.isfinite(incumbent) else 0.0
        )
        if child_q_f is not None:
            incumbent = max(incumbent, child_q_f)
        stage = attempt.get("stage")
        parent_for_route = attempt.get("anchor_id")
        child_for_route = attempt.get("child_id")
        route_id: int | None = None
        if isinstance(parent_for_route, int):
            route_id = int(anchors[parent_for_route]["root_id"])
        elif isinstance(child_for_route, int):
            route_id = int(anchors[child_for_route]["root_id"])
        route_q_before = route_frontier[route_id] if route_id is not None else None

        if stage != "search":
            if route_id is not None and child_q_f is not None:
                route_frontier[route_id] = max(route_frontier[route_id], child_q_f)
            continue

        parent_id = attempt.get("anchor_id")
        child_anchor_id = attempt.get("child_id")
        parent_q = (
            float(programs[int(anchors[parent_id]["program_id"])]["q"])
            if isinstance(parent_id, int)
            else None
        )
        child_program_id = attempt.get("program_id")
        has_child = isinstance(child_anchor_id, int) and isinstance(
            child_program_id, int
        )
        parent_tags = (
            mechanism_tags(
                task, programs[int(anchors[parent_id]["program_id"])]["code"]
            )
            if isinstance(parent_id, int)
            else frozenset()
        )
        child_tags = (
            mechanism_tags(task, programs[int(child_program_id)]["code"])
            if has_child
            else None
        )
        parent_family = macro_family(task, parent_tags)
        child_family = (
            macro_family(task, child_tags) if child_tags is not None else None
        )
        descendant = subtrees[int(child_anchor_id)] if has_child else None
        descendant_best = descendant["descendant_best_q"] if descendant else None
        outcome = attempt.get("outcome")
        dead_on_arrival = (
            has_child
            and child_q_f is not None
            and route_q_before is not None
            and math.isfinite(route_q_before)
            and child_q_f + scale < route_q_before - EPS
        )
        rows.append(
            {
                "intent": attempt.get("intent"),
                "outcome": outcome,
                "has_child": has_child,
                "dq": float(attempt["dq"]) if attempt.get("dq") is not None else None,
                "change_lines": int(attempt.get("added") or 0)
                + int(attempt.get("removed") or 0),
                "breakthrough": breakthrough,
                "global_increment": increment,
                "parent_q": parent_q,
                "child_q": child_q_f,
                "tag_switch": (
                    child_tags != parent_tags if child_tags is not None else None
                ),
                "tag_distance": (
                    jaccard_distance(parent_tags, child_tags)
                    if child_tags is not None
                    else None
                ),
                "macro_switch": (
                    child_family != parent_family if child_family is not None else None
                ),
                "parent_family": parent_family,
                "child_family": child_family,
                "child_reexpanded": (
                    int(anchors[int(child_anchor_id)]["n"]) > 0 if has_child else None
                ),
                "dead_on_arrival": dead_on_arrival if has_child else None,
                "descendant_count": (
                    int(descendant["descendant_count"]) if descendant else None
                ),
                "descendant_best_q": descendant_best,
                "descendant_beats_parent": (
                    descendant_best is not None
                    and parent_q is not None
                    and float(descendant_best) > parent_q + EPS
                    if descendant is not None
                    else None
                ),
                "descendant_beats_child": (
                    descendant_best is not None
                    and child_q_f is not None
                    and float(descendant_best) > child_q_f + EPS
                    if descendant is not None
                    else None
                ),
                "on_final_best_path": (
                    int(child_anchor_id) in best_path if has_child else False
                ),
            }
        )
        if route_id is not None and child_q_f is not None:
            route_frontier[route_id] = max(route_frontier[route_id], child_q_f)

    best_program_id = int(checkpoint["best_id"])
    creators = [
        attempt
        for attempt in attempts
        if attempt.get("stage") == "search"
        and int(attempt.get("program_id") or -1) == best_program_id
        and isinstance(attempt.get("child_id"), int)
    ]
    final_birth_intent = creators[0].get("intent") if creators else None
    final_program_tags = mechanism_tags(task, programs[best_program_id]["code"])
    final_family = macro_family(task, final_program_tags)
    root_anchor_id = best_path_ordered[0]
    root_tags = mechanism_tags(
        task, programs[int(anchors[root_anchor_id]["program_id"])]["code"]
    )
    root_family = macro_family(task, root_tags)
    path_transitions: list[dict[str, Any]] = []
    for anchor_id in best_path_ordered[1:]:
        attempt = attempts_by_child[anchor_id]
        parent_id = int(anchors[anchor_id]["parent_id"])
        parent_tags = mechanism_tags(
            task, programs[int(anchors[parent_id]["program_id"])]["code"]
        )
        child_tags = mechanism_tags(
            task, programs[int(anchors[anchor_id]["program_id"])]["code"]
        )
        path_transitions.append(
            {
                "anchor_id": anchor_id,
                "intent": attempt.get("intent"),
                "idea": attempt.get("idea"),
                "outcome": attempt.get("outcome"),
                "dq": attempt.get("dq"),
                "order": attempt.get("order"),
                "parent_family": macro_family(task, parent_tags),
                "child_family": macro_family(task, child_tags),
                "parent_tags": sorted(parent_tags),
                "child_tags": sorted(child_tags),
            }
        )
    first_final_family = next(
        (
            row
            for row in path_transitions
            if row["child_family"] == final_family
            and row["parent_family"] != final_family
        ),
        None,
    )
    return {
        "rows": rows,
        "final_birth_intent": final_birth_intent,
        "final_path": {
            "root_family": root_family,
            "final_family": final_family,
            "root_to_final_macro_changed": root_family != final_family,
            "first_entry_final_family_intent": (
                first_final_family["intent"] if first_final_family else None
            ),
            "macro_switches_by_intent": dict(
                Counter(
                    str(row["intent"])
                    for row in path_transitions
                    if row["parent_family"] != row["child_family"]
                )
            ),
            "transition_count": len(path_transitions),
            "first_entry_final_family_event": first_final_family,
        },
    }


def summarize_proposals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for intent in ("refine", "explore"):
        subset = [row for row in rows if row["intent"] == intent]
        valid = [row for row in subset if row["has_child"]]
        regress = [row for row in valid if row["outcome"] == "regress"]
        with_descendants = [
            row for row in valid if int(row["descendant_count"] or 0) > 0
        ]
        out[intent] = {
            "attempts": len(subset),
            "new_child_count": len(valid),
            "invalid_count": sum(row["outcome"] == "invalid" for row in subset),
            "improve_count": sum(row["outcome"] == "improve" for row in subset),
            "improve_rate_per_attempt": (
                sum(row["outcome"] == "improve" for row in subset) / len(subset)
                if subset
                else None
            ),
            "global_breakthrough_count": sum(row["breakthrough"] for row in subset),
            "global_breakthrough_rate_per_attempt": (
                sum(row["breakthrough"] for row in subset) / len(subset)
                if subset
                else None
            ),
            "global_gain": sum(float(row["global_increment"]) for row in subset),
            "median_global_increment": median_or_none(
                row["global_increment"] for row in subset if row["breakthrough"]
            ),
            "largest_global_increment": max(
                (
                    float(row["global_increment"])
                    for row in subset
                    if row["breakthrough"]
                ),
                default=None,
            ),
            "median_change_lines": median_or_none(row["change_lines"] for row in valid),
            "median_dq": median_or_none(row["dq"] for row in valid),
            "tag_switch_rate": (
                sum(bool(row["tag_switch"]) for row in valid) / len(valid)
                if valid
                else None
            ),
            "macro_switch_count": sum(bool(row["macro_switch"]) for row in valid),
            "macro_switch_rate": (
                sum(bool(row["macro_switch"]) for row in valid) / len(valid)
                if valid
                else None
            ),
            "median_tag_distance": median_or_none(row["tag_distance"] for row in valid),
            "child_reexpanded_rate": (
                sum(bool(row["child_reexpanded"]) for row in valid) / len(valid)
                if valid
                else None
            ),
            "dead_on_arrival_count": sum(bool(row["dead_on_arrival"]) for row in valid),
            "dead_on_arrival_rate": (
                sum(bool(row["dead_on_arrival"]) for row in valid) / len(valid)
                if valid
                else None
            ),
            "child_has_descendant_rate": (
                len(with_descendants) / len(valid) if valid else None
            ),
            "descendant_beats_parent_rate_all_children": (
                sum(bool(row["descendant_beats_parent"]) for row in valid) / len(valid)
                if valid
                else None
            ),
            "descendant_beats_parent_rate_if_expanded": (
                sum(bool(row["descendant_beats_parent"]) for row in with_descendants)
                / len(with_descendants)
                if with_descendants
                else None
            ),
            "regress_child_count": len(regress),
            "regress_rescued_above_parent_count": sum(
                bool(row["descendant_beats_parent"]) for row in regress
            ),
            "regress_rescued_above_parent_rate": (
                sum(bool(row["descendant_beats_parent"]) for row in regress)
                / len(regress)
                if regress
                else None
            ),
            "final_path_child_count": sum(row["on_final_best_path"] for row in valid),
        }

    macro_groups: dict[str, Any] = {}
    valid_all = [row for row in rows if row["has_child"]]
    for label, subset in (
        ("macro_switch", [row for row in valid_all if row["macro_switch"]]),
        ("macro_stay", [row for row in valid_all if not row["macro_switch"]]),
    ):
        with_descendants = [
            row for row in subset if int(row["descendant_count"] or 0) > 0
        ]
        macro_groups[label] = {
            "n": len(subset),
            "improve_rate": (
                sum(row["outcome"] == "improve" for row in subset) / len(subset)
                if subset
                else None
            ),
            "global_breakthrough_rate": (
                sum(row["breakthrough"] for row in subset) / len(subset)
                if subset
                else None
            ),
            "child_has_descendant_rate": (
                len(with_descendants) / len(subset) if subset else None
            ),
            "dead_on_arrival_rate": (
                sum(bool(row["dead_on_arrival"]) for row in subset) / len(subset)
                if subset
                else None
            ),
            "descendant_beats_parent_rate_all_children": (
                sum(bool(row["descendant_beats_parent"]) for row in subset)
                / len(subset)
                if subset
                else None
            ),
        }
    out["by_macro_transition"] = macro_groups
    total_global_gain = sum(
        float(out[intent]["global_gain"]) for intent in ("refine", "explore")
    )
    out["global_gain_share"] = {
        intent: (
            float(out[intent]["global_gain"]) / total_global_gain
            if total_global_gain > 0
            else None
        )
        for intent in ("refine", "explore")
    }
    return out


def aggregate_allocation(run_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "top_route_share",
        "routes_selected",
        "stable_top_route_start",
        "irreversible_single_viable_start",
        "viable_routes_at_start",
        "dead_routes_at_start",
        "initial_rank_of_top_route",
        "initial_rank_of_final_best_route",
        "initial_top_second_gap_over_s",
        "initial_q_range_over_s",
        "initial_q_sd_over_s",
        "initial_best_tie_count",
        "routes_with_zero_search_visits",
        "routes_with_at_most_five_visits",
        "routes_with_positive_observed_gain",
        "observed_route_best_range_over_s",
        "root_macro_family_count",
        "root_dominant_macro_share",
        "root_signature_count",
        "root_mean_pairwise_tag_distance",
        "root_to_final_tag_distance",
    )
    return {
        "n_runs": len(run_rows),
        "initial_best_is_top_route_count": sum(
            row["initial_best_is_top_route"] for row in run_rows
        ),
        "top_route_is_final_best_route_count": sum(
            row["top_route_is_final_best_route"] for row in run_rows
        ),
        "root_to_final_macro_changed_count": sum(
            row["root_to_final_macro_changed"] for row in run_rows
        ),
        "distributions": {
            key: distribution(row[key] for row in run_rows) for key in keys
        },
        "runs": run_rows,
    }


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            ranks[order[offset]] = average_rank
        position = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    ranked_left = _rank(left)
    ranked_right = _rank(right)
    mean_left = statistics.mean(ranked_left)
    mean_right = statistics.mean(ranked_right)
    numerator = sum(
        (x - mean_left) * (y - mean_right) for x, y in zip(ranked_left, ranked_right)
    )
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in ranked_left)
        * sum((y - mean_right) ** 2 for y in ranked_right)
    )
    return numerator / denominator if denominator > 0 else None


FITNESS_LINE = re.compile(
    r"Fitness:\s*(?P<parent>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)"
    r"\s*->\s*(?P<child>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)",
    flags=re.IGNORECASE,
)


def probe_conditions() -> dict[str, Any]:
    probe = REPO / "experiments" / "generation_probe" / "20260813_v96_parent_path_probe"
    anchors = {row["anchor_id"]: row for row in iter_jsonl(probe / "anchors.jsonl")}
    metrics: dict[tuple[str, str], dict[str, Any]] = {}
    with (probe / "analysis" / "anchor_condition_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            metrics[(row["anchor_id"], row["condition"])] = row

    summary = read_json(probe / "analysis" / "summary.json")
    paired = summary["contrasts"]["parent_path_minus_code_only"]["paired_summaries"]
    strata: dict[str, Any] = {}
    for task in TASKS:
        strata[task] = {}
        for stratum in ("low", "middle", "high"):
            block = paired[f"task:{task}:stratum:{stratum}"]["metrics"]
            metric = block["conditional_delta_q"]
            strata[task][stratum] = {
                "n_anchors": paired[f"task:{task}:stratum:{stratum}"]["anchor_count"],
                "paired_delta_q_effect": metric["paired_mean_difference_b_minus_a"],
                "bootstrap_ci95": metric["paired_mean_difference_bootstrap_95ci"],
            }

    features_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor_id, anchor in anchors.items():
        control = metrics[(anchor_id, "code_only")]
        treatment = metrics[(anchor_id, "parent_path")]
        if not control["conditional_delta_q"] or not treatment["conditional_delta_q"]:
            continue
        effect = float(treatment["conditional_delta_q"]) - float(
            control["conditional_delta_q"]
        )
        outcomes = re.findall(
            r"Result:\s*(improve|regress|plateau)",
            anchor["formation_block"],
            flags=re.IGNORECASE,
        )
        transitions = [
            (float(match.group("parent")), float(match.group("child")))
            for match in FITNESS_LINE.finditer(anchor["formation_block"])
        ]
        first_parent = transitions[0][0] if transitions else float(anchor["q"])
        last_dq = transitions[-1][1] - transitions[-1][0] if transitions else 0.0
        positive_gain = sum(max(child - parent, 0.0) for parent, child in transitions)
        negative_gain = sum(max(parent - child, 0.0) for parent, child in transitions)
        n_events = max(len(outcomes), 1)
        features_by_task[str(anchor["task"])].append(
            {
                "effect": effect,
                "depth": float(anchor["depth"]),
                "source_iteration": float(anchor["source_iteration"]),
                "formation_event_count": float(anchor["formation_event_count"]),
                "history_improve_fraction": outcomes.count("improve") / n_events,
                "history_regress_fraction": outcomes.count("regress") / n_events,
                "path_net_gain": float(anchor["q"]) - first_parent,
                "path_positive_gain": positive_gain,
                "path_negative_gain": negative_gain,
                "last_step_dq": last_dq,
            }
        )

    correlations: dict[str, Any] = {}
    feature_names = (
        "depth",
        "source_iteration",
        "formation_event_count",
        "history_improve_fraction",
        "history_regress_fraction",
        "path_net_gain",
        "path_positive_gain",
        "path_negative_gain",
        "last_step_dq",
    )
    for task, rows in features_by_task.items():
        effects = [float(row["effect"]) for row in rows]
        correlations[task] = {
            "n_anchors": len(rows),
            "spearman_with_parent_path_effect": {
                feature: spearman([float(row[feature]) for row in rows], effects)
                for feature in feature_names
            },
        }
    return {
        "quality_strata": strata,
        "exploratory_correlations": correlations,
        "boundary": (
            "Quality strata were built into the paired probe. Correlations with "
            "path features are post hoc, use only 18 anchors per task, and are "
            "descriptive rather than validated effect modifiers."
        ),
    }


def analyze() -> dict[str, Any]:
    allocation: dict[str, Any] = {}
    proposal: dict[str, Any] = {}
    classification_audit: dict[str, Any] = {}
    for task in TASKS:
        dirs = run_dirs(task)
        if len(dirs) != 3:
            raise ValueError(f"expected 3 complete {task} runs, found {len(dirs)}")
        allocation_runs: list[dict[str, Any]] = []
        proposal_rows: list[dict[str, Any]] = []
        final_births: Counter[str] = Counter()
        final_family_entries: Counter[str] = Counter()
        changed_final_family_entries: Counter[str] = Counter()
        final_path_switches: Counter[str] = Counter()
        classification_audit[task] = {}
        for run in dirs:
            checkpoint = read_json(run / "checkpoints" / "latest.json")
            alloc = route_geometry(run, checkpoint)
            classification_audit[task][run.name] = alloc.pop("audit")
            alloc["run"] = run.name
            allocation_runs.append(alloc)

            proposals = proposal_geometry(run, checkpoint)
            proposal_rows.extend(proposals["rows"])
            classification_audit[task][run.name]["final_path"] = proposals["final_path"]
            final_births[str(proposals["final_birth_intent"])] += 1
            final_family_entries[
                str(proposals["final_path"]["first_entry_final_family_intent"])
            ] += 1
            if proposals["final_path"]["root_to_final_macro_changed"]:
                changed_final_family_entries[
                    str(proposals["final_path"]["first_entry_final_family_intent"])
                ] += 1
            final_path_switches.update(
                proposals["final_path"]["macro_switches_by_intent"]
            )
        allocation[task] = aggregate_allocation(allocation_runs)
        proposal[task] = {
            "n_runs": len(dirs),
            "final_best_birth_intent": dict(final_births),
            "first_entry_final_family_intent": dict(final_family_entries),
            "root_to_final_changed_entry_intent": dict(changed_final_family_entries),
            "final_path_macro_switches_by_intent": dict(final_path_switches),
            "summary": summarize_proposals(proposal_rows),
        }

    return {
        "analysis": "traceaad_v97_search_geometry",
        "batch": BATCH,
        "evidence_boundary": {
            "allocation": (
                "Observed route histories are policy-selected. Abandoned routes are "
                "right-censored, so their counterfactual continuation value is not "
                "identified by these logs."
            ),
            "proposal": (
                "Intent comparisons in complete search are confounded by selected "
                "anchors and downstream allocation. Static code mechanism tags are "
                "an interpretable proxy, not a validated semantic family label."
            ),
        },
        "allocation": allocation,
        "proposal": proposal,
        "history_effect_conditions": probe_conditions(),
        "classification_audit": classification_audit,
    }


def main() -> None:
    result = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
