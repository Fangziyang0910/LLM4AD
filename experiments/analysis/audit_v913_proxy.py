#!/usr/bin/env python3
"""Blind audit worksheet for the frozen V9.13 proxy mechanism tags.

Design section 4.2: before formal Stage P, 24 programs per task are sampled
stratified by source run and proxy region; a reviewer who cannot see
fitness, generation intent, or final placement checks whether the assigned
tags are supported by the actual code.  This script builds that worksheet
and attaches, per tag, the exact lexical evidence the frozen rules matched,
plus a keyword scan for mechanism vocabulary present in the code but outside
the frozen rules (candidate misses).

Usage:

    uv run python experiments/analysis/audit_v913_proxy.py [--seed 913401]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm4ad.method.traceaad_v9_13.regions import code_tokens, macro_family, mechanism_tags

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "analysis" / "traceaad_v913_proxy_audit"
TASKS = ("tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing")
BATCH = "20260814_150927"
PER_TASK = 24
DEFAULT_SEED = 913401

# Mechanism vocabulary plausibly present in generated code but NOT part of
# the frozen rules; used only to flag candidate misses for the reviewer.
EXTRA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tsp_construct": (
        "or_opt", "3_opt", "3opt", "two_opt", "2opt", "nearest_neighbor",
        "greedy", "random_restart", "simulated", "anneal", "christofides",
        "insertion", "savings", "regret", "lookahead",
    ),
    "cvrp_aco": (
        "or_opt", "two_opt", "2opt", "exchange", "relocate", "split",
        "giant_tour", "sector", "polar", "cluster", "seed_customer",
        "gini", "entropy", "bias",
    ),
    "op_aco": (
        "orienteering", "knapsack", "ratio", "greedy", "cheapest",
        "insertion", "penalty", "slack", "margin", "horizon",
    ),
    "online_bin_packing": (
        "first_fit", "best_fit", "worst_fit", "next_fit", "harmonic",
        "classifier", "quantile", "cluster", "dual", "learning",
        "anticipate", "reserve",
    ),
}


def _runs(task: str) -> list[Path]:
    root = REPO / "experiments" / task / "traceaad_v9_7"
    return sorted(root.glob(f"v9_7_{BATCH}_{task}_rep*"))


def _tag_evidence(task: str, code: str, tags: frozenset[str]) -> dict[str, list[str]]:
    """Which concrete tokens support each assigned tag (from the code text)."""

    text, names = code_tokens(code)
    evidence: dict[str, list[str]] = {}
    lowered = text.lower()
    for tag in sorted(tags):
        hits: list[str] = []
        if tag == "destination_aware":
            hits = [f"name 'destination_node' x{names['destination_node']}"]
        elif tag == "prize_distance":
            hits = [
                f"name 'prize' x{names['prize']}",
                f"name 'distance' x{names['distance']}",
            ]
        elif tag == "demand_capacity":
            hits = [
                f"name 'demands' x{names['demands']}",
                f"name 'capacity' x{names['capacity']}",
            ]
        else:
            for token in re.findall(r"[a-z_0-9]+", lowered):
                if token in (
                    "lru_cache", "remaining_mask", "bitmask", "beam_search",
                    "beam_width", "beam", "completion_cost", "temp_remaining",
                    "sampled_nn", "cost_original", "argpartition", "top_k",
                    "candidates_to_check", "dist_unvisited", "forward_cost",
                    "connectivity", "isolation", "future_cost", "remaining_count",
                    "progress", "stage_factor", "cluster_ids",
                    "sorted_customer_indices", "clarke", "saving", "savings",
                    "arctan2", "angle_diff", "angular", "radial", "dist_to_depot",
                    "depot_distance", "feasible_mask", "infeasible_mask",
                    "cap_factor", "neighbor", "cluster_bonus", "density",
                    "round_trip", "dist_j_to_depot", "remaining_budget",
                    "budget_factor", "cost_pressure", "budget_tightness",
                    "top_k_indices", "target_scores", "directional_factor",
                    "cluster_prize", "near_prizes", "forward_factor",
                    "normalized_potential", "softmax", "exp_scores",
                    "item_history", "history", "percentile", "pdf_score",
                    "probab", "likelihood", "future", "fit_score", "fragment",
                    "awkward", "closure_bonus", "gap_penalty", "tiny_threshold",
                    "diversity", "spread", "balance", "remaining_after",
                    "leftover", "remainder", "threshold",
                ):
                    if token in lowered and _rule_covers(task, tag, token):
                        hits.append(f"'{token}'")
                        break
        evidence[tag] = hits or ["<no lexical hit found>"]
    return evidence


def _rule_covers(task: str, tag: str, token: str) -> bool:
    """Coarse check that the token is one of the tag's trigger words."""

    table = {
        "tsp_construct": {
            "exact_dp": {"lru_cache", "remaining_mask", "bitmask"},
            "beam_search": {"beam_search", "beam_width", "beam"},
            "tour_rollout": {"completion_cost", "temp_remaining", "sampled_nn"},
            "two_opt": {"cost_original"},
            "candidate_screening": {"argpartition", "top_k", "candidates_to_check"},
            "one_step_lookahead": {
                "dist_unvisited", "forward_cost", "connectivity", "isolation",
                "future_cost",
            },
            "stage_adaptive": {"remaining_count", "progress", "stage_factor"},
            "destination_aware": set(),
        },
        "cvrp_aco": {
            "sweep_partition": {"cluster_ids", "sorted_customer_indices"},
            "clarke_wright_savings": {"clarke", "saving", "savings"},
            "angular_geometry": {"arctan2", "angle_diff", "angular"},
            "depot_radial": {"radial", "dist_to_depot", "depot_distance"},
            "demand_capacity": set(),
            "pair_feasibility": {"feasible_mask", "infeasible_mask", "cap_factor"},
            "neighborhood_density": {"neighbor", "cluster_bonus", "density"},
        },
        "op_aco": {
            "prize_distance": set(),
            "return_feasibility": {"round_trip", "dist_j_to_depot"},
            "budget_pressure": {
                "remaining_budget", "budget_factor", "cost_pressure",
                "budget_tightness",
            },
            "directional_targets": {"top_k_indices", "target_scores", "directional_factor"},
            "neighborhood_potential": {
                "cluster_prize", "near_prizes", "forward_factor",
                "normalized_potential",
            },
            "nonlinear_emphasis": {"softmax", "exp_scores"},
        },
        "online_bin_packing": {
            "stateful_history": {"item_history", "history"},
            "distribution_model": {"percentile", "pdf_score"},
            "future_fit": {"probab", "likelihood", "future", "fit_score"},
            "fragmentation_control": {
                "fragment", "awkward", "closure_bonus", "gap_penalty",
                "tiny_threshold",
            },
            "balance_spread": {"diversity", "spread", "balance"},
            "best_fit": {"remaining_after", "leftover", "remainder"},
            "piecewise_bands": {"threshold"},
        },
    }
    return token in table[task].get(tag, set())


def build_worksheet(seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASKS):
        rng = random.Random(seed + 31 * (task_index + 1))
        programs: list[dict[str, Any]] = []
        for run in _runs(task):
            checkpoint = json.loads(
                (run / "checkpoints" / "latest.json").read_text(encoding="utf-8")
            )
            for item in checkpoint["forest"]["programs"]:
                programs.append(
                    {
                        "task": task,
                        "source_run": run.name,
                        "program_id": int(item["id"]),
                        "code": item["code"],
                    }
                )
        # stratify by (source run, proxy region); blind: no fitness/intent/order
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for program in programs:
            tags = mechanism_tags(task, program["code"])
            program["tags"] = sorted(tags)
            program["family"] = macro_family(task, tags)
            strata[(program["source_run"], program["family"])].append(program)
        sampled: list[dict[str, Any]] = []
        leftovers: list[tuple[str, str, dict[str, Any]]] = []
        stratum_items = sorted(strata.items())
        quotas: dict[int, int] = {}
        base, extra = divmod(PER_TASK, len(stratum_items))
        for index in range(len(stratum_items)):
            quotas[index] = base + (1 if index < extra else 0)
        for index, ((run_name, family), items) in enumerate(stratum_items):
            rng.shuffle(items)
            take = min(quotas[index], len(items))
            for item in items[:take]:
                sampled.append(
                    {
                        "audit_id": f"{task}:{run_name}:p{item['program_id']}",
                        "task": task,
                        "source_run": run_name,
                        "program_id": item["program_id"],
                        "family": family,
                        "assigned_tags": item["tags"],
                        "tag_evidence": _tag_evidence(
                            task, item["code"], frozenset(item["tags"])
                        ),
                        "unmatched_keywords": sorted(
                            {
                                keyword
                                for keyword in EXTRA_KEYWORDS[task]
                                if re.search(
                                    re.escape(keyword).replace(r"\_", r"[ _]"),
                                    item["code"].lower(),
                                )
                            }
                        ),
                        "code": item["code"],
                    }
                )
            # thin strata hand their unmet quota back to the task pool
            for item in items[take:]:
                leftovers.append((run_name, family, item))
        rng.shuffle(leftovers)
        while len(sampled) + int(bool(leftovers)) <= PER_TASK and leftovers:
            run_name, family, item = leftovers.pop()
            sampled.append(
                {
                    "audit_id": f"{task}:{run_name}:p{item['program_id']}",
                    "task": task,
                    "source_run": run_name,
                    "program_id": item["program_id"],
                    "family": family,
                    "assigned_tags": item["tags"],
                    "tag_evidence": _tag_evidence(
                        task, item["code"], frozenset(item["tags"])
                    ),
                    "unmatched_keywords": sorted(
                        {
                            keyword
                            for keyword in EXTRA_KEYWORDS[task]
                            if re.search(
                                re.escape(keyword).replace(r"\_", r"[ _]"),
                                item["code"].lower(),
                            )
                        }
                    ),
                    "code": item["code"],
                }
            )
        if len(sampled) != PER_TASK:
            raise AssertionError(
                f"{task}: sampled {len(sampled)} programs, expected {PER_TASK}"
            )
        rows.extend(sampled)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    worksheet = build_worksheet(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "worksheet.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in worksheet),
        encoding="utf-8",
    )
    summary = {
        "purpose": "design 4.2 blind proxy-tag audit before formal Stage P",
        "source_batch": BATCH,
        "per_task": PER_TASK,
        "seed": args.seed,
        "blind_fields_removed": ["fitness", "q", "intent", "idea", "order"],
        "task_family_coverage": {
            task: dict(
                Counter(row["family"] for row in worksheet if row["task"] == task)
            )
            for task in TASKS
        },
        "rows_with_unmatched_keywords": sum(
            bool(row["unmatched_keywords"]) for row in worksheet
        ),
        "rows_with_missing_lexical_evidence": sum(
            any(
                any(hit.startswith("<no lexical") for hit in hits)
                for hits in row["tag_evidence"].values()
            )
            for row in worksheet
        ),
    }
    (OUT / "worksheet_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
