#!/usr/bin/env python3
"""Gate measurement for search-global generation information on V9.7.

Replays the official ``20260814_150927`` runs in program-creation order and
asks, for every code-novel child, whether its static mechanism proxy region
had already been visited earlier in the same run.  The numbers gate the
"global experience / cross-source operator" research line: a high revisit
share means prompts lack information the search already paid for; a low
share means there is no lever.  Static tags are the auditable proxy, not
semantic ground truth.

Usage:

    uv run python experiments/analysis/analyze_v97_region_revisit.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_v97_search_geometry import (
    TASKS,
    TASK_LABELS,
    iter_jsonl,
    macro_family,
    mechanism_tags,
    read_json,
    run_dirs,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs" / "analysis" / "traceaad_v97_region_revisit"


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def replay_run(task: str, run: Path) -> dict[str, Any]:
    checkpoint = read_json(run / "checkpoints" / "latest.json")
    forest = checkpoint["forest"]
    programs = {int(row["id"]): row for row in forest["programs"]}
    anchors = {int(row["id"]): row for row in forest["anchors"]}
    attempts = sorted(forest["attempts"], key=lambda row: int(row["order"]))

    scale = float(checkpoint["s"])
    best_program_id = int(checkpoint["best_id"])
    run_best_q = float(programs[best_program_id]["q"])

    seen_tagsets: set[frozenset[str]] = set()
    seen_families: set[str] = set()
    tag_union: set[str] = set()
    family_programs: dict[str, list[int]] = defaultdict(list)
    program_family: dict[int, str] = {}

    children: list[dict[str, Any]] = []
    family_first_entry: list[dict[str, Any]] = []
    explore_responses = 0
    search_responses = 0
    first_creator: set[int] = set()

    for attempt in attempts:
        intent = attempt.get("intent")
        stage = attempt.get("stage")
        if stage == "search":
            search_responses += 1
            if intent == "explore":
                explore_responses += 1

        program_id = attempt.get("program_id")
        if not isinstance(program_id, int):
            continue
        program_id = int(program_id)
        if program_id in first_creator:
            continue
        first_creator.add(program_id)

        raw_iteration = attempt.get("iteration")
        iteration = int(raw_iteration) if raw_iteration is not None else 0

        tags = mechanism_tags(task, programs[program_id]["code"])
        family = macro_family(task, tags)
        program_family[program_id] = family
        family_programs[family].append(program_id)

        family_is_new = family not in seen_families
        parent_anchor = attempt.get("anchor_id")
        parent_family = (
            program_family.get(int(anchors[int(parent_anchor)]["program_id"]))
            if isinstance(parent_anchor, int)
            else family
        )

        if stage == "search" and intent in {"refine", "explore"}:
            children.append(
                {
                    "intent": intent,
                    "iteration": iteration,
                    "program_id": program_id,
                    "family": family,
                    "switched": family != parent_family,
                    "no_new_tags": tags <= tag_union,
                    "tagset_revisit": tags in seen_tagsets,
                    "family_revisit": not family_is_new,
                    "q": float(programs[program_id]["q"]),
                }
            )

        if family_is_new:
            source = "init" if stage in {"root_generation", "bootstrap"} else intent
            family_first_entry.append(
                {"family": family, "source": source, "iteration": iteration}
            )
        seen_families.add(family)
        seen_tagsets.add(tags)
        tag_union |= set(tags)

    family_best = {
        family: max(float(programs[p]["q"]) for p in program_ids)
        for family, program_ids in family_programs.items()
    }
    family_count = {family: len(ids) for family, ids in family_programs.items()}

    initial_families = {
        program_family[int(anchors[int(root)]["program_id"])]
        for root in forest["root_ids"]
    }
    new_family_entries = [
        {
            "family": entry["family"],
            "source": entry["source"],
            "iteration": entry["iteration"],
            "family_best_q": family_best[entry["family"]],
            "gap_to_run_best_s": (run_best_q - family_best[entry["family"]]) / scale,
        }
        for entry in family_first_entry
        if entry["family"] not in initial_families
    ]

    route_families: dict[int, set[str]] = defaultdict(set)
    for anchor in anchors.values():
        route_families[int(anchor["root_id"])].add(
            program_family[int(anchor["program_id"])]
        )
    root_ids = [int(root) for root in forest["root_ids"]]

    route_counts: Counter[int] = Counter()
    for row in iter_jsonl(run / "artifacts" / "decisions.jsonl"):
        if row.get("event") == "route_selected":
            route_counts[int(row["selected_root_state_id"])] += 1
    top_route = route_counts.most_common(1)[0][0]
    top_route_families = route_families[top_route]
    off_route_only_families = (
        {family for root in root_ids if root != top_route for family in route_families[root]}
        - top_route_families
    )

    return {
        "run": run.name,
        "n_search_responses": search_responses,
        "n_explore_responses": explore_responses,
        "children": children,
        "distinct_tagsets": len(seen_tagsets),
        "distinct_families": len(family_programs),
        "initial_families": sorted(initial_families),
        "new_family_entries": new_family_entries,
        "family_best": family_best,
        "family_count": family_count,
        "route_families": {str(root): sorted(route_families[root]) for root in root_ids},
        "top_route": top_route,
        "off_route_only_families": sorted(off_route_only_families),
        "run_best_q": run_best_q,
        "scale": scale,
    }


def intent_stats(children: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(rows: list[dict[str, Any]], key: str) -> float | None:
        return sum(1 for row in rows if row[key]) / len(rows) if rows else None

    out: dict[str, Any] = {}
    for intent in ("explore", "refine"):
        rows = [row for row in children if row["intent"] == intent]
        switchers = [row for row in rows if row["switched"]]
        out[intent] = {
            "n_children": len(rows),
            "switch_rate": rate(rows, "switched"),
            "no_new_tags": rate(rows, "no_new_tags"),
            "tagset_revisit": rate(rows, "tagset_revisit"),
            "family_revisit_all": rate(rows, "family_revisit"),
            "family_revisit_given_switch": rate(switchers, "family_revisit"),
            "n_switchers": len(switchers),
        }
    return out


def analyze() -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task in TASKS:
        runs = run_dirs(task)
        replays = [replay_run(task, run) for run in runs]
        all_children = [child for replay in replays for child in replay["children"]]

        revisit_children = sum(
            1
            for replay in replays
            for child in replay["children"]
            if child["intent"] == "explore" and child["no_new_tags"]
        )
        total_responses = sum(replay["n_search_responses"] for replay in replays)

        entries = [entry for replay in replays for entry in replay["new_family_entries"]]
        entry_iterations = [float(entry["iteration"]) for entry in entries]
        entry_gaps = [float(entry["gap_to_run_best_s"]) for entry in entries]

        family_rows: dict[str, dict[str, Any]] = {}
        for replay in replays:
            for family, best_q in replay["family_best"].items():
                row = family_rows.setdefault(
                    family, {"best_q": [], "counts": []}
                )
                row["best_q"].append(best_q)
                row["counts"].append(replay["family_count"][family])

        per_task[task] = {
            "label": TASK_LABELS[task],
            "n_runs": len(runs),
            "intent_stats": intent_stats(all_children),
            "explore_revisit_children": revisit_children,
            "explore_revisit_budget_share": revisit_children / total_responses,
            "new_family_entries_per_run": [
                len(replay["new_family_entries"]) for replay in replays
            ],
            "new_family_entry_source": dict(Counter(e["source"] for e in entries)),
            "new_family_entry_iteration": {
                "median": statistics.median(entry_iterations) if entry_iterations else None,
                "q25": quantile(entry_iterations, 0.25),
                "q75": quantile(entry_iterations, 0.75),
                "max": max(entry_iterations) if entry_iterations else None,
            },
            "new_family_gap_to_run_best_s": {
                "median": statistics.median(entry_gaps) if entry_gaps else None,
                "min": min(entry_gaps) if entry_gaps else None,
                "within_2s": sum(1 for gap in entry_gaps if gap <= 2.0) / len(entry_gaps)
                if entry_gaps
                else None,
            },
            "families": {
                family: {
                    "runs": len(row["best_q"]),
                    "median_best_q": statistics.median(row["best_q"]),
                    "median_programs": statistics.median(row["counts"]),
                }
                for family, row in sorted(
                    family_rows.items(), key=lambda item: -statistics.median(item[1]["best_q"])
                )
            },
            "distinct_tagsets_per_run": [replay["distinct_tagsets"] for replay in replays],
            "distinct_families_per_run": [replay["distinct_families"] for replay in replays],
            "off_route_only_families_per_run": [
                {
                    "n": len(replay["off_route_only_families"]),
                    "gap_s": [
                        (replay["run_best_q"] - replay["family_best"][family]) / replay["scale"]
                        for family in replay["off_route_only_families"]
                    ],
                }
                for replay in replays
            ],
            "runs": [
                {key: replay[key] for key in (
                    "run",
                    "n_search_responses",
                    "n_explore_responses",
                    "initial_families",
                    "top_route",
                    "off_route_only_families",
                )}
                for replay in replays
            ],
        }
    return per_task


def main() -> None:
    per_task = analyze()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "summary.json"
    path.write_text(json.dumps(per_task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(REPO)}\n")

    print(
        f"{'task':<6}{'exChild':>8}{'noNewTag':>9}{'switch':>8}"
        f"{'revst|sw':>9}{'budget':>8}{'newFam':>8}{'gapS':>7}"
    )
    for task, block in per_task.items():
        explore = block["intent_stats"]["explore"]
        refine = block["intent_stats"]["refine"]
        gaps = block["new_family_gap_to_run_best_s"]
        revisit_or_dash = (
            f"{explore['family_revisit_given_switch']:.1%}"
            if explore["family_revisit_given_switch"] is not None
            else "-"
        )
        print(
            f"{block['label']:<6}"
            f"{explore['n_children']:>8}"
            f"{explore['no_new_tags']:>9.1%}"
            f"{explore['switch_rate']:>8.1%}"
            f"{revisit_or_dash:>9}"
            f"{block['explore_revisit_budget_share']:>8.1%}"
            f"{statistics.mean(block['new_family_entries_per_run']):>8.1f}"
            f"{gaps['median'] if gaps['median'] is not None else float('nan'):>7.1f}"
        )
        refine_revisit_or_dash = (
            f"{refine['family_revisit_given_switch']:.1%}"
            if refine["family_revisit_given_switch"] is not None
            else "-"
        )
        print(
            f"{'  refine ref:':<14}"
            f"n={refine['n_children']}"
            f" noNewTag={refine['no_new_tags']:.1%}"
            f" switch={refine['switch_rate']:.1%}"
            f" revst|sw={refine_revisit_or_dash}"
        )


if __name__ == "__main__":
    main()
