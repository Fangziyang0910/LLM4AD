"""Combine two probe panels and aggregate corrected BehaveSim run statistics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from experiments.analysis.behavesim_profiler import summarize_distance_matrix

DEFAULT_ROOT = Path("experiments/_logs/behavesim_v3")
CAMPAIGNS = ("traceaad_v916", "traceaad_versions", "external_methods")
EXPECTED_COMBINED_RUNS = {
    "traceaad_v916": 15,
    "traceaad_versions": 42,
    "external_methods": 75,
}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aligned_matrix(panel_dir: Path, target_keys: list[str]) -> np.ndarray:
    summary = _load_json(panel_dir / "summary.json")
    keys = summary["distribution_profile_keys"]
    matrix = np.load(panel_dir / "distance_matrix.npy")
    index = {key: position for position, key in enumerate(keys)}
    positions = [index[key] for key in target_keys]
    return matrix[np.ix_(positions, positions)]


def _aligned_candidates(panel_dir: Path, target_keys: list[str]) -> list[dict[str, Any]]:
    profiles = _load_json(panel_dir / "profiles.json")
    by_key = {row["candidate"]["key"]: row["candidate"] for row in profiles}
    return [by_key[key] for key in target_keys]


def summarize_temporal_behavior(
    matrix: np.ndarray, candidates: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate behavioral revisits from sampled quality breakthroughs."""
    if len(matrix) != len(candidates):
        raise ValueError("candidate sequence and distance matrix size differ")
    events = []
    running_best = float(candidates[0]["fitness"])
    longest_no_breakthrough = 0
    current_no_breakthrough = 0
    for index in range(1, len(candidates)):
        fitness = float(candidates[index]["fitness"])
        novelty = float(np.min(matrix[index, :index]))
        breakthrough = fitness > running_best + 1e-12
        events.append(
            {
                "key": candidates[index]["key"],
                "order": candidates[index]["order"],
                "fitness": fitness,
                "prior_sampled_best": running_best,
                "fitness_delta_to_prior_best": fitness - running_best,
                "novelty_to_sampled_history": novelty,
                "sampled_quality_breakthrough": breakthrough,
            }
        )
        if breakthrough:
            running_best = fitness
            current_no_breakthrough = 0
        else:
            current_no_breakthrough += 1
            longest_no_breakthrough = max(longest_no_breakthrough, current_no_breakthrough)

    novelties = [row["novelty_to_sampled_history"] for row in events]
    breakthrough_novelties = [
        row["novelty_to_sampled_history"]
        for row in events
        if row["sampled_quality_breakthrough"]
    ]
    non_breakthrough_novelties = [
        row["novelty_to_sampled_history"]
        for row in events
        if not row["sampled_quality_breakthrough"]
    ]
    threshold_curve = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        below = [value <= threshold for value in novelties]
        no_gain_below = [
            value <= threshold and not row["sampled_quality_breakthrough"]
            for value, row in zip(novelties, events)
        ]
        threshold_curve.append(
            {
                "threshold": round(float(threshold), 2),
                "behavioral_revisit_share": float(np.mean(below)),
                "revisit_without_breakthrough_share": float(np.mean(no_gain_below)),
            }
        )
    summary = {
        "n_transitions": len(events),
        "mean_novelty_to_history": float(np.mean(novelties)),
        "median_novelty_to_history": float(np.median(novelties)),
        "exact_revisit_share": float(np.mean(np.asarray(novelties) <= 1e-8)),
        "exact_revisit_without_breakthrough_share": float(
            np.mean(
                [
                    value <= 1e-8 and not row["sampled_quality_breakthrough"]
                    for value, row in zip(novelties, events)
                ]
            )
        ),
        "sampled_breakthrough_count": len(breakthrough_novelties),
        "sampled_breakthrough_share": len(breakthrough_novelties) / len(events),
        "median_novelty_on_breakthrough": (
            float(np.median(breakthrough_novelties)) if breakthrough_novelties else None
        ),
        "median_novelty_without_breakthrough": (
            float(np.median(non_breakthrough_novelties))
            if non_breakthrough_novelties
            else None
        ),
        "longest_sampled_non_breakthrough_run": longest_no_breakthrough,
        "threshold_curve": threshold_curve,
    }
    return summary, events


def _combine_operator_edges(summary_a: dict, summary_b: dict) -> list[dict]:
    def index_rows(rows):
        return {
            (row["operator"], row["parent_key"], row["child_key"]): row for row in rows
        }

    rows_a = index_rows(summary_a.get("operator_edges", []))
    rows_b = index_rows(summary_b.get("operator_edges", []))
    combined = []
    for key in sorted(set(rows_a) & set(rows_b)):
        left, right = rows_a[key], rows_b[key]
        combined.append(
            {
                **left,
                "distance_panel_a": left["distance"],
                "distance_panel_b": right["distance"],
                "distance": (left["distance"] + right["distance"]) / 2.0,
            }
        )
    return combined


def combine_run(rep_dir: Path) -> tuple[dict, list[dict]]:
    panel_a = rep_dir / "panel_a"
    panel_b = rep_dir / "panel_b"
    summary_a = _load_json(panel_a / "summary.json")
    summary_b = _load_json(panel_b / "summary.json")
    common_keys = [
        key
        for key in summary_a["distribution_profile_keys"]
        if key in set(summary_b["distribution_profile_keys"])
    ]
    if len(common_keys) < 2:
        raise RuntimeError(f"Fewer than two common panel profiles under {rep_dir}")
    matrix_a = _aligned_matrix(panel_a, common_keys)
    matrix_b = _aligned_matrix(panel_b, common_keys)
    combined_matrix = (matrix_a + matrix_b) / 2.0
    metrics = summarize_distance_matrix(combined_matrix)
    candidates = _aligned_candidates(panel_a, common_keys)
    if summary_a["population_scope"] == "final_archive_only":
        temporal_behavior = None
        temporal_events = []
    else:
        temporal_behavior, temporal_events = summarize_temporal_behavior(
            combined_matrix, candidates
        )
    upper = np.triu_indices(len(common_keys), k=1)
    raw_panel_spearman = float(spearmanr(matrix_a[upper], matrix_b[upper]).statistic)
    panel_spearman = raw_panel_spearman if np.isfinite(raw_panel_spearman) else None
    operator_edges = _combine_operator_edges(summary_a, summary_b)
    operator_summary = {}
    for operator in ("explore", "refine"):
        rows = [row for row in operator_edges if row["operator"] == operator]
        distances = [row["distance"] for row in rows]
        fitness_deltas = [row["fitness_delta"] for row in rows]
        operator_summary[operator] = {
            "n": len(distances),
            "mean_parent_child_distance": float(np.mean(distances)) if distances else None,
            "median_parent_child_distance": float(np.median(distances)) if distances else None,
            "median_fitness_delta": (
                float(np.median(fitness_deltas)) if fitness_deltas else None
            ),
            "improvement_share": (
                float(np.mean(np.asarray(fitness_deltas) > 1e-12))
                if fitness_deltas
                else None
            ),
        }
    output_dir = rep_dir / "combined"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "distance_matrix.npy", combined_matrix)
    run = {
        "schema_version": 3,
        "campaign": summary_a["campaign"],
        "task": summary_a["task"],
        "label": summary_a["label"],
        "repeat": summary_a["repeat"],
        "run_dir": summary_a["run_dir"],
        "method": summary_a.get("method"),
        "source_format": summary_a["source_format"],
        "population_scope": summary_a["population_scope"],
        "loaded_candidate_count": summary_a["loaded_candidate_count"],
        "selected_distribution_count": summary_a["selected_distribution_count"],
        "combined_profile_count": len(common_keys),
        "combined_candidate_sequence": [
            {
                "key": candidate["key"],
                "order": candidate["order"],
                "fitness": candidate["fitness"],
            }
            for candidate in candidates
        ],
        "combined_artifact_dir": str(output_dir.resolve()),
        "combined_coverage": len(common_keys) / summary_a["selected_distribution_count"],
        "panel_a_coverage": summary_a["distribution_coverage"],
        "panel_b_coverage": summary_b["distribution_coverage"],
        "panel_spearman": panel_spearman,
        "failure_counts_panel_a": summary_a["failure_counts"],
        "failure_counts_panel_b": summary_b["failure_counts"],
        "distance_metrics": metrics,
        "temporal_behavior": temporal_behavior,
        "operator_summary": operator_summary,
        "search_best_fitness": summary_a["search_best_audit"]["fitness"],
        "search_best_profiled_both": (
            summary_a["search_best_audit"]["profiled"]
            and summary_b["search_best_audit"]["profiled"]
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run, handle, indent=2, ensure_ascii=False)
    with (output_dir / "operator_edges.json").open("w", encoding="utf-8") as handle:
        json.dump(operator_edges, handle, indent=2, ensure_ascii=False)
    with (output_dir / "temporal_events.json").open("w", encoding="utf-8") as handle:
        json.dump(temporal_events, handle, indent=2, ensure_ascii=False)
    return run, operator_edges


def _metric_stats(values: list[float | None]) -> dict[str, Any]:
    available = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "per_repeat": values,
        "n_available": len(available),
        "mean": float(np.mean(available)) if available else None,
        "sample_std": (
            float(np.std(available, ddof=1)) if len(available) > 1 else 0.0
        ),
        "min": float(np.min(available)) if available else None,
        "max": float(np.max(available)) if available else None,
    }


def attach_balanced_comparisons(runs: list[dict[str, Any]]) -> None:
    """Use a common successful sample count within each formal comparison unit."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        run["balanced_comparison"] = None
        run["metric_profile_count"] = run["combined_profile_count"]
        run["comparison_scope"] = run["population_scope"]
        if run["population_scope"] != "final_archive_only":
            grouped[run["task"]].append(run)

    for rows in grouped.values():
        target = min(row["combined_profile_count"] for row in rows)
        for row in rows:
            count = row["combined_profile_count"]
            indices = np.rint(np.linspace(0, count - 1, target)).astype(int)
            if len(set(indices.tolist())) != target:
                raise AssertionError("balanced sample produced duplicate indices")
            matrix = np.load(Path(row["combined_artifact_dir"]) / "distance_matrix.npy")
            balanced_matrix = matrix[np.ix_(indices, indices)]
            candidates = [
                row["combined_candidate_sequence"][int(index)] for index in indices
            ]
            temporal_behavior, _ = summarize_temporal_behavior(
                balanced_matrix, candidates
            )
            row["balanced_comparison"] = {
                "profile_count": target,
                "distance_metrics": summarize_distance_matrix(balanced_matrix),
                "temporal_behavior": temporal_behavior,
            }
            row["metric_profile_count"] = target
            row["comparison_scope"] = "balanced_recorded_search_candidates"


def _comparison_distance_metrics(row: dict[str, Any]) -> dict[str, Any]:
    balanced = row.get("balanced_comparison")
    return balanced["distance_metrics"] if balanced is not None else row["distance_metrics"]


def _comparison_temporal_behavior(row: dict[str, Any]) -> dict[str, Any] | None:
    balanced = row.get("balanced_comparison")
    return balanced["temporal_behavior"] if balanced is not None else row["temporal_behavior"]


def aggregate_groups(runs: list[dict]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for run in runs:
        grouped[(run["campaign"], run["task"], run["label"])].append(run)
    output: dict[str, Any] = {}
    metric_names = (
        "mean_pairwise_distance",
        "median_nearest_neighbor_distance",
        "exact_duplicate_share",
        "cluster_curve_auc",
    )
    for (campaign, task, label), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["repeat"])
        key = f"{campaign}/{task}/{label}"
        output[key] = {
            "campaign": campaign,
            "task": task,
            "label": label,
            "population_scope": rows[0]["population_scope"],
            "n_repeats": len(rows),
            "repeat_ids": [row["repeat"] for row in rows],
            "combined_profile_count": [row["combined_profile_count"] for row in rows],
            "comparison_profile_count": [
                row["metric_profile_count"] for row in rows
            ],
            "coverage": _metric_stats([row["combined_coverage"] for row in rows]),
            "panel_spearman": _metric_stats([row["panel_spearman"] for row in rows]),
            "metrics": {
                metric: _metric_stats(
                    [_comparison_distance_metrics(row)[metric] for row in rows]
                )
                for metric in metric_names
            },
            "temporal_behavior": (
                {
                    metric: _metric_stats(
                        [
                            _comparison_temporal_behavior(row)[metric]
                            if _comparison_temporal_behavior(row) is not None
                            else None
                            for row in rows
                        ]
                    )
                    for metric in (
                        "mean_novelty_to_history",
                        "median_novelty_to_history",
                        "exact_revisit_share",
                        "exact_revisit_without_breakthrough_share",
                        "sampled_breakthrough_share",
                        "median_novelty_on_breakthrough",
                        "median_novelty_without_breakthrough",
                        "longest_sampled_non_breakthrough_run",
                    )
                }
                if any(_comparison_temporal_behavior(row) is not None for row in rows)
                else None
            ),
            "search_best_fitness": _metric_stats(
                [row["search_best_fitness"] for row in rows]
            ),
            "threshold_curve": [
                {
                    "threshold": threshold_row["threshold"],
                    "cluster_fraction_mean": float(
                        np.mean(
                            [
                                _comparison_distance_metrics(row)["threshold_curve"][index][
                                    "cluster_fraction"
                                ]
                                for row in rows
                            ]
                        )
                    ),
                    "top1_share_mean": float(
                        np.mean(
                            [
                                _comparison_distance_metrics(row)["threshold_curve"][index][
                                    "top1_share"
                                ]
                                for row in rows
                            ]
                        )
                    ),
                }
                for index, threshold_row in enumerate(
                    _comparison_distance_metrics(rows[0])["threshold_curve"]
                )
            ],
            "operator_summary": {
                operator: {
                    "n_per_repeat": [row["operator_summary"][operator]["n"] for row in rows],
                    "median_distance_per_repeat": [
                        row["operator_summary"][operator]["median_parent_child_distance"]
                        for row in rows
                    ],
                    "median_fitness_delta_per_repeat": [
                        row["operator_summary"][operator]["median_fitness_delta"]
                        for row in rows
                    ],
                    "improvement_share_per_repeat": [
                        row["operator_summary"][operator]["improvement_share"]
                        for row in rows
                    ],
                }
                for operator in ("explore", "refine")
            },
        }
    return output


def _write_run_csv(path: Path, runs: list[dict]) -> None:
    fields = (
        "campaign",
        "task",
        "label",
        "repeat",
        "population_scope",
        "loaded_candidate_count",
        "selected_distribution_count",
        "combined_profile_count",
        "metric_profile_count",
        "comparison_scope",
        "combined_coverage",
        "panel_spearman",
        "mean_pairwise_distance",
        "median_nearest_neighbor_distance",
        "exact_duplicate_share",
        "cluster_curve_auc",
        "mean_novelty_to_history",
        "exact_revisit_without_breakthrough_share",
        "sampled_breakthrough_share",
        "search_best_fitness",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "campaign": run["campaign"],
                    "task": run["task"],
                    "label": run["label"],
                    "repeat": run["repeat"],
                    "population_scope": run["population_scope"],
                    "loaded_candidate_count": run["loaded_candidate_count"],
                    "selected_distribution_count": run["selected_distribution_count"],
                    "combined_profile_count": run["combined_profile_count"],
                    "metric_profile_count": run["metric_profile_count"],
                    "comparison_scope": run["comparison_scope"],
                    "combined_coverage": run["combined_coverage"],
                    "panel_spearman": run["panel_spearman"],
                    "mean_pairwise_distance": _comparison_distance_metrics(run)[
                        "mean_pairwise_distance"
                    ],
                    "median_nearest_neighbor_distance": _comparison_distance_metrics(
                        run
                    )["median_nearest_neighbor_distance"],
                    "exact_duplicate_share": _comparison_distance_metrics(run)[
                        "exact_duplicate_share"
                    ],
                    "cluster_curve_auc": _comparison_distance_metrics(run)[
                        "cluster_curve_auc"
                    ],
                    "mean_novelty_to_history": (
                        _comparison_temporal_behavior(run)["mean_novelty_to_history"]
                        if _comparison_temporal_behavior(run) is not None
                        else None
                    ),
                    "exact_revisit_without_breakthrough_share": (
                        _comparison_temporal_behavior(run)[
                            "exact_revisit_without_breakthrough_share"
                        ]
                        if _comparison_temporal_behavior(run) is not None
                        else None
                    ),
                    "sampled_breakthrough_share": (
                        _comparison_temporal_behavior(run)["sampled_breakthrough_share"]
                        if _comparison_temporal_behavior(run) is not None
                        else None
                    ),
                    "search_best_fitness": run["search_best_fitness"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--campaigns", nargs="*", default=CAMPAIGNS)
    args = parser.parse_args()
    runs = []
    all_operator_edges = []
    for campaign in args.campaigns:
        campaign_dir = args.root / campaign
        if not campaign_dir.exists():
            raise FileNotFoundError(campaign_dir)
        campaign_run_count = 0
        for rep_dir in sorted(campaign_dir.glob("*/*/rep*")):
            if not (rep_dir / "panel_a" / "summary.json").exists():
                continue
            if not (rep_dir / "panel_b" / "summary.json").exists():
                continue
            run, operator_edges = combine_run(rep_dir)
            runs.append(run)
            campaign_run_count += 1
            for edge in operator_edges:
                all_operator_edges.append(
                    {
                        "campaign": run["campaign"],
                        "task": run["task"],
                        "label": run["label"],
                        "repeat": run["repeat"],
                        **edge,
                    }
                )
        expected = EXPECTED_COMBINED_RUNS.get(campaign)
        if expected is not None and campaign_run_count != expected:
            raise RuntimeError(
                f"Expected {expected} complete A/B runs for {campaign}, "
                f"found {campaign_run_count}"
            )
    runs.sort(key=lambda row: (row["campaign"], row["task"], row["label"], row["repeat"]))
    attach_balanced_comparisons(runs)
    groups = aggregate_groups(runs)
    with (args.root / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"schema_version": 3, "runs": runs, "groups": groups},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    _write_run_csv(args.root / "run_metrics.csv", runs)
    if all_operator_edges:
        fields = list(all_operator_edges[0])
        with (args.root / "operator_edges.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_operator_edges)
    print(f"combined_runs={len(runs)} groups={len(groups)} root={args.root}")


if __name__ == "__main__":
    main()
