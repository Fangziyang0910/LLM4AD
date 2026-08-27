from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from experiments.analysis import behavesim_profiler as profiler
from experiments.analysis.aggregate_behavesim_results import (
    attach_balanced_comparisons,
    combine_run,
    summarize_temporal_behavior,
)
from experiments.analysis.behavesim_batch import assert_completed_run
from llm4ad.task.optimization.cvrp_aco.evaluation import ACO as CVRPACO
from llm4ad.task.optimization.op_aco.evaluation import ACO as OPACO


def _profile(trajectories):
    return {"trajectories": trajectories}


def test_paper_distance_definition() -> None:
    assert profiler.normalized_edit_distance([0, 1], [0, 2]) == 0.5
    assert profiler.normalized_edit_distance([], []) == 0.0
    left = [[0], [0, 1]]
    right = [[0], [0, 2]]
    assert profiler.pstraj_dtw_distance(left, right) == 0.25


def test_numba_distance_matches_definition_for_prefix_trajectories() -> None:
    profiles = [
        _profile([[[0], [0, 1], [0, 1, 2]]]),
        _profile([[[0], [0, 2], [0, 2, 1]]]),
    ]
    expected = profiler.profile_distance(profiles[0], profiles[1])
    matrix = profiler.compute_distance_matrix(profiles, prefix_mode=True)
    assert matrix.shape == (2, 2)
    assert np.isclose(matrix[0, 1], expected)


def test_numba_distance_matches_definition_for_general_trajectories() -> None:
    profiles = [
        _profile([[[0, 1, 2], [0, 2, 1]]]),
        _profile([[[0, 2, 1], [0, 1, 2]]]),
    ]
    expected = profiler.profile_distance(profiles[0], profiles[1])
    matrix = profiler.compute_distance_matrix(profiles, prefix_mode=False)
    assert np.isclose(matrix[0, 1], expected)


def test_obp_probe_exposes_all_precreated_bins() -> None:
    profiler._init_worker("online_bin_packing", "A", max_points=12, timeout_seconds=30)

    def always_open_new_bin(item, valid_bins):
        return valid_bins

    trajectories, score = profiler._profile_obp(always_open_new_bin)
    assert score == -256.0
    assert len(trajectories) == 4
    final_choices = trajectories[0][-1]
    assert final_choices == list(range(256))


def test_tsp_probe_records_sampled_partial_solutions() -> None:
    profiler._init_worker("tsp_construct", "A", max_points=12, timeout_seconds=30)

    def nearest_neighbor(current_node, destination_node, unvisited_nodes, distance_matrix):
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    trajectories, score = profiler._profile_tsp(nearest_neighbor)
    assert np.isclose(score, profiler._GLOBAL_DATA["evaluator"].evaluate(nearest_neighbor))
    assert np.isfinite(score)
    assert len(trajectories) == 4
    for trajectory in trajectories:
        assert len(trajectory) == 12
        final_route = trajectory[-1]
        assert len(final_route) == 50
        assert all(final_route[: len(state)] == state for state in trajectory)


def test_op_aco_probe_matches_real_aco_incumbent() -> None:
    profiler._init_worker("op_aco", "A", max_points=2, timeout_seconds=30)
    assert profiler._GLOBAL_DATA is not None
    evaluator = profiler._GLOBAL_DATA["evaluator"]
    evaluator.n_iterations = 2
    profiler._GLOBAL_DATA["instances"] = profiler._GLOBAL_DATA["instances"][:1]

    def heuristic(prizes, distances, max_len):
        return prizes[np.newaxis, :] / np.maximum(distances, 1e-9)

    trajectories, probe_score = profiler._profile_op_aco(heuristic)
    coordinates = profiler._GLOBAL_DATA["instances"][0]
    prizes, distances, prior = evaluator._build_prior(coordinates, heuristic)
    expected = OPACO(
        prizes,
        distances,
        evaluator.max_len,
        prior,
        n_ants=evaluator.n_ants,
        rng=np.random.default_rng(evaluator.aco_seed),
    ).run(2)
    assert np.isclose(probe_score, expected)
    assert len(trajectories[0]) == 2


def test_cvrp_aco_probe_matches_real_aco_incumbent() -> None:
    profiler._init_worker("cvrp_aco", "A", max_points=2, timeout_seconds=30)
    assert profiler._GLOBAL_DATA is not None
    evaluator = profiler._GLOBAL_DATA["evaluator"]
    evaluator.n_iterations = 2
    profiler._GLOBAL_DATA["instances"] = profiler._GLOBAL_DATA["instances"][:1]

    def heuristic(distances, coordinates, demands, capacity):
        return 1.0 / np.maximum(distances, 1e-9)

    trajectories, probe_score = profiler._profile_cvrp_aco(heuristic)
    instance = profiler._GLOBAL_DATA["instances"][0]
    distances, demands, prior = evaluator._build_prior(instance, heuristic)
    expected = CVRPACO(
        distances,
        demands,
        prior,
        evaluator.capacity,
        n_ants=evaluator.n_ants,
        rng=np.random.default_rng(evaluator.aco_seed),
    ).run(2)
    assert np.isclose(probe_score, -expected)
    assert len(trajectories[0]) == 2


def test_random_candidate_is_reproducible_under_common_seed() -> None:
    profiler._init_worker("tsp_construct", "A", max_points=12, timeout_seconds=30)
    candidate = {
        "key": "random",
        "id": 1,
        "order": 1,
        "fitness": -1.0,
        "idea": None,
        "parent_id": None,
        "entry_id": None,
        "created_by": None,
        "code": """
import numpy as np
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    return int(np.random.choice(unvisited_nodes))
""",
    }
    first = profiler._profile_candidate(candidate)
    second = profiler._profile_candidate(candidate)
    assert first["ok"] and second["ok"]
    assert first["trajectories"] == second["trajectories"]
    assert first["probe_score"] == second["probe_score"]


def test_aco_panel_b_uses_validation_not_test_instances() -> None:
    for task in ("op_aco", "cvrp_aco"):
        probe = profiler._build_probe_data(task, "B")
        assert probe["probe_metadata"]["split"] == "val_50"
        assert probe["probe_metadata"]["role"] == "validation"


def test_v97_loader_preserves_reused_program_events(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True)
    rows = [
        {
            "order": 1,
            "status": "ok",
            "kind": "new",
            "program_id": 7,
            "program": "def f(): return 1",
            "child_fitness": 1.0,
        },
        {
            "order": 2,
            "status": "ok",
            "kind": "no_op",
            "program_id": 7,
            "program": "def f(): return 1",
            "child_fitness": 1.0,
        },
    ]
    (artifacts / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    candidates, _ = profiler.load_run_algorithms(run_dir)
    assert [row["key"] for row in candidates] == ["event:1", "event:2"]
    assert [row["id"] for row in candidates] == [7, 7]


def test_calm_loader_marks_final_archive_only() -> None:
    run_dir = Path(
        "experiments/其他实验/基线重跑-20260824/tsp_construct/calm/"
        "20260824_rerun_tsp_calm_rep1"
    )
    candidates, metadata = profiler.load_run_algorithms(run_dir)
    assert len(candidates) == 62
    assert metadata["population_scope"] == "final_archive_only"


def test_distance_summary_is_finite() -> None:
    matrix = np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]])
    summary = profiler.summarize_distance_matrix(matrix)
    assert summary["n"] == 3
    assert np.isfinite(summary["mean_pairwise_distance"])
    assert np.isfinite(summary["median_nearest_neighbor_distance"])
    assert len(summary["threshold_curve"]) == 19


def test_temporal_summary_separates_revisit_and_breakthrough() -> None:
    matrix = np.array(
        [
            [0.0, 0.0, 0.8],
            [0.0, 0.0, 0.8],
            [0.8, 0.8, 0.0],
        ]
    )
    candidates = [
        {"key": "a", "order": 1, "fitness": 1.0},
        {"key": "b", "order": 2, "fitness": 1.0},
        {"key": "c", "order": 3, "fitness": 2.0},
    ]
    summary, events = summarize_temporal_behavior(matrix, candidates)
    assert summary["exact_revisit_without_breakthrough_share"] == 0.5
    assert summary["sampled_breakthrough_share"] == 0.5
    assert events[0]["sampled_quality_breakthrough"] is False
    assert events[1]["sampled_quality_breakthrough"] is True


def test_completion_check_rejects_partial_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"method_params": {"max_sample_nums": 1000}}), encoding="utf-8"
    )
    summary_path = run_dir / "logs" / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "status": "finished",
                "search_aborted": False,
                "method_sample_count": 999,
            }
        ),
        encoding="utf-8",
    )
    try:
        assert_completed_run(run_dir)
    except RuntimeError as exc:
        assert "Incomplete baseline run" in str(exc)
    else:
        raise AssertionError("partial baseline was accepted")


def test_balanced_comparison_uses_common_success_count(tmp_path: Path) -> None:
    runs = []
    for repeat, matrix in enumerate(
        (
            np.array([[0.0, 0.2, 0.4], [0.2, 0.0, 0.3], [0.4, 0.3, 0.0]]),
            np.array([[0.0, 0.5], [0.5, 0.0]]),
        ),
        1,
    ):
        artifact_dir = tmp_path / f"rep{repeat}"
        artifact_dir.mkdir()
        np.save(artifact_dir / "distance_matrix.npy", matrix)
        runs.append(
            {
                "campaign": "campaign",
                "task": "task",
                "population_scope": "recorded_valid_search_candidates",
                "combined_profile_count": len(matrix),
                "combined_artifact_dir": str(artifact_dir),
                "combined_candidate_sequence": [
                    {"key": str(i), "order": i, "fitness": float(i)}
                    for i in range(len(matrix))
                ],
            }
        )
    attach_balanced_comparisons(runs)
    assert [run["metric_profile_count"] for run in runs] == [2, 2]
    assert all(
        run["comparison_scope"] == "balanced_recorded_search_candidates"
        for run in runs
    )


def test_panel_aggregation_aligns_candidates(tmp_path: Path) -> None:
    rep_dir = tmp_path / "rep1"
    base_summary = {
        "campaign": "test",
        "task": "tsp_construct",
        "label": "method",
        "repeat": 1,
        "run_dir": "/tmp/run",
        "method": "method",
        "source_format": "test",
        "population_scope": "recorded_valid_search_candidates",
        "loaded_candidate_count": 2,
        "selected_distribution_count": 2,
        "distribution_coverage": 1.0,
        "failure_counts": {},
        "operator_edges": [],
        "search_best_audit": {"fitness": 1.0, "profiled": True},
    }
    for panel, keys, distance in (
        ("panel_a", ["a", "b"], 0.2),
        ("panel_b", ["b", "a"], 0.4),
    ):
        panel_dir = rep_dir / panel
        panel_dir.mkdir(parents=True)
        summary = {**base_summary, "distribution_profile_keys": keys}
        (panel_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        profiles = [
            {
                "candidate": {
                    "key": key,
                    "order": index + 1,
                    "fitness": float(index + 1),
                },
                "roles": ["distribution"],
            }
            for index, key in enumerate(keys)
        ]
        (panel_dir / "profiles.json").write_text(json.dumps(profiles), encoding="utf-8")
        np.save(panel_dir / "distance_matrix.npy", np.array([[0.0, distance], [distance, 0.0]]))
    combined, edges = combine_run(rep_dir)
    assert edges == []
    assert combined["combined_profile_count"] == 2
    assert np.isclose(combined["distance_metrics"]["mean_pairwise_distance"], 0.3)
