from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from experiments.analysis.analyze_behavesim_landscape import (
    _edge_key,
    build_behavior_knn_graph,
    build_compressed_lineage_graph,
    compute_fitness_locality,
    load_landscape_run,
    normalize_fitness,
    aggregate_repeats,
)


def _candidate(key: str, order: int, fitness: float, parent_id: int | None) -> dict:
    return {
        "key": key,
        "id": int(key),
        "order": order,
        "fitness": fitness,
        "parent_id": parent_id,
        "idea": None,
        "entry_id": None,
        "created_by": None,
    }


def test_normalize_fitness_preserves_maximize_direction() -> None:
    values = normalize_fitness([-4.0, -2.0, -1.0])
    assert values[2] > values[1] > values[0]
    assert np.isclose(np.mean(values), 0.0)
    assert np.isclose(np.std(values), 1.0)


def test_behavior_knn_graph_has_symmetric_edges_and_expected_k() -> None:
    nodes = [_candidate(str(index), index, float(index), 0) for index in range(5)]
    matrix = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 0.0, 0.1, 0.2, 0.3],
            [0.2, 0.1, 0.0, 0.1, 0.2],
            [0.3, 0.2, 0.1, 0.0, 0.1],
            [0.4, 0.3, 0.2, 0.1, 0.0],
        ]
    )
    graph, edges, k = build_behavior_knn_graph(nodes, matrix, k_rate=0.5)
    assert k == 2
    assert len(edges) == graph.number_of_edges()
    assert all(left != right for left, right in graph.edges)
    assert all(graph.has_edge(right, left) for left, right in graph.edges)
    assert all(row["distance"] > 0.0 for row in edges)


def test_compressed_lineage_connects_nearest_sampled_ancestor() -> None:
    all_candidates = [
        _candidate("1", 1, 1.0, 0),
        _candidate("2", 2, 2.0, 1),
        _candidate("3", 3, 3.0, 2),
        _candidate("4", 4, 4.0, 3),
    ]
    sampled = [all_candidates[0], all_candidates[3]]
    graph, edges = build_compressed_lineage_graph(sampled, all_candidates)
    assert set(graph.edges) == {("1", "4")}
    assert edges[0]["lineage_hops"] == 3
    assert edges[0]["parent_key"] == "1"


def test_compressed_lineage_rejects_parent_cycle() -> None:
    candidates = [_candidate("1", 1, 1.0, 2), _candidate("2", 2, 2.0, 1)]
    with pytest.raises(ValueError, match="cycle"):
        build_compressed_lineage_graph(candidates, candidates)


def test_fitness_locality_identifies_maximize_local_optimum() -> None:
    nodes = [
        _candidate("1", 1, 1.0, 0),
        _candidate("2", 2, 3.0, 0),
        _candidate("3", 3, 2.0, 0),
    ]
    matrix = np.array([[0.0, 0.2, 0.8], [0.2, 0.0, 0.3], [0.8, 0.3, 0.0]])
    graph, _, _ = build_behavior_knn_graph(nodes, matrix, k_rate=0.5)
    graph.graph["distance_matrix"] = matrix
    locality, _, _, aux = compute_fitness_locality(nodes, graph, matrix, seed=1)
    assert aux["local_optima"][0]["key"] == "2"
    assert locality["local_optima"]["count"] == 1
    assert locality["local_smoothness"]["behavior_edge_count"] == graph.number_of_edges()


def test_load_landscape_run_uses_run_root_for_checkpoint(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps({"tree": {"algorithms": []}}))
    artifact_dir = tmp_path / "combined"
    artifact_dir.mkdir()
    np.save(artifact_dir / "distance_matrix.npy", np.zeros((1, 1)))
    record = {
        "campaign": "traceaad_v916",
        "task": "tsp_construct",
        "repeat": 1,
        "run_dir": str(run_dir),
        "combined_artifact_dir": str(artifact_dir),
        "combined_candidate_sequence": [],
    }
    called: list[Path] = []

    def fake_loader(path: Path):
        called.append(path)
        return [], {}

    monkeypatch.setattr("experiments.analysis.behavesim_profiler.load_run_algorithms", fake_loader)
    with pytest.raises(ValueError, match="matrix shape"):
        load_landscape_run(record)
    assert called == []


def test_edge_key_is_order_independent() -> None:
    assert _edge_key("b", "a") == ("a", "b")


def test_aggregate_repeats_reports_task_metrics() -> None:
    base = {
        "task": "tsp_construct",
        "repeat": 1,
        "node_count": 4,
        "topology": {
            "lineage": {"connected_component_count": 2},
            "behavior": {"connected_component_count": 1},
            "semantic_regions": {"region_count": 3},
            "neighborhood_jaccard": 0.2,
            "edge_jaccard": 0.1,
            "lineage_behavior_alignment": {
                "candidate_share_in_multi_lineage_clusters": 0.5,
                "lineage_share_crossing_multiple_clusters": 0.25,
            },
        },
        "fitness_landscape": {
            "local_smoothness": {"delta_random_minus_behavior": 0.3},
            "distance_to_best": {"distance_gap_spearman": 0.4},
            "local_optima": {"share": 0.2},
        },
        "quality_regions": {"region_count": 2},
        "temporal": {
            "near_revisit_without_breakthrough_share": 0.6,
            "exact_revisit_without_breakthrough_share": 0.1,
            "large_step_without_breakthrough_share": 0.05,
            "sampled_breakthrough_share": 0.1,
        },
        "operator": {
            "explore": {"median_parent_child_distance": 0.7},
            "refine": {"median_parent_child_distance": 0.3},
        },
    }
    aggregate = aggregate_repeats([base, {**base, "repeat": 2}])
    task = aggregate["tasks"]["tsp_construct"]
    assert task["n_repeats"] == 2
    metrics = task["metrics"]
    assert metrics["semantic_region_count"]["per_repeat"] == [3, 3]
    assert metrics["explore_minus_refine_distance"]["per_repeat"] == [0.39999999999999997, 0.39999999999999997]
    assert aggregate["scope"]["best_distance_analysis"] == "sampled_best_only"
