"""Construct and inspect BehaveSim semantic fitness landscapes offline.

The analyzer reuses the exact candidate nodes, fitness values, and combined
BehaveSim matrices produced by ``behavesim_v3``. It does not profile programs,
call an LLM, or consume evaluator budget.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.manifold import MDS

from experiments.analysis.behavesim_profiler import TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATE = REPO_ROOT / "experiments/_logs/behavesim_v3/aggregate.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/_logs/behavesim_landscape"
CAMPAIGN = "traceaad_v916"
SCHEMA_VERSION = 1
BEHAVIOR_K_RATES = (0.025, 0.05, 0.10)
TOP_QUALITY_FRACTION = 0.10
EXACT_DISTANCE_EPSILON = 1e-8
FITNESS_EPSILON = 1e-12


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _id(value: Any) -> str:
    return str(value)


def _is_root_parent(value: Any) -> bool:
    return value is None or _id(value) == "0"


def normalize_fitness(values: Sequence[float]) -> np.ndarray:
    """Standardize within one run while preserving the maximize direction."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("fitness values must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("fitness values must be finite")
    scale = float(np.std(array))
    if scale == 0.0:
        return np.zeros_like(array)
    return (array - float(np.mean(array))) / scale


def _edge_key(left: str, right: str) -> tuple[str, str]:
    return tuple(sorted((str(left), str(right))))


def _node_by_key(nodes: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(node["key"]): node for node in nodes}
    if len(result) != len(nodes):
        raise ValueError("candidate keys must be unique")
    return result


def _all_candidate_by_id(candidates: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {_id(candidate["id"]): candidate for candidate in candidates}
    if len(result) != len(candidates):
        raise ValueError("candidate ids must be unique")
    return result


def _lineage_root(
    candidate: dict[str, Any], by_id: dict[str, dict[str, Any]]
) -> str:
    current = candidate
    seen: set[str] = set()
    while not _is_root_parent(current.get("parent_id")):
        current_key = _id(current["id"])
        if current_key in seen:
            raise ValueError(f"parent cycle detected at candidate {current_key}")
        seen.add(current_key)
        parent = by_id.get(_id(current["parent_id"]))
        if parent is None:
            raise ValueError(
                f"missing parent {current['parent_id']} for candidate {current['id']}"
            )
        current = parent
    return _id(current["id"])


def build_compressed_lineage_graph(
    sampled_nodes: Sequence[dict[str, Any]],
    all_candidates: Sequence[dict[str, Any]],
) -> tuple[nx.Graph, list[dict[str, Any]]]:
    """Connect each sampled node to its nearest sampled ancestor."""
    nodes = _node_by_key(sampled_nodes)
    by_id = _all_candidate_by_id(all_candidates)
    for candidate in all_candidates:
        _lineage_root(candidate, by_id)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    rows: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for child in sampled_nodes:
        current = child
        hops = 0
        while not _is_root_parent(current.get("parent_id")):
            parent = by_id.get(_id(current["parent_id"]))
            if parent is None:
                raise ValueError(
                    f"missing parent {current['parent_id']} for candidate {current['id']}"
                )
            hops += 1
            parent_key = _id(parent["id"])
            if parent_key in nodes:
                child_key = str(child["key"])
                edge_key = _edge_key(parent_key, child_key)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    row = {
                        "parent_key": parent_key,
                        "child_key": child_key,
                        "source_key": parent_key,
                        "target_key": child_key,
                        "lineage_hops": hops,
                        "distance": None,
                        "fitness_difference": abs(
                            float(parent["fitness"]) - float(child["fitness"])
                        ),
                        "order_difference": int(child["order"]) - int(parent["order"]),
                    }
                    rows.append(row)
                    graph.add_edge(parent_key, child_key, **row)
                break
            current = parent
    rows.sort(key=lambda row: (int(nodes[row["child_key"]]["order"]), row["child_key"]))
    return graph, rows


def build_behavior_knn_graph(
    nodes: Sequence[dict[str, Any]],
    distance_matrix: np.ndarray,
    k_rate: float = 0.05,
) -> tuple[nx.Graph, list[dict[str, Any]], int]:
    """Build the undirected union of each candidate's k nearest neighbors."""
    matrix = np.asarray(distance_matrix, dtype=float)
    n = len(nodes)
    if matrix.shape != (n, n):
        raise ValueError("distance matrix shape does not match candidate count")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("distance matrix must be finite")
    if np.any(np.diag(matrix) != 0.0):
        raise ValueError("distance matrix diagonal must be zero")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-6):
        raise ValueError("distance matrix must be symmetric")
    if not 0.0 < k_rate:
        raise ValueError("k_rate must be positive")
    k = min(max(2, int(math.ceil(k_rate * max(0, n - 1))),), max(0, n - 1))
    graph = nx.Graph()
    graph.add_nodes_from(str(node["key"]) for node in nodes)
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for index, node in enumerate(nodes):
        order = np.argsort(matrix[index], kind="stable")
        neighbors = [int(position) for position in order if int(position) != index][:k]
        for neighbor_index in neighbors:
            neighbor = nodes[neighbor_index]
            left_key, right_key = str(node["key"]), str(neighbor["key"])
            pair = _edge_key(left_key, right_key)
            if pair in rows_by_key:
                continue
            left_index = min(index, neighbor_index)
            right_index = max(index, neighbor_index)
            row = {
                "source_key": left_key,
                "target_key": right_key,
                "distance": float(matrix[index, neighbor_index]),
                "fitness_difference": abs(
                    float(node["fitness"]) - float(neighbor["fitness"])
                ),
                "order_difference": int(node["order"]) - int(neighbor["order"]),
                "is_lineage_edge": False,
                "left_index": left_index,
                "right_index": right_index,
            }
            rows_by_key[pair] = row
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (row["distance"], row["source_key"], row["target_key"]))
    for row in rows:
        graph.add_edge(row["source_key"], row["target_key"], **row)
    return graph, rows, k


def _graph_density(graph: nx.Graph) -> float:
    return float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0


def _average_neighborhood_jaccard(left: nx.Graph, right: nx.Graph) -> float:
    keys = sorted(set(left.nodes) | set(right.nodes))
    values = []
    for key in keys:
        left_neighbors = set(left.neighbors(key))
        right_neighbors = set(right.neighbors(key))
        union = left_neighbors | right_neighbors
        values.append(1.0 if not union else len(left_neighbors & right_neighbors) / len(union))
    return float(np.mean(values)) if values else 1.0


def _edge_jaccard(left: nx.Graph, right: nx.Graph) -> float:
    left_edges = {_edge_key(*edge) for edge in left.edges}
    right_edges = {_edge_key(*edge) for edge in right.edges}
    union = left_edges | right_edges
    return 1.0 if not union else len(left_edges & right_edges) / len(union)


def _component_labels(graph: nx.Graph, nodes: Sequence[dict[str, Any]]) -> dict[str, int]:
    components = sorted(
        (sorted(component) for component in nx.connected_components(graph)),
        key=lambda component: min(int(_node_by_key(nodes)[key]["order"]) for key in component),
    )
    return {
        key: component_index
        for component_index, component in enumerate(components)
        for key in component
    }


def _threshold_components(
    nodes: Sequence[dict[str, Any]], matrix: np.ndarray, threshold: float
) -> tuple[nx.Graph, dict[str, int]]:
    graph = nx.Graph()
    keys = [str(node["key"]) for node in nodes]
    graph.add_nodes_from(keys)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if float(matrix[i, j]) <= threshold:
                graph.add_edge(keys[i], keys[j], distance=float(matrix[i, j]))
    return graph, _component_labels(graph, nodes)


def _cluster_curve(matrix: np.ndarray) -> list[dict[str, Any]]:
    n = len(matrix)
    if n < 2:
        return []
    tree = linkage(squareform(matrix, checks=False), method="average")
    rows = []
    for threshold in np.arange(0.05, 0.96, 0.05):
        labels = fcluster(tree, t=float(threshold), criterion="distance")
        counts = np.bincount(labels)[1:]
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "n_clusters": int(len(counts)),
                "cluster_fraction": float(len(counts) / n),
                "top1_share": float(np.max(counts) / n),
            }
        )
    return rows


def _pairwise_upper(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.triu_indices(len(matrix), k=1)
    return rows, cols, np.asarray(matrix[rows, cols], dtype=float)


def _finite_spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    result = spearmanr(left, right).statistic
    return float(result) if np.isfinite(result) else None


def _sample_non_neighbor_pairs(
    n: int, behavior_edges: Iterable[dict[str, Any]], sample_size: int, seed: int
) -> list[tuple[int, int]]:
    blocked = {
        tuple(sorted((int(row["left_index"]), int(row["right_index"]))))
        for row in behavior_edges
    }
    candidates = [
        (left, right)
        for left in range(n)
        for right in range(left + 1, n)
        if (left, right) not in blocked
    ]
    if sample_size >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=sample_size, replace=False)
    return [candidates[int(index)] for index in sorted(selected.tolist())]


def _distance_bins(
    distances: np.ndarray, fitness_differences: np.ndarray, bins: int = 10
) -> list[dict[str, Any]]:
    if not len(distances):
        return []
    edges = np.unique(np.quantile(distances, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return [
            {
                "lower": float(edges[0]),
                "upper": float(edges[0]),
                "count": int(len(distances)),
                "mean_abs_fitness_difference": float(np.mean(fitness_differences)),
                "median_abs_fitness_difference": float(np.median(fitness_differences)),
            }
        ]
    labels = np.digitize(distances, edges[1:-1], right=True)
    result = []
    for index in range(len(edges) - 1):
        values = fitness_differences[labels == index]
        if not len(values):
            continue
        result.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(len(values)),
                "mean_abs_fitness_difference": float(np.mean(values)),
                "median_abs_fitness_difference": float(np.median(values)),
            }
        )
    return result


def _lineage_cluster_metrics(
    nodes: Sequence[dict[str, Any]], cluster_labels: dict[str, int]
) -> dict[str, Any]:
    by_key = _node_by_key(nodes)
    cluster_roots: dict[int, set[str]] = defaultdict(set)
    root_clusters: dict[str, set[int]] = defaultdict(set)
    for key, cluster in cluster_labels.items():
        root = str(by_key[key]["lineage_root_id"])
        cluster_roots[cluster].add(root)
        root_clusters[root].add(cluster)
    multi_clusters = [cluster for cluster, roots in cluster_roots.items() if len(roots) > 1]
    return {
        "cluster_count": len(cluster_roots),
        "clusters_with_multiple_lineages": len(multi_clusters),
        "candidate_share_in_multi_lineage_clusters": float(
            np.mean([cluster_labels[key] in multi_clusters for key in cluster_labels])
        )
        if cluster_labels
        else 0.0,
        "lineages": {
            root: {
                "semantic_cluster_count": len(clusters),
                "semantic_clusters": sorted(clusters),
            }
            for root, clusters in sorted(root_clusters.items())
        },
        "mean_clusters_per_lineage": (
            float(np.mean([len(clusters) for clusters in root_clusters.values()]))
            if root_clusters
            else 0.0
        ),
        "lineage_share_crossing_multiple_clusters": (
            float(np.mean([len(clusters) > 1 for clusters in root_clusters.values()]))
            if root_clusters
            else 0.0
        ),
    }


def _quality_regions(
    nodes: Sequence[dict[str, Any]], behavior_graph: nx.Graph, z_values: np.ndarray
) -> dict[str, Any]:
    count = max(1, int(math.ceil(TOP_QUALITY_FRACTION * len(nodes))))
    top_indices = np.argsort(-z_values, kind="stable")[:count]
    top_keys = {str(nodes[index]["key"]) for index in top_indices}
    subgraph = behavior_graph.subgraph(top_keys).copy()
    components = sorted(
        (sorted(component) for component in nx.connected_components(subgraph)),
        key=lambda component: min(int(_node_by_key(nodes)[key]["order"]) for key in component),
    )
    region_distances = []
    index_by_key = {str(node["key"]): index for index, node in enumerate(nodes)}
    for left_index, left_component in enumerate(components):
        for right_component in components[left_index + 1 :]:
            values = [
                float(
                    behavior_graph.graph["distance_matrix"][index_by_key[left], index_by_key[right]]
                )
                for left in left_component
                for right in right_component
            ]
            if values:
                region_distances.append(float(min(values)))
    return {
        "top_fraction": TOP_QUALITY_FRACTION,
        "top_count": count,
        "top_keys": sorted(top_keys, key=lambda key: int(_node_by_key(nodes)[key]["order"])),
        "region_count": len(components),
        "region_sizes": [len(component) for component in components],
        "minimum_between_region_distances": region_distances,
        "regions": components,
    }


def _temporal_events(
    nodes: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    region_labels: dict[str, int],
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = len(nodes)
    _, _, all_distances = _pairwise_upper(matrix)
    if len(all_distances):
        small_threshold = float(np.quantile(all_distances, 0.25))
        large_threshold = float(np.quantile(all_distances, 0.75))
    else:
        small_threshold = large_threshold = 0.0
    events: list[dict[str, Any]] = []
    running_best = float(nodes[0]["fitness"])
    seen_regions: set[int] = {region_labels[str(nodes[0]["key"])]} if nodes else set()
    longest_non_breakthrough = 0
    current_non_breakthrough = 0
    for index, node in enumerate(nodes):
        key = str(node["key"])
        if index == 0:
            novelty = None
            breakthrough = True
            prior_best = None
            fitness_delta = None
            exact_revisit = False
            near_revisit = False
        else:
            novelty = float(np.min(matrix[index, :index]))
            prior_best = running_best
            fitness_delta = float(node["fitness"]) - running_best
            breakthrough = fitness_delta > FITNESS_EPSILON
            exact_revisit = novelty <= EXACT_DISTANCE_EPSILON
            near_revisit = novelty <= small_threshold
            if breakthrough:
                running_best = float(node["fitness"])
            current_non_breakthrough = 0 if breakthrough else current_non_breakthrough + 1
            longest_non_breakthrough = max(longest_non_breakthrough, current_non_breakthrough)
            seen_regions.add(region_labels[key])
        if index == 0:
            current_non_breakthrough = 0
        if novelty is None:
            step_class = "initial"
        elif novelty <= small_threshold:
            step_class = "small"
        elif novelty >= large_threshold:
            step_class = "large"
        else:
            step_class = "medium"
        if index == 0:
            quality_change = "initial"
            category = "initial"
        elif breakthrough and step_class == "small":
            quality_change = "improvement"
            category = "small_improvement"
        elif not breakthrough and step_class == "small":
            quality_change = "no_improvement"
            category = "small_no_improvement"
        elif breakthrough and step_class == "large":
            quality_change = "improvement"
            category = "large_improvement"
        elif not breakthrough and step_class == "large":
            quality_change = "no_improvement"
            category = "large_no_improvement"
        else:
            quality_change = "improvement" if breakthrough else "no_improvement"
            category = f"{step_class}_{quality_change}"
        events.append(
            {
                "key": key,
                "order": int(node["order"]),
                "fitness": float(node["fitness"]),
                "prior_sampled_best": prior_best,
                "fitness_delta_to_prior_best": fitness_delta,
                "novelty_to_sampled_history": novelty,
                "sampled_quality_breakthrough": breakthrough,
                "exact_behavior_revisit": exact_revisit,
                "near_behavior_revisit": near_revisit,
                "behavior_step_class": step_class,
                "quality_change": quality_change,
                "event_category": category,
                "cumulative_semantic_region_count": len(seen_regions),
            }
        )
    novelty_values = [
        float(event["novelty_to_sampled_history"])
        for event in events
        if event["novelty_to_sampled_history"] is not None
    ]
    breakthrough_novelty = [
        float(event["novelty_to_sampled_history"])
        for event in events
        if event["sampled_quality_breakthrough"] and event["novelty_to_sampled_history"] is not None
    ]
    non_breakthrough_novelty = [
        float(event["novelty_to_sampled_history"])
        for event in events
        if not event["sampled_quality_breakthrough"] and event["novelty_to_sampled_history"] is not None
    ]
    summary = {
        "n_transitions": max(0, n - 1),
        "small_step_threshold_q25": small_threshold,
        "large_step_threshold_q75": large_threshold,
        "mean_novelty_to_history": float(np.mean(novelty_values)) if novelty_values else None,
        "median_novelty_to_history": float(np.median(novelty_values)) if novelty_values else None,
        "exact_revisit_share": float(
            np.mean([event["exact_behavior_revisit"] for event in events[1:]])
        )
        if n > 1
        else None,
        "near_revisit_share": float(
            np.mean([event["near_behavior_revisit"] for event in events[1:]])
        )
        if n > 1
        else None,
        "exact_revisit_without_breakthrough_share": float(
            np.mean(
                [
                    event["exact_behavior_revisit"]
                    and not event["sampled_quality_breakthrough"]
                    for event in events[1:]
                ]
            )
        )
        if n > 1
        else None,
        "near_revisit_without_breakthrough_share": float(
            np.mean(
                [
                    event["near_behavior_revisit"]
                    and not event["sampled_quality_breakthrough"]
                    for event in events[1:]
                ]
            )
        )
        if n > 1
        else None,
        "large_step_without_breakthrough_share": float(
            np.mean(
                [
                    event["behavior_step_class"] == "large"
                    and not event["sampled_quality_breakthrough"]
                    for event in events[1:]
                ]
            )
        )
        if n > 1
        else None,
        "sampled_breakthrough_count": int(
            sum(event["sampled_quality_breakthrough"] for event in events[1:])
        ),
        "sampled_breakthrough_share": (
            sum(event["sampled_quality_breakthrough"] for event in events[1:]) / (n - 1)
            if n > 1
            else None
        ),
        "median_novelty_on_breakthrough": (
            float(np.median(breakthrough_novelty)) if breakthrough_novelty else None
        ),
        "median_novelty_without_breakthrough": (
            float(np.median(non_breakthrough_novelty)) if non_breakthrough_novelty else None
        ),
        "longest_sampled_non_breakthrough_run": longest_non_breakthrough,
        "event_category_counts": dict(
            sorted(
                {
                    category: sum(event["event_category"] == category for event in events)
                    for category in {event["event_category"] for event in events}
                }.items()
            )
        ),
    }
    return summary, events


def _operator_summary(combined_dir: Path) -> dict[str, Any]:
    path = combined_dir / "operator_edges.json"
    if not path.exists():
        return {}
    rows = _load_json(path)
    output: dict[str, Any] = {}
    for operator in ("explore", "refine"):
        values = [row for row in rows if str(row.get("operator", "")).lower() == operator]
        distances = [float(row["distance"]) for row in values if row.get("distance") is not None]
        deltas = [float(row["fitness_delta"]) for row in values if row.get("fitness_delta") is not None]
        output[operator] = {
            "n": len(values),
            "mean_parent_child_distance": float(np.mean(distances)) if distances else None,
            "median_parent_child_distance": float(np.median(distances)) if distances else None,
            "median_fitness_delta": float(np.median(deltas)) if deltas else None,
            "improvement_share": float(np.mean(np.asarray(deltas) > FITNESS_EPSILON)) if deltas else None,
            "rows": values,
        }
    return output


def _behavior_graph_rate_metrics(
    nodes: Sequence[dict[str, Any]],
    matrix: np.ndarray,
    k_rate: float,
) -> dict[str, Any]:
    graph, edges, k = build_behavior_knn_graph(nodes, matrix, k_rate)
    return {
        "k_rate": k_rate,
        "k": k,
        "edge_count": graph.number_of_edges(),
        "connected_component_count": nx.number_connected_components(graph),
        "average_clustering": float(nx.average_clustering(graph)),
        "density": _graph_density(graph),
        "edges": [
            {
                "source_key": row["source_key"],
                "target_key": row["target_key"],
                "distance": row["distance"],
            }
            for row in edges
        ],
    }


def _load_checkpoint_candidates(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from experiments.analysis.behavesim_profiler import load_run_algorithms

    return load_run_algorithms(run_dir)


def load_landscape_run(run_record: dict[str, Any]) -> dict[str, Any]:
    """Load and strictly align one combined run with its checkpoint."""
    if run_record.get("campaign") != CAMPAIGN:
        raise ValueError(f"expected campaign {CAMPAIGN}, got {run_record.get('campaign')}")
    sequence = list(run_record["combined_candidate_sequence"])
    artifact_dir = Path(run_record["combined_artifact_dir"])
    matrix = np.asarray(np.load(artifact_dir / "distance_matrix.npy"), dtype=float)
    if matrix.shape != (len(sequence), len(sequence)):
        raise ValueError("combined matrix shape and candidate sequence differ")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-6):
        raise ValueError("combined distance matrix is not symmetric")
    checkpoint = Path(run_record["run_dir"]) / "checkpoints" / "latest.json"
    all_candidates, loader_metadata = _load_checkpoint_candidates(
        Path(run_record["run_dir"])
    )
    if not all_candidates:
        raise ValueError("checkpoint contains no valid candidates")
    run_search_best = max(all_candidates, key=lambda candidate: float(candidate["fitness"]))
    run_search_best_record = {
        "key": str(run_search_best["key"]),
        "id": run_search_best["id"],
        "order": int(run_search_best["order"]),
        "fitness": float(run_search_best["fitness"]),
        "profiled_in_combined": any(
            str(row["key"]) == str(run_search_best["key"]) for row in sequence
        ),
    }
    by_id = _all_candidate_by_id(all_candidates)
    nodes = []
    for sequence_row in sequence:
        key = str(sequence_row["key"])
        candidate = by_id.get(key)
        if candidate is None:
            raise ValueError(f"combined candidate {key} missing from checkpoint")
        if not np.isclose(
            float(sequence_row["fitness"]), float(candidate["fitness"]), rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"fitness mismatch for candidate {key}")
        nodes.append(
            {
                "key": key,
                "id": candidate["id"],
                "order": int(sequence_row["order"]),
                "fitness": float(sequence_row["fitness"]),
                "idea": candidate.get("idea"),
                "parent_id": candidate.get("parent_id"),
                "entry_id": candidate.get("entry_id"),
                "created_by": candidate.get("created_by"),
                "lineage_root_id": _lineage_root(candidate, by_id),
            }
        )
    if len(_node_by_key(nodes)) != len(nodes):
        raise ValueError("combined candidate sequence contains duplicate keys")
    if [node["key"] for node in nodes] != [str(row["key"]) for row in sequence]:
        raise ValueError("candidate order changed during alignment")
    return {
        "record": run_record,
        "nodes": nodes,
        "all_candidates": all_candidates,
        "loader_metadata": loader_metadata,
        "distance_matrix": matrix,
        "artifact_dir": artifact_dir,
        "run_search_best": run_search_best_record,
    }


def compute_topology_metrics(
    nodes: Sequence[dict[str, Any]],
    lineage_graph: nx.Graph,
    behavior_graph: nx.Graph,
    semantic_region_labels: dict[str, int],
    semantic_region_graph: nx.Graph,
) -> dict[str, Any]:
    behavior_labels = _component_labels(behavior_graph, nodes)
    lineage_metrics = _lineage_cluster_metrics(nodes, semantic_region_labels)
    return {
        "lineage": {
            "node_count": lineage_graph.number_of_nodes(),
            "edge_count": lineage_graph.number_of_edges(),
            "connected_component_count": nx.number_connected_components(lineage_graph),
            "average_clustering": float(nx.average_clustering(lineage_graph)),
            "density": _graph_density(lineage_graph),
        },
        "behavior": {
            "node_count": behavior_graph.number_of_nodes(),
            "edge_count": behavior_graph.number_of_edges(),
            "connected_component_count": nx.number_connected_components(behavior_graph),
            "average_clustering": float(nx.average_clustering(behavior_graph)),
            "density": _graph_density(behavior_graph),
        },
        "neighborhood_jaccard": _average_neighborhood_jaccard(lineage_graph, behavior_graph),
        "edge_jaccard": _edge_jaccard(lineage_graph, behavior_graph),
        "behavior_cluster_labels": behavior_labels,
        "semantic_regions": {
            "node_count": semantic_region_graph.number_of_nodes(),
            "edge_count": semantic_region_graph.number_of_edges(),
            "region_count": nx.number_connected_components(semantic_region_graph),
        },
        "lineage_behavior_alignment": lineage_metrics,
    }


def compute_fitness_locality(
    nodes: Sequence[dict[str, Any]], behavior_graph: nx.Graph, matrix: np.ndarray, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    z_values = normalize_fitness([node["fitness"] for node in nodes])
    _, _, pair_distances = _pairwise_upper(matrix)
    _, _, pair_z_differences = _pairwise_upper(np.abs(z_values[:, None] - z_values[None, :]))
    behavior_edge_rows = [dict(behavior_graph.edges[edge]) for edge in behavior_graph.edges]
    behavior_indices = [
        (int(row["left_index"]), int(row["right_index"])) for row in behavior_edge_rows
    ]
    behavior_z_differences = [
        abs(float(z_values[left]) - float(z_values[right])) for left, right in behavior_indices
    ]
    random_pairs = _sample_non_neighbor_pairs(
        len(nodes), behavior_edge_rows, len(behavior_edge_rows), seed
    )
    random_z_differences = [
        abs(float(z_values[left]) - float(z_values[right])) for left, right in random_pairs
    ]
    local_smoothness = {
        "behavior_neighbor_mean_abs_z_difference": (
            float(np.mean(behavior_z_differences)) if behavior_z_differences else None
        ),
        "non_neighbor_random_mean_abs_z_difference": (
            float(np.mean(random_z_differences)) if random_z_differences else None
        ),
        "delta_random_minus_behavior": (
            float(np.mean(random_z_differences) - np.mean(behavior_z_differences))
            if behavior_z_differences and random_z_differences
            else None
        ),
        "behavior_edge_count": len(behavior_z_differences),
        "random_pair_count": len(random_z_differences),
        "random_seed": seed,
    }
    best_index = int(np.argmax(z_values))
    best_key = str(nodes[best_index]["key"])
    best_fitness = float(nodes[best_index]["fitness"])
    distance_to_best = []
    for index, node in enumerate(nodes):
        distance_to_best.append(
            {
                "key": str(node["key"]),
                "order": int(node["order"]),
                "fitness": float(node["fitness"]),
                "fitness_std": float(z_values[index]),
                "fitness_gap_to_best": float(best_fitness - node["fitness"]),
                "fitness_std_gap_to_best": float(z_values[best_index] - z_values[index]),
                "distance_to_best": float(matrix[index, best_index]),
            }
        )
    gap_values = np.asarray([row["fitness_std_gap_to_best"] for row in distance_to_best])
    best_distance_values = np.asarray([row["distance_to_best"] for row in distance_to_best])
    distance_best_summary = {
        "best_key": best_key,
        "best_order": int(nodes[best_index]["order"]),
        "best_fitness": best_fitness,
        "best_is_search_best": best_key == str(
            behavior_graph.graph.get("search_best_key")
        ),
        "distance_gap_spearman": _finite_spearman(best_distance_values, gap_values),
        "distance_to_best_mean": float(np.mean(best_distance_values)),
        "distance_to_best_median": float(np.median(best_distance_values)),
    }
    local_optima = []
    behavior_labels = _component_labels(behavior_graph, nodes)
    node_by_key = _node_by_key(nodes)
    index_by_key = {str(node["key"]): index for index, node in enumerate(nodes)}
    for index, node in enumerate(nodes):
        neighbors = list(behavior_graph.neighbors(str(node["key"])))
        is_local = not any(
            z_values[index_by_key[neighbor]] > z_values[index] + FITNESS_EPSILON
            for neighbor in neighbors
        )
        if is_local:
            local_optima.append(
                {
                    "key": str(node["key"]),
                    "order": int(node["order"]),
                    "fitness": float(node["fitness"]),
                    "fitness_std": float(z_values[index]),
                    "behavior_cluster": behavior_labels[str(node["key"])],
                    "neighbor_count": len(neighbors),
                    "lineage_root_id": str(node.get("lineage_root_id", node["key"])),
                }
            )
    local_pairs = []
    for left_index, left in enumerate(local_optima):
        for right in local_optima[left_index + 1 :]:
            local_pairs.append(
                float(
                        matrix[
                        index_by_key[left["key"]],
                        index_by_key[right["key"]],
                    ]
                )
            )
    local_summary = {
        "count": len(local_optima),
        "share": len(local_optima) / len(nodes) if nodes else None,
        "pairwise_distance_mean": float(np.mean(local_pairs)) if local_pairs else None,
        "pairwise_distance_min": float(np.min(local_pairs)) if local_pairs else None,
        "pairwise_distance_max": float(np.max(local_pairs)) if local_pairs else None,
    }
    bins = _distance_bins(pair_distances, pair_z_differences)
    locality = {
        "fitness_standardization": {
            "direction": "maximize",
            "mean": float(np.mean([node["fitness"] for node in nodes])),
            "std": float(np.std([node["fitness"] for node in nodes])),
        },
        "local_smoothness": local_smoothness,
        "distance_to_best": distance_best_summary,
        "fitness_distance_bins": bins,
        "distance_abs_fitness_difference_spearman": _finite_spearman(
            pair_distances, pair_z_differences
        ),
        "local_optima": local_summary,
        "z_values": {str(node["key"]): float(z_values[index]) for index, node in enumerate(nodes)},
    }
    return locality, distance_to_best, behavior_labels, {
        "local_optima": local_optima,
        "z_values": z_values,
        "best_index": best_index,
    }


def analyze_landscape_run(run_data: dict[str, Any], seed: int = 730241) -> dict[str, Any]:
    nodes = [dict(node) for node in run_data["nodes"]]
    matrix = np.asarray(run_data["distance_matrix"], dtype=float)
    for index, node in enumerate(nodes):
        node["order_index"] = index
    lineage_graph, lineage_edges = build_compressed_lineage_graph(
        nodes, run_data["all_candidates"]
    )
    behavior_graph, behavior_edges, k = build_behavior_knn_graph(nodes, matrix, 0.05)
    behavior_graph.graph["distance_matrix"] = matrix
    search_best_key = run_data["record"].get("search_best_key")
    if search_best_key is None:
        search_best_key = run_data["record"].get("search_best_audit", {}).get("key")
    behavior_graph.graph["search_best_key"] = search_best_key
    pair_distances = _pairwise_upper(matrix)[2]
    region_threshold = float(np.quantile(pair_distances, 0.25)) if len(pair_distances) else 0.0
    region_graph, region_labels = _threshold_components(nodes, matrix, region_threshold)
    topology = compute_topology_metrics(
        nodes, lineage_graph, behavior_graph, region_labels, region_graph
    )
    topology["behavior_knn_sensitivity"] = {
        f"{rate:.3f}": _behavior_graph_rate_metrics(nodes, matrix, rate)
        for rate in BEHAVIOR_K_RATES
    }
    behavior_labels = topology["behavior_cluster_labels"]
    locality, distance_to_best, _, aux = compute_fitness_locality(
        nodes, behavior_graph, matrix, seed
    )
    quality = _quality_regions(nodes, behavior_graph, aux["z_values"])
    temporal_summary, temporal_events = _temporal_events(nodes, matrix, region_labels, seed)
    local_optima = aux["local_optima"]
    local_keys = {row["key"] for row in local_optima}
    top_keys = set(quality["top_keys"])
    best_so_far = -float("inf")
    for index, node in enumerate(nodes):
        value = float(node["fitness"])
        node["fitness_std"] = float(aux["z_values"][index])
        node["best_so_far"] = value > best_so_far + FITNESS_EPSILON
        best_so_far = max(best_so_far, value)
        key = str(node["key"])
        node["behavior_cluster"] = behavior_labels[key]
        node["semantic_region"] = region_labels[key]
        node["behavior_local_optimum"] = key in local_keys
        node["top_quality"] = key in top_keys
        node.pop("order_index", None)
    lineage_edge_keys = {_edge_key(row["source_key"], row["target_key"]) for row in lineage_edges}
    for row in behavior_edges:
        row["is_lineage_edge"] = _edge_key(row["source_key"], row["target_key"]) in lineage_edge_keys
        row.pop("left_index", None)
        row.pop("right_index", None)
    for row in lineage_edges:
        pair = _edge_key(row["source_key"], row["target_key"])
        row["is_behavior_edge"] = pair in {
            _edge_key(edge["source_key"], edge["target_key"]) for edge in behavior_edges
        }
    record = run_data["record"]
    locality["distance_to_best"]["search_best_key"] = search_best_key
    locality["distance_to_best"]["search_best_in_nodes"] = str(search_best_key) in {
        str(node["key"]) for node in nodes
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": record["campaign"],
        "task": record["task"],
        "label": record.get("label"),
        "repeat": record["repeat"],
        "run_dir": record["run_dir"],
        "combined_artifact_dir": record["combined_artifact_dir"],
        "node_count": len(nodes),
        "candidate_scope": "combined_behavesim_distribution_sample",
        "distance_matrix_scope": "combined_panel_mean",
        "run_search_best": run_data["run_search_best"],
        "distance_matrix_checks": {
            "symmetric": True,
            "diagonal_zero": bool(np.allclose(np.diag(matrix), 0.0)),
            "candidate_order_matches_matrix": True,
        },
        "behavior_knn": {
            "k_rate": 0.05,
            "k": k,
            "alternative_k_rates": list(BEHAVIOR_K_RATES),
            "connectivity_is_not_a_primary_finding": True,
        },
        "semantic_region_definition": {
            "kind": "distance_threshold_connected_components",
            "threshold_quantile": 0.25,
            "threshold": region_threshold,
            "used_for_temporal_coverage": True,
        },
        "search_best_audit": record.get("search_best_audit", {}),
        "topology": topology,
        "fitness_landscape": locality,
        "distance_to_best": distance_to_best,
        "quality_regions": quality,
        "temporal": temporal_summary,
        "operator": _operator_summary(run_data["artifact_dir"]),
        "nodes": nodes,
        "lineage_edges": lineage_edges,
        "behavior_edges": behavior_edges,
        "local_optima": local_optima,
        "temporal_events": temporal_events,
        "cluster_curve": _cluster_curve(matrix),
    }


def _plot_edges(ax: Any, edges: Sequence[dict[str, Any]], positions: dict[str, np.ndarray], **kwargs: Any) -> None:
    for edge in edges:
        left = positions[edge["source_key"]]
        right = positions[edge["target_key"]]
        ax.plot([left[0], right[0]], [left[1], right[1]], **kwargs)


def _mds_coordinates(nodes: Sequence[dict[str, Any]], matrix: np.ndarray) -> dict[str, np.ndarray]:
    if len(nodes) == 1:
        return {str(nodes[0]["key"]): np.zeros(2)}
    coordinates = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        n_init=4,
        max_iter=300,
    ).fit_transform(matrix)
    return {str(node["key"]): coordinates[index] for index, node in enumerate(nodes)}


def plot_landscape_comparison(result: dict[str, Any], path: Path) -> None:
    nodes = result["nodes"]
    matrix = np.asarray(np.load(Path(result["combined_artifact_dir"]) / "distance_matrix.npy"))
    positions = _mds_coordinates(nodes, matrix)
    z_values = np.asarray([node["fitness_std"] for node in nodes])
    sizes = np.asarray([45.0 if node["behavior_local_optimum"] else 18.0 for node in nodes])
    colors = plt.get_cmap("viridis")((z_values - z_values.min()) / max(1e-12, z_values.max() - z_values.min()))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, edges, title in (
        (axes[0], result["lineage_edges"], "Compressed lineage"),
        (axes[1], result["behavior_edges"], "BehaveSim kNN"),
    ):
        _plot_edges(axis, edges, positions, color="#9aa0a6", alpha=0.20, linewidth=0.5)
        xy = np.asarray([positions[str(node["key"])] for node in nodes])
        axis.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors, edgecolors="white", linewidths=0.25)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{result['task']} rep{result['repeat']}: same nodes, different neighborhoods")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_fitness_distance(result: dict[str, Any], path: Path) -> None:
    rows = result["distance_to_best"]
    x = np.asarray([row["distance_to_best"] for row in rows])
    y = np.asarray([row["fitness_std_gap_to_best"] for row in rows])
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.scatter(x, y, c=np.arange(len(rows)), cmap="plasma", s=18, alpha=0.75)
    axis.set_xlabel("BehaveSim distance to sampled best")
    axis.set_ylabel("Standardized fitness gap to sampled best")
    axis.set_title(f"{result['task']} rep{result['repeat']}: semantic funnel")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_distance_fitness(result: dict[str, Any], path: Path) -> None:
    rows = result["fitness_landscape"]["fitness_distance_bins"]
    figure, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    if rows:
        x = [0.5 * (row["lower"] + row["upper"]) for row in rows]
        y = [row["median_abs_fitness_difference"] for row in rows]
        axis.plot(x, y, marker="o", color="#0072B2")
    axis.set_xlabel("BehaveSim distance bin midpoint")
    axis.set_ylabel("Median absolute standardized fitness difference")
    axis.set_title(f"{result['task']} rep{result['repeat']}: distance and fitness difference")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_semantic_coverage(result: dict[str, Any], path: Path) -> None:
    events = result["temporal_events"]
    orders = np.asarray([event["order"] for event in events])
    regions = np.asarray([event["cumulative_semantic_region_count"] for event in events])
    novelty = np.asarray(
        [
            np.nan if event["novelty_to_sampled_history"] is None else event["novelty_to_sampled_history"]
            for event in events
        ]
    )
    best_so_far = np.maximum.accumulate(np.asarray([event["fitness"] for event in events]))
    breakthrough = np.asarray([event["sampled_quality_breakthrough"] for event in events])
    figure, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    axes[0].plot(orders, regions, color="#009E73")
    axes[0].set_ylabel("Regions seen")
    axes[1].plot(orders, novelty, color="#D55E00")
    axes[1].set_ylabel("Nearest history distance")
    axes[2].plot(orders, best_so_far, color="#0072B2")
    axes[2].scatter(orders[breakthrough], best_so_far[breakthrough], s=14, color="#CC79A7")
    axes[2].set_ylabel("Best-so-far fitness")
    axes[2].set_xlabel("Candidate order")
    figure.suptitle(f"{result['task']} rep{result['repeat']}: semantic coverage over search order")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_run_outputs(result: dict[str, Any], output_dir: Path, plots: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics = {key: value for key, value in result.items() if key not in {
        "nodes", "lineage_edges", "behavior_edges", "local_optima", "temporal_events"
    }}
    _write_json(output_dir / "run_metrics.json", run_metrics)
    _write_json(output_dir / "nodes.json", result["nodes"])
    _write_json(output_dir / "lineage_edges.json", result["lineage_edges"])
    _write_json(output_dir / "behavior_edges.json", result["behavior_edges"])
    _write_json(output_dir / "local_optima.json", result["local_optima"])
    _write_json(output_dir / "temporal_events.json", result["temporal_events"])
    if plots:
        plot_landscape_comparison(result, output_dir / "landscape_comparison.png")
        plot_fitness_distance(result, output_dir / "fitness_distance.png")
        plot_distance_fitness(result, output_dir / "distance_fitness.png")
        plot_semantic_coverage(result, output_dir / "semantic_coverage.png")


def _stats(values: Sequence[Any]) -> dict[str, Any]:
    available = [float(value) for value in values if value is not None and np.isfinite(value)]
    return {
        "per_repeat": list(values),
        "n_available": len(available),
        "mean": float(np.mean(available)) if available else None,
        "min": float(np.min(available)) if available else None,
        "max": float(np.max(available)) if available else None,
    }


def _get_metric(result: dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = result
    for key in path:
        if value is None:
            return None
        value = value.get(key) if isinstance(value, dict) else None
    return value


def aggregate_repeats(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["task"])].append(result)
    metric_paths = {
        "lineage_component_count": ("topology", "lineage", "connected_component_count"),
        "behavior_component_count": ("topology", "behavior", "connected_component_count"),
        "semantic_region_count": ("topology", "semantic_regions", "region_count"),
        "lineage_average_clustering": ("topology", "lineage", "average_clustering"),
        "behavior_average_clustering": ("topology", "behavior", "average_clustering"),
        "neighborhood_jaccard": ("topology", "neighborhood_jaccard"),
        "edge_jaccard": ("topology", "edge_jaccard"),
        "multi_lineage_cluster_share": (
            "topology", "lineage_behavior_alignment", "candidate_share_in_multi_lineage_clusters"
        ),
        "lineage_share_crossing_multiple_clusters": (
            "topology", "lineage_behavior_alignment", "lineage_share_crossing_multiple_clusters"
        ),
        "local_smoothness_delta": (
            "fitness_landscape", "local_smoothness", "delta_random_minus_behavior"
        ),
        "distance_gap_spearman": (
            "fitness_landscape", "distance_to_best", "distance_gap_spearman"
        ),
        "local_optima_share": ("fitness_landscape", "local_optima", "share"),
        "high_quality_region_count": ("quality_regions", "region_count"),
        "semantic_region_count_threshold": ("semantic_region_definition", "threshold"),
        "near_revisit_without_breakthrough_share": (
            "temporal", "near_revisit_without_breakthrough_share"
        ),
        "exact_revisit_without_breakthrough_share": (
            "temporal", "exact_revisit_without_breakthrough_share"
        ),
        "large_step_without_breakthrough_share": (
            "temporal", "large_step_without_breakthrough_share"
        ),
        "sampled_breakthrough_share": ("temporal", "sampled_breakthrough_share"),
        "explore_median_parent_child_distance": (
            "operator", "explore", "median_parent_child_distance"
        ),
        "refine_median_parent_child_distance": (
            "operator", "refine", "median_parent_child_distance"
        ),
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "tasks": {},
        "scope": {
            "best_distance_analysis": "sampled_best_only",
            "run_search_best": "reported_from_full_checkpoint_without_behavesim_distance",
        },
    }
    for task, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: int(row["repeat"]))
        metrics = {
            name: _stats([_get_metric(row, path) for row in rows])
            for name, path in metric_paths.items()
        }
        explore = metrics["explore_median_parent_child_distance"]["per_repeat"]
        refine = metrics["refine_median_parent_child_distance"]["per_repeat"]
        metrics["explore_minus_refine_distance"] = _stats(
            [
                None if left is None or right is None else float(left) - float(right)
                for left, right in zip(explore, refine)
            ]
        )
        output["tasks"][task] = {
            "n_repeats": len(rows),
            "repeat_ids": [row["repeat"] for row in rows],
            "node_count": [row["node_count"] for row in rows],
            "metrics": metrics,
        }
    return output


def select_v916_records(
    aggregate_path: Path,
    tasks: Sequence[str] | None = None,
    repeats: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    data = _load_json(aggregate_path)
    task_set = set(tasks) if tasks else set(TASKS)
    repeat_set = set(repeats) if repeats else None
    records = [
        record
        for record in data["runs"]
        if record.get("campaign") == CAMPAIGN
        and record.get("task") in task_set
        and (repeat_set is None or int(record.get("repeat")) in repeat_set)
    ]
    records.sort(key=lambda record: (str(record["task"]), int(record["repeat"])))
    if not records:
        raise RuntimeError("no V9.16 BehaveSim records matched the requested selection")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tasks", nargs="*", choices=TASKS)
    parser.add_argument("--repeats", nargs="*", type=int)
    parser.add_argument("--seed", type=int, default=730241)
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = select_v916_records(args.aggregate, args.tasks, args.repeats)
    results = []
    for record in records:
        run_data = load_landscape_run(record)
        result = analyze_landscape_run(run_data, seed=args.seed + int(record["repeat"]))
        output_dir = args.output_root / str(record["task"]) / f"rep{record['repeat']}"
        write_run_outputs(result, output_dir, plots=not args.skip_plots)
        results.append(result)
        print(
            json.dumps(
                {
                    "task": result["task"],
                    "repeat": result["repeat"],
                    "nodes": result["node_count"],
                    "lineage_components": result["topology"]["lineage"]["connected_component_count"],
                    "behavior_components": result["topology"]["behavior"]["connected_component_count"],
                    "local_smoothness_delta": result["fitness_landscape"]["local_smoothness"]["delta_random_minus_behavior"],
                    "near_revisit_without_breakthrough_share": result["temporal"]["near_revisit_without_breakthrough_share"],
                },
                ensure_ascii=False,
            )
        )
    _write_json(args.output_root / "aggregate.json", aggregate_repeats(results))


if __name__ == "__main__":
    main()
