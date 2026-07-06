from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, file_sha256

DEFAULT_DATASET_ID = "gnn_aggregation_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "n_instances": 20,
        "n_nodes": 30,
        "n_feat": 4,
        "n_layers": 3,
        "p_in": 0.4,
        "p_out": 0.1,
        "noise_std": 5.0,
        "seed": 2024,
        "note": "Original EoH search evaluator: 20 SBM graph instances generated with seed 2024.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.npz",
        "n_instances": 64,
        "n_nodes": 30,
        "n_feat": 4,
        "n_layers": 3,
        "p_in": 0.4,
        "p_out": 0.1,
        "noise_std": 5.0,
        "seed": 2025,
        "note": (
            "Fixed LLM4AD post-hoc split generated from the same EoH SBM distribution. "
            "The original EoH example does not ship a separate test split."
        ),
    },
}


def _generate_instances(
        *,
        n_instances: int,
        n_nodes: int,
        n_feat: int,
        p_in: float,
        p_out: float,
        noise_std: float,
        seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    half = n_nodes // 2
    base_labels = np.array([0] * half + [1] * (n_nodes - half), dtype=int)

    adj_matrices = np.zeros((n_instances, n_nodes, n_nodes), dtype=float)
    node_features = np.zeros((n_instances, n_nodes, n_feat), dtype=float)
    labels = np.zeros((n_instances, n_nodes), dtype=int)

    for instance_idx in range(n_instances):
        adj = np.zeros((n_nodes, n_nodes), dtype=float)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                p = p_in if base_labels[i] == base_labels[j] else p_out
                if rng.rand() < p:
                    adj[i, j] = adj[j, i] = 1.0

        signal = np.where(base_labels == 0, 1.0, -1.0)[:, np.newaxis]
        signal = np.tile(signal, (1, n_feat))
        noise = rng.randn(n_nodes, n_feat) * noise_std

        adj_matrices[instance_idx] = adj
        node_features[instance_idx] = signal + noise
        labels[instance_idx] = base_labels

    return adj_matrices, node_features, labels


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        adj_matrices, node_features, labels = _generate_instances(
            n_instances=int(spec["n_instances"]),
            n_nodes=int(spec["n_nodes"]),
            n_feat=int(spec["n_feat"]),
            p_in=float(spec["p_in"]),
            p_out=float(spec["p_out"]),
            noise_std=float(spec["noise_std"]),
            seed=int(spec["seed"]),
        )
        path = DATA_DIR / spec["filename"]
        np.savez_compressed(
            path,
            adj_matrices=adj_matrices,
            node_features=node_features,
            labels=labels,
        )

        splits[split] = {
            **spec,
            "format": "npz",
            "adj_matrices_shape": list(adj_matrices.shape),
            "node_features_shape": list(node_features.shape),
            "labels_shape": list(labels.shape),
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "gnn_aggregation",
        "version": 1,
        "description": (
            "Fixed EoH GNN neighborhood-aggregation benchmark. The searched function "
            "updates node features for community detection on stochastic block model graphs."
        ),
        "generator": "llm4ad.task.optimization.gnn_aggregation.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/gnn_aggregation",
        "splits": splits,
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"GNN aggregation manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.gnn_aggregation.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown GNN aggregation split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"GNN aggregation dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"GNN aggregation dataset checksum mismatch: {path}")

    with np.load(path) as data:
        adj_matrices = np.asarray(data["adj_matrices"], dtype=float)
        node_features = np.asarray(data["node_features"], dtype=float)
        labels = np.asarray(data["labels"], dtype=int)

    instances = [
        {
            "instance_id": idx,
            "adj_matrix": adj_matrices[idx],
            "node_features": node_features[idx],
            "labels": labels[idx],
        }
        for idx in range(adj_matrices.shape[0])
    ]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
