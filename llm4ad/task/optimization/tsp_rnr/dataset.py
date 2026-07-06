from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, file_sha256

DEFAULT_DATASET_ID = "tsp_rnr_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "n_instances": 5,
        "n_nodes": 50,
        "n_destroy": 10,
        "iter_max": 100,
        "time_max": 5.0,
        "seed": 2024,
        "run_seed": 9101,
        "note": "Original EoH search evaluator scale: 5 instances, 50 nodes, 100 RnR iterations.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.npz",
        "n_instances": 16,
        "n_nodes": 50,
        "n_destroy": 10,
        "iter_max": 200,
        "time_max": 10.0,
        "seed": 2025,
        "run_seed": 9102,
        "note": (
            "Fixed LLM4AD post-hoc split following the EoH evaluation scale but using "
            "a different seed so test instances do not overlap the train prefix."
        ),
    },
}


def _generate_instances(n_instances: int, n_nodes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    coordinates = rng.rand(n_instances, n_nodes, 2)
    diff = coordinates[:, :, np.newaxis, :] - coordinates[:, np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=3)
    return coordinates.astype(float), distances.astype(float)


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        coordinates, distance_matrices = _generate_instances(
            n_instances=int(spec["n_instances"]),
            n_nodes=int(spec["n_nodes"]),
            seed=int(spec["seed"]),
        )
        path = DATA_DIR / spec["filename"]
        np.savez_compressed(
            path,
            coordinates=coordinates,
            distance_matrices=distance_matrices,
        )

        splits[split] = {
            **spec,
            "format": "npz",
            "distribution": "uniform_unit_square",
            "coordinates_shape": list(coordinates.shape),
            "distance_matrices_shape": list(distance_matrices.shape),
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tsp_rnr",
        "version": 1,
        "description": (
            "Fixed EoH TSP ruin-and-recreate benchmark. The searched function "
            "selects nodes to remove during the ruin phase."
        ),
        "generator": "llm4ad.task.optimization.tsp_rnr.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/tsp_rnr",
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
            f"TSP RnR manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.tsp_rnr.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown TSP RnR split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"TSP RnR dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"TSP RnR dataset checksum mismatch: {path}")

    with np.load(path) as data:
        coordinates = np.asarray(data["coordinates"], dtype=float)
        distance_matrices = np.asarray(data["distance_matrices"], dtype=float)

    instances = [
        {
            "instance_id": idx,
            "coordinates": coordinates[idx],
            "distance_matrix": distance_matrices[idx],
        }
        for idx in range(distance_matrices.shape[0])
    ]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
