from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, file_sha256

DEFAULT_DATASET_ID = "aco_pheromone_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.npz",
        "n_cities": 20,
        "n_instances": 3,
        "seed": 2024,
        "n_ants": 20,
        "iter_max": 100,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "n_runs": 3,
        "seed_start": 0,
        "note": "Original EoH search evaluator: three fixed 20-city Euclidean TSP instances.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.npz",
        "n_cities": 50,
        "n_instances": 5,
        "seed": 2024,
        "n_ants": 25,
        "iter_max": 200,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "n_runs": 10,
        "seed_start": 0,
        "note": "Post-hoc evaluator from the EoH example: five fixed 50-city instances.",
    },
}


def _generate_instances(n_instances: int, n_cities: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    coordinates = rng.rand(n_instances, n_cities, 2)
    distances = np.linalg.norm(
        coordinates[:, :, None, :] - coordinates[:, None, :, :],
        axis=3,
    )
    return coordinates.astype(float), distances.astype(float)


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        filename = spec["filename"]
        coordinates, distances = _generate_instances(
            n_instances=int(spec["n_instances"]),
            n_cities=int(spec["n_cities"]),
            seed=int(spec["seed"]),
        )
        path = DATA_DIR / filename
        np.savez_compressed(path, coordinates=coordinates, distances=distances)

        split_info = {
            **spec,
            "format": "npz",
            "coordinate_shape": list(coordinates.shape),
            "distance_shape": list(distances.shape),
            "sha256": file_sha256(path),
        }
        splits[split] = split_info

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "aco_pheromone",
        "version": 1,
        "description": (
            "Fixed EoH ACO pheromone-update benchmark on Euclidean TSP. "
            "The searched function updates the pheromone matrix after each ACO iteration."
        ),
        "generator": "llm4ad.task.optimization.aco_pheromone.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/aco_pheromone",
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
            f"ACO pheromone manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.aco_pheromone.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown ACO pheromone split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"ACO pheromone dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"ACO pheromone dataset checksum mismatch: {path}")

    with np.load(path) as data:
        coordinates = np.asarray(data["coordinates"], dtype=float)
        distances = np.asarray(data["distances"], dtype=float)

    instances = [
        {
            "instance_id": idx,
            "coordinates": coordinates[idx],
            "distances": distances[idx],
            "n_cities": int(split_info["n_cities"]),
        }
        for idx in range(coordinates.shape[0])
    ]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
