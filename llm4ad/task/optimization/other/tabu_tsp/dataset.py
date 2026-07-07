from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest, file_sha256

DEFAULT_DATASET_ID = "tabu_tsp_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.pkl.gz",
        "n_instances": 5,
        "n_iter": 200,
        "tabu_tenure": 7,
        "n_runs": 3,
        "groups": [
            {
                "label": "20-node",
                "n_nodes": 20,
                "n_instances": 5,
                "seed": 42,
            },
        ],
        "note": "Original EoH search evaluator: five 20-node Euclidean TSP instances.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.pkl.gz",
        "n_instances": 20,
        "n_iter": 500,
        "tabu_tenure": 7,
        "n_runs": 10,
        "groups": [
            {
                "label": "20-node",
                "n_nodes": 20,
                "n_instances": 10,
                "seed": 100,
            },
            {
                "label": "30-node",
                "n_nodes": 30,
                "n_instances": 10,
                "seed": 200,
            },
        ],
        "note": "Post-hoc evaluator from the EoH example: 20-node and 30-node instances.",
    },
}


def _generate_group(group: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.RandomState(int(group["seed"]))
    instances = []
    n_nodes = int(group["n_nodes"])
    for local_id in range(int(group["n_instances"])):
        coordinates = rng.rand(n_nodes, 2) * 100.0
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        distances = np.sqrt((diff ** 2).sum(-1))
        instances.append({
            "group_label": group["label"],
            "local_id": local_id,
            "n_nodes": n_nodes,
            "coordinates": coordinates,
            "distances": distances,
        })
    return instances


def _generate_split(spec: dict[str, Any]) -> list[dict[str, Any]]:
    instances = []
    for group in spec["groups"]:
        instances.extend(_generate_group(group))
    for instance_id, instance in enumerate(instances):
        instance["instance_id"] = instance_id
    return instances


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        instances = _generate_split(spec)
        path = DATA_DIR / spec["filename"]
        with gzip.open(path, "wb") as handle:
            pickle.dump(instances, handle, protocol=pickle.HIGHEST_PROTOCOL)

        splits[split] = {
            **spec,
            "format": "pickle_gzip",
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tabu_tsp",
        "version": 1,
        "description": (
            "Fixed EoH Tabu Search TSP benchmark. The searched function scores "
            "all candidate 2-opt moves at each Tabu Search iteration."
        ),
        "generator": "llm4ad.task.optimization.other.tabu_tsp.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/tabu_tsp",
        "splits": splits,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Tabu TSP manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.tabu_tsp.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown Tabu TSP split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Tabu TSP dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"Tabu TSP dataset checksum mismatch: {path}")

    with gzip.open(path, "rb") as handle:
        instances = pickle.load(handle)

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return instances, metadata
