from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest, file_sha256

DEFAULT_DATASET_ID = "evo_dynamic_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_SEED = 2024

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.pkl.gz",
        "n_instances": 5,
        "pop_size": 30,
        "k_iter": 30,
        "run_seed_mode": "global",
        "run_seed": 42,
        "groups": [
            {
                "label": "10D medium",
                "n_dims": 10,
                "n_changes": 10,
                "sigma_change": 0.5,
                "n_instances": 5,
                "data_seed": DATA_SEED,
            },
        ],
        "note": "Original EoH search evaluator: five 10D dynamic optimisation trajectories.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.pkl.gz",
        "n_instances": 64,
        "pop_size": 30,
        "k_iter": 50,
        "run_seed_mode": "local_id",
        "groups": [
            {
                "label": "10D slow",
                "n_dims": 10,
                "n_changes": 15,
                "sigma_change": 0.3,
                "n_instances": 16,
                "data_seed": DATA_SEED,
            },
            {
                "label": "10D fast",
                "n_dims": 10,
                "n_changes": 15,
                "sigma_change": 0.8,
                "n_instances": 16,
                "data_seed": DATA_SEED,
            },
            {
                "label": "20D medium",
                "n_dims": 20,
                "n_changes": 15,
                "sigma_change": 0.5,
                "n_instances": 16,
                "data_seed": DATA_SEED,
            },
            {
                "label": "20D rapid",
                "n_dims": 20,
                "n_changes": 15,
                "sigma_change": 1.0,
                "n_instances": 16,
                "data_seed": DATA_SEED,
            },
        ],
        "note": "Post-hoc evaluator from the EoH example: four dynamic scenarios.",
    },
}


def _generate_group(group: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.RandomState(int(group["data_seed"]))
    instances = []
    n_dims = int(group["n_dims"])
    n_changes = int(group["n_changes"])
    sigma_change = float(group["sigma_change"])

    for local_id in range(int(group["n_instances"])):
        optimum = rng.uniform(-3.0, 3.0, n_dims)
        trajectory = [optimum.copy()]
        for _ in range(n_changes - 1):
            step = rng.normal(0.0, sigma_change, n_dims)
            optimum = np.clip(optimum + step, -4.0, 4.0)
            trajectory.append(optimum.copy())

        instances.append({
            "group_label": group["label"],
            "local_id": local_id,
            "n_dims": n_dims,
            "n_changes": n_changes,
            "sigma_change": sigma_change,
            "trajectory": trajectory,
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
        "task": "evo_dynamic",
        "version": 1,
        "description": (
            "Fixed EoH dynamic evolutionary optimisation benchmark. The searched "
            "function regenerates or adjusts a population after each objective shift."
        ),
        "generator": "llm4ad.task.optimization.other.evo_dynamic.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/evo_dynamic",
        "splits": splits,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Evo dynamic manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.evo_dynamic.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown evo dynamic split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Evo dynamic dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"Evo dynamic dataset checksum mismatch: {path}")

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
