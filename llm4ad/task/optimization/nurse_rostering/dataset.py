from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, file_sha256

DEFAULT_DATASET_ID = "nurse_rostering_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"
REQUIREMENTS = np.array([2, 2, 1], dtype=int)

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.pkl.gz",
        "n_instances": 5,
        "groups": [
            {
                "label": "8 nurses x 14 days",
                "n_nurses": 8,
                "n_days": 14,
                "n_instances": 5,
                "seed": 42,
            },
        ],
        "note": "Original EoH search evaluator: five 8-nurse, 14-day instances.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.pkl.gz",
        "n_instances": 20,
        "groups": [
            {
                "label": "8 nurses x 14 days",
                "n_nurses": 8,
                "n_days": 14,
                "n_instances": 10,
                "seed": 200,
            },
            {
                "label": "8 nurses x 21 days",
                "n_nurses": 8,
                "n_days": 21,
                "n_instances": 10,
                "seed": 300,
            },
        ],
        "note": "Post-hoc evaluator from the EoH example: 14-day and 21-day rosters.",
    },
}


def _generate_group(group: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.RandomState(int(group["seed"]))
    instances = []
    for local_id in range(int(group["n_instances"])):
        n_nurses = int(group["n_nurses"])
        preferences = rng.randint(-1, 2, size=(n_nurses, 3)).astype(float)
        instances.append({
            "group_label": group["label"],
            "local_id": local_id,
            "n_nurses": n_nurses,
            "n_days": int(group["n_days"]),
            "n_shift_types": 3,
            "requirements": REQUIREMENTS.copy(),
            "preferences": preferences,
            "max_consecutive": 5,
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
        "task": "nurse_rostering",
        "version": 1,
        "description": (
            "Fixed EoH nurse-rostering benchmark. The searched function scores "
            "eligible nurse-shift assignments inside a greedy roster constructor."
        ),
        "generator": "llm4ad.task.optimization.nurse_rostering.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/nurse_rostering",
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
            f"Nurse rostering manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.nurse_rostering.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown nurse rostering split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Nurse rostering dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"Nurse rostering dataset checksum mismatch: {path}")

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
