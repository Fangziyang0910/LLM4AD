from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

DEFAULT_DATASET_ID = "nsga2_crowding_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "pop_size": 100,
        "n_gen": 100,
        "n_runs": 3,
        "seed_start": 0,
        "configs": [
            {"name": "ZDT1", "n_var": 30, "n_obj": 2, "ref_point": [1.1, 1.1]},
            {"name": "ZDT2", "n_var": 30, "n_obj": 2, "ref_point": [1.1, 1.1]},
        ],
        "n_instances": 2,
        "note": "Original EoH search evaluator: ZDT1 and ZDT2 with 30 variables.",
    },
    "test_full": {
        "role": "test",
        "pop_size": 100,
        "n_gen": 200,
        "n_runs": 10,
        "seed_start": 0,
        "configs": [
            {"name": "ZDT1", "n_var": 30, "n_obj": 2, "ref_point": [1.1, 1.1]},
            {"name": "ZDT2", "n_var": 30, "n_obj": 2, "ref_point": [1.1, 1.1]},
            {"name": "ZDT3", "n_var": 30, "n_obj": 2, "ref_point": [1.1, 1.1]},
        ],
        "n_instances": 3,
        "note": "Post-hoc evaluator from the EoH example: ZDT1, ZDT2, and ZDT3.",
    },
}


class ZDT1:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g))
        return np.array([f1, f2])


class ZDT2:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2])


class ZDT3:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (len(x) - 1)
        f2 = g * (1.0 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10.0 * np.pi * f1))
        return np.array([f1, f2])


FUNCTION_CLASSES = {
    "ZDT1": ZDT1,
    "ZDT2": ZDT2,
    "ZDT3": ZDT3,
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "nsga2_crowding",
        "version": 1,
        "description": (
            "Fixed EoH NSGA-II crowding-operator benchmark. The searched function "
            "computes a diversity score for each solution in one non-dominated front."
        ),
        "generator": "llm4ad.task.optimization.nsga2_crowding.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/nsga2_crowding",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    with (DATA_DIR / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"NSGA-II crowding manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.nsga2_crowding.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown NSGA-II crowding split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances = []
    for config in split_info["configs"]:
        name = config["name"]
        instances.append({
            "name": name,
            "func": FUNCTION_CLASSES[name](),
            "n_var": int(config["n_var"]),
            "n_obj": int(config["n_obj"]),
            "ref_point": np.array(config["ref_point"], dtype=float),
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
