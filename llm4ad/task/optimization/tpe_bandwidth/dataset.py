from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

DEFAULT_DATASET_ID = "tpe_bandwidth_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

TRAIN_CONFIGS = [
    {"name": "sphere", "lo": -5.12, "hi": 5.12},
    {"name": "rastrigin", "lo": -5.12, "hi": 5.12},
    {"name": "ackley", "lo": -32.768, "hi": 32.768},
    {"name": "griewank", "lo": -100.0, "hi": 100.0},
    {"name": "narrow", "lo": 0.0, "hi": 1.0},
]

TEST_CONFIGS = [
    {"name": "sphere", "lo": -5.12, "hi": 5.12},
    {"name": "sphere", "lo": -50.0, "hi": 50.0},
    {"name": "rastrigin", "lo": -5.12, "hi": 5.12},
    {"name": "rastrigin", "lo": -2.0, "hi": 2.0},
    {"name": "ackley", "lo": -32.768, "hi": 32.768},
    {"name": "ackley", "lo": -5.0, "hi": 5.0},
    {"name": "griewank", "lo": -100.0, "hi": 100.0},
    {"name": "griewank", "lo": -10.0, "hi": 10.0},
    {"name": "narrow", "lo": 0.0, "hi": 1.0},
    {"name": "narrow", "lo": -0.5, "hi": 1.5},
]

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_startup": 10,
        "n_iter": 30,
        "n_ei_candidates": 64,
        "n_runs": 3,
        "configs": TRAIN_CONFIGS,
        "n_instances": len(TRAIN_CONFIGS),
        "note": "Original EoH search evaluator: five 1-D benchmarks.",
    },
    "test_full": {
        "role": "test",
        "n_startup": 20,
        "n_iter": 60,
        "n_ei_candidates": 64,
        "n_runs": 10,
        "configs": TEST_CONFIGS,
        "n_instances": len(TEST_CONFIGS),
        "note": "Original EoH post-hoc evaluator: base and wider-domain variants.",
    },
}


def sphere(x: float) -> float:
    return float(x ** 2)


def rastrigin(x: float) -> float:
    return float(10.0 + x ** 2 - 10.0 * np.cos(2.0 * np.pi * x))


def ackley(x: float) -> float:
    return float(
        -20.0 * np.exp(-0.2 * abs(x))
        - np.exp(np.cos(2.0 * np.pi * x))
        + 20.0 + np.e
    )


def griewank(x: float) -> float:
    return float(x ** 2 / 4000.0 - np.cos(x) + 1.0)


def narrow(x: float) -> float:
    return float(1.0 - np.exp(-200.0 * (x - 0.3) ** 2))


FUNCTIONS = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "ackley": ackley,
    "griewank": griewank,
    "narrow": narrow,
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "tpe_bandwidth",
        "version": 1,
        "description": (
            "Fixed EoH Optuna TPE observation-weight benchmark. The searched "
            "function computes weights for the good-group Parzen estimator."
        ),
        "generator": "llm4ad.task.optimization.tpe_bandwidth.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/tpe_bandwidth",
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
            f"TPE bandwidth manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.tpe_bandwidth.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown TPE bandwidth split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances = []
    for config in split_info["configs"]:
        name = config["name"]
        instances.append({
            "name": name,
            "func": FUNCTIONS[name],
            "lo": float(config["lo"]),
            "hi": float(config["hi"]),
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
