from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "moead_decomposition_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "n_gen": 100,
        "n_runs": 3,
        "seed_start": 0,
        "T": 5,
        "H": 5,
        "hv_samples": 20000,
        "configs": [
            {"name": "DTLZ2", "n_var": 7, "n_obj": 3, "ref_point": [2.0, 2.0, 2.0]},
            {"name": "DTLZ2", "n_var": 12, "n_obj": 3, "ref_point": [2.0, 2.0, 2.0]},
        ],
        "n_instances": 2,
        "note": "Original EoH search evaluator: DTLZ2 with 7 and 12 variables.",
    },
    "test_full": {
        "role": "test",
        "n_gen": 200,
        "n_runs": 10,
        "seed_start": 0,
        "T": 5,
        "H": 5,
        "hv_samples": 30000,
        "configs": [
            {"name": "DTLZ2", "n_var": 7, "n_obj": 3, "ref_point": [2.0, 2.0, 2.0]},
            {"name": "DTLZ2", "n_var": 12, "n_obj": 3, "ref_point": [2.0, 2.0, 2.0]},
            {"name": "DTLZ2", "n_var": 20, "n_obj": 3, "ref_point": [2.0, 2.0, 2.0]},
            {"name": "DTLZ2", "n_var": 30, "n_obj": 3, "ref_point": [2.0, 2.0, 2.0]},
        ],
        "n_instances": 4,
        "note": "Post-hoc evaluator from the EoH example: DTLZ2 with 7/12/20/30 variables.",
    },
}


class DTLZ2:
    def __init__(self, n_obj: int = 3):
        self.n_obj = n_obj

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n_obj = self.n_obj
        g = float(np.sum((x[n_obj - 1:] - 0.5) ** 2))
        values = np.empty(n_obj)
        values[0] = (1.0 + g) * np.prod(np.cos(x[:n_obj - 1] * np.pi / 2.0))
        for objective in range(1, n_obj - 1):
            values[objective] = (
                (1.0 + g)
                * np.prod(np.cos(x[:n_obj - 1 - objective] * np.pi / 2.0))
                * np.sin(x[n_obj - 1 - objective] * np.pi / 2.0)
            )
        values[n_obj - 1] = (1.0 + g) * np.sin(x[0] * np.pi / 2.0)
        return values


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "moead_decomposition",
        "version": 1,
        "description": (
            "Fixed EoH MOEA/D decomposition benchmark. The searched function maps "
            "objective vectors and weight vectors to scalar subproblem scores."
        ),
        "generator": "llm4ad.task.optimization.other.moead_decomposition.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/moead_decomposition",
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"MOEA/D decomposition manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.other.moead_decomposition.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown MOEA/D decomposition split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances = []
    for config in split_info["configs"]:
        n_obj = int(config["n_obj"])
        n_var = int(config["n_var"])
        instances.append({
            "name": config["name"],
            "func": DTLZ2(n_obj=n_obj),
            "n_var": n_var,
            "n_obj": n_obj,
            "ref_point": np.array(config["ref_point"], dtype=float),
        })

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
