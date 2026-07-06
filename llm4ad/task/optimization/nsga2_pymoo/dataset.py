from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

DEFAULT_DATASET_ID = "nsga2_pymoo_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "pop_size": 100,
        "n_gen": 100,
        "n_runs": 3,
        "seed_start": 0,
        "configs": [
            {"name": "zdt1", "n_var": 30, "ref_point": [1.1, 1.1]},
            {"name": "zdt2", "n_var": 30, "ref_point": [1.1, 1.1]},
        ],
        "n_instances": 2,
        "note": "Original EoH search evaluator: pymoo ZDT1 and ZDT2.",
    },
    "test_full": {
        "role": "test",
        "pop_size": 100,
        "n_gen": 200,
        "n_runs": 10,
        "seed_start": 0,
        "configs": [
            {"name": "zdt1", "n_var": 30, "ref_point": [1.1, 1.1]},
            {"name": "zdt2", "n_var": 30, "ref_point": [1.1, 1.1]},
            {"name": "zdt3", "n_var": 30, "ref_point": [1.1, 1.1]},
        ],
        "n_instances": 3,
        "note": "Post-hoc evaluator from the EoH example: pymoo ZDT1, ZDT2, and ZDT3.",
    },
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "nsga2_pymoo",
        "version": 1,
        "description": (
            "Fixed EoH pymoo-backed NSGA-II crossover benchmark. The searched "
            "function recombines two parent decision vectors into two offspring."
        ),
        "generator": "llm4ad.task.optimization.nsga2_pymoo.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/nsga2_pymoo",
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
            f"NSGA-II pymoo manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.nsga2_pymoo.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown NSGA-II pymoo split `{split}`. Available splits: {available}")

    split_info = splits[split]
    instances = [
        {
            "name": config["name"],
            "n_var": int(config["n_var"]),
            "ref_point": np.array(config["ref_point"], dtype=float),
        }
        for config in split_info["configs"]
    ]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return instances, metadata
