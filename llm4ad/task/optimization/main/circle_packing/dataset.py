from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest

DEFAULT_DATASET_ID = "circle_packing_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"

INSTANCE = {
    "name": "circle_packing_n26_unit_square",
    "n_circles": 26,
    "square_size": 1.0,
    "objective": "maximize_sum_of_radii",
}

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "instances": [INSTANCE["name"]],
        "n_instances": 1,
        "note": "Fixed single-instance benchmark used by the ShinkaEvolve circle-packing task.",
    },
    "test_exact": {
        "role": "test",
        "instances": [INSTANCE["name"]],
        "n_instances": 1,
        "note": "Same fixed problem instance, reserved for final exact verification rather than generalization.",
    },
}


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "circle_packing",
        "version": 1,
        "description": (
            "Fixed ShinkaEvolve circle-packing benchmark: place 26 circles in "
            "a unit square and maximize the sum of radii under exact geometry constraints."
        ),
        "generator": "llm4ad.task.optimization.main.circle_packing.dataset.write_default_dataset",
        "paper": [
            "papers/ShinkaEvolve/sections/domains/04_a_circle.tex",
            "papers/ShinkaEvolve/sections/appendix.tex",
        ],
        "source": "reference_code/ShinkaEvolve/examples/circle_packing",
        "instance": INSTANCE,
        "splits": DEFAULT_SPLIT_SPECS,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Circle packing manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.main.circle_packing.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown circle packing split `{split}`. Available splits: {available}")

    split_info = splits[split]
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    return [dict(manifest["instance"])], metadata
