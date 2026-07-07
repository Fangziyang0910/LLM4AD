from __future__ import annotations

import gzip
import importlib.util
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT, write_manifest, file_sha256

DEFAULT_DATASET_ID = "bp_online_v1"
DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_TASK_DIR = Path(__file__).resolve().parents[5] / "reference_code" / "EoH" / "examples" / "bp_online"
SOURCE_TEST_DATA_DIR = SOURCE_TASK_DIR / "testingdata"

DEFAULT_SPLIT_SPECS = {
    "train": {
        "role": "train",
        "filename": "train.pkl.gz",
        "source": "reference_code/EoH/examples/bp_online/get_instance.py",
        "n_instances": 5,
        "capacity": 100,
        "n_items": 5000,
        "note": "Original EoH search evaluator: hardcoded Weibull 5k training set at capacity 100.",
    },
    "test_full": {
        "role": "test",
        "filename": "test_full.pkl.gz",
        "source": "reference_code/EoH/examples/bp_online/testingdata",
        "n_instances": 30,
        "capacities": [100, 500],
        "sizes": ["1k", "5k", "10k"],
        "items_per_size": {"1k": 1000, "5k": 5000, "10k": 10000},
        "instances_per_cell": 5,
        "note": "Original EoH post-hoc evaluator: 1k/5k/10k item sets under capacities 100 and 500.",
    },
}


def _source_get_data_class():
    path = SOURCE_TASK_DIR / "get_instance.py"
    if not path.exists():
        raise FileNotFoundError(
            f"EoH bp_online source get_instance.py not found: {path}. "
            "The committed LLM4AD data can still be loaded without regenerating it."
        )
    spec = importlib.util.spec_from_file_location("_eoh_bp_online_get_instance", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import EoH bp_online source data module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GetData


def _l1_bound(items: np.ndarray, capacity: int) -> float:
    return float(np.ceil(np.sum(items) / capacity))


def _records_for_dataset(
        *,
        item_lists: list[list[int] | np.ndarray],
        capacity: int,
        group_label: str,
) -> list[dict[str, Any]]:
    records = []
    for idx, items in enumerate(item_lists, 1):
        item_array = np.asarray(items, dtype=int)
        records.append({
            "instance_id": f"{group_label}_{idx}",
            "group_label": group_label,
            "capacity": int(capacity),
            "num_items": int(len(item_array)),
            "items": item_array,
            "l1_bound": _l1_bound(item_array, int(capacity)),
        })
    return records


def _generate_train_records() -> list[dict[str, Any]]:
    get_data = _source_get_data_class()
    datasets, _ = get_data().get_instances(capacity=100)
    item_lists = [
        datasets["Weibull 5k"][f"test_{idx}"]["items"]
        for idx in range(5)
    ]
    return _records_for_dataset(
        item_lists=item_lists,
        capacity=100,
        group_label="5k_c100",
    )


def _load_raw_test_items(size_label: str) -> list[list[int]]:
    filename = f"test_dataset_{size_label}.pkl"
    path = SOURCE_TEST_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"EoH bp_online source test data not found: {path}")
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    item_lists = []
    for instances in raw.values():
        item_lists.extend(list(items) for items in instances)
    return item_lists


def _generate_test_records() -> list[dict[str, Any]]:
    records = []
    for capacity in DEFAULT_SPLIT_SPECS["test_full"]["capacities"]:
        for size_label in DEFAULT_SPLIT_SPECS["test_full"]["sizes"]:
            item_lists = _load_raw_test_items(size_label)
            records.extend(_records_for_dataset(
                item_lists=item_lists,
                capacity=int(capacity),
                group_label=f"{size_label}_c{capacity}",
            ))
    return records


def _generate_split(split: str) -> list[dict[str, Any]]:
    if split == "train":
        return _generate_train_records()
    if split == "test_full":
        return _generate_test_records()
    raise ValueError(f"Unknown bp_online split: {split}")


def _group_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for group_label in sorted({record["group_label"] for record in records}):
        group = [record for record in records if record["group_label"] == group_label]
        summaries.append({
            "group_label": group_label,
            "n_instances": len(group),
            "capacity": int(group[0]["capacity"]),
            "num_items": int(group[0]["num_items"]),
            "mean_l1_bound": float(np.mean([record["l1_bound"] for record in group])),
        })
    return summaries


def write_default_dataset() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in DEFAULT_SPLIT_SPECS.items():
        records = _generate_split(split)
        path = DATA_DIR / spec["filename"]
        with gzip.open(path, "wb") as handle:
            pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)

        splits[split] = {
            **spec,
            "format": "pickle_gzip",
            "n_instances": len(records),
            "groups": _group_summaries(records),
            "sha256": file_sha256(path),
        }

    manifest = {
        "dataset_id": DEFAULT_DATASET_ID,
        "task": "bp_online",
        "version": 1,
        "description": (
            "Fixed EoH online bin-packing benchmark. The searched function scores "
            "feasible bins for each arriving item; evaluation reports excess over "
            "the L1 lower bound."
        ),
        "generator": "llm4ad.task.optimization.main.bp_online.dataset.write_default_dataset",
        "paper": "papers/EoH",
        "source": "reference_code/EoH/examples/bp_online",
        "splits": splits,
    }
    return write_manifest(DATA_DIR, manifest)


def load_manifest() -> dict[str, Any]:
    path = DATA_DIR / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"bp_online manifest not found: {path}. "
            "Run `uv run python -m llm4ad.task.optimization.main.bp_online.generate_dataset`."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_split_instances(split: str = DEFAULT_SPLIT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_manifest()
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown bp_online split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = DATA_DIR / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"bp_online dataset split not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"bp_online dataset checksum mismatch: {path}")

    with gzip.open(path, "rb") as handle:
        records = pickle.load(handle)

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        "path": str(path),
        **split_info,
    }
    return records, metadata
