from __future__ import annotations

import gzip
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable


DEFAULT_SPLIT = "train"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_data_dir(module_file: str) -> Path:
    return Path(module_file).resolve().parent / "data"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def write_pickle_splits(
        *,
        data_dir: Path,
        dataset_id: str,
        task: str,
        version: int,
        description: str,
        split_specs: dict[str, dict[str, Any]],
        generate_split: Callable[[str, dict[str, Any]], Any],
        generator: str,
) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in split_specs.items():
        filename = spec.get("filename", f"{split}.pkl.gz")
        path = data_dir / filename
        instances = generate_split(split, spec)
        with gzip.open(path, "wb") as handle:
            pickle.dump(instances, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if "n_instances" in spec:
            n_instances = spec["n_instances"]
        else:
            n_instances = len(instances)

        split_info = {
            **{k: v for k, v in spec.items() if k != "filename"},
            "filename": filename,
            "n_instances": n_instances,
            "format": "pickle_gzip",
            "sha256": file_sha256(path),
        }
        splits[split] = _json_safe(split_info)

    manifest = {
        "dataset_id": dataset_id,
        "task": task,
        "version": version,
        "description": description,
        "generator": generator,
        "splits": splits,
    }
    manifest_path = data_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return manifest


def load_manifest(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pickle_split(
        *,
        data_dir: Path,
        split: str = DEFAULT_SPLIT,
) -> tuple[Any, dict[str, Any]]:
    manifest = load_manifest(data_dir)
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown dataset split `{split}`. Available splits: {available}")

    split_info = splits[split]
    path = data_dir / split_info["filename"]
    if not path.exists():
        raise FileNotFoundError(f"Dataset split file not found: {path}")
    if file_sha256(path) != split_info["sha256"]:
        raise ValueError(f"Dataset split checksum mismatch: {path}")

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
