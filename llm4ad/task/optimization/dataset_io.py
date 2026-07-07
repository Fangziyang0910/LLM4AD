from __future__ import annotations

import argparse
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


def make_manifest(
        *,
        dataset_id: str,
        task: str,
        version: int,
        description: str,
        generator: str,
        splits: dict[str, Any],
        **extra_fields: Any,
) -> dict[str, Any]:
    manifest = {
        "dataset_id": dataset_id,
        "task": task,
        "version": version,
        "description": description,
        "generator": generator,
        "splits": splits,
    }
    manifest.update(extra_fields)
    return _json_safe(manifest)


def write_manifest(data_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_manifest = _json_safe(manifest)
    with (data_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(safe_manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return safe_manifest


def write_dataset_manifest(
        *,
        data_dir: Path,
        dataset_id: str,
        task: str,
        version: int,
        description: str,
        generator: str,
        splits: dict[str, Any],
        **extra_fields: Any,
) -> dict[str, Any]:
    return write_manifest(
        data_dir,
        make_manifest(
            dataset_id=dataset_id,
            task=task,
            version=version,
            description=description,
            generator=generator,
            splits=splits,
            **extra_fields,
        ),
    )


def get_split_info(
        manifest: dict[str, Any],
        split: str = DEFAULT_SPLIT,
        *,
        label: str = "dataset",
) -> dict[str, Any]:
    splits = manifest.get("splits", {})
    if split not in splits:
        available = ", ".join(sorted(splits))
        raise ValueError(f"Unknown {label} split `{split}`. Available splits: {available}")
    return splits[split]


def build_split_metadata(
        manifest: dict[str, Any],
        split: str,
        split_info: dict[str, Any],
        *,
        path: Path | None = None,
) -> dict[str, Any]:
    metadata = {
        "dataset_id": manifest["dataset_id"],
        "task": manifest["task"],
        "split": split,
        **split_info,
    }
    if path is not None:
        metadata["path"] = str(path)
    return metadata


def verify_file_sha256(
        path: Path,
        expected_sha256: str,
        *,
        label: str = "Dataset file",
) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"{label} checksum mismatch: {path}")


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

    return write_dataset_manifest(
        data_dir=data_dir,
        dataset_id=dataset_id,
        task=task,
        version=version,
        description=description,
        generator=generator,
        splits=splits,
    )


def write_npz_splits(
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
    import numpy as np

    data_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Any] = {}
    for split, spec in split_specs.items():
        filename = spec.get("filename", f"{split}.npz")
        path = data_dir / filename
        generated = generate_split(split, spec)
        if isinstance(generated, tuple):
            arrays, extra_info = generated
        else:
            arrays, extra_info = generated, {}
        np.savez_compressed(path, **arrays)

        split_info = {
            **{k: v for k, v in spec.items() if k != "filename"},
            **extra_info,
            "filename": filename,
            "format": "npz_compressed",
            "sha256": file_sha256(path),
        }
        splits[split] = _json_safe(split_info)

    return write_dataset_manifest(
        data_dir=data_dir,
        dataset_id=dataset_id,
        task=task,
        version=version,
        description=description,
        generator=generator,
        splits=splits,
    )


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
    split_info = get_split_info(manifest, split)
    path = data_dir / split_info["filename"]
    verify_file_sha256(path, split_info["sha256"], label="Dataset split file")

    with gzip.open(path, "rb") as handle:
        instances = pickle.load(handle)

    return instances, build_split_metadata(manifest, split, split_info, path=path)


def load_npz_split(
        *,
        data_dir: Path,
        split: str = DEFAULT_SPLIT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    manifest = load_manifest(data_dir)
    split_info = get_split_info(manifest, split)
    path = data_dir / split_info["filename"]
    verify_file_sha256(path, split_info["sha256"], label="Dataset split file")

    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}

    return arrays, build_split_metadata(manifest, split, split_info, path=path)


def format_dataset_summary(manifest: dict[str, Any], *, note: str | None = None) -> str:
    dataset_id = manifest["dataset_id"]
    n_splits = len(manifest.get("splits", {}))
    if n_splits:
        message = f"Wrote {dataset_id} with {n_splits} splits."
    else:
        message = f"Wrote {dataset_id} manifest."

    files = manifest.get("files")
    if isinstance(files, dict) and files:
        size_bytes = sum(int(info.get("bytes", 0)) for info in files.values())
        message = message[:-1] + f" and {size_bytes / 1024 / 1024:.1f} MiB of data."

    if note:
        message = message[:-1] + f". {note}"
    return message


def run_dataset_cli(
        write_default_dataset: Callable[..., dict[str, Any]],
        *,
        description: str | None = None,
        source_dir_help: str | None = None,
        note: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {}
    if source_dir_help is not None:
        parser = argparse.ArgumentParser(description=description)
        parser.add_argument("--source-dir", default=None, help=source_dir_help)
        args = parser.parse_args()
        kwargs["source_dir"] = args.source_dir

    manifest = write_default_dataset(**kwargs)
    print(format_dataset_summary(manifest, note=note))
