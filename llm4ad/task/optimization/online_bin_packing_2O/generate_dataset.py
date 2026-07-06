from __future__ import annotations

from llm4ad.task.optimization.online_bin_packing_2O.dataset import write_default_dataset


def main() -> None:
    manifest = write_default_dataset()
    print(f"Wrote {manifest['dataset_id']} manifest with {len(manifest['splits'])} shared splits.")


if __name__ == "__main__":
    main()
