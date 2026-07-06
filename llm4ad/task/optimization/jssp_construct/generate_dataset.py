from __future__ import annotations

from llm4ad.task.optimization.jssp_construct.dataset import write_default_dataset


def main() -> None:
    manifest = write_default_dataset()
    print(f"Wrote {manifest['dataset_id']} with {len(manifest['splits'])} splits.")


if __name__ == "__main__":
    main()
