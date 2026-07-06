from __future__ import annotations

import argparse

from llm4ad.task.optimization.op_aco.dataset import write_default_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed OP-ACO dataset splits.")
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Path to HSEvo/problems/op_aco/dataset.",
    )
    args = parser.parse_args()

    manifest = write_default_dataset(source_dir=args.source_dir)
    print(f"Wrote {manifest['dataset_id']} with {len(manifest['splits'])} splits.")


if __name__ == "__main__":
    main()
