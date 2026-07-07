from __future__ import annotations

from llm4ad.task.optimization.dataset_io import run_dataset_cli
from llm4ad.task.optimization.main.op_aco.dataset import write_default_dataset


def main() -> None:
    run_dataset_cli(
        write_default_dataset,
        description='Generate fixed OP-ACO dataset splits.',
        source_dir_help='Path to HSEvo/problems/op_aco/dataset.',
    )


if __name__ == "__main__":
    main()
