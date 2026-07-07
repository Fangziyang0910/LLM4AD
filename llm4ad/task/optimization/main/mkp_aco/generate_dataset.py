from __future__ import annotations

from llm4ad.task.optimization.dataset_io import run_dataset_cli
from llm4ad.task.optimization.main.mkp_aco.dataset import write_default_dataset


def main() -> None:
    run_dataset_cli(
        write_default_dataset,
        description='Generate fixed MKP-ACO dataset splits.',
        source_dir_help='Path to MCTS-AHD-master/problems/mkp_aco/dataset.',
    )


if __name__ == "__main__":
    main()
