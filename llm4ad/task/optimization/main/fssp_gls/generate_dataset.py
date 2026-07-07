from __future__ import annotations

from llm4ad.task.optimization.dataset_io import run_dataset_cli
from llm4ad.task.optimization.main.fssp_gls.dataset import write_default_dataset


def main() -> None:
    run_dataset_cli(
        write_default_dataset,
        description='Generate fixed FSSP-GLS dataset splits.',
        source_dir_help='Path to reference_code/EoH/examples/fssp_gls. Defaults to the workspace reference path.',
    )


if __name__ == "__main__":
    main()
