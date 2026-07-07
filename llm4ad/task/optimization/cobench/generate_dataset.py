from __future__ import annotations

from llm4ad.task.optimization.dataset_io import run_dataset_cli
from llm4ad.task.optimization.cobench.dataset import write_default_dataset


def main() -> None:
    run_dataset_cli(
        write_default_dataset,
        note='Raw CO-Bench data is not generated.',
    )


if __name__ == "__main__":
    main()
