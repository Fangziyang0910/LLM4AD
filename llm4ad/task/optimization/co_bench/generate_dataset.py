from __future__ import annotations

from llm4ad.task.optimization.co_bench.dataset import write_default_dataset


def main() -> None:
    manifest = write_default_dataset()
    print(f"Wrote {manifest['dataset_id']} manifest. Raw CO-Bench data is not generated.")


if __name__ == "__main__":
    main()
