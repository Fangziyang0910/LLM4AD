from __future__ import annotations

from llm4ad.task.optimization.dpp_ga.dataset import write_default_dataset


def main() -> None:
    manifest = write_default_dataset()
    size_mb = sum(info["bytes"] for info in manifest["files"].values()) / 1024 / 1024
    print(
        f"Wrote {manifest['dataset_id']} with {len(manifest['splits'])} splits "
        f"and {size_mb:.1f} MiB of data."
    )


if __name__ == "__main__":
    main()
