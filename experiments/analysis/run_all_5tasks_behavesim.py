"""Profile TraceAAD-V9.16 on all five tasks with corrected BehaveSim."""

from __future__ import annotations

import argparse

from experiments.analysis.behavesim_batch import (
    parse_tasks,
    run_targets,
    traceaad_v916_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--panel", choices=("A", "B"), default="A")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_targets(
        traceaad_v916_targets(parse_tasks(args.tasks)),
        panel=args.panel,
        workers=args.workers,
        force=args.force,
        sample_size_override=args.sample_size,
    )


if __name__ == "__main__":
    main()
