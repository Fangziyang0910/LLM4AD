"""Run corrected BehaveSim campaigns sequentially for selected tasks and panels."""

from __future__ import annotations

import argparse

from experiments.analysis.behavesim_batch import (
    baseline_targets,
    parse_tasks,
    run_targets,
    traceaad_v916_targets,
    traceaad_version_targets,
)

CAMPAIGNS = ("v916", "versions", "baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--panels", nargs="*", choices=("A", "B"), default=("A", "B"))
    parser.add_argument("--campaigns", nargs="*", choices=CAMPAIGNS, default=CAMPAIGNS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    factories = {
        "v916": traceaad_v916_targets,
        "versions": traceaad_version_targets,
        "baselines": baseline_targets,
    }
    for panel in args.panels:
        for campaign in args.campaigns:
            run_targets(
                factories[campaign](tasks),
                panel=panel,
                workers=args.workers,
                force=args.force,
            )


if __name__ == "__main__":
    main()
