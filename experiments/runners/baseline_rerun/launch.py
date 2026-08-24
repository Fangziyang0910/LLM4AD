"""Launch the baseline re-run batch (5 methods x 4 tasks x 3 repeats) into free slots.

Each run goes through the unified runner command template, so tmux sessions,
run naming, backend assignment, retries and watch-mode slot filling behave
like the per-method launchers. Original batch directories keep their names;
this batch writes new ``<batch>_<short>_<method>_rep<N>`` directories next to
them. VRPTW is excluded (its baseline histories survive).

    uv run python -m experiments.runners.baseline_rerun.launch [--dry-run] [--watch]
"""

from __future__ import annotations

import argparse
from datetime import datetime

from .._common import (
    LaunchItem,
    REPO_ROOT,
    TASK_SHORT,
    TaskName,
    add_launch_parser_args,
    fill_once,
    validate_new_items,
    watch_and_fill,
)

METHODS = ("eoh", "reevo", "mcts_ahd", "pathwise", "calm")
RERUN_TASKS: tuple[TaskName, ...] = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
)


def build_plan(batch: str, session_prefix: str) -> list[LaunchItem]:
    # Method-major within each repeat: every method's first repeat starts
    # before any method's second repeat, so early slots spread across methods.
    plan: list[LaunchItem] = []
    for repeat in range(1, 4):
        for method in METHODS:
            for task in RERUN_TASKS:
                short = TASK_SHORT[task]
                run_name = f"{batch}_{short}_{method}_rep{repeat}"
                plan.append(
                    LaunchItem(
                        task=task,
                        repeat=repeat,
                        backend=None,
                        session=f"{session_prefix}_{method}_{short}_r{repeat}",
                        run_name=run_name,
                        run_dir=REPO_ROOT / "experiments" / task / method / run_name,
                        seed=repeat - 1,
                        module=f"experiments.runners.{method}.run",
                    )
                )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_parser_args(parser, watch=True, session_prefix="brerun")
    args = parser.parse_args()
    batch = args.batch or datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_plan(batch, args.session_prefix)
    validate_new_items(plan)
    print(
        f"baseline re-run batch={batch}: "
        f"{len(METHODS)} methods x {len(RERUN_TASKS)} tasks x 3 repeats "
        f"= {len(plan)} runs",
        flush=True,
    )
    if args.watch:
        watch_and_fill(
            plan,
            interval_sec=args.watch_interval,
            dry_run=args.dry_run,
            method_label=f"baseline re-run {batch}",
        )
    else:
        fill_once(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
