"""Launch the baseline re-run batch (5 methods x 4 tasks x 3 repeats) into free slots.

Each run goes through the unified runner command template, so tmux sessions,
run naming, retries and slot filling behave like the per-method launchers.
Original batch directories keep their names; this batch writes new
``<batch>_<short>_<method>_rep<N>`` directories next to them. VRPTW is
excluded (its baseline histories survive).

The watch loop fills both the primary server slots and the local
llama-server slots; local runs cap output tokens at 8192 to fit the 32k
context.

    uv run python -m experiments.runners.baseline_rerun.launch [--dry-run] [--watch]
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime

from .._common import (
    BACKEND_CAPACITY,
    LaunchItem,
    REPO_ROOT,
    TASK_SHORT,
    TaskName,
    add_launch_parser_args,
    count_backend_usage,
    fill_once,
    launch_items,
    pending_items,
    validate_new_items,
)

METHODS = ("eoh", "reevo", "mcts_ahd", "pathwise", "calm")
RERUN_TASKS: tuple[TaskName, ...] = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
)
LOCAL_OUTPUT_TOKENS = 8192


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


def fill_local_once(plan: list[LaunchItem], *, dry_run: bool) -> None:
    free_local = BACKEND_CAPACITY["local"] - count_backend_usage()["local"]
    if free_local <= 0:
        return
    assigned = [
        replace(
            item.with_backend("local"),
            extra_args=("--output-tokens", str(LOCAL_OUTPUT_TOKENS)),
        )
        for item in pending_items(plan)[:free_local]
    ]
    if not assigned:
        return
    print(
        f"local fill: {len(assigned)} run(s) at output-tokens {LOCAL_OUTPUT_TOKENS}",
        flush=True,
    )
    launch_items(assigned, dry_run=dry_run)


def watch(plan: list[LaunchItem], *, interval_sec: int, dry_run: bool, batch: str) -> None:
    print(
        f"watching baseline re-run {batch}; interval={interval_sec}s; "
        f"{len(plan)} runs; fills primary servers and local",
        flush=True,
    )
    while pending_items(plan):
        fill_once(plan, dry_run=dry_run)
        fill_local_once(plan, dry_run=dry_run)
        pending = len(pending_items(plan))
        running = sum(1 for item in plan if item.run_dir.exists())
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"pending={pending} started={running}",
            flush=True,
        )
        if dry_run:
            return
        time.sleep(interval_sec)
    print("baseline re-run batch complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launch_parser_args(parser, watch=True, session_prefix="brerun")
    args = parser.parse_args()
    batch = args.batch or datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = build_plan(batch, args.session_prefix)
    if not any(item.run_dir.exists() for item in plan):
        validate_new_items(plan)
    print(
        f"baseline re-run batch={batch}: "
        f"{len(METHODS)} methods x {len(RERUN_TASKS)} tasks x 3 repeats "
        f"= {len(plan)} runs",
        flush=True,
    )
    if args.watch:
        watch(plan, interval_sec=args.watch_interval, dry_run=args.dry_run, batch=batch)
    else:
        fill_once(plan, dry_run=args.dry_run)
        fill_local_once(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
