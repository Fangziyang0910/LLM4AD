"""Launch paired V9.17 FixedCycle runs as initialization bundles become ready."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._common import (
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    count_backend_usage,
    free_slots,
)

FIXED_BACKEND_ORDER = ("server3b", "local", "server1", "server3")
LOCAL_FIXED_LIMIT = 3
SERVER1_FIXED_LIMIT = 9


@dataclass(frozen=True, slots=True)
class Item:
    task: str
    repeat: int
    session: str
    run_name: str
    run_dir: Path
    initialization_checkpoint: Path


def build_plan(
    *,
    adaptive_batch: str,
    fixed_batch: str,
    session_prefix: str = "v917f",
) -> list[Item]:
    plan: list[Item] = []
    for repeat in range(1, 4):
        for task in TASKS:
            adaptive_name = f"v9_17_{adaptive_batch}_{task}_rep{repeat}"
            adaptive_run = (
                REPO_ROOT / "experiments" / task / "traceaad_v9_17" / adaptive_name
            )
            run_name = f"v9_17_fixed_cycle_{fixed_batch}_{task}_rep{repeat}"
            plan.append(
                Item(
                    task=task,
                    repeat=repeat,
                    session=f"{session_prefix}_{TASK_SHORT[task]}_r{repeat}",
                    run_name=run_name,
                    run_dir=(
                        REPO_ROOT
                        / "experiments"
                        / task
                        / "traceaad_v9_17_fixed_cycle"
                        / run_name
                    ),
                    initialization_checkpoint=(
                        adaptive_run / "paired_initialization" / "latest.json"
                    ),
                )
            )
    return plan


def _running(item: Item) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={item.session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _done(item: Item) -> bool:
    summary_path = item.run_dir / "logs" / "summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return summary.get("status") == "finished" and summary.get("budget_slots") == 1000


def backend_launch_capacity(
    usage: dict[str, int], available: dict[str, int]
) -> dict[str, int]:
    return {
        "server3": available.get("server3", 0),
        "server3b": available.get("server3b", 0),
        "local": min(
            available.get("local", 0), max(0, LOCAL_FIXED_LIMIT - usage.get("local", 0))
        ),
        "server1": min(
            available.get("server1", 0),
            max(0, SERVER1_FIXED_LIMIT - usage.get("server1", 0)),
        ),
    }


def launch(item: Item, *, backend: str, dry_run: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        "v9_17_fixed_cycle",
        "--backend",
        backend,
        "--budget",
        "1000",
        "--repeat",
        str(item.repeat),
        "--seed",
        str(item.repeat - 1),
    ]
    checkpoint = item.run_dir / "checkpoints" / "latest.json"
    if checkpoint.is_file():
        command.extend(("--resume-from", str(item.run_dir)))
        action = "resume"
    elif item.run_dir.exists():
        raise RuntimeError(
            f"FixedCycle run directory exists without a checkpoint: {item.run_dir}"
        )
    else:
        command.extend(("--run-name", item.run_name))
        command.extend(
            ("--initialization-checkpoint", str(item.initialization_checkpoint))
        )
        action = "launch"
    print(
        f"{action} task={item.task} rep={item.repeat} backend={backend} "
        f"session={item.session}",
        flush=True,
    )
    if dry_run:
        return
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            item.session,
            "-c",
            str(REPO_ROOT),
            *command,
        ],
        check=True,
    )


def launch_ready(
    plan: list[Item], *, dry_run: bool = False, cursor: int = 0
) -> tuple[int, int]:
    usage = {str(key): value for key, value in count_backend_usage().items()}
    available = {str(key): value for key, value in free_slots().items()}
    capacity = backend_launch_capacity(usage, available)
    launched = 0
    for item in plan:
        if _done(item) or _running(item) or not item.initialization_checkpoint.is_file():
            continue
        backend = None
        for offset in range(len(FIXED_BACKEND_ORDER)):
            candidate_index = (cursor + offset) % len(FIXED_BACKEND_ORDER)
            candidate = FIXED_BACKEND_ORDER[candidate_index]
            if capacity.get(candidate, 0) > 0:
                backend = candidate
                cursor = (candidate_index + 1) % len(FIXED_BACKEND_ORDER)
                break
        if backend is None:
            break
        launch(item, backend=backend, dry_run=dry_run)
        capacity[backend] -= 1
        launched += 1
    return launched, cursor


def watch(
    plan: list[Item], *, poll_seconds: float, dry_run: bool = False
) -> None:
    cursor = 0
    while True:
        launched, cursor = launch_ready(plan, dry_run=dry_run, cursor=cursor)
        done = sum(_done(item) for item in plan)
        running = sum(_running(item) for item in plan)
        ready = sum(item.initialization_checkpoint.is_file() for item in plan)
        print(
            f"fixed_cycle total={len(plan)} ready={ready} done={done} "
            f"running={running} launched={launched}",
            flush=True,
        )
        if dry_run or done == len(plan):
            return
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-batch", required=True)
    parser.add_argument("--fixed-batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default="v917f")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    plan = build_plan(
        adaptive_batch=args.adaptive_batch,
        fixed_batch=args.fixed_batch,
        session_prefix=args.session_prefix,
    )
    watch(plan, poll_seconds=args.poll_seconds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
