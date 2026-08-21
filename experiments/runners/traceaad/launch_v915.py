"""Launch the four-task, three-repeat TraceAAD V9.15 batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._common import REPO_ROOT, TASKS, TASK_SHORT

BACKEND_LAYOUT = {
    1: {
        "tsp_construct": "server3",
        "cvrp_aco": "server3b",
        "op_aco": "server3",
        "online_bin_packing": "server3b",
    },
    2: {
        "tsp_construct": "server3b",
        "cvrp_aco": "server3",
        "op_aco": "server3b",
        "online_bin_packing": "server3",
    },
    3: {
        "tsp_construct": "server3",
        "cvrp_aco": "server3b",
        "op_aco": "server3b",
        "online_bin_packing": "server3",
    },
}


@dataclass(frozen=True, slots=True)
class Item:
    task: str
    repeat: int
    backend: str
    session: str
    run_name: str
    run_dir: Path


def build_plan(*, batch: str, session_prefix: str = "v915") -> list[Item]:
    plan: list[Item] = []
    for repeat in range(1, 4):
        for task in TASKS:
            run_name = f"v9_15_{batch}_{task}_rep{repeat}"
            plan.append(
                Item(
                    task=task,
                    repeat=repeat,
                    backend=BACKEND_LAYOUT[repeat][task],
                    session=f"{session_prefix}_{TASK_SHORT[task]}_r{repeat}",
                    run_name=run_name,
                    run_dir=(
                        REPO_ROOT
                        / "experiments"
                        / task
                        / "traceaad_v9_15"
                        / run_name
                    ),
                )
            )
    return plan


def _running(item: Item) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", f"={item.session}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _done(item: Item) -> bool:
    path = item.run_dir / "logs" / "summary.json"
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding="utf-8"))
    return (
        summary.get("status") == "finished"
        and summary.get("evaluator_call_count") == 1000
    )


def launch(item: Item, *, dry_run: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        "v9_15",
        "--backend",
        item.backend,
        "--budget",
        "1000",
        "--repeat",
        str(item.repeat),
        "--seed",
        str(item.repeat - 1),
    ]
    if item.run_dir.exists():
        command.extend(("--resume-from", str(item.run_dir)))
        action = "resume"
    else:
        command.extend(("--run-name", item.run_name))
        action = "launch"
    print(
        f"{action} task={item.task} rep={item.repeat} "
        f"backend={item.backend} session={item.session}",
        flush=True,
    )
    if not dry_run:
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


def launch_pending(
    plan: list[Item], *, limits: dict[str, int], dry_run: bool = False
) -> int:
    launched = 0
    for item in plan:
        if limits[item.backend] <= 0 or _done(item) or _running(item):
            continue
        launch(item, dry_run=dry_run)
        limits[item.backend] -= 1
        launched += 1
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default="v915")
    parser.add_argument("--server3", type=int, default=6)
    parser.add_argument("--server3b", type=int, default=6)
    parser.add_argument("--server1", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    limits = {
        "server3": args.server3,
        "server3b": args.server3b,
        "server1": args.server1,
    }
    if any(value < 0 for value in limits.values()):
        raise ValueError("backend launch limits must be non-negative")

    plan = build_plan(batch=args.batch, session_prefix=args.session_prefix)
    launched = launch_pending(plan, limits=limits, dry_run=args.dry_run)
    done = sum(_done(item) for item in plan)
    running = sum(_running(item) for item in plan)
    print(
        f"batch={args.batch} total={len(plan)} done={done} "
        f"running={running} launched={launched}",
        flush=True,
    )


if __name__ == "__main__":
    main()
