"""Launch the three-repeat TSP-only TraceAAD V9.15-EH batch."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._common import REPO_ROOT, TASK_SHORT

TASK = "tsp_construct"
BACKEND_LAYOUT = {
    1: "server3",
    2: "server3b",
    3: "server3",
}


@dataclass(frozen=True, slots=True)
class Item:
    task: str
    repeat: int
    backend: str
    session: str
    run_name: str
    run_dir: Path


def build_plan(*, batch: str, session_prefix: str = "v915eh") -> list[Item]:
    return [
        Item(
            task=TASK,
            repeat=repeat,
            backend=BACKEND_LAYOUT[repeat],
            session=f"{session_prefix}_{TASK_SHORT[TASK]}_r{repeat}",
            run_name=f"v9_15_eh_{batch}_{TASK}_rep{repeat}",
            run_dir=(
                REPO_ROOT
                / "experiments"
                / TASK
                / "traceaad_v9_15_eh"
                / f"v9_15_eh_{batch}_{TASK}_rep{repeat}"
            ),
        )
        for repeat in range(1, 4)
    ]


def _summary(item: Item) -> dict | None:
    path = item.run_dir / "logs" / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


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
    summary = _summary(item)
    return bool(
        summary
        and summary.get("status") == "finished"
        and int(summary.get("evaluator_call_count", -1)) == 1000
    )


def launch(item: Item, *, dry_run: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        "v9_15_eh",
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
        if limits.get(item.backend, 0) <= 0 or _done(item) or _running(item):
            continue
        launch(item, dry_run=dry_run)
        limits[item.backend] -= 1
        launched += 1
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default="v915eh")
    parser.add_argument("--server3", type=int, default=2)
    parser.add_argument("--server3b", type=int, default=1)
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
