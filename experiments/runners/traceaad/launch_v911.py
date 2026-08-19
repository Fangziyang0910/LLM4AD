"""Launch and continuously fill a four-task, three-repeat V9.11 batch."""

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
    PRIMARY_BACKENDS,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    free_slots,
    select_backend,
)


@dataclass(frozen=True, slots=True)
class Item:
    task: str
    repeat: int
    seed: int
    session: str
    run_name: str
    run_dir: Path


def build_plan(*, batch: str, repeats: int, session_prefix: str) -> list[Item]:
    items: list[Item] = []
    for repeat in range(1, repeats + 1):
        for task in TASKS:
            run_name = f"v9_11_{batch}_{task}_rep{repeat}"
            items.append(
                Item(
                    task=task,
                    repeat=repeat,
                    seed=repeat - 1,
                    session=f"{session_prefix}_{TASK_SHORT[task]}_r{repeat}",
                    run_name=run_name,
                    run_dir=REPO_ROOT
                    / "experiments"
                    / task
                    / "traceaad_v9_11"
                    / run_name,
                )
            )
    return items


def _summary(item: Item) -> dict | None:
    path = item.run_dir / "logs" / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _running(item: Item) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={item.session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _done(item: Item) -> bool:
    summary = _summary(item)
    return bool(
        summary
        and summary.get("status") == "finished"
        and int(summary.get("evaluator_call_count", -1)) == 1000
    )


def launch(item: Item, *, backend: str, dry_run: bool) -> None:
    resume = item.run_dir.exists()
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        "v9_11",
        "--backend",
        backend,
        "--budget",
        "1000",
        "--repeat",
        str(item.repeat),
        "--seed",
        str(item.seed),
    ]
    if resume:
        command.extend(("--resume-from", str(item.run_dir)))
    else:
        command.extend(("--run-name", item.run_name))
    print(
        f"{'resume' if resume else 'launch'} task={item.task} rep={item.repeat} "
        f"backend={backend} session={item.session} run={item.run_name}",
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


def fill_once(plan: list[Item], *, backends: tuple[str, ...], dry_run: bool) -> int:
    remaining = {name: free for name, free in free_slots().items() if name in backends}
    launched = 0
    for item in plan:
        if _done(item) or _running(item):
            continue
        backend = select_backend(remaining)
        if backend is None:
            break
        launch(item, backend=backend, dry_run=dry_run)
        remaining[backend] -= 1
        launched += 1
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=("server3", "server3b"),
        default=None,
        help="restrict filling to one backend (default: balance across server3 and server3b)",
    )
    parser.add_argument("--session-prefix", default="v911")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeats != 3:
        raise ValueError("formal V9.11 batch requires exactly three repeats")
    plan = build_plan(
        batch=args.batch,
        repeats=args.repeats,
        session_prefix=args.session_prefix,
    )
    backends = (args.backend,) if args.backend else PRIMARY_BACKENDS
    while True:
        done = sum(_done(item) for item in plan)
        running = sum(_running(item) for item in plan)
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] done={done}/{len(plan)} "
            f"running={running} free={free_slots()}",
            flush=True,
        )
        if done == len(plan):
            return
        fill_once(plan, backends=backends, dry_run=args.dry_run)
        if not args.watch or args.dry_run:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
