"""Launch the parent-path full-search ablation (V9.7-PP vs V9.7-CO).

Protocol: docs/experiments/父代来时路完整搜索消融.md.  Per task and seed,
one paired unit: the PP arm runs the unchanged V9.7, the CO arm runs the
code-only variant; both share the V9.7 intent schedule, the generation-seed
formula, and the budget (1000 real evaluations).  4 tasks x 3 seeds x 2
arms = 24 runs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .._common import (
    PRIMARY_BACKENDS,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    free_slots,
    select_backend,
)

ARMS = ("pp", "co")
VERSIONS = {"pp": "v9_7", "co": "v9_7_co"}
BUDGET = 1000


@dataclass(frozen=True, slots=True)
class Item:
    task: str
    seed: int
    arm: str
    version: str
    session: str
    run_name: str
    run_dir: Path


def build_plan(*, batch: str, session_prefix: str) -> list[Item]:
    items: list[Item] = []
    for seed in range(3):
        for task in TASKS:
            for arm in ARMS:
                run_name = f"ppab_{batch}_{task}_{arm}_s{seed}"
                items.append(
                    Item(
                        task=task,
                        seed=seed,
                        arm=arm,
                        version=VERSIONS[arm],
                        session=f"{session_prefix}_{arm}_{TASK_SHORT[task]}_s{seed}",
                        run_name=run_name,
                        run_dir=(
                            REPO_ROOT
                            / "experiments"
                            / task
                            / f"traceaad_{VERSIONS[arm]}"
                            / run_name
                        ),
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
        and int(summary.get("evaluator_call_count", -1)) == BUDGET
    )


def launch(item: Item, *, backend: str) -> None:
    resume = item.run_dir.exists()
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        item.version,
        "--backend",
        backend,
        "--budget",
        str(BUDGET),
        "--seed",
        str(item.seed),
    ]
    if resume:
        command.extend(("--resume-from", str(item.run_dir)))
    else:
        command.extend(("--run-name", item.run_name))
    print(
        f"launch arm={item.arm} task={item.task} seed={item.seed} "
        f"backend={backend} session={item.session}",
        flush=True,
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", item.session, "-c", str(REPO_ROOT), *command],
        check=True,
    )


def fill_once(plan: list[Item], *, backends: tuple[str, ...]) -> int:
    remaining = {name: free for name, free in free_slots().items() if name in backends}
    launched = 0
    for item in plan:
        if _done(item) or _running(item):
            continue
        backend = select_backend(remaining)
        if backend is None:
            break
        launch(item, backend=backend)
        remaining[backend] -= 1
        launched += 1
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument(
        "--backend",
        choices=("server3", "server3b"),
        default=None,
        help="restrict filling to one backend (default: balance)",
    )
    parser.add_argument("--session-prefix", default="ppab")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=120)
    args = parser.parse_args()
    plan = build_plan(batch=args.batch, session_prefix=args.session_prefix)
    backends = (args.backend,) if args.backend else PRIMARY_BACKENDS
    while True:
        done = sum(_done(item) for item in plan)
        print(
            f"[{datetime.datetime.now().isoformat(timespec='seconds')}] "
            f"done={done}/{len(plan)} running={sum(_running(item) for item in plan)} "
            f"free={free_slots()}",
            flush=True,
        )
        if done == len(plan):
            return
        fill_once(plan, backends=backends)
        if not args.watch:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
