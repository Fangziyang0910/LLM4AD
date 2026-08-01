"""Unified fair-budget launcher for PathWise / ReEvo / ShinkaEvolve.

Fills free LLM backend slots from a combined 36-run queue
(3 methods × 4 tasks × 3 repeats), each with max_sample_nums=1000.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[3]
TaskName = Literal["tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing"]
BackendName = Literal["local", "server1", "zhong"]
MethodName = Literal["pathwise", "reevo", "shinka_evo"]

TASKS: tuple[TaskName, ...] = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
)
METHODS: tuple[MethodName, ...] = ("pathwise", "reevo", "shinka_evo")
TASK_SHORT = {
    "tsp_construct": "tsp",
    "cvrp_aco": "cvrp",
    "op_aco": "op",
    "online_bin_packing": "obp",
}
METHOD_SHORT = {
    "pathwise": "pw",
    "reevo": "reevo",
    "shinka_evo": "shinka",
}
METHOD_MODULE = {
    "pathwise": "experiments.runners.pathwise.run",
    "reevo": "experiments.runners.reevo.run",
    "shinka_evo": "experiments.runners.shinka_evo.run",
}
METHOD_DIR = {
    "pathwise": "pathwise",
    "reevo": "reevo",
    "shinka_evo": "shinka_evo",
}
BACKEND_CAPACITY: dict[BackendName, int] = {
    "zhong": 6,
    "server1": 6,
    "local": 3,
}
BACKEND_MARKERS: dict[BackendName, tuple[str, ...]] = {
    "zhong": ("--backend zhong", "183.36.243.124"),
    "server1": ("--backend server1", "222.201.145.8"),
    "local": ("--backend local", "127.0.0.1:8001"),
}
MAX_RETRIES = 12


@dataclass(frozen=True, slots=True)
class LaunchItem:
    method: MethodName
    task: TaskName
    repeat: int
    backend: BackendName | None
    session: str
    run_name: str
    run_dir: Path
    seed: int

    def with_backend(self, backend: BackendName) -> LaunchItem:
        return LaunchItem(
            method=self.method,
            task=self.task,
            repeat=self.repeat,
            backend=backend,
            session=self.session,
            run_name=self.run_name,
            run_dir=self.run_dir,
            seed=self.seed,
        )

    def command(self) -> tuple[str, ...]:
        if self.backend is None:
            raise ValueError(f"backend not assigned for {self.run_name}")
        return (
            sys.executable,
            "-m",
            METHOD_MODULE[self.method],
            "--task",
            self.task,
            "--backend",
            self.backend,
            "--repeat",
            str(self.repeat),
            "--seed",
            str(self.seed),
            "--run-name",
            self.run_name,
            "--max-sample-nums",
            "1000",
        )


def build_launch_plan(args: argparse.Namespace) -> list[LaunchItem]:
    plan: list[LaunchItem] = []
    for method in METHODS:
        for repeat in range(1, args.repeats + 1):
            for task in TASKS:
                short = TASK_SHORT[task]
                mshort = METHOD_SHORT[method]
                run_name = f"{args.batch}_{short}_{mshort}_rep{repeat}"
                session = f"f1k_{mshort}_{short}_r{repeat}"
                run_dir = (
                    REPO_ROOT
                    / "experiments"
                    / task
                    / METHOD_DIR[method]
                    / run_name
                )
                plan.append(
                    LaunchItem(
                        method=method,
                        task=task,
                        repeat=repeat,
                        backend=None,
                        session=session,
                        run_name=run_name,
                        run_dir=run_dir,
                        seed=repeat - 1,
                    )
                )
    return plan


def _process_cmdlines() -> list[str]:
    result = subprocess.run(
        ["ps", "-eo", "args="],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = []
    for line in result.stdout.splitlines():
        text = line.strip()
        if not text or "uv run" in text or "python" not in text:
            continue
        if "experiments." not in text and "run_experiment" not in text:
            continue
        lines.append(text)
    return lines


def count_backend_usage() -> dict[BackendName, int]:
    counts: dict[BackendName, int] = {name: 0 for name in BACKEND_CAPACITY}
    for cmdline in _process_cmdlines():
        matched: BackendName | None = None
        for backend, markers in BACKEND_MARKERS.items():
            if any(marker in cmdline for marker in markers):
                matched = backend
                break
        if matched is None and "traceaad_v4.run_experiment" in cmdline:
            matched = "local"
        if matched is not None:
            counts[matched] += 1
    return counts


def free_slots() -> dict[BackendName, int]:
    usage = count_backend_usage()
    return {
        backend: max(0, BACKEND_CAPACITY[backend] - usage[backend])
        for backend in BACKEND_CAPACITY
    }


def _summary_status(item: LaunchItem) -> str | None:
    summary = item.run_dir / "logs" / "run_summary.json"
    if not summary.exists():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def item_is_done(item: LaunchItem) -> bool:
    return _summary_status(item) == "finished"


def item_is_failed(item: LaunchItem) -> bool:
    return _summary_status(item) in {"error", "aborted", "interrupted"}


def item_is_running(item: LaunchItem) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={item.session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _retry_candidate(item: LaunchItem, retry: int) -> LaunchItem:
    return LaunchItem(
        method=item.method,
        task=item.task,
        repeat=item.repeat,
        backend=None,
        session=f"{item.session}_retry{retry}",
        run_name=f"{item.run_name}_retry{retry}",
        run_dir=item.run_dir.parent / f"{item.run_name}_retry{retry}",
        seed=item.seed,
    )


def item_active_attempt(item: LaunchItem) -> LaunchItem | None:
    if item_is_running(item):
        return item
    for retry in range(2, MAX_RETRIES + 1):
        candidate = _retry_candidate(item, retry)
        if item_is_running(candidate):
            return candidate
    return None


def item_has_successful_result(item: LaunchItem) -> bool:
    if item_is_done(item):
        return True
    for retry in range(2, MAX_RETRIES + 1):
        if item_is_done(_retry_candidate(item, retry)):
            return True
    return False


def pending_items(plan: list[LaunchItem]) -> list[LaunchItem]:
    pending = []
    for item in plan:
        if item_has_successful_result(item):
            continue
        if item_active_attempt(item) is not None:
            continue
        attempts = [item] + [
            _retry_candidate(item, retry) for retry in range(2, MAX_RETRIES + 1)
        ]
        for candidate in attempts:
            if item_is_done(candidate):
                break
            if item_is_running(candidate):
                break
            if candidate.run_dir.exists() and not item_is_failed(candidate):
                print(f"skip incomplete run_dir={candidate.run_dir}", flush=True)
                break
            if item_is_failed(candidate):
                continue
            pending.append(candidate)
            break
        else:
            print(
                f"skip {item.run_name}: exceeded MAX_RETRIES={MAX_RETRIES}",
                flush=True,
            )
    return pending


def assign_backends(
    pending: list[LaunchItem],
    *,
    preferred: tuple[BackendName, ...] = ("zhong", "server1", "local"),
) -> list[LaunchItem]:
    remaining = dict(free_slots())
    assigned: list[LaunchItem] = []
    for item in pending:
        chosen: BackendName | None = None
        for backend in preferred:
            if remaining.get(backend, 0) > 0:
                chosen = backend
                remaining[backend] -= 1
                break
        if chosen is None:
            break
        assigned.append(item.with_backend(chosen))
    return assigned


def validate_new_items(items: list[LaunchItem]) -> None:
    sessions = [item.session for item in items]
    run_dirs = [item.run_dir for item in items]
    if len(sessions) != len(set(sessions)):
        raise ValueError("tmux session names must be unique")
    if len(run_dirs) != len(set(run_dirs)):
        raise ValueError("run directories must be unique")
    collisions = [str(path) for path in run_dirs if path.exists()]
    if collisions:
        raise FileExistsError(f"run directories already exist: {collisions}")
    active = [item.session for item in items if item_is_running(item)]
    if active:
        raise RuntimeError(f"tmux sessions already exist: {active}")


def launch_items(items: list[LaunchItem], *, dry_run: bool) -> None:
    for item in items:
        printable = shlex.join(item.command())
        print(
            f"{item.backend:7s} {item.method:10s} {item.task:20s} "
            f"rep={item.repeat} session={item.session} command={printable}",
            flush=True,
        )
        if dry_run:
            continue
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                item.session,
                "-c",
                str(REPO_ROOT),
                printable,
            ],
            check=True,
        )


def fill_once(plan: list[LaunchItem], *, dry_run: bool) -> list[LaunchItem]:
    pending = pending_items(plan)
    assigned = assign_backends(pending)
    if not assigned:
        print(
            "no free slots or no pending runs; "
            f"free={free_slots()} pending={len(pending)}",
            flush=True,
        )
        return []
    if not dry_run:
        validate_new_items(assigned)
    launch_items(assigned, dry_run=dry_run)
    return assigned


def watch_and_fill(
    plan: list[LaunchItem],
    *,
    interval_sec: int,
    dry_run: bool,
) -> None:
    print(
        f"watching fair-1000 batch; interval={interval_sec}s; "
        f"capacity={BACKEND_CAPACITY}; total={len(plan)}",
        flush=True,
    )
    while True:
        remaining = pending_items(plan)
        running = sum(1 for item in plan if item_active_attempt(item) is not None)
        done = sum(1 for item in plan if item_has_successful_result(item))
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"done={done} running={running} pending={len(remaining)} "
            f"free={free_slots()}",
            flush=True,
        )
        if done == len(plan):
            print("all fair-1000 runs finished", flush=True)
            return
        fill_once(plan, dry_run=dry_run)
        if dry_run:
            return
        time.sleep(interval_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch PathWise+ReEvo+ShinkaEvolve fair-budget (1000) experiments "
            "into free LLM backend slots."
        )
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--batch",
        default=datetime.now().strftime("fair1000_%Y%m%d_%H%M%S"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats != 3:
        raise ValueError("fair comparison batch uses exactly 3 repeats")
    plan = build_launch_plan(args)
    print(f"batch={args.batch} total={len(plan)} free={free_slots()}", flush=True)
    if args.watch:
        watch_and_fill(plan, interval_sec=args.watch_interval, dry_run=args.dry_run)
        return
    fill_once(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
