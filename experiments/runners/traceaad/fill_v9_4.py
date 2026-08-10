"""Continuously fill shared LLM slots with one formal TraceAAD V9.4 batch."""

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

from llm4ad.method.traceaad_v9_4 import PROTOCOL_ID

from .._common import (
    BACKEND_CAPACITY,
    BACKEND_MARKERS,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    BackendName,
    TaskName,
)

MODULE = "experiments.runners.traceaad.run"
METHOD = "traceaad_v9_4"
TERMINAL_FAILURES = frozenset({"error", "aborted", "interrupted", "stalled"})
PREFERRED_BACKENDS: tuple[BackendName, ...] = ("zhong", "server1", "local")


@dataclass(frozen=True, slots=True)
class V94Run:
    task: TaskName
    repeat: int
    session: str
    run_name: str
    run_dir: Path

    def command(self, backend: BackendName, *, budget: int) -> tuple[str, ...]:
        return (
            sys.executable,
            "-m",
            MODULE,
            "--task",
            self.task,
            "--version",
            "v9_4",
            "--backend",
            backend,
            "--budget",
            str(budget),
            "--n-init",
            "8",
            "--output-tokens",
            "8192",
            "--context-token-limit",
            "24576",
            "--seed",
            str(self.repeat),
            "--repeat",
            str(self.repeat),
            "--run-name",
            self.run_name,
        )


RunState = Literal["pending", "running", "finished", "failed", "orphaned"]


def usage_from_cmdlines(cmdlines: list[str]) -> dict[BackendName, int]:
    """Count experiments once even when evaluator children inherit the command."""
    seen: dict[BackendName, set[str]] = {backend: set() for backend in BACKEND_CAPACITY}
    for line in cmdlines:
        if "python" not in line or "uv run" in line:
            continue
        backend = next(
            (
                name
                for name, markers in BACKEND_MARKERS.items()
                if any(marker in line for marker in markers)
            ),
            None,
        )
        if backend is None:
            continue
        try:
            tokens = shlex.split(line)
            index = tokens.index("--run-name")
            identity = tokens[index + 1]
        except (ValueError, IndexError):
            identity = line
        seen[backend].add(identity)
    return {backend: len(identities) for backend, identities in seen.items()}


def backend_usage() -> dict[BackendName, int]:
    result = subprocess.run(
        ("ps", "-eo", "args="),
        check=True,
        capture_output=True,
        text=True,
    )
    return usage_from_cmdlines(result.stdout.splitlines())


def available_slots() -> dict[BackendName, int]:
    usage = backend_usage()
    return {
        backend: max(0, capacity - usage[backend])
        for backend, capacity in BACKEND_CAPACITY.items()
    }


def build_plan(*, batch: str, repeats: int) -> tuple[V94Run, ...]:
    return tuple(
        V94Run(
            task=task,
            repeat=repeat,
            session=f"v94_{batch}_{TASK_SHORT[task]}_r{repeat}",
            run_name=f"v9_4_{batch}_{TASK_SHORT[task]}_rep{repeat}",
            run_dir=(
                REPO_ROOT
                / "experiments"
                / task
                / METHOD
                / f"v9_4_{batch}_{TASK_SHORT[task]}_rep{repeat}"
            ),
        )
        for repeat in range(1, repeats + 1)
        for task in TASKS
    )


def summary_status(item: V94Run) -> str | None:
    path = item.run_dir / "logs" / "summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def session_running(item: V94Run) -> bool:
    result = subprocess.run(
        ("tmux", "has-session", "-t", f"={item.session}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def run_state(item: V94Run) -> RunState:
    status = summary_status(item)
    if status == "finished":
        return "finished"
    if session_running(item):
        return "running"
    if status in TERMINAL_FAILURES:
        return "failed"
    if item.run_dir.exists():
        return "orphaned"
    return "pending"


def assign_pending(
    pending: tuple[V94Run, ...],
    available: dict[BackendName, int],
) -> tuple[tuple[V94Run, BackendName], ...]:
    remaining = dict(available)
    assigned: list[tuple[V94Run, BackendName]] = []
    for item in pending:
        backend = next(
            (name for name in PREFERRED_BACKENDS if remaining.get(name, 0) > 0),
            None,
        )
        if backend is None:
            break
        remaining[backend] -= 1
        assigned.append((item, backend))
    return tuple(assigned)


def launch_assigned(
    assigned: tuple[tuple[V94Run, BackendName], ...],
    *,
    budget: int,
    dry_run: bool,
) -> None:
    for item, backend in assigned:
        command = item.command(backend, budget=budget)
        printable = shlex.join(command)
        print(
            f"launch backend={backend:7s} task={item.task:20s} "
            f"repeat={item.repeat} session={item.session}\n  {printable}",
            flush=True,
        )
        if dry_run:
            continue
        if item.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {item.run_dir}")
        subprocess.run(
            (
                "tmux",
                "new-session",
                "-d",
                "-s",
                item.session,
                "-c",
                str(REPO_ROOT),
                printable,
            ),
            check=True,
        )


def fill_once(
    plan: tuple[V94Run, ...],
    *,
    budget: int,
    dry_run: bool,
) -> tuple[tuple[V94Run, BackendName], ...]:
    pending = tuple(item for item in plan if run_state(item) == "pending")
    available = available_slots()
    assigned = assign_pending(pending, available)
    print(
        f"free={available} pending={len(pending)} launching={len(assigned)}",
        flush=True,
    )
    launch_assigned(assigned, budget=budget, dry_run=dry_run)
    return assigned


def state_counts(plan: tuple[V94Run, ...]) -> dict[RunState, int]:
    counts: dict[RunState, int] = {
        "pending": 0,
        "running": 0,
        "finished": 0,
        "failed": 0,
        "orphaned": 0,
    }
    for item in plan:
        counts[run_state(item)] += 1
    return counts


def watch(
    plan: tuple[V94Run, ...],
    *,
    budget: int,
    interval_seconds: int,
    dry_run: bool,
) -> None:
    while True:
        counts = state_counts(plan)
        usage = backend_usage()
        free = {
            backend: max(0, BACKEND_CAPACITY[backend] - usage[backend])
            for backend in BACKEND_CAPACITY
        }
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"states={counts} usage={usage} free={free}",
            flush=True,
        )
        if counts["finished"] == len(plan):
            print("all TraceAAD V9.4 runs finished", flush=True)
            return
        terminal = counts["finished"] + counts["failed"] + counts["orphaned"]
        if terminal == len(plan):
            blocked = [
                item.run_name
                for item in plan
                if run_state(item) in {"failed", "orphaned"}
            ]
            raise RuntimeError(
                "V9.4 batch contains failed or orphaned runs: " + ", ".join(blocked)
            )
        fill_once(plan, budget=budget, dry_run=dry_run)
        if dry_run:
            return
        time.sleep(interval_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously fill all shared LLM slots with one four-task, "
            "three-repeat TraceAAD V9.4 batch."
        )
    )
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.repeats != 3:
        raise ValueError("the formal V9.4 batch requires exactly three repeats")
    if args.budget <= 0 or args.interval_seconds <= 0:
        raise ValueError("budget and interval-seconds must be positive")
    plan = build_plan(batch=args.batch, repeats=args.repeats)
    print(
        f"protocol={PROTOCOL_ID} batch={args.batch} runs={len(plan)} "
        f"capacity={BACKEND_CAPACITY}",
        flush=True,
    )
    watch(
        plan,
        budget=args.budget,
        interval_seconds=args.interval_seconds,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
