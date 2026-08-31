"""Launch the 15-run TraceAAD V9.22 batch across available LLM backends.

The launcher itself stays on the local workstation.  A run uses the local
checkout and sends generation to the selected backend, so local, server1,
server3, and server3b capacity can be filled from one scheduler.  Existing
runs on the workstation and on ``B3-server1`` are counted before assignment.

Examples::

    uv run python -m experiments.runners.traceaad.launch_v922 --dry-run
    uv run python -m experiments.runners.traceaad.launch_v922 --once
    uv run python -m experiments.runners.traceaad.launch_v922 --watch
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .._common import (
    BACKEND_CAPACITY,
    EXPERIMENTS_ROOT,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    _process_cmdlines,
    detect_backend,
)

VERSION = "v9_22"
BUDGET = 1000
INITIAL_ROOTS = 8
EVAL_WORKERS = 4
OUTPUT_TOKENS = 8192
CONTEXT_TOKEN_LIMIT = 32768
REMOTE_HOST = "B3-server1"
POLL_SECONDS = 60.0
BACKENDS = ("local", "server1", "server3", "server3b")


@dataclass(frozen=True, slots=True)
class RunItem:
    task: str
    repeat: int
    run_name: str
    session: str
    run_dir: Path


def build_plan(
    *,
    experiments_root: Path = EXPERIMENTS_ROOT,
    batch: str | None = None,
    session_prefix: str = "v922",
    repeats: int = 3,
) -> list[RunItem]:
    batch = batch or f"v9_22_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not batch or any(char in batch for char in "/\\"):
        raise ValueError("batch must be a non-empty path-free name")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    plan: list[RunItem] = []
    for repeat in range(1, repeats + 1):
        for task in TASKS:
            run_name = f"{batch}_{TASK_SHORT[task]}_rep{repeat}"
            plan.append(
                RunItem(
                    task=task,
                    repeat=repeat,
                    run_name=run_name,
                    session=f"{session_prefix}_{TASK_SHORT[task]}_r{repeat}",
                    run_dir=experiments_root / task / "traceaad_v9_22" / run_name,
                )
            )
    return plan


def _remote_process_lines() -> list[str]:
    command = "ps -eo args= || true"
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", REMOTE_HOST, command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _run_key(cmdline: str) -> str:
    match = re.search(r"--run-name\s+(\S+)", cmdline)
    if match:
        return match.group(1)
    match = re.search(r"--resume-from\s+(\S+)", cmdline)
    if match:
        return match.group(1)
    return cmdline


def _usage(lines: list[str]) -> dict[str, int]:
    counts = {name: 0 for name in BACKENDS}
    seen: set[tuple[str, str]] = set()
    for line in lines:
        text = line.strip()
        if not text or "python" not in text:
            continue
        if "experiments.runners" not in text and "run_experiment" not in text:
            continue
        if ".launch" in text:
            continue
        backend = detect_backend(text)
        if backend not in counts:
            continue
        key = (backend, _run_key(text))
        if key in seen:
            continue
        seen.add(key)
        counts[backend] += 1
    return counts


def available_slots() -> dict[str, int]:
    local = _usage(_process_cmdlines())
    remote = _usage(_remote_process_lines())
    return {
        backend: max(0, BACKEND_CAPACITY[backend] - local[backend] - remote[backend])
        for backend in BACKENDS
    }


def _summary(item: RunItem) -> dict[str, object]:
    path = item.run_dir / "logs" / "summary.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_finished(item: RunItem) -> bool:
    payload = _summary(item)
    return payload.get("status") == "finished" and payload.get("budget_slots") == BUDGET


def is_running(item: RunItem) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", f"={item.session}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _checkpoint(item: RunItem) -> Path:
    return item.run_dir / "checkpoints" / "latest.json"


def command_for(item: RunItem, backend: str) -> list[str]:
    if backend not in BACKENDS:
        raise ValueError(f"unsupported backend: {backend}")
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        VERSION,
        "--backend",
        backend,
        "--budget",
        str(BUDGET),
        "--n-init",
        str(INITIAL_ROOTS),
        "--eval-workers",
        str(EVAL_WORKERS),
        "--output-tokens",
        str(OUTPUT_TOKENS),
        "--context-token-limit",
        str(CONTEXT_TOKEN_LIMIT),
        "--repeat",
        str(item.repeat),
        "--seed",
        str(item.repeat - 1),
        "--experiments-root",
        str(EXPERIMENTS_ROOT),
    ]
    if _checkpoint(item).is_file():
        command.extend(("--resume-from", str(item.run_dir)))
    else:
        if item.run_dir.exists():
            raise RuntimeError(
                "run directory exists without a V9.22 checkpoint; inspect before "
                f"launching: {item.run_dir}"
            )
        command.extend(("--run-name", item.run_name))
    return command


def pending_items(plan: list[RunItem]) -> list[RunItem]:
    return [item for item in plan if not is_finished(item) and not is_running(item)]


def assign_backends(items: list[RunItem]) -> list[tuple[RunItem, str]]:
    remaining = available_slots()
    assigned: list[tuple[RunItem, str]] = []
    for item in items:
        candidates = [backend for backend in BACKENDS if remaining[backend] > 0]
        if not candidates:
            break
        backend = max(candidates, key=lambda name: (remaining[name], -BACKENDS.index(name)))
        remaining[backend] -= 1
        assigned.append((item, backend))
    return assigned


def launch_available(plan: list[RunItem], *, dry_run: bool = False) -> int:
    assignments = assign_backends(pending_items(plan))
    for item, backend in assignments:
        command = command_for(item, backend)
        print(
            f"{'resume' if _checkpoint(item).is_file() else 'launch'} "
            f"backend={backend} task={item.task} rep={item.repeat} "
            f"session={item.session}",
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
                "--",
                shlex.join(command),
            ],
            check=True,
        )
    return len(assignments)


def run_scheduler(
    plan: list[RunItem], *, dry_run: bool, watch: bool, poll_seconds: float
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    while True:
        done = sum(is_finished(item) for item in plan)
        running = sum(is_running(item) for item in plan)
        pending = len(plan) - done - running
        try:
            free = available_slots()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("could not query remote backend usage") from exc
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] total={len(plan)} "
            f"done={done} running={running} pending={pending} free={free}",
            flush=True,
        )
        if done == len(plan):
            return
        if not watch:
            launch_available(plan, dry_run=dry_run)
            return
        launched = launch_available(plan, dry_run=dry_run)
        if dry_run:
            return
        if launched == 0:
            time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_ROOT)
    parser.add_argument("--batch")
    parser.add_argument("--session-prefix", default="v922")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true", help="launch one capacity fill and return")
    parser.add_argument("--watch", action="store_true", help="keep filling slots until all runs finish")
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    args = parser.parse_args(argv)
    plan = build_plan(
        experiments_root=args.experiments_root.resolve(),
        batch=args.batch,
        session_prefix=args.session_prefix,
        repeats=args.repeats,
    )
    run_scheduler(
        plan,
        dry_run=args.dry_run,
        watch=args.watch and not args.once,
        poll_seconds=args.poll_seconds,
    )


if __name__ == "__main__":
    main()
