"""Schedule a fresh, repaired TraceAAD V9.19 batch.

This launcher uses ``server3``, ``server1``, ``server3b`` and ``local``.
It keeps the repaired batch in distinct run directories, fills every currently
available backend slot, and polls until a completed run frees another slot.

Examples::

    uv run python -m experiments.runners.traceaad_v9_19.launch_fixed
    uv run python -m experiments.runners.traceaad_v9_19.launch_fixed --dry-run
    uv run python -m experiments.runners.traceaad_v9_19.launch_fixed --once
"""

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
    BACKEND_CAPACITY,
    EXPERIMENTS_ROOT,
    REPO_ROOT,
    _process_cmdlines,
    detect_backend,
    free_slots,
)

BUDGET = 1000
INITIAL_ROOTS = 8
VERSION = "v9_19"
DEFAULT_BATCH = "v9_19_fixed_20260829"
DEFAULT_SESSION_PREFIX = "v919fix"
POLL_SECONDS = 30.0

# server3 currently has three user-approved free slots.  Keep this guard so a
# stale process listing cannot consume the fourth slot that is reserved by the
# user; subsequent completions are filled one at a time on later polls.
ALLOWED_BACKENDS = ("server3", "server1", "server3b", "local")
SERVER3_FREE_LIMIT = 3


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
    batch: str = DEFAULT_BATCH,
    session_prefix: str = DEFAULT_SESSION_PREFIX,
    repeats: int = 3,
) -> list[RunItem]:
    if not batch or any(char in batch for char in "/\\"):
        raise ValueError("batch must be a non-empty path-free name")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    tasks = (
        "tsp_construct",
        "cvrp_aco",
        "op_aco",
        "online_bin_packing",
        "vrptw_construct",
    )
    plan: list[RunItem] = []
    for repeat in range(1, repeats + 1):
        for task in tasks:
            run_name = f"{batch}_{task}_rep{repeat}"
            plan.append(
                RunItem(
                    task=task,
                    repeat=repeat,
                    run_name=run_name,
                    session=f"{session_prefix}_{task}_{repeat}",
                    run_dir=experiments_root / task / f"traceaad_{VERSION}" / run_name,
                )
            )
    return plan


def _checkpoint(item: RunItem) -> Path:
    return item.run_dir / "checkpoints" / "latest.json"


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
    summary = _summary(item)
    return summary.get("status") == "finished" and summary.get("budget_slots") == BUDGET


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


def command_for(item: RunItem, backend: str) -> list[str]:
    if backend not in BACKEND_CAPACITY or backend not in ALLOWED_BACKENDS:
        raise ValueError(f"backend is not allowed for the repaired batch: {backend}")
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad_v9_19.run",
        "--task",
        item.task,
        "--backend",
        backend,
        "--budget",
        str(BUDGET),
        "--n-init",
        str(INITIAL_ROOTS),
        "--repeat",
        str(item.repeat),
        "--seed",
        str(item.repeat - 1),
        "--experiments-root",
        str(item.run_dir.parents[2]),
    ]
    checkpoint = _checkpoint(item)
    if checkpoint.is_file():
        command.extend(("--resume-from", str(item.run_dir)))
    else:
        if item.run_dir.exists():
            raise RuntimeError(
                "run directory exists without a checkpoint; inspect before "
                f"launching: {item.run_dir}"
            )
        command.extend(("--run-name", item.run_name))
    return command


def _pending(plan: list[RunItem]) -> list[RunItem]:
    return [item for item in plan if not is_finished(item) and not is_running(item)]


def _batch_backend_usage(batch: str) -> dict[str, int]:
    usage = {backend: 0 for backend in ALLOWED_BACKENDS}
    for cmdline in _process_cmdlines():
        if batch not in cmdline:
            continue
        backend = detect_backend(cmdline)
        if backend in usage:
            usage[backend] += 1
    return usage


def available_slots(batch: str | None = None) -> dict[str, int]:
    # ``free_slots`` already subtracts every live run on the host, including
    # runs from this batch.  Do not subtract batch-local processes a second
    # time: doing so made a batch appear to have fewer slots than the host
    # actually exposed and prevented immediate backfilling after completion.
    slots = free_slots()
    slots["server3"] = min(slots.get("server3", 0), SERVER3_FREE_LIMIT)
    return {backend: max(0, slots.get(backend, 0)) for backend in ALLOWED_BACKENDS}


def assign_backends(
    items: list[RunItem], *, batch: str | None = None
) -> list[tuple[RunItem, str]]:
    remaining = available_slots(batch)
    assigned: list[tuple[RunItem, str]] = []
    for item in items:
        candidates = [name for name in ALLOWED_BACKENDS if remaining[name] > 0]
        if not candidates:
            break
        backend = max(
            candidates,
            key=lambda name: (remaining[name], -ALLOWED_BACKENDS.index(name)),
        )
        remaining[backend] -= 1
        assigned.append((item, backend))
    return assigned


def launch_available(
    plan: list[RunItem], *, batch: str, dry_run: bool = False
) -> int:
    assigned = assign_backends(_pending(plan), batch=batch)
    for item, backend in assigned:
        command = command_for(item, backend)
        action = "resume" if _checkpoint(item).is_file() else "launch"
        print(
            f"{action} backend={backend} task={item.task} rep={item.repeat} "
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
                *command,
            ],
            check=True,
        )
    return len(assigned)


def watch(
    plan: list[RunItem],
    *,
    batch: str,
    poll_seconds: float,
    dry_run: bool,
    once: bool,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    while True:
        done = sum(is_finished(item) for item in plan)
        running = sum(is_running(item) for item in plan)
        pending = len(plan) - done - running
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"total={len(plan)} done={done} running={running} pending={pending} "
            f"free={available_slots(batch)}",
            flush=True,
        )
        if done == len(plan):
            return
        launched = launch_available(plan, batch=batch, dry_run=dry_run)
        if dry_run or once:
            return
        if launched == 0:
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_ROOT)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--session-prefix", default=DEFAULT_SESSION_PREFIX)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=POLL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        experiments_root=args.experiments_root.resolve(),
        batch=args.batch,
        session_prefix=args.session_prefix,
        repeats=args.repeats,
    )
    watch(
        plan,
        batch=args.batch,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        once=args.once,
    )


if __name__ == "__main__":
    main()
