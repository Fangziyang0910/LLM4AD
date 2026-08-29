"""Schedule the TraceAAD V9.20 batch across machines.

The scheduler runs on the local workstation and launches every runner on
server1 (``~/code/LLM4AD/LLM4AD``), so evaluation CPU lands on server1
while generation fans out to the server1/server3/server3b vLLM slots.
Slot accounting is global: a backend counts as busy if either the local
workstation or server1 runs a runner against it, which keeps the batch
inside the capacity that the local V9.19 searches are still occupying.
The scheduler is restartable: it skips finished runs, leaves active tmux
sessions alone, resumes directories holding a V9.20 checkpoint, and with
``--watch`` fills slots as earlier searches finish, then evaluates all
five tasks on the held-out splits once the search batch closes.

Examples::

    uv run python -m experiments.runners.traceaad.launch_v920 --dry-run
    uv run python -m experiments.runners.traceaad.launch_v920 --watch
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .._common import (
    BACKEND_CAPACITY,
    TASKS,
    count_backend_usage,
    detect_backend,
)

BUDGET = 1000
INITIAL_ROOTS = 8
PRIMARY_BACKENDS = ("server3", "server3b", "server1")
VERSION = "v9_20"
SESSION_PREFIX = "v920"

REMOTE_HOST = "B3-server1"
REMOTE_REPO = Path("/home/fzy/code/LLM4AD/LLM4AD")
REMOTE_PYTHON = REMOTE_REPO / ".venv" / "bin" / "python"
REMOTE_EXPERIMENTS_ROOT = REMOTE_REPO / "experiments"


@dataclass(frozen=True, slots=True)
class RunItem:
    task: str
    repeat: int
    run_name: str
    session: str
    run_dir: Path

    @property
    def method_dir(self) -> Path:
        return self.run_dir.parent


@dataclass(slots=True)
class RemoteState:
    sessions: set[str] = field(default_factory=set)
    existing_dirs: set[Path] = field(default_factory=set)
    checkpoints: set[Path] = field(default_factory=set)
    summaries: dict[Path, dict] = field(default_factory=dict)
    ps_lines: list[str] = field(default_factory=list)


def build_plan(
    *,
    experiments_root: Path = REMOTE_EXPERIMENTS_ROOT,
    session_prefix: str = SESSION_PREFIX,
    repeats: int = 3,
) -> list[RunItem]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    plan: list[RunItem] = []
    for repeat in range(1, repeats + 1):
        for task in TASKS:
            run_name = f"{VERSION}_{task}_rep{repeat}"
            run_dir = experiments_root / task / f"traceaad_{VERSION}" / run_name
            plan.append(
                RunItem(
                    task=task,
                    repeat=repeat,
                    run_name=run_name,
                    session=f"{session_prefix}_{task}_{repeat}",
                    run_dir=run_dir,
                )
            )
    return plan


def _ssh(command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", REMOTE_HOST, command],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def fetch_remote_state(plan: list[RunItem]) -> RemoteState:
    """One round trip: tmux sessions, dirs, checkpoints, summaries, ps."""
    dirs = [str(item.run_dir) for item in plan]
    script = [
        "echo @@TMUX@@",
        "tmux ls -F '#{session_name}' 2>/dev/null || true",
        "echo @@PS@@",
        "ps -eo args= || true",
    ]
    for directory in dirs:
        script.append(f'[ -d "{directory}" ] && echo "DIR {directory}" || true')
        script.append(
            f'[ -f "{directory}/checkpoints/latest.json" ] '
            f'&& echo "CKPT {directory}" || true'
        )
    for directory in dirs:
        script.append(f'echo "SUM {directory}"')
        script.append(f'cat "{directory}/logs/summary.json" 2>/dev/null || true')
        script.append("echo @@ENDSUM@@")
    state = RemoteState()
    section = None
    pending_dir: Path | None = None
    summary_lines: list[str] = []
    for line in _ssh("\n".join(script)).splitlines():
        if line == "@@TMUX@@":
            section = "tmux"
            continue
        if line == "@@PS@@":
            section = "ps"
            continue
        if line == "@@ENDSUM@@":
            if pending_dir is not None:
                text = "\n".join(summary_lines).strip()
                if text:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        state.summaries[pending_dir] = payload
            pending_dir = None
            summary_lines = []
            continue
        if line.startswith("DIR "):
            state.existing_dirs.add(Path(line[4:]))
        elif line.startswith("CKPT "):
            state.checkpoints.add(Path(line[5:]))
        elif line.startswith("SUM "):
            pending_dir = Path(line[4:])
            state.summaries.setdefault(pending_dir, {})
            summary_lines = []
        elif pending_dir is not None:
            summary_lines.append(line)
        elif section == "tmux" and line.strip():
            state.sessions.add(line.strip())
        elif section == "ps":
            state.ps_lines.append(line)
    return state


def remote_backend_usage(state: RemoteState) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in BACKEND_CAPACITY}
    seen: set[str] = set()
    for line in state.ps_lines:
        text = line.strip()
        if not text or "uv run" in text or "python" not in text:
            continue
        if "experiments." not in text and "run_experiment" not in text:
            continue
        if ".launch" in text:
            continue
        if text in seen:
            continue
        seen.add(text)
        backend = detect_backend(text)
        if backend is not None:
            counts[backend] += 1
    return counts


def free_slots(state: RemoteState) -> dict[str, int]:
    local = count_backend_usage()
    remote = remote_backend_usage(state)
    return {
        backend: max(
            0,
            BACKEND_CAPACITY[backend] - local.get(backend, 0) - remote.get(backend, 0),
        )
        for backend in PRIMARY_BACKENDS
    }


def is_finished(item: RunItem, state: RemoteState) -> bool:
    summary = state.summaries.get(item.run_dir, {})
    return summary.get("status") == "finished" and summary.get("budget_slots") == BUDGET


def is_running(item: RunItem, state: RemoteState) -> bool:
    return item.session in state.sessions


def command_for(item: RunItem) -> list[str]:
    return [
        str(REMOTE_PYTHON),
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        item.task,
        "--version",
        VERSION,
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


def _launch(item: RunItem, backend: str, *, dry_run: bool) -> None:
    command = command_for(item)
    if item.run_dir in _state.checkpoints:
        command.extend(("--resume-from", str(item.run_dir)))
        action = "resume"
    else:
        if item.run_dir in _state.existing_dirs:
            raise RuntimeError(
                "run directory exists without a checkpoint; inspect before "
                f"resuming: {item.run_dir}"
            )
        command.extend(("--run-name", item.run_name))
        action = "launch"
    command.extend(("--backend", backend))
    print(
        f"{action} backend={backend} task={item.task} rep={item.repeat} "
        f"session={item.session}",
        flush=True,
    )
    if dry_run:
        return
    remote = " ".join(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            item.session,
            "-c",
            str(REMOTE_REPO),
            "--",
            shlex.join(command),
        ]
    )
    _ssh(remote)


_state = RemoteState()


def _pending(plan: list[RunItem]) -> list[RunItem]:
    return [
        item
        for item in plan
        if not is_finished(item, _state) and not is_running(item, _state)
    ]


def _assign_backends(items: list[RunItem]) -> list[tuple[RunItem, str]]:
    remaining = free_slots(_state)
    assigned: list[tuple[RunItem, str]] = []
    for item in items:
        candidates = [
            backend for backend in PRIMARY_BACKENDS if remaining.get(backend, 0) > 0
        ]
        if not candidates:
            break
        backend = max(
            candidates,
            key=lambda name: (remaining[name], -PRIMARY_BACKENDS.index(name)),
        )
        remaining[backend] -= 1
        assigned.append((item, backend))
    return assigned


def launch_available(plan: list[RunItem], *, dry_run: bool = False) -> int:
    global _state
    _state = fetch_remote_state(plan)
    pending = _pending(plan)
    assigned = _assign_backends(pending)
    for item, backend in assigned:
        _launch(item, backend, dry_run=dry_run)
    if not assigned:
        print(
            f"no launch: pending={len(pending)} free={free_slots(_state)} "
            f"local={dict(count_backend_usage())}",
            flush=True,
        )
    return len(assigned)


def _grouped(plan: list[RunItem]) -> dict[str, list[RunItem]]:
    groups: dict[str, list[RunItem]] = {}
    for item in plan:
        groups.setdefault(item.task, []).append(item)
    return groups


def _heldout_complete(output_dir: Path, items: list[RunItem]) -> bool:
    try:
        text = _ssh(f'cat "{output_dir / "results.json"}" 2>/dev/null || true')
    except subprocess.CalledProcessError:
        return False
    if not text.strip():
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    names = {
        row.get("run_name")
        for row in payload.get("run_records", [])
        if isinstance(row, dict)
    }
    if not all(item.run_name in names for item in items):
        return False
    n_runs = len(items)
    sections = (
        payload.get("eval_results_by_size"),
        payload.get("eval_results_by_scale"),
        payload.get("results_by_size"),
        payload.get("results_by_split"),
    )
    section = next((value for value in sections if isinstance(value, dict)), None)
    if not section:
        return False
    for unit in section.values():
        if not isinstance(unit, dict):
            return False
        summary = unit.get("summary")
        if not isinstance(summary, dict):
            return False
        if summary.get("num_runs") != n_runs:
            return False
        if summary.get("num_successful_eval_runs") != n_runs:
            return False
    return True


def run_heldout(plan: list[RunItem], *, eval_workers: int, dry_run: bool = False) -> int:
    completed_groups = 0
    for task, items in sorted(_grouped(plan).items()):
        if not all(is_finished(item, _state) for item in items):
            continue
        output_dir = items[0].method_dir / f"eval_best_{VERSION}_{task}_complete"
        if _heldout_complete(output_dir, items):
            completed_groups += 1
            continue
        command = " ".join(
            [
                f"cd {REMOTE_REPO}",
                "&&",
                str(REMOTE_PYTHON),
                "experiments/evaluate_best.py",
                *[f'"{item.run_dir}"' for item in items],
                "--output-dir",
                f'"{output_dir}"',
                "--workers",
                str(eval_workers),
            ]
        )
        print(f"heldout task={task} output={output_dir}", flush=True)
        if dry_run:
            continue
        try:
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", REMOTE_HOST, command],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"heldout failed task={task} "
                f"returncode={exc.returncode}; will retry",
                flush=True,
            )
            continue
        completed_groups += 1
    return completed_groups


def all_heldout_complete(plan: list[RunItem]) -> bool:
    for task, items in _grouped(plan).items():
        if not all(is_finished(item, _state) for item in items):
            return False
        output_dir = items[0].method_dir / f"eval_best_{VERSION}_{task}_complete"
        if not _heldout_complete(output_dir, items):
            return False
    return True


def watch(
    plan: list[RunItem],
    *,
    poll_seconds: float,
    eval_workers: int,
    dry_run: bool,
    auto_heldout: bool,
) -> None:
    global _state
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    while True:
        _state = fetch_remote_state(plan)
        done = sum(is_finished(item, _state) for item in plan)
        running = sum(is_running(item, _state) for item in plan)
        pending = len(plan) - done - running
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"total={len(plan)} done={done} running={running} pending={pending} "
            f"free={free_slots(_state)}",
            flush=True,
        )
        if done == len(plan):
            if auto_heldout:
                run_heldout(plan, eval_workers=eval_workers, dry_run=dry_run)
                if not dry_run and not all_heldout_complete(plan):
                    time.sleep(poll_seconds)
                    continue
            return
        launched = launch_available(plan, dry_run=dry_run)
        if dry_run:
            return
        if launched == 0:
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=REMOTE_EXPERIMENTS_ROOT)
    parser.add_argument("--session-prefix", default=SESSION_PREFIX)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-heldout", action="store_true")
    args = parser.parse_args()
    if args.eval_workers <= 0:
        raise ValueError("eval-workers must be positive")
    plan = build_plan(
        experiments_root=args.experiments_root,
        session_prefix=args.session_prefix,
        repeats=args.repeats,
    )
    if args.watch:
        watch(
            plan,
            poll_seconds=args.poll_seconds,
            eval_workers=args.eval_workers,
            dry_run=args.dry_run,
            auto_heldout=not args.no_heldout,
        )
    else:
        launch_available(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
