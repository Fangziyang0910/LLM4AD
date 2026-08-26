"""Schedule the TraceAAD V9.18-R0 A-stage batch.

The scheduler is restartable.  It skips finished runs, leaves active tmux
sessions alone, resumes only directories with a V9.18 checkpoint, and uses
free backend capacity for any pending run.  With ``--watch`` it also evaluates
all five tasks for both arms after the search batch closes and writes the
process audit.

Examples::

    uv run python -m experiments.runners.traceaad.launch_v918 --dry-run
    uv run python -m experiments.runners.traceaad.launch_v918 --watch
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
    TASKS,
    count_backend_usage,
    free_slots,
)

BUDGET = 1000
INITIAL_ROOTS = 8
PRIMARY_BACKENDS = ("server3", "server3b", "server1", "local")
ARMS = (
    ("q_atomic", "v9_18_q_atomic", "A0q"),
    ("q_opportunity", "v9_18_q_opportunity", "A1o"),
)


@dataclass(frozen=True, slots=True)
class RunItem:
    task: str
    repeat: int
    version: str
    arm_tag: str
    run_name: str
    session: str
    run_dir: Path
    initialization_checkpoint: Path

    @property
    def method_dir(self) -> Path:
        return self.run_dir.parent


def build_plan(
    *,
    experiments_root: Path = EXPERIMENTS_ROOT,
    session_prefix: str = "v918",
    repeats: int = 3,
) -> list[RunItem]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    plan: list[RunItem] = []
    for _, version, arm_tag in ARMS:
        for repeat in range(1, repeats + 1):
            for task in TASKS:
                run_name = f"v9_18_{arm_tag}_{task}_rep{repeat}"
                run_dir = (
                    experiments_root
                    / task
                    / f"traceaad_{version}"
                    / run_name
                )
                bootstrap = (
                    experiments_root
                    / task
                    / "traceaad_v9_18_q_atomic"
                    / f"v9_18_bootstrap_{task}_rep{repeat}"
                    / "checkpoints"
                    / "latest.json"
                )
                plan.append(
                    RunItem(
                        task=task,
                        repeat=repeat,
                        version=version,
                        arm_tag=arm_tag,
                        run_name=run_name,
                        session=f"{session_prefix}_{arm_tag}_{task}_{repeat}",
                        run_dir=run_dir,
                        initialization_checkpoint=bootstrap,
                    )
                )
    return plan


def _read_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "logs" / "summary.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_finished(item: RunItem) -> bool:
    summary = _read_summary(item.run_dir)
    return summary.get("status") == "finished" and summary.get("budget_slots") == BUDGET


def is_running(item: RunItem) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={item.session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _checkpoint(item: RunItem) -> Path:
    return item.run_dir / "checkpoints" / "latest.json"


def command_for(item: RunItem, backend: str) -> list[str]:
    if backend not in BACKEND_CAPACITY:
        raise ValueError(f"unknown backend: {backend}")
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
                f"run directory exists without a checkpoint; inspect before resuming: {item.run_dir}"
            )
        if not item.initialization_checkpoint.is_file():
            raise FileNotFoundError(
                f"bootstrap checkpoint is missing: {item.initialization_checkpoint}"
            )
        command.extend(
            (
                "--run-name",
                item.run_name,
                "--initialization-checkpoint",
                str(item.initialization_checkpoint),
            )
        )
    return command


def _launch(item: RunItem, backend: str, *, dry_run: bool) -> None:
    command = command_for(item, backend)
    action = "resume" if _checkpoint(item).is_file() else "launch"
    print(
        f"{action} backend={backend} task={item.task} arm={item.arm_tag} "
        f"rep={item.repeat} session={item.session}",
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


def _pending(plan: list[RunItem]) -> list[RunItem]:
    pending: list[RunItem] = []
    for item in plan:
        if is_finished(item) or is_running(item):
            continue
        pending.append(item)
    return pending


def _assign_backends(items: list[RunItem]) -> list[tuple[RunItem, str]]:
    remaining = free_slots()
    assigned: list[tuple[RunItem, str]] = []
    for item in items:
        candidates = [
            backend
            for backend in PRIMARY_BACKENDS
            if remaining.get(backend, 0) > 0
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
    pending = _pending(plan)
    assigned = _assign_backends(pending)
    for item, backend in assigned:
        _launch(item, backend, dry_run=dry_run)
    if not assigned:
        print(
            f"no launch: pending={len(pending)} free={free_slots()} "
            f"usage={count_backend_usage()}",
            flush=True,
        )
    return len(assigned)


def _grouped(plan: list[RunItem]) -> dict[tuple[str, str], list[RunItem]]:
    groups: dict[tuple[str, str], list[RunItem]] = {}
    for item in plan:
        groups.setdefault((item.task, item.arm_tag), []).append(item)
    return groups


def _heldout_complete(output_dir: Path, items: list[RunItem]) -> bool:
    path = output_dir / "results.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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


def run_heldout(
    plan: list[RunItem], *, eval_workers: int, dry_run: bool = False
) -> int:
    completed_groups = 0
    for (task, arm_tag), items in sorted(_grouped(plan).items()):
        if not all(is_finished(item) for item in items):
            continue
        output_dir = (
            items[0].method_dir / f"eval_best_v918_{arm_tag}_{task}_complete"
        )
        if _heldout_complete(output_dir, items):
            completed_groups += 1
            continue
        command = [
            sys.executable,
            "experiments/evaluate_best.py",
            *(str(item.run_dir) for item in items),
            "--output-dir",
            str(output_dir),
            "--workers",
            str(eval_workers),
        ]
        print(f"heldout task={task} arm={arm_tag} output={output_dir}", flush=True)
        if dry_run:
            continue
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"heldout failed task={task} arm={arm_tag} "
                f"returncode={exc.returncode}; will retry",
                flush=True,
            )
            continue
        completed_groups += 1
    return completed_groups


def all_heldout_complete(plan: list[RunItem]) -> bool:
    for items in _grouped(plan).values():
        if not all(is_finished(item) for item in items):
            return False
        output_dir = items[0].method_dir / (
            f"eval_best_v918_{items[0].arm_tag}_{items[0].task}_complete"
        )
        if not _heldout_complete(output_dir, items):
            return False
    return True


def run_process_audit(plan: list[RunItem], *, dry_run: bool = False) -> None:
    run_dirs = [str(item.run_dir) for item in plan]
    json_output = REPO_ROOT / "docs" / "analysis" / "traceaad_v918_process" / "summary.json"
    markdown_output = REPO_ROOT / "docs" / "analysis" / "TraceAAD-V9.18-A阶段过程审计.md"
    command = [
        sys.executable,
        "experiments/analysis/analyze_v918_process.py",
        *sum((["--run-dir", path] for path in run_dirs), []),
        "--json-output",
        str(json_output),
        "--markdown-output",
        str(markdown_output),
    ]
    print(f"process-audit output={markdown_output}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def watch(
    plan: list[RunItem],
    *,
    poll_seconds: float,
    eval_workers: int,
    dry_run: bool,
    auto_heldout: bool,
) -> None:
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    audit_done = False
    while True:
        done = sum(is_finished(item) for item in plan)
        running = sum(is_running(item) for item in plan)
        pending = len(plan) - done - running
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"total={len(plan)} done={done} running={running} pending={pending} "
            f"free={free_slots()}",
            flush=True,
        )
        if done == len(plan):
            if auto_heldout:
                run_heldout(plan, eval_workers=eval_workers, dry_run=dry_run)
                if not dry_run and not all_heldout_complete(plan):
                    time.sleep(poll_seconds)
                    continue
            if not dry_run and not audit_done:
                run_process_audit(plan)
                audit_done = True
            return
        launched = launch_available(plan, dry_run=dry_run)
        if dry_run:
            return
        if launched == 0:
            time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_ROOT)
    parser.add_argument("--session-prefix", default="v918")
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
        experiments_root=args.experiments_root.resolve(),
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
