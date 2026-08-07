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

from . import run

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_SHORT = {
    "tsp_construct": "tsp",
    "cvrp_aco": "cvrp",
    "op_aco": "op",
    "online_bin_packing": "obp",
}
BACKEND_CAPACITY: dict[run.BackendName, int] = {
    "zhong": 6,
    "server1": 6,
    "local": 3,
}
BACKEND_MARKERS: dict[run.BackendName, tuple[str, ...]] = {
    "zhong": ("--backend zhong", "183.36.243.124"),
    "server1": ("--backend server1", "222.201.145.8"),
    "local": ("--backend local", "127.0.0.1:8001"),
}


@dataclass(frozen=True, slots=True)
class LaunchItem:
    task: run.TaskName
    repeat: int
    backend: run.BackendName | None
    session: str
    run_name: str
    run_dir: Path
    seed: int

    def with_backend(self, backend: run.BackendName) -> LaunchItem:
        return LaunchItem(
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
            "experiments.runners.calm.run",
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
        )


def build_launch_plan(args: argparse.Namespace) -> list[LaunchItem]:
    plan = []
    for repeat in range(1, args.repeats + 1):
        for task in run.TASKS:
            short = TASK_SHORT[task]
            run_name = f"{args.batch}_{short}_calm_rep{repeat}"
            session = f"{args.session_prefix}_{short}_r{repeat}"
            run_dir = REPO_ROOT / "experiments" / task / "calm" / run_name
            plan.append(
                LaunchItem(
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
        if not text:
            continue
        if "uv run" in text:
            continue
        if "python" not in text:
            continue
        if "experiments." not in text and "run_experiment" not in text:
            continue
        lines.append(text)
    return lines


def count_backend_usage() -> dict[run.BackendName, int]:
    counts: dict[run.BackendName, int] = {name: 0 for name in BACKEND_CAPACITY}
    for cmdline in _process_cmdlines():
        matched: run.BackendName | None = None
        for backend, markers in BACKEND_MARKERS.items():
            if any(marker in cmdline for marker in markers):
                matched = backend
                break
        if matched is not None:
            counts[matched] += 1
    return counts


def free_slots() -> dict[run.BackendName, int]:
    usage = count_backend_usage()
    return {
        backend: max(0, BACKEND_CAPACITY[backend] - usage[backend])
        for backend in BACKEND_CAPACITY
    }


def item_is_done(item: LaunchItem) -> bool:
    summary = item.run_dir / "logs" / "run_summary.json"
    if not summary.exists():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "finished"


def item_is_running(item: LaunchItem) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", item.session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def assign_backends(
    pending: list[LaunchItem],
    *,
    preferred: tuple[run.BackendName, ...] = ("zhong", "server1", "local"),
) -> list[LaunchItem]:
    remaining = dict(free_slots())
    assigned: list[LaunchItem] = []
    for item in pending:
        chosen: run.BackendName | None = None
        for backend in preferred:
            if remaining.get(backend, 0) > 0:
                chosen = backend
                remaining[backend] -= 1
                break
        if chosen is None:
            break
        assigned.append(item.with_backend(chosen))
    return assigned


def launch_item(item: LaunchItem) -> None:
    # Do not pre-create run_dir: run.resolve_run_dir uses mkdir(exist_ok=False).
    cmd = " ".join(shlex.quote(part) for part in item.command())
    inner = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"uv run {cmd}; echo exit:$?"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", item.session, "bash", "-lc", inner],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch CALM (w/o GRPO) runs at 1000-eval budget in tmux."
    )
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default="calm")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = build_launch_plan(args)
    pending = [item for item in plan if not item_is_done(item) and not item_is_running(item)]
    assigned = assign_backends(pending)
    print(f"plan={len(plan)} pending={len(pending)} launch_now={len(assigned)}")
    for item in assigned:
        print(f"launch {item.session} backend={item.backend} -> {item.run_dir}")
        if not args.dry_run:
            launch_item(item)
            time.sleep(1)


if __name__ == "__main__":
    main()
