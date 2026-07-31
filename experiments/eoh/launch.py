from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import run

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_SHORT = {
    "tsp_construct": "tsp",
    "cvrp_aco": "cvrp",
    "op_aco": "op",
    "online_bin_packing": "obp",
}

# Every task has runs on both servers; each server receives exactly six runs.
BACKEND_ASSIGNMENT = {
    (1, "tsp_construct"): "zhong",
    (1, "cvrp_aco"): "zhong",
    (1, "op_aco"): "server1",
    (1, "online_bin_packing"): "server1",
    (2, "tsp_construct"): "zhong",
    (2, "cvrp_aco"): "server1",
    (2, "op_aco"): "zhong",
    (2, "online_bin_packing"): "server1",
    (3, "tsp_construct"): "server1",
    (3, "cvrp_aco"): "zhong",
    (3, "op_aco"): "server1",
    (3, "online_bin_packing"): "zhong",
}


@dataclass(frozen=True, slots=True)
class LaunchItem:
    task: run.TaskName
    repeat: int
    backend: run.BackendName
    session: str
    run_name: str
    run_dir: Path
    command: tuple[str, ...]


def build_launch_plan(args: argparse.Namespace) -> list[LaunchItem]:
    plan = []
    for repeat in range(1, args.repeats + 1):
        for task in run.TASKS:
            backend = BACKEND_ASSIGNMENT[(repeat, task)]
            short = TASK_SHORT[task]
            run_name = f"{args.batch}_{short}_eoh_rep{repeat}"
            session = f"{args.session_prefix}_{short}_r{repeat}"
            run_dir = REPO_ROOT / "experiments" / task / "eoh" / run_name
            command = (
                sys.executable,
                "-m",
                "experiments.eoh.run",
                "--task",
                task,
                "--backend",
                backend,
                "--repeat",
                str(repeat),
                "--seed",
                str(repeat - 1),
                "--run-name",
                run_name,
            )
            plan.append(
                LaunchItem(
                    task=task,
                    repeat=repeat,
                    backend=backend,
                    session=session,
                    run_name=run_name,
                    run_dir=run_dir,
                    command=command,
                )
            )
    return plan


def validate_plan(plan: list[LaunchItem]) -> None:
    sessions = [item.session for item in plan]
    run_dirs = [item.run_dir for item in plan]
    if len(sessions) != len(set(sessions)):
        raise ValueError("tmux session names must be unique")
    if len(run_dirs) != len(set(run_dirs)):
        raise ValueError("run directories must be unique")
    collisions = [str(path) for path in run_dirs if path.exists()]
    if collisions:
        raise FileExistsError(f"run directories already exist: {collisions}")
    active = []
    for session in sessions:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode == 0:
            active.append(session)
    if active:
        raise RuntimeError(f"tmux sessions already exist: {active}")


def launch(plan: list[LaunchItem], *, dry_run: bool) -> None:
    for item in plan:
        printable = shlex.join(item.command)
        print(
            f"{item.backend:7s} {item.task:20s} rep={item.repeat} "
            f"session={item.session} command={printable}"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch four-task, three-repeat EoH experiments across two servers."
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default="eoh")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats != 3:
        raise ValueError("the balanced two-server assignment requires exactly 3 repeats")
    plan = build_launch_plan(args)
    validate_plan(plan)
    launch(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
