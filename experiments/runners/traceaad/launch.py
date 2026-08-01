from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .run import BACKENDS, EXPERIMENTS_ROOT, TASKS, VERSIONS

TASK_ALIASES = {
    "tsp_construct": "tspc",
    "cvrp_aco": "cvrp",
    "op_aco": "opaco",
    "online_bin_packing": "obp",
}


@dataclass(frozen=True, slots=True)
class LaunchItem:
    repeat: int
    session: str
    run_name: str
    run_dir: Path
    command: tuple[str, ...]


def build_launch_plan(args: argparse.Namespace) -> tuple[LaunchItem, ...]:
    batch = args.batch or datetime.now().strftime("%Y%m%d_%H%M%S")
    alias = TASK_ALIASES[args.task]
    method_name = f"traceaad_{args.version}"
    experiment_version = f"version{args.version.removeprefix('v')}"
    prefix = args.session_prefix or f"{alias}_{method_name}_{batch}"
    items = []
    for repeat in range(1, args.repeats + 1):
        run_name = f"{batch}_{alias}_{args.version}_rep{repeat}"
        run_dir = (
            EXPERIMENTS_ROOT
            / args.task
            / method_name
            / experiment_version
            / run_name
        )
        command = [
            sys.executable,
            "-m",
            "experiments.runners.traceaad.run",
            "--task",
            args.task,
            "--version",
            args.version,
            "--backend",
            args.backend,
            "--budget",
            str(args.budget),
            "--n-init",
            str(args.n_init),
            "--seed",
            str(repeat),
            "--repeat",
            str(repeat),
            "--run-name",
            run_name,
        ]
        _append_optional(command, "--eval-workers", args.eval_workers)
        _append_optional(command, "--output-tokens", args.output_tokens)
        _append_optional(command, "--action-max-tokens", args.action_max_tokens)
        _append_optional(command, "--base-url", args.base_url)
        _append_optional(command, "--model", args.model)
        _append_optional(command, "--no-proxy", args.no_proxy)
        items.append(
            LaunchItem(
                repeat=repeat,
                session=f"{prefix}_r{repeat}",
                run_name=run_name,
                run_dir=run_dir,
                command=tuple(command),
            )
        )
    return tuple(items)


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def validate_launch_plan(plan: tuple[LaunchItem, ...]) -> None:
    for item in plan:
        if item.run_dir.exists():
            raise FileExistsError(f"run directory already exists: {item.run_dir}")
        result = subprocess.run(
            ("tmux", "has-session", "-t", item.session),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            raise FileExistsError(f"tmux session already exists: {item.session}")


def launch(plan: tuple[LaunchItem, ...], *, dry_run: bool) -> None:
    for item in plan:
        rendered = shlex.join(item.command)
        if dry_run:
            print(f"{item.session}: {rendered}")
            continue
        subprocess.run(
            (
                "tmux",
                "new-session",
                "-d",
                "-c",
                str(EXPERIMENTS_ROOT.parent),
                "-s",
                item.session,
                "zsh",
                "-lc",
                rendered,
            ),
            check=True,
        )
        print(f"{item.session}: {item.run_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch independent TraceAAD repeats in tmux."
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--version", required=True, choices=VERSIONS)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch")
    parser.add_argument("--session-prefix")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--n-init", type=int, default=30)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--action-max-tokens", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name in ("repeats", "budget", "n_init"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in (
        "eval_workers",
        "output_tokens",
        "action_max_tokens",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    plan = build_launch_plan(args)
    if not args.dry_run:
        validate_launch_plan(plan)
    launch(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
