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
from .._common import count_backend_usage

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_SHORT = {
    "tsp_construct": "tsp",
    "cvrp_aco": "cvrp",
    "op_aco": "op",
    "online_bin_packing": "obp",
}
CAPACITY: dict[run.BackendName, int] = {
    "zhong": 6,
    "server1": 4,
    "local": 2,
}
PREFERRED: tuple[run.BackendName, ...] = ("zhong", "server1", "local")
STATE_FILE = Path(__file__).resolve().with_name(".dispatch_state.json")


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


def build_queue(batch: str, repeats: int) -> list[LaunchItem]:
    queue = []
    for task in run.TASKS:
        for repeat in range(1, repeats + 1):
            short = TASK_SHORT[task]
            run_name = f"{batch}_{short}_calm_rep{repeat}"
            session = f"calm_{short}_r{repeat}"
            run_dir = REPO_ROOT / "experiments" / task / "calm" / run_name
            queue.append(
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
    return queue


def load_state() -> dict[str, str]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(state: dict[str, str]) -> None:
    STATE_FILE.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def session_alive(name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def item_finished(item: LaunchItem) -> bool:
    for name in ("logs/run_summary.json", "logs/summary.json"):
        summary = item.run_dir / name
        if not summary.exists():
            continue
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("status") == "finished":
            return True
    return False


def launch_item(item: LaunchItem) -> None:
    cmd = " ".join(shlex.quote(part) for part in item.command())
    inner = (
        f"cd {shlex.quote(str(REPO_ROOT))} && "
        f"uv run {cmd}; echo exit:$?"
    )
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", item.session, "bash", "-lc", inner],
        check=True,
    )


def free_slots() -> dict[run.BackendName, int]:
    usage = count_backend_usage()
    return {
        backend: max(0, CAPACITY[backend] - usage[backend])
        for backend in CAPACITY
    }


def dispatch_once(
    queue: list[LaunchItem],
    state: dict[str, str],
    *,
    dry_run: bool,
) -> list[LaunchItem]:
    launched: list[LaunchItem] = []
    free = free_slots()
    if sum(free.values()) <= 0:
        return launched
    remaining = dict(free)
    for item in queue:
        if item.run_name in state:
            continue
        chosen: run.BackendName | None = None
        for backend in PREFERRED:
            if remaining.get(backend, 0) > 0:
                chosen = backend
                remaining[backend] -= 1
                break
        if chosen is None:
            break
        item = item.with_backend(chosen)
        print(
            f"launch {item.session} backend={chosen} -> {item.run_dir}",
            flush=True,
        )
        if not dry_run:
            launch_item(item)
            state[item.run_name] = chosen
            time.sleep(1)
        launched.append(item)
    if not dry_run:
        save_state(state)
    return launched


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dispatch CALM runs onto slots released by the running V9 batch: "
            "poll free backend capacity, launch one pending CALM run per free slot."
        )
    )
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    queue = build_queue(args.batch, args.repeats)
    state = load_state()
    print(
        f"queue={len(queue)} already_dispatched={len(state)} "
        f"free={free_slots()}",
        flush=True,
    )
    while True:
        pending = [item for item in queue if item.run_name not in state]
        if not pending:
            print("all CALM runs dispatched; exiting", flush=True)
            return
        launched = dispatch_once(queue, state, dry_run=args.dry_run)
        if launched or args.once or args.dry_run:
            if not launched:
                print(
                    f"no free slot (free={free_slots()}, pending={len(pending)})",
                    flush=True,
                )
            if args.once or args.dry_run:
                return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
