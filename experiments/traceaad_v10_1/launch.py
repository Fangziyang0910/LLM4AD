"""Launch the formal TraceAAD V10.1 batch.

5 tasks x 3 repeats = 15 runs. Backends are assigned by fixed rotation over
(server1, server3, server3b, local), task-major, which yields the quota
server1 x4 / server3 x4 / server3b x4 / local x3 and puts every task's three
repeats on three different backends. The watcher relaunches a dead run in the
same run directory, where run.py resumes from tree_state.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from experiments.infra.base import (
    BACKENDS,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    BackendName,
    LaunchItem,
    launch_items,
)
from llm4ad.tools.env import resolve_llm_api_key

MODULE = "experiments.traceaad_v10_1.run"
METHOD = "v101"
BACKEND_ROTATION: tuple[BackendName, ...] = ("server1", "server3b", "local")
MAX_ATTEMPTS = 5


def build_plan(args: argparse.Namespace) -> list[LaunchItem]:
    plan: list[LaunchItem] = []
    index = 0
    for task in TASKS:
        for repeat in range(1, args.repeats + 1):
            short = TASK_SHORT[task]
            run_name = f"{args.batch}_{short}_{METHOD}_rep{repeat}"
            plan.append(
                LaunchItem(
                    task=task,
                    repeat=repeat,
                    backend=BACKEND_ROTATION[index % len(BACKEND_ROTATION)],
                    session=f"{args.session_prefix}_{short}_r{repeat}",
                    run_name=run_name,
                    run_dir=REPO_ROOT / "experiments" / "traceaad_v10_1" / "results" / task / run_name,
                    seed=repeat - 1,
                    module=MODULE,
                )
            )
            index += 1
    return plan


def check_backends(plan: list[LaunchItem]) -> None:
    """Read each backend's /v1/models once (no load); abort if any is down."""
    profiles = {item.backend: BACKENDS[item.backend] for item in plan}
    for name, profile in profiles.items():
        url = profile.base_url.rstrip("/") + "/models"
        api_key = resolve_llm_api_key(base_url=profile.base_url)
        headers = (
            {}
            if not api_key or api_key == "EMPTY"
            else {"Authorization": f"Bearer {api_key}"}
        )
        request = Request(url, headers=headers)
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=15) as response:
                ok = response.status == 200
        except Exception as exc:
            raise RuntimeError(f"backend {name} unreachable at {url}: {exc}") from exc
        if not ok:
            raise RuntimeError(f"backend {name} returned {response.status} at {url}")
        print(f"backend {name} reachable", flush=True)


def _summary_status(item: LaunchItem) -> str | None:
    path = item.run_dir / "logs" / "run_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def _session_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _tmux_launch(item: LaunchItem, session: str) -> None:
    printable = shlex.join(item.command())
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(REPO_ROOT),
            printable,
        ],
        check=True,
    )


def _live_session(item: LaunchItem, attempts: int) -> str:
    """Session name of the current attempt: base name, or latest _rN."""
    return item.session if attempts == 1 else f"{item.session}_r{attempts}"


def _any_session_alive(item: LaunchItem) -> bool:
    """True if any attempt session (base or _rN) still runs; robust to a
    watcher restart that loses its in-memory attempt counts."""
    if _session_alive(item.session):
        return True
    return any(
        _session_alive(f"{item.session}_r{retry}")
        for retry in range(2, MAX_ATTEMPTS + 1)
    )


def watch(plan: list[LaunchItem], *, interval_sec: int) -> None:
    """Relaunch dead unfinished runs in place (resume) until all finish."""
    attempts = {item.run_name: 1 for item in plan}
    while True:
        done = stopped = running = 0
        relaunch: list[LaunchItem] = []
        for item in plan:
            if _summary_status(item) == "finished":
                done += 1
                continue
            if attempts[item.run_name] >= MAX_ATTEMPTS:
                stopped += 1
                continue
            if _any_session_alive(item):
                running += 1
                continue
            relaunch.append(item)
        for item in relaunch:
            attempts[item.run_name] += 1
            session = _live_session(item, attempts[item.run_name])
            print(
                f"relaunch (resume) {item.run_name} as session={session}",
                flush=True,
            )
            _tmux_launch(item, session)
            running += 1
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"done={done} running={running} stopped={stopped} total={len(plan)}",
            flush=True,
        )
        if done + stopped == len(plan):
            if stopped:
                print(f"{stopped} runs exceeded MAX_ATTEMPTS={MAX_ATTEMPTS}", flush=True)
            print(f"all {METHOD} runs finished", flush=True)
            return
        time.sleep(interval_sec)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the V10.1 formal batch.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default="v101")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=120)
    return parser


def ensure_launchable(items: list[LaunchItem]) -> None:
    """Guard only against double-launching a live session; existing run
    directories are fine (run.py resumes them or starts over in place)."""
    sessions = [item.session for item in items]
    if len(sessions) != len(set(sessions)):
        raise ValueError("tmux session names must be unique")
    alive = [item.session for item in items if _any_session_alive(item)]
    if alive:
        raise RuntimeError(f"tmux sessions already exist: {alive}")


def main() -> None:
    args = build_parser().parse_args()
    plan = build_plan(args)
    print(
        f"plan: {len(plan)} runs, rotation={BACKEND_ROTATION} -> "
        + ", ".join(
            f"{name}x{sum(1 for i in plan if i.backend == name)}"
            for name in BACKEND_ROTATION
        ),
        flush=True,
    )
    if args.dry_run:
        launch_items(plan, dry_run=True)
        return
    check_backends(plan)
    # first launch starts everything; a restarted launcher adopts runs that
    # are already alive or finished and only (re)launches the rest
    to_launch = [
        item
        for item in plan
        if _summary_status(item) != "finished" and not _any_session_alive(item)
    ]
    ensure_launchable(to_launch)
    launch_items(to_launch, dry_run=False)
    if args.watch:
        watch(plan, interval_sec=args.watch_interval)


if __name__ == "__main__":
    main()
