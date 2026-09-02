"""Standard batch launcher and watchdog.

Provides multi-backend rotation, backend reachability checks, tmux session
management, and automatic in-place relaunch (resume) until all runs finish.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from collections.abc import Iterable, Sequence
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

DEFAULT_MAX_ATTEMPTS = 5


def build_rotated_plan(
    *,
    module: str,
    method: str,
    results_root: Path,
    backend_rotation: Sequence[BackendName],
    repeats: int = 3,
    batch: str | None = None,
    session_prefix: str = "batch",
    extra_args: tuple[str, ...] = (),
    backend_map: dict[tuple[str, int], BackendName] | None = None,
) -> list[LaunchItem]:
    """Build a standard multi-task multi-repeat plan with rotated backend assignments."""
    batch_name = batch or datetime.now().strftime("%Y%m%d_%H%M%S")
    plan: list[LaunchItem] = []
    index = 0
    for task in TASKS:
        for repeat in range(1, repeats + 1):
            short = TASK_SHORT[task]
            run_name = f"{batch_name}_{short}_{method}_rep{repeat}"
            if backend_map and (task, repeat) in backend_map:
                backend = backend_map[(task, repeat)]
            else:
                backend = backend_rotation[index % len(backend_rotation)]
            plan.append(
                LaunchItem(
                    task=task,
                    repeat=repeat,
                    backend=backend,
                    session=f"{session_prefix}_{short}_r{repeat}",
                    run_name=run_name,
                    run_dir=results_root / task / run_name,
                    seed=repeat - 1,
                    module=module,
                    extra_args=extra_args,
                )
            )
            index += 1
    return plan


def check_backends(backends: Iterable[BackendName]) -> None:
    """Read each backend's /v1/models once; abort if any is unreachable."""
    unique_backends = {b: BACKENDS[b] for b in backends}
    for name, profile in unique_backends.items():
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


def get_summary_status(run_dir: Path) -> str | None:
    """Return status from logs/run_summary.json or logs/summary.json."""
    for filename in ("run_summary.json", "summary.json"):
        path = run_dir / "logs" / filename
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = payload.get("status")
                if isinstance(status, str):
                    return status
            except json.JSONDecodeError:
                pass
    return None


def is_session_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def live_session_name(base_session: str, attempt: int) -> str:
    return base_session if attempt == 1 else f"{base_session}_r{attempt}"


def is_any_attempt_alive(item: LaunchItem, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> bool:
    if is_session_alive(item.session):
        return True
    return any(
        is_session_alive(live_session_name(item.session, retry))
        for retry in range(2, max_attempts + 1)
    )


def tmux_launch(item: LaunchItem, session: str) -> None:
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


def ensure_launchable(items: list[LaunchItem], max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
    """Guard against duplicate tmux sessions."""
    sessions = [item.session for item in items]
    if len(sessions) != len(set(sessions)):
        raise ValueError("tmux session names must be unique")
    alive = [item.session for item in items if is_any_attempt_alive(item, max_attempts)]
    if alive:
        raise RuntimeError(f"tmux sessions already exist: {alive}")


def watch_batch(
    plan: list[LaunchItem],
    *,
    interval_sec: int,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    method_label: str = "batch",
) -> None:
    """Relaunch dead unfinished runs in place (resume) until all finish."""
    attempts = {item.run_name: 1 for item in plan}
    while True:
        done = stopped = running = 0
        relaunch: list[LaunchItem] = []
        for item in plan:
            if get_summary_status(item.run_dir) == "finished":
                done += 1
                continue
            if attempts[item.run_name] >= max_attempts:
                stopped += 1
                continue
            if is_any_attempt_alive(item, max_attempts):
                running += 1
                continue
            relaunch.append(item)
        for item in relaunch:
            attempts[item.run_name] += 1
            session = live_session_name(item.session, attempts[item.run_name])
            print(
                f"relaunch (resume) {item.run_name} as session={session}",
                flush=True,
            )
            tmux_launch(item, session)
            running += 1
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] "
            f"done={done} running={running} stopped={stopped} total={len(plan)}",
            flush=True,
        )
        if done + stopped == len(plan):
            if stopped:
                print(f"{stopped} runs exceeded max_attempts={max_attempts}", flush=True)
            print(f"all {method_label} runs finished", flush=True)
            return
        time.sleep(interval_sec)


def build_batch_parser(
    description: str = "Launch a batch experiment.",
    default_session_prefix: str = "batch",
    default_watch_interval: int = 120,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default=default_session_prefix)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=default_watch_interval)
    return parser


def launch_batch(
    *,
    method: str,
    module: str,
    results_root: Path,
    backend_rotation: Sequence[BackendName],
    default_session_prefix: str = "batch",
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    default_watch_interval: int = 120,
    parser: argparse.ArgumentParser | None = None,
    backend_map: dict[tuple[str, int], BackendName] | None = None,
) -> None:
    """Standard entry point for launching and watching a batch experiment."""
    if parser is None:
        parser = build_batch_parser(
            description=f"Launch {method} batch experiments.",
            default_session_prefix=default_session_prefix,
            default_watch_interval=default_watch_interval,
        )
    args = parser.parse_args()
    plan = build_rotated_plan(
        module=module,
        method=method,
        results_root=results_root,
        backend_rotation=backend_rotation,
        repeats=args.repeats,
        batch=args.batch,
        session_prefix=args.session_prefix,
        backend_map=backend_map,
    )
    print(
        f"plan: {len(plan)} runs, rotation={backend_rotation} -> "
        + ", ".join(
            f"{name}x{sum(1 for i in plan if i.backend == name)}"
            for name in backend_rotation
        ),
        flush=True,
    )
    if args.dry_run:
        launch_items(plan, dry_run=True)
        return
    check_backends({item.backend for item in plan if item.backend})
    to_launch = [
        item
        for item in plan
        if get_summary_status(item.run_dir) != "finished"
        and not is_any_attempt_alive(item, max_attempts)
    ]
    ensure_launchable(to_launch, max_attempts=max_attempts)
    launch_items(to_launch, dry_run=False)
    if args.watch:
        watch_batch(
            plan,
            interval_sec=args.watch_interval,
            max_attempts=max_attempts,
            method_label=method,
        )

