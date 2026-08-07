"""Fill configured LLM backend slots with a 12-run TraceAAD tree batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import re
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from llm4ad.method.traceaad_v8 import PROTOCOL_ID as V8_PROTOCOL_ID
from llm4ad.method.traceaad_v8_3 import PROTOCOL_ID as V83_PROTOCOL_ID

from . import run

STATE_SCHEMA = 1
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
BACKEND_ORDER: tuple[run.BackendName, ...] = ("zhong", "server1", "local")
BACKEND_URL_MARKERS: dict[run.BackendName, str] = {
    "zhong": "183.36.243.124",
    "server1": "222.201.145.8",
    "local": "127.0.0.1:8001",
}
TERMINAL_STATUSES = {"finished", "failed", "stalled", "launch_failed"}
FAILED_SUMMARY_STATUSES = {"error", "aborted", "interrupted"}
DEFAULT_STATE_DIR = run.EXPERIMENTS_ROOT / ".traceaad_v8_scheduler"
PROTOCOL_IDS = {"v8": V8_PROTOCOL_ID, "v8_3": V83_PROTOCOL_ID}
PROTOCOL_ID = V8_PROTOCOL_ID


def _v83_backend(task: run.TaskName, repeat: int) -> run.BackendName:
    if task in {"tsp_construct", "cvrp_aco"}:
        return "zhong"
    if task == "op_aco" or repeat == 1:
        return "server1"
    return "local"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(message: str, *, log_path: Path | None = None) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")


def _process_rows() -> list[tuple[int, int, str]]:
    result = subprocess.run(
        ("ps", "-eo", "pid=,ppid=,args="),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) == 3:
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def _backend_from_command(command: str) -> run.BackendName | None:
    if "--backend" not in command:
        return None
    if "experiments.runners." not in command and "run_experiment" not in command:
        return None
    for backend in BACKEND_ORDER:
        pattern = rf"(?:^|\s)--backend(?:=|\s+){backend}(?:\s|$)"
        if re.search(pattern, command):
            return backend
    for backend, marker in BACKEND_URL_MARKERS.items():
        if marker in command:
            return backend
    return None


def count_backend_usage(
    rows: list[tuple[int, int, str]] | None = None,
) -> dict[run.BackendName, int]:
    """Count top-level experiments once, excluding inherited worker cmdlines."""
    rows = _process_rows() if rows is None else rows
    matched = {
        pid: (ppid, backend)
        for pid, ppid, command in rows
        if (backend := _backend_from_command(command)) is not None
    }
    counts: dict[run.BackendName, int] = {backend: 0 for backend in BACKEND_ORDER}
    for ppid, backend in matched.values():
        if ppid not in matched:
            counts[backend] += 1
    return counts


def free_slots(
    capacity: dict[run.BackendName, int],
    usage: dict[run.BackendName, int] | None = None,
) -> dict[run.BackendName, int]:
    usage = count_backend_usage() if usage is None else usage
    return {
        backend: max(0, capacity[backend] - usage.get(backend, 0))
        for backend in BACKEND_ORDER
    }


def _job_command(job: dict[str, object]) -> tuple[str, ...]:
    backend = job.get("backend")
    if backend not in BACKEND_ORDER:
        raise ValueError(f"backend not assigned for {job['run_name']}")
    return (
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        str(job["task"]),
        "--version",
        str(job.get("version", "v8")),
        "--backend",
        str(backend),
        "--budget",
        str(job["budget"]),
        "--n-init",
        str(job["n_init"]),
        "--seed",
        str(job["seed"]),
        "--repeat",
        str(job["repeat"]),
        "--run-name",
        str(job["run_name"]),
        "--context-token-limit",
        str(job["context_token_limit"]),
    )


def build_state(
    *,
    batch: str,
    budget: int,
    n_init: int,
    context_token_limit: int,
    version: run.VersionName = "v8",
    experiments_root: Path = run.EXPERIMENTS_ROOT,
) -> dict[str, object]:
    if version not in PROTOCOL_IDS:
        raise ValueError(
            f"scheduler only supports tree versions: {sorted(PROTOCOL_IDS)}"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", batch):
        raise ValueError("batch may contain only letters, numbers, '.', '_' and '-'")
    jobs = []
    for repeat in range(1, 4):
        for task in run.TASKS:
            short = TASK_SHORT[task]
            tag = "v82" if version == "v8" else "v83"
            run_name = f"{tag}_{batch}_{short}_rep{repeat}"
            jobs.append(
                {
                    "task": task,
                    "repeat": repeat,
                    "seed": repeat,
                    "backend": None,
                    "preferred_backend": (
                        _v83_backend(task, repeat) if version == "v8_3" else None
                    ),
                    "version": version,
                    "session": f"traceaad_{tag}_{batch}_{short}_r{repeat}",
                    "run_name": run_name,
                    "run_dir": str(
                        experiments_root
                        / task
                        / f"traceaad_{version}"
                        / f"version{version.removeprefix('v')}"
                        / run_name
                    ),
                    "budget": budget,
                    "n_init": n_init,
                    "context_token_limit": context_token_limit,
                    "status": "pending",
                    "launched_at": None,
                    "terminal_at": None,
                    "error": None,
                }
            )
    now = _now()
    return {
        "schema": STATE_SCHEMA,
        "protocol_id": PROTOCOL_IDS[version],
        "batch": batch,
        "created_at": now,
        "updated_at": now,
        "jobs": jobs,
    }


def save_state(state: dict[str, object], path: Path) -> None:
    state["updated_at"] = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_state(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STATE_SCHEMA:
        raise ValueError(f"unsupported scheduler state schema: {payload.get('schema')}")
    if payload.get("protocol_id") not in PROTOCOL_IDS.values():
        raise ValueError("scheduler state belongs to a different V8 protocol")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 12:
        raise ValueError("scheduler state must contain exactly 12 jobs")
    return payload


def _session_exists(session: str) -> bool:
    result = subprocess.run(
        ("tmux", "has-session", "-t", f"={session}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _summary_status(job: dict[str, object]) -> str | None:
    path = Path(str(job["run_dir"])) / "logs" / "summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    status = payload.get("status")
    return status if isinstance(status, str) else None


def reconcile_job(job: dict[str, object]) -> bool:
    previous = job["status"]
    summary_status = _summary_status(job)
    if summary_status == "finished":
        job["status"] = "finished"
        job["terminal_at"] = job.get("terminal_at") or _now()
    elif summary_status in FAILED_SUMMARY_STATUSES:
        job["status"] = "failed"
        job["terminal_at"] = job.get("terminal_at") or _now()
        job["error"] = f"run summary status={summary_status}"
    elif previous in TERMINAL_STATUSES:
        pass
    elif _session_exists(str(job["session"])):
        job["status"] = "running"
    elif previous == "launching" and not Path(str(job["run_dir"])).exists():
        job["status"] = "pending"
        job["backend"] = None
        job["launched_at"] = None
    elif previous == "pending" and not Path(str(job["run_dir"])).exists():
        pass
    else:
        job["status"] = "stalled"
        job["terminal_at"] = job.get("terminal_at") or _now()
        job["error"] = "tmux session ended without a terminal summary"
    return job["status"] != previous


def reconcile_state(state: dict[str, object]) -> bool:
    jobs = state["jobs"]
    assert isinstance(jobs, list)
    changed = False
    for job in jobs:
        changed = reconcile_job(job) or changed
    return changed


def assign_pending(
    state: dict[str, object],
    available: dict[run.BackendName, int],
) -> list[dict[str, object]]:
    jobs = state["jobs"]
    assert isinstance(jobs, list)
    assigned = []
    for job in jobs:
        if job["status"] != "pending":
            continue
        preferred = job.get("preferred_backend")
        if preferred in BACKEND_ORDER:
            backend = preferred if available.get(preferred, 0) > 0 else None
            if backend is None:
                continue
        else:
            backend = next(
                (name for name in BACKEND_ORDER if available.get(name, 0) > 0),
                None,
            )
        if backend is None:
            break
        available[backend] -= 1
        job["backend"] = backend
        assigned.append(job)
    return assigned


def launch_job(
    job: dict[str, object],
    *,
    state: dict[str, object],
    state_path: Path,
    log_path: Path,
) -> None:
    run_dir = Path(str(job["run_dir"]))
    if run_dir.exists() or _session_exists(str(job["session"])):
        job["status"] = "stalled"
        job["terminal_at"] = _now()
        job["error"] = "run directory or tmux session already exists"
        save_state(state, state_path)
        _log(f"collision: {job['run_name']}", log_path=log_path)
        return

    job["status"] = "launching"
    job["launched_at"] = _now()
    save_state(state, state_path)
    command = _job_command(job)
    rendered = shlex.join(command)
    try:
        subprocess.run(
            (
                "tmux",
                "new-session",
                "-d",
                "-s",
                str(job["session"]),
                "-c",
                str(run.REPO_ROOT),
                rendered,
            ),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        job["status"] = "launch_failed"
        job["terminal_at"] = _now()
        job["error"] = exc.stderr.strip() or str(exc)
        save_state(state, state_path)
        _log(
            f"launch failed: {job['run_name']} error={job['error']}",
            log_path=log_path,
        )
        return
    job["status"] = "running"
    save_state(state, state_path)
    _log(
        f"launched backend={job['backend']} task={job['task']} "
        f"repeat={job['repeat']} session={job['session']}",
        log_path=log_path,
    )


def _status_counts(state: dict[str, object]) -> dict[str, int]:
    jobs = state["jobs"]
    assert isinstance(jobs, list)
    counts: dict[str, int] = {}
    for job in jobs:
        status = str(job["status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_once(
    state: dict[str, object],
    *,
    capacity: dict[run.BackendName, int],
    state_path: Path,
    log_path: Path,
    dry_run: bool,
) -> int:
    if reconcile_state(state) and not dry_run:
        save_state(state, state_path)
    usage = count_backend_usage()
    available = free_slots(capacity, usage)
    assigned = assign_pending(state, dict(available))
    _log(
        f"status={_status_counts(state)} usage={usage} free={available} "
        f"next={len(assigned)}",
        log_path=None if dry_run else log_path,
    )
    if dry_run:
        for job in assigned:
            print(
                f"DRY-RUN backend={job['backend']} session={job['session']} "
                f"command={shlex.join(_job_command(job))}",
                flush=True,
            )
        return len(assigned)
    for job in assigned:
        launch_job(job, state=state, state_path=state_path, log_path=log_path)
    return len(assigned)


@contextmanager
def scheduler_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"scheduler already running: {path}") from exc
        yield


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill backend slots with 12 TraceAAD tree-method runs."
    )
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--version", choices=("v8", "v8_3"), default="v8_3")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--context-token-limit", type=int, default=24576)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--zhong-capacity", type=int, default=6)
    parser.add_argument("--server1-capacity", type=int, default=4)
    parser.add_argument("--local-capacity", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for name in (
        "budget",
        "n_init",
        "context_token_limit",
        "poll_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("zhong_capacity", "server1_capacity", "local_capacity"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if not any(
        getattr(args, name) > 0
        for name in ("zhong_capacity", "server1_capacity", "local_capacity")
    ):
        raise ValueError("at least one backend capacity must be positive")
    capacity: dict[run.BackendName, int] = {
        "zhong": args.zhong_capacity,
        "server1": args.server1_capacity,
        "local": args.local_capacity,
    }
    state_path = args.state_dir / f"{args.batch}.json"
    log_path = args.state_dir / f"{args.batch}.log"
    lock_path = args.state_dir / f"{args.batch}.lock"

    def load_or_build() -> dict[str, object]:
        if state_path.exists():
            return load_state(state_path)
        return build_state(
            batch=args.batch,
            budget=args.budget,
            n_init=args.n_init,
            context_token_limit=args.context_token_limit,
            version=args.version,
        )

    if args.dry_run:
        state = load_or_build()
        run_once(
            state,
            capacity=capacity,
            state_path=state_path,
            log_path=log_path,
            dry_run=True,
        )
        return

    with scheduler_lock(lock_path):
        state = load_or_build()
        if not state_path.exists():
            save_state(state, state_path)
        _log(
            f"scheduler started batch={args.batch} protocol={state['protocol_id']} "
            f"capacity={capacity}",
            log_path=log_path,
        )
        while True:
            run_once(
                state,
                capacity=capacity,
                state_path=state_path,
                log_path=log_path,
                dry_run=False,
            )
            counts = _status_counts(state)
            if sum(counts.get(status, 0) for status in TERMINAL_STATUSES) == 12:
                _log(f"scheduler finished status={counts}", log_path=log_path)
                return
            if args.once:
                return
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
