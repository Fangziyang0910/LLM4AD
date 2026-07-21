#!/usr/bin/env python3
"""One-shot TraceAAD v3 experiment orchestrator.

Fills free LLM slots (max 3 per source), launches pending TraceAAD v3 runs,
and evaluates finished 3-rep batches. Safe to call repeatedly.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = Path(__file__).resolve().parent / ".orchestrate_traceaad_v3.json"
ZHONG_KEY_FILE = Path(__file__).resolve().parent / ".zhong_api_key"
MAX_PER_SOURCE = 3
NO_PROXY = "183.36.243.124,222.201.145.8,localhost,127.0.0.1,::1"

SOURCES = {
    "zhong": {
        "base_url": "http://183.36.243.124:9000/v1",
        "model": "Qwen3.6-27B-Q4_K_M",
        "api_key_env": "ZHONG_API_KEY",
        "api_key_file": ZHONG_KEY_FILE,
    },
    "server1": {
        "base_url": "http://222.201.145.8:8080/v1",
        "model": "qwen3.6-27b-awq",
        "api_key": "EMPTY",
    },
    "fang": {
        "base_url": "http://127.0.0.1:8001/v1",
        "model": "Qwen3.6-27B",
        "api_key": "EMPTY",
    },
}

# Already-run tasks, launch order.
TASKS = [
    "tsp_construct",
    "cvrp_aco",
    "orienteering_construct",
    "online_bin_packing",
    "knapsack_construct",
    "tsp_gls",
]

TASK_SHORT = {
    "tsp_construct": "tspc",
    "cvrp_aco": "cvrp",
    "orienteering_construct": "op",
    "online_bin_packing": "obp",
    "knapsack_construct": "kp",
    "tsp_gls": "tspgls",
}

EVAL_SCRIPTS = {
    "tsp_construct": "experiments/tsp_construct/evaluate_best_on_test.py",
    "cvrp_aco": "experiments/cvrp_aco/evaluate_best_on_test.py",
    "orienteering_construct": "experiments/orienteering_construct/evaluate_best_on_test.py",
    "online_bin_packing": "experiments/online_bin_packing/evaluate_best_on_test.py",
    "knapsack_construct": "experiments/knapsack_construct/evaluate_best_on_test.py",
    "tsp_gls": "experiments/tsp_gls/evaluate_best_on_test.py",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _default_state() -> dict:
    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    for task in TASKS:
        for rep in (1, 2, 3):
            jobs.append(
                {
                    "task": task,
                    "rep": rep,
                    "status": "pending",  # pending|running|done|evaluated|failed
                    "session": f"{TASK_SHORT[task]}_traceaad_v3_rep{rep}",
                    "run_ts": f"{batch_ts}_{TASK_SHORT[task]}_v3_rep{rep}",
                    "run_dir": None,
                    "source": None,
                    "started_at": None,
                    "finished_at": None,
                    "evaluated_at": None,
                    "eval_dir": None,
                    "error": None,
                }
            )
    return {
        "created_at": _now(),
        "updated_at": _now(),
        "batch_ts": batch_ts,
        "version": "version3",
        "jobs": jobs,
        "notes": [],
    }


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state = _default_state()
    save_state(state)
    return state


def save_state(state: dict) -> None:
    state["updated_at"] = _now()
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _proc_env(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="ignore")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for item in raw.split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k] = v
    return out


def running_llm_jobs() -> list[dict]:
    """Top-level run_experiment.py jobs (exclude evaluator child processes)."""
    ps = subprocess.check_output(["ps", "-eo", "pid,ppid,cmd"], text=True)
    rows: list[tuple[int, int, str]] = []
    for line in ps.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid_s, ppid_s, cmd = parts
        if "run_experiment.py" not in cmd:
            continue
        if "uv run" in cmd:
            continue
        if "python" not in cmd:
            continue
        rows.append((int(pid_s), int(ppid_s), cmd))
    python_pids = {pid for pid, _, _ in rows}
    jobs = []
    for pid, ppid, cmd in rows:
        if ppid in python_pids:
            continue  # evaluator / worker child
        env = _proc_env(pid)
        run_ts = env.get("RUN_TIMESTAMP", f"pid:{pid}")
        base = env.get("LLM_BASE_URL", "")
        source = None
        for name, cfg in SOURCES.items():
            if cfg["base_url"] in base:
                source = name
                break
        jobs.append(
            {
                "pid": pid,
                "source": source,
                "cmd": cmd,
                "base_url": base,
                "run_ts": run_ts,
            }
        )
    return jobs


def free_slots() -> dict[str, int]:
    used = {name: 0 for name in SOURCES}
    for job in running_llm_jobs():
        if job["source"] in used:
            used[job["source"]] += 1
    return {name: max(0, MAX_PER_SOURCE - n) for name, n in used.items()}


def _api_key_for(source: str) -> str:
    cfg = SOURCES[source]
    if "api_key" in cfg:
        return cfg["api_key"]
    env_name = cfg.get("api_key_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    key_file = cfg.get("api_key_file")
    if key_file and Path(key_file).exists():
        return Path(key_file).read_text(encoding="utf-8").strip()
    raise RuntimeError(f"missing API key for source={source}")


def _tmux_has(session: str) -> bool:
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", session],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _session_exit_ok(session: str) -> bool | None:
    """True if EXIT:0 visible, False if EXIT:nonzero, None if still running/unknown."""
    if not _tmux_has(session):
        return None
    out = subprocess.check_output(
        ["tmux", "capture-pane", "-t", session, "-p", "-S", "-30"], text=True
    )
    if "EXIT:0" in out:
        return True
    if "EXIT:" in out:
        for line in out.splitlines():
            if line.strip().startswith("EXIT:") and line.strip() != "EXIT:0":
                return False
    return None


def expected_run_dir(job: dict) -> Path:
    return (
        ROOT
        / "experiments"
        / job["task"]
        / "traceaad"
        / "version3"
        / job["run_ts"]
    )


def _run_finished(run_dir: Path) -> bool:
    summary = run_dir / "logs" / "run_summary.json"
    if not summary.exists():
        return False
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("status") == "finished" and not data.get("search_aborted")


def refresh_job_statuses(state: dict) -> list[str]:
    messages = []
    for job in state["jobs"]:
        if job["status"] not in {"running", "pending"}:
            continue
        run_dir = Path(job["run_dir"]) if job["run_dir"] else expected_run_dir(job)
        if job["status"] == "running":
            exit_ok = _session_exit_ok(job["session"])
            if _run_finished(run_dir) or exit_ok is True:
                job["status"] = "done"
                job["finished_at"] = _now()
                job["run_dir"] = str(run_dir)
                messages.append(f"DONE {job['session']} -> {run_dir}")
            elif exit_ok is False:
                job["status"] = "failed"
                job["finished_at"] = _now()
                job["error"] = "nonzero exit"
                messages.append(f"FAILED {job['session']}")
        elif job["status"] == "pending" and _run_finished(run_dir):
            job["status"] = "done"
            job["run_dir"] = str(run_dir)
            job["finished_at"] = _now()
            messages.append(f"FOUND_DONE {job['session']}")
    return messages


def launch_job(job: dict, source: str) -> None:
    cfg = SOURCES[source]
    api_key = _api_key_for(source)
    script = ROOT / "experiments" / job["task"] / "traceaad" / "run_experiment.py"
    session = job["session"]
    if _tmux_has(session):
        raise RuntimeError(f"session already exists: {session}")

    cmd = (
        f"cd '{ROOT}' && "
        f"export NO_PROXY='{NO_PROXY}' no_proxy='{NO_PROXY}' "
        f"LLM_BASE_URL='{cfg['base_url']}' LLM_MODEL='{cfg['model']}' "
        f"LLM_API_KEY='{api_key}' "
        f"RUN_TIMESTAMP='{job['run_ts']}' EXPERIMENT_VERSION='version3' "
        f"SEARCH_SEED='2024' && "
        f"echo starting method=traceaad version=3 task={job['task']} "
        f"rep={job['rep']} source={source} run_ts={job['run_ts']} && "
        f"uv run python '{script}'; echo EXIT:$?; exec bash"
    )
    subprocess.check_call(["tmux", "new-session", "-d", "-s", session, cmd])
    job["status"] = "running"
    job["source"] = source
    job["started_at"] = _now()
    job["run_dir"] = str(expected_run_dir(job))


def fill_slots(state: dict) -> list[str]:
    messages = []
    free = free_slots()
    pending = [j for j in state["jobs"] if j["status"] == "pending"]
    for job in pending:
        source = next((s for s, n in free.items() if n > 0), None)
        if source is None:
            break
        try:
            launch_job(job, source)
            free[source] -= 1
            messages.append(
                f"LAUNCHED {job['session']} on {source} -> {job['run_dir']}"
            )
            time.sleep(2)
        except Exception as exc:  # noqa: BLE001
            job["status"] = "failed"
            job["error"] = str(exc)
            messages.append(f"LAUNCH_FAIL {job['session']}: {exc}")
    return messages


def evaluate_ready_tasks(state: dict) -> list[str]:
    messages = []
    by_task: dict[str, list[dict]] = {t: [] for t in TASKS}
    for job in state["jobs"]:
        by_task[job["task"]].append(job)

    for task, jobs in by_task.items():
        if any(j["status"] == "evaluated" for j in jobs) and all(
            j["status"] in {"evaluated", "done"} for j in jobs
        ):
            # already handled
            pass
        if not all(j["status"] in {"done", "evaluated"} for j in jobs):
            continue
        if all(j["status"] == "evaluated" for j in jobs):
            continue
        run_dirs = [Path(j["run_dir"] or expected_run_dir(j)) for j in jobs]
        for d in run_dirs:
            if not _run_finished(d):
                messages.append(f"EVAL_SKIP {task}: unfinished {d}")
                break
        else:
            out_dir = (
                ROOT
                / "experiments"
                / task
                / "traceaad"
                / "version3"
                / f"eval_best_{state['batch_ts']}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            script = ROOT / EVAL_SCRIPTS[task]
            cmd = [
                "uv",
                "run",
                "python",
                str(script),
                *[str(d) for d in run_dirs],
                "--output-dir",
                str(out_dir),
            ]
            messages.append(f"EVAL_START {task} -> {out_dir}")
            try:
                subprocess.check_call(cmd, cwd=ROOT)
                for j in jobs:
                    j["status"] = "evaluated"
                    j["evaluated_at"] = _now()
                    j["eval_dir"] = str(out_dir)
                messages.append(f"EVAL_DONE {task}")
            except subprocess.CalledProcessError as exc:
                messages.append(f"EVAL_FAIL {task}: {exc}")
    return messages


def maybe_eval_tspgls_v2() -> list[str]:
    """When tsp_gls v2 mcts/traceaad finish, evaluate if not yet done."""
    messages = []
    base = ROOT / "experiments" / "tsp_gls"
    for method, pattern, out_name in [
        (
            "mcts_ahd",
            "mcts_ahd/20260720_140109_tspgls_rep*",
            "mcts_ahd/eval_best_20260720_140109",
        ),
        (
            "traceaad",
            "traceaad/version2/20260720_140109_tspgls_rep*",
            "traceaad/version2/eval_best_20260720_140109",
        ),
    ]:
        dirs = sorted(base.glob(pattern))
        if len(dirs) < 3:
            continue
        out_dir = base.joinpath(*out_name.split("/"))
        if (out_dir / "results.json").exists():
            continue
        if not all(_run_finished(d) for d in dirs):
            continue
        script = ROOT / "experiments/tsp_gls/evaluate_best_on_test.py"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv",
            "run",
            "python",
            str(script),
            *[str(d) for d in dirs],
            "--output-dir",
            str(out_dir),
        ]
        messages.append(f"EVAL_TSPGLS_V2_START {method}")
        try:
            subprocess.check_call(cmd, cwd=ROOT)
            messages.append(f"EVAL_TSPGLS_V2_DONE {method} -> {out_dir}")
        except subprocess.CalledProcessError as exc:
            messages.append(f"EVAL_TSPGLS_V2_FAIL {method}: {exc}")
    return messages


def summarize(state: dict) -> dict:
    counts = {"pending": 0, "running": 0, "done": 0, "evaluated": 0, "failed": 0}
    for j in state["jobs"]:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    return {
        "free_slots": free_slots(),
        "running_llm": [
            {"pid": j["pid"], "source": j["source"]} for j in running_llm_jobs()
        ],
        "job_counts": counts,
        "batch_ts": state["batch_ts"],
    }


def step() -> dict:
    state = load_state()
    messages: list[str] = []
    messages.extend(refresh_job_statuses(state))
    messages.extend(maybe_eval_tspgls_v2())
    messages.extend(evaluate_ready_tasks(state))
    messages.extend(fill_slots(state))
    # refresh again after launches
    summary = summarize(state)
    state["last_messages"] = messages
    state["last_summary"] = summary
    save_state(state)
    return {"messages": messages, "summary": summary}


if __name__ == "__main__":
    result = step()
    print(json.dumps(result, indent=2, ensure_ascii=False))
