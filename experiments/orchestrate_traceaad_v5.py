#!/usr/bin/env python3
"""TraceAAD v5 训练实验巡检与调度。

12 个 run = 4 task (tsp_construct, cvrp_aco, op_aco, online_bin_packing) × 3 rep，
全部走 zhong (LLM_BASE_URL=http://183.36.243.124:9000/v1)，zhong 6 路并发。
v4 收尾 run 可能仍占 zhong 槽，按实际进程计数（见 count_zhong_runs），本脚本不动它们。

zhong 占用判定用「进程口径」而非 tmux session：v4 收尾的 session 可能是空壳
（resume 跑完只剩 shell），且真实 v4 进程可能脱离 tmux 独立运行。进程口径能自动
感知 v4 完成腾槽。batch 时间戳共享自 experiments/.traceaad_v5_batch.txt。

用法:
  python experiments/orchestrate_traceaad_v5.py             # 巡检 + 自动填空槽
  python experiments/orchestrate_traceaad_v5.py --no-launch # 仅巡检，不启动
  python experiments/orchestrate_traceaad_v5.py --summary   # 单行摘要（cron 用）
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
BATCH_FILE = EXP / ".traceaad_v5_batch.txt"
VERSION = "version5_1"

ZHONG_URL = "http://183.36.243.124:9000/v1"
ZHONG_MODEL = "/home/fzy/models/Qwen3.6-27B-NVFP4"
ZHONG_HOST = "183.36.243.124"
ZHONG_CAP = 6
MAX_SAMPLES = 1000

TMUX = "/usr/local/bin/tmux"

TASKS = [
    ("tsp_construct", "tspc", "experiments/tsp_construct/traceaad_v5/run_experiment.py"),
    ("cvrp_aco", "cvrp", "experiments/cvrp_aco/traceaad_v5/run_experiment.py"),
    ("op_aco", "opaco", "experiments/op_aco/traceaad_v5/run_experiment.py"),
    ("online_bin_packing", "obp", "experiments/online_bin_packing/traceaad_v5/run_experiment.py"),
]
REPS = [1, 2, 3]


def read_batch() -> str:
    if not BATCH_FILE.is_file():
        raise SystemExit(f"[orchestrate] 缺少 {BATCH_FILE}，请先写入 batch 时间戳。")
    return BATCH_FILE.read_text().strip()


def build_plan() -> list[dict]:
    # round-robin：rep 优先（各 task rep1→rep2→rep3），让 4 个 task 均衡推进
    batch = read_batch()
    plan = []
    for rep in REPS:
        for task_dir, short, entry in TASKS:
            run_ts = f"{batch}_{short}_v5_rep{rep}"
            plan.append({
                "task": task_dir, "short": short, "entry": entry,
                "rep": rep, "seed": rep,
                "session": f"{short}_traceaad_v5_rep{rep}",
                "run_ts": run_ts,
                "run_dir": EXP / task_dir / "traceaad_v5" / VERSION / run_ts,
            })
    return plan


def tmux_alive_sessions() -> set[str]:
    out = subprocess.run([TMUX, "ls"], capture_output=True, text=True)
    if out.returncode != 0:
        return set()
    return {line.split(":", 1)[0] for line in out.stdout.splitlines() if ":" in line}


def _all_run_procs() -> list[tuple[str, str, str]]:
    """所有 run_experiment.py 进程: [(pid, ppid, args), ...]。"""
    out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True)
    procs = []
    for line in out.stdout.splitlines():
        if "run_experiment.py" not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        procs.append((parts[0], parts[1], parts[2]))
    return procs


def _read_proc_env(pid: str, key: str) -> str | None:
    try:
        data = (Path("/proc") / pid / "environ").read_bytes()
    except OSError:
        return None
    for tok in data.split(b"\0"):
        s = tok.decode("utf-8", "replace")
        if s.startswith(key + "="):
            return s.split("=", 1)[1]
    return None


def count_zhong_runs() -> int:
    """挂 zhong 的活跃 run_experiment 主进程数（排除 fork 评估子进程与 uv 的 .venv 子进程）。

    主进程 = ppid 不是另一个 run_experiment 进程的那些。对 uv run 父进程，LLM_BASE_URL
    在它启动的 .venv 子进程 environ 里，故回退查子进程。
    """
    procs = _all_run_procs()
    pids = {p[0] for p in procs}
    by_pid = {p[0]: (p[1], p[2]) for p in procs}
    mains = [(pid, args) for pid, ppid, args in procs if ppid not in pids]
    zhong = 0
    for pid, _ in mains:
        base = _read_proc_env(pid, "LLM_BASE_URL")
        if base is None:
            for cpid, (cppid, _) in by_pid.items():
                if cppid == pid:
                    base = _read_proc_env(cpid, "LLM_BASE_URL")
                    if base is not None:
                        break
        if base and ZHONG_HOST in base:
            zhong += 1
    return zhong


def parse_progress(run_dir: Path) -> dict:
    """从 method_events.jsonl 解析 budget 进度、best 与结束标志。"""
    info = {"budget": 0, "max_order": 0, "best": None,
            "last_event": None, "last_status": None,
            "finished": False, "aborted": False}
    events_f = run_dir / "logs" / "method_events.jsonl"
    if not events_f.is_file():
        return info
    best = None
    for raw in events_f.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        info["last_event"] = ev.get("event")
        info["last_status"] = ev.get("status")
        if ev.get("counts_budget"):
            info["budget"] += 1
        order = ev.get("sample_order") or ev.get("profiler_sample_order")
        if isinstance(order, int) and order > info["max_order"]:
            info["max_order"] = order
        if ev.get("status") == "ok":
            sc = ev.get("score")
            if isinstance(sc, (int, float)) and (best is None or sc > best):
                best = sc
        st = ev.get("status")
        if st == "finished":
            info["finished"] = True
        elif st == "aborted":
            info["aborted"] = True
    info["best"] = best
    return info


def classify(item: dict, alive: set[str], prog: dict) -> str:
    if item["session"] in alive:
        return "running"
    if not item["run_dir"].is_dir():
        return "queued"
    if prog["finished"] or prog["budget"] >= MAX_SAMPLES:
        return "done"
    if prog["aborted"]:
        return "aborted"
    return "exited"


def launch(item: dict) -> None:
    cmd = (
        f"cd {REPO} && "
        f"LLM_BASE_URL={ZHONG_URL} "
        f"LLM_MODEL={ZHONG_MODEL} "
        f"RUN_TIMESTAMP={item['run_ts']} "
        f"TRACEAAD_RANDOM_SEED={item['seed']} "
        f"python {item['entry']}"
    )
    subprocess.run([TMUX, "new-session", "-d", "-s", item["session"], cmd], check=False)
    print(f"  [launch] {item['session']}  ({item['run_ts']})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-launch", action="store_true", help="仅巡检，不自动填槽")
    ap.add_argument("--summary", action="store_true", help="只输出单行摘要")
    args = ap.parse_args()

    plan = build_plan()
    alive = tmux_alive_sessions()

    rows = []
    v5_running = 0
    for item in plan:
        prog = parse_progress(item["run_dir"])
        status = classify(item, alive, prog)
        if status == "running":
            v5_running += 1
        rows.append((item, status, prog))

    n_done = sum(1 for _, s, _ in rows if s == "done")
    n_aborted = sum(1 for _, s, _ in rows if s == "aborted")
    n_exited = sum(1 for _, s, _ in rows if s == "exited")
    n_queued = sum(1 for _, s, _ in rows if s == "queued")

    zhong_runs = count_zhong_runs()
    v4_zhong = max(0, zhong_runs - v5_running)
    free_slots = ZHONG_CAP - zhong_runs

    if args.summary:
        print(f"v5: running={v5_running} done={n_done} aborted={n_aborted} "
              f"exited={n_exited} queued={n_queued} | "
              f"zhong v4收尾={v4_zhong} v5={v5_running} free={free_slots}/{ZHONG_CAP}")
        return 0

    print("=" * 100)
    print(f"{'session':<26} {'status':<9} {'budget':<12} {'order':<7} {'best':<14} run_ts")
    print("-" * 100)
    for item, status, prog in rows:
        best = f"{prog['best']:.4f}" if prog["best"] is not None else "-"
        budget = f"{prog['budget']}/{MAX_SAMPLES}" if status != "queued" else "-"
        order = str(prog["max_order"]) if status != "queued" else "-"
        print(f"{item['session']:<26} {status:<9} {budget:<12} {order:<7} {best:<14} {item['run_ts']}")
    print("=" * 100)
    print(f"v5: running={v5_running} done={n_done} aborted={n_aborted} "
          f"exited={n_exited} queued={n_queued}")
    print(f"zhong: v4收尾占用={v4_zhong} v5={v5_running} 总占用={zhong_runs} 空槽={free_slots}/{ZHONG_CAP}")

    if not args.no_launch and free_slots > 0 and n_queued > 0:
        to_launch = [it for it, s, _ in rows if s == "queued"][:free_slots]
        print(f"\n将启动 {len(to_launch)} 个 queued run 填空槽:")
        for it in to_launch:
            launch(it)
    elif args.no_launch:
        print("\n(--no-launch，未自动启动)")
    else:
        print(f"\n无需启动: 空槽={free_slots} queued={n_queued}")

    if n_exited:
        print("\n[警告] 有 run 异常退出（未达预算且非 aborted），需人工检查:")
        for it, s, _ in rows:
            if s == "exited":
                print(f"  - {it['session']}  {it['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
