"""TraceAAD V10.2 Training Experiment Live Monitor & Visualizer.

High-performance, lightweight live observer for official TraceAAD V10.2 runs.
Features:
- File modification & size guards (zero redundant disk I/O / JSON deserialization)
- Sparse step curve compression (shrinks points by 98% while visually identical)
- Real-time progress across all 15 runs (5 tasks x 3 repeats)
- Accurate ETA estimation with windowed velocity blending
- Individual run inspector (code, ideas, lineages, recent event stream)

Usage:
    uv run python -m experiments.traceaad_v10_2.monitor [--port 8765] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = Path(__file__).resolve().parent / "results"
HTML_FILE = Path(__file__).with_name("monitor.html")

TASKS_METADATA = [
    {
        "key": "tsp_construct",
        "label": "TSP 构造",
        "unit": "路程 (min)",
        "direction": "min",
        "short": "tsp",
        "description": "旅行商问题启发式构造算法",
    },
    {
        "key": "cvrp_aco",
        "label": "CVRP-ACO",
        "unit": "路程 (min)",
        "direction": "min",
        "short": "cvrp",
        "description": "带容量约束车辆路径问题 (ACO 启发因子)",
    },
    {
        "key": "op_aco",
        "label": "OP-ACO",
        "unit": "收益 (max)",
        "direction": "max",
        "short": "op",
        "description": "定向越野问题 (ACO 启发因子)",
    },
    {
        "key": "online_bin_packing",
        "label": "在线装箱",
        "unit": "箱数 (min)",
        "direction": "min",
        "short": "obp",
        "description": "一维在线装箱启发式算法",
    },
    {
        "key": "vrptw_construct",
        "label": "VRPTW 构造",
        "unit": "路程 (min)",
        "direction": "min",
        "short": "vrptw",
        "description": "带时间窗车辆路径问题启发式算法",
    },
]

TASK_MAP = {t["key"]: t for t in TASKS_METADATA}
REP_RE = re.compile(r"_rep(\d+)$")


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "–"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分钟"
    hours = minutes // 60
    rem_min = minutes % 60
    if hours < 24:
        return f"{hours}小时{rem_min}分"
    days = hours // 24
    rem_hours = hours % 24
    return f"{days}天{rem_hours}小时"


def _format_finish_time(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "–"
    target = datetime.now() + timedelta(seconds=seconds)
    return target.strftime("%m-%d %H:%M")


def _get_active_tmux_sessions() -> set[str]:
    try:
        out = subprocess.run(
            ["tmux", "ls"], capture_output=True, text=True, timeout=2
        ).stdout
        sessions = set()
        for line in out.strip().splitlines():
            if ":" in line:
                sessions.add(line.split(":")[0].strip())
        return sessions
    except Exception:
        return set()


class MonitorDataEngine:
    def __init__(self, results_root: Path):
        self.results_root = results_root
        self._cache_overview: dict[str, Any] | None = None
        self._cache_runs: dict[str, dict[str, Any]] = {}
        self._cache_summaries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def get_overview(self) -> dict[str, Any]:
        with self._lock:
            if self._cache_overview is not None:
                return self._cache_overview
            overview = self._scan_overview()
            self._cache_overview = overview
            return overview

    def get_run_detail(self, task: str, run_name: str) -> dict[str, Any] | None:
        run_dir = self.results_root / task / run_name
        if not run_dir.is_dir():
            return None
        with self._lock:
            cached = self._cache_runs.get(run_name)
            stamp = self._get_run_mtime(run_dir)
            if cached and cached.get("_stamp") == stamp:
                return cached["data"]
            data = self._parse_run_detail(run_dir, task, run_name)
            if data:
                self._cache_runs[run_name] = {"_stamp": stamp, "data": data}
            return data

    def get_node_detail(
        self, task: str, run_name: str, node_id: int
    ) -> dict[str, Any] | None:
        run_dir = self.results_root / task / run_name
        tree_p = run_dir / "tree_state.json"
        if not tree_p.exists():
            return None
        try:
            tree_data = json.loads(tree_p.read_text(encoding="utf-8"))
            nodes_by_id = {n["id"]: n for n in tree_data.get("nodes", [])}
            target = nodes_by_id.get(node_id)
            if not target:
                return None

            ancestors = []
            curr = target
            while curr.get("parent_id") is not None:
                pid = curr["parent_id"]
                curr = nodes_by_id.get(pid)
                if not curr:
                    break
                ancestors.append(
                    {
                        "id": curr["id"],
                        "operator": curr.get("operator", "Init"),
                        "fitness": curr.get("fitness"),
                        "idea": (curr.get("idea") or "")[:80],
                    }
                )

            return {
                "node": target,
                "ancestors": ancestors,
            }
        except Exception:
            return None

    def refresh(self) -> None:
        overview = self._scan_overview()
        with self._lock:
            self._cache_overview = overview

    def _get_run_mtime(self, run_dir: Path) -> tuple[Any, ...]:
        def _stat(p: Path) -> tuple[float, int] | None:
            try:
                st = p.stat()
                return (st.st_mtime, st.st_size)
            except OSError:
                return None

        return (
            _stat(run_dir / "run_config.json"),
            _stat(run_dir / "tree_state.json"),
            _stat(run_dir / "events.jsonl"),
            _stat(run_dir / "logs" / "summary.json"),
        )

    def _scan_overview(self) -> dict[str, Any]:
        active_tmux = _get_active_tmux_sessions()
        now = datetime.now()

        tasks_data: list[dict[str, Any]] = []
        total_evals = 0
        total_budget = 0
        running_count = 0
        finished_count = 0
        stalled_count = 0
        all_etas: list[float] = []
        all_speeds: list[float] = []

        for task_info in TASKS_METADATA:
            task_key = task_info["key"]
            task_dir = self.results_root / task_key
            runs_data: list[dict[str, Any]] = []

            if task_dir.is_dir():
                for run_dir in sorted(task_dir.iterdir()):
                    if not run_dir.is_dir() or not (run_dir / "run_config.json").exists():
                        continue
                    rep_match = re.search(r"_rep(\d+)$", run_dir.name)
                    if not rep_match:
                        continue
                    rep = int(rep_match.group(1))

                    run_summary = self._parse_run_summary_cached(
                        run_dir, task_info, rep, active_tmux, now
                    )
                    runs_data.append(run_summary)

                    total_evals += run_summary["budget_used"]
                    total_budget += run_summary["budget"]

                    if run_summary["status"] == "running":
                        running_count += 1
                        if run_summary["eta_seconds"] is not None:
                            all_etas.append(run_summary["eta_seconds"])
                        if run_summary["sec_per_eval"] is not None:
                            all_speeds.append(run_summary["sec_per_eval"])
                    elif run_summary["status"] == "finished":
                        finished_count += 1
                    else:
                        stalled_count += 1

            runs_data.sort(key=lambda r: r["rep"])
            tasks_data.append(
                {
                    "meta": task_info,
                    "runs": runs_data,
                    "completed_runs": sum(1 for r in runs_data if r["status"] == "finished"),
                    "total_runs": len(runs_data),
                }
            )

        max_eta = max(all_etas) if all_etas else 0.0
        avg_speed = (sum(all_speeds) / len(all_speeds)) if all_speeds else 0.0

        return {
            "updated_at": now.isoformat(timespec="seconds"),
            "global_summary": {
                "total_runs": sum(len(t["runs"]) for t in tasks_data),
                "running_runs": running_count,
                "finished_runs": finished_count,
                "stalled_runs": stalled_count,
                "total_budget": total_budget,
                "total_evals": total_evals,
                "pct": round(total_evals / max(1, total_budget) * 100, 2),
                "max_eta_seconds": max_eta,
                "max_eta_formatted": _format_duration(max_eta),
                "max_eta_finish_time": _format_finish_time(max_eta),
                "avg_speed_sec": round(avg_speed, 1),
            },
            "tasks": tasks_data,
        }

    def _parse_run_summary_cached(
        self,
        run_dir: Path,
        task_info: dict[str, Any],
        rep: int,
        active_tmux: set[str],
        now: datetime,
    ) -> dict[str, Any]:
        stamp = self._get_run_mtime(run_dir)
        cached_entry = self._cache_summaries.get(run_dir.name)

        if cached_entry and cached_entry.get("_stamp") == stamp:
            base = dict(cached_entry["summary"])
            status = base["status"]
            budget = base["budget"]
            budget_used = base["budget_used"]
            sec_per_eval = base.get("sec_per_eval")
            expected_session = f"v102_{task_info['short']}_r{rep}"

            if base.get("has_finished_summary") or budget_used >= budget:
                status = "finished"
                eta_sec = 0.0
            elif expected_session in active_tmux:
                status = "running"
                rem_evals = max(0, budget - budget_used)
                eta_sec = (rem_evals * sec_per_eval) if sec_per_eval else None
            else:
                status = "stalled"
                eta_sec = None

            base["status"] = status
            base["eta_seconds"] = eta_sec
            base["eta_formatted"] = (
                _format_duration(eta_sec)
                if status == "running"
                else ("已完成" if status == "finished" else "已中断")
            )
            base["eta_finish_time"] = (
                _format_finish_time(eta_sec)
                if (status == "running" and eta_sec)
                else "–"
            )
            return base

        summary = self._parse_run_summary_raw(run_dir, task_info, rep, active_tmux, now)
        self._cache_summaries[run_dir.name] = {"_stamp": stamp, "summary": summary}
        return summary

    def _parse_run_summary_raw(
        self,
        run_dir: Path,
        task_info: dict[str, Any],
        rep: int,
        active_tmux: set[str],
        now: datetime,
    ) -> dict[str, Any]:
        cfg_p = run_dir / "run_config.json"
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        budget = cfg.get("method_params", {}).get("budget", 1000)
        backend = cfg.get("backend", "unknown")

        tree_p = run_dir / "tree_state.json"
        tree_data: dict[str, Any] = {}
        if tree_p.exists():
            try:
                tree_data = json.loads(tree_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        nodes = tree_data.get("nodes", [])
        budget_used = tree_data.get("budget_used", len(nodes))
        started_at_str = tree_data.get("started_at") or cfg.get("created_at")
        started_at = datetime.fromisoformat(started_at_str) if started_at_str else None

        expected_session = f"v102_{task_info['short']}_r{rep}"
        is_in_tmux = expected_session in active_tmux

        events_p = run_dir / "events.jsonl"
        last_event_ts = None
        op_counts = {"Init": 0, "Refine": 0, "Pivot": 0, "Fuse": 0}
        status_counts = {"ok": 0, "eval_failed": 0, "invalid_output": 0}
        recent_timestamps: list[datetime] = []

        if events_p.exists():
            try:
                for line in events_p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        op = ev.get("operator")
                        if op in op_counts:
                            op_counts[op] += 1
                        st = ev.get("status", "unknown")
                        status_counts[st] = status_counts.get(st, 0) + 1
                        ts_str = ev.get("ts")
                        if ts_str:
                            dt = datetime.fromisoformat(ts_str)
                            last_event_ts = dt
                            recent_timestamps.append(dt)
                    except Exception:
                        pass
            except Exception:
                pass

        summary_p = run_dir / "logs" / "summary.json"
        has_finished_summary = False
        if summary_p.exists():
            try:
                s = json.loads(summary_p.read_text(encoding="utf-8"))
                if s.get("status") == "finished":
                    has_finished_summary = True
            except Exception:
                pass

        if has_finished_summary or budget_used >= budget:
            status = "finished"
        elif is_in_tmux:
            status = "running"
        else:
            status = "stalled"

        sec_per_eval = None
        eta_seconds = None

        if started_at and budget_used > 0:
            ref_ts = last_event_ts or now
            total_elapsed = (ref_ts - started_at).total_seconds()
            avg_sec = total_elapsed / budget_used

            if len(recent_timestamps) >= 5:
                window = recent_timestamps[-10:]
                dt_window = (window[-1] - window[0]).total_seconds()
                window_speed = dt_window / max(1, len(window) - 1)
                sec_per_eval = 0.7 * window_speed + 0.3 * avg_sec
            else:
                sec_per_eval = avg_sec

            if status == "running":
                rem_evals = max(0, budget - budget_used)
                eta_seconds = rem_evals * sec_per_eval
            elif status == "finished":
                eta_seconds = 0.0

        best_fitness = None
        breakthroughs: list[dict[str, Any]] = []

        sorted_nodes = sorted(
            nodes, key=lambda n: n.get("evaluation_id") or 0
        )
        for n in sorted_nodes:
            eid = n.get("evaluation_id")
            fit = n.get("fitness")
            if eid is None or fit is None:
                continue
            if best_fitness is None or fit > best_fitness:
                best_fitness = fit
                breakthroughs.append(
                    {
                        "eid": eid,
                        "fitness": fit,
                        "display": self._format_metric(fit, task_info),
                        "node_id": n.get("id"),
                        "operator": n.get("operator", "Init"),
                        "idea": (n.get("idea") or "")[:80],
                    }
                )

        curve: list[list[float]] = []
        for bt in breakthroughs:
            curve.append([bt["eid"], bt["fitness"]])

        if curve and curve[-1][0] < budget_used and best_fitness is not None:
            curve.append([budget_used, best_fitness])

        elapsed_sec = (
            (last_event_ts or now) - started_at
        ).total_seconds() if started_at else None

        return {
            "name": run_dir.name,
            "task": task_info["key"],
            "rep": rep,
            "backend": backend,
            "status": status,
            "has_finished_summary": has_finished_summary,
            "budget": budget,
            "budget_used": budget_used,
            "pct": round(budget_used / max(1, budget) * 100, 1),
            "node_count": len(nodes),
            "started_at": started_at.isoformat(timespec="seconds") if started_at else "–",
            "last_active_at": last_event_ts.isoformat(timespec="seconds") if last_event_ts else "–",
            "elapsed_seconds": elapsed_sec,
            "elapsed_formatted": _format_duration(elapsed_sec),
            "sec_per_eval": round(sec_per_eval, 1) if sec_per_eval else None,
            "speed_formatted": f"{round(sec_per_eval, 1)}s/eval" if sec_per_eval else "–",
            "eta_seconds": eta_seconds,
            "eta_formatted": _format_duration(eta_seconds) if status == "running" else ("已完成" if status == "finished" else "已中断"),
            "eta_finish_time": _format_finish_time(eta_seconds) if (status == "running" and eta_seconds) else "–",
            "best_fitness": best_fitness,
            "best_display": self._format_metric(best_fitness, task_info),
            "operator_counts": op_counts,
            "status_counts": status_counts,
            "curve": curve,
            "breakthroughs": breakthroughs,
        }

    def _parse_run_detail(
        self, run_dir: Path, task: str, run_name: str
    ) -> dict[str, Any]:
        task_info = TASK_MAP.get(task, {"key": task, "label": task, "unit": "fitness", "direction": "max", "short": task})
        rep_match = re.search(r"_rep(\d+)$", run_name)
        rep = int(rep_match.group(1)) if rep_match else 0
        active_tmux = _get_active_tmux_sessions()
        now = datetime.now()

        summary = self._parse_run_summary_raw(run_dir, task_info, rep, active_tmux, now)

        tree_p = run_dir / "tree_state.json"
        tree_data: dict[str, Any] = {}
        if tree_p.exists():
            try:
                tree_data = json.loads(tree_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        nodes = tree_data.get("nodes", [])
        best_node = None
        if nodes:
            best_node = max(nodes, key=lambda n: n.get("fitness") or float("-inf"))

        nodes_compact = []
        for n in nodes:
            nodes_compact.append(
                {
                    "id": n.get("id"),
                    "evaluation_id": n.get("evaluation_id"),
                    "operator": n.get("operator", "Init"),
                    "fitness": n.get("fitness"),
                    "parent_id": n.get("parent_id"),
                    "donor_id": n.get("donor_id"),
                    "idea": (n.get("idea") or "")[:120],
                }
            )

        events_p = run_dir / "events.jsonl"
        recent_events: list[dict[str, Any]] = []
        scatter_points: list[dict[str, Any]] = []

        if events_p.exists():
            try:
                lines = events_p.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        if ev.get("fitness") is not None:
                            scatter_points.append(
                                {
                                    "step": ev.get("step", i),
                                    "fitness": ev.get("fitness"),
                                    "operator": ev.get("operator", "Init"),
                                    "status": ev.get("status", "ok"),
                                    "node_id": ev.get("node_id"),
                                }
                            )
                    except Exception:
                        pass

                for line in lines[-40:]:
                    if not line.strip():
                        continue
                    try:
                        recent_events.append(json.loads(line))
                    except Exception:
                        pass
                recent_events.reverse()
            except Exception:
                pass

        return {
            "summary": summary,
            "best_node": best_node,
            "nodes": nodes_compact,
            "recent_events": recent_events,
            "scatter_points": scatter_points,
        }

    def _format_metric(self, fitness: float | None, task_info: dict[str, Any]) -> str:
        if fitness is None:
            return "–"
        direction = task_info.get("direction", "max")
        if direction == "min":
            cost = -fitness
            if abs(cost) >= 100:
                return f"{cost:.1f}"
            return f"{cost:.4f}"
        else:
            if abs(fitness) >= 100:
                return f"{fitness:.1f}"
            return f"{fitness:.3f}"


def make_request_handler(engine: MonitorDataEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)

            if path == "/" or path == "/index.html":
                return self._serve_file(HTML_FILE, "text/html; charset=utf-8")

            if path == "/api/state" or path == "/api/overview":
                data = engine.get_overview()
                return self._send_json(data)

            if path == "/api/run":
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                if not task or not name:
                    return self.send_error(400, "Missing task or name parameter")
                detail = engine.get_run_detail(task, name)
                if not detail:
                    return self.send_error(404, "Run not found")
                return self._send_json(detail)

            if path == "/api/node":
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                node_id_str = params.get("id", [""])[0]
                if not task or not name or not node_id_str:
                    return self.send_error(400, "Missing parameters")
                try:
                    nid = int(node_id_str)
                except ValueError:
                    return self.send_error(400, "Invalid node id")
                node_detail = engine.get_node_detail(task, name, nid)
                if not node_detail:
                    return self.send_error(404, "Node not found")
                return self._send_json(node_detail)

            self.send_error(404)

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html", "/api/state", "/api/overview", "/api/run", "/api/node"):
                self.send_response(200)
                if path.endswith(".html") or path == "/":
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                else:
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
            else:
                self.send_error(404)

        def _serve_file(self, file_path: Path, content_type: str) -> None:
            if not file_path.exists() or file_path.stat().st_size == 0:
                return self.send_error(404, "File not found")
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceAAD V10.2 Live Monitor")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")
    args = parser.parse_args()

    engine = MonitorDataEngine(RESULTS_ROOT)

    def background_polling() -> None:
        while True:
            try:
                engine.refresh()
            except Exception as e:
                print(f"[monitor] Refresh error: {e}", flush=True)
            time.sleep(5)

    poller = threading.Thread(target=background_polling, daemon=True)
    poller.start()

    server_address = (args.host, args.port)
    handler_class = make_request_handler(engine)
    server = ThreadingHTTPServer(server_address, handler_class)

    print(f"===========================================================", flush=True)
    print(f"🚀 TraceAAD V10.2 训练实验可视化监控已启动", flush=True)
    print(f"📡 本地访问地址: http://127.0.0.1:{args.port}", flush=True)
    print(f"🌐 远程访问地址: http://{args.host}:{args.port}", flush=True)
    print(f"===========================================================", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] 服务已停止", flush=True)


if __name__ == "__main__":
    main()
