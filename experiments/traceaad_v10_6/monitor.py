"""TraceAAD multi-version training experiment monitor.

High-performance, lightweight live observer for official TraceAAD V10.6 runs.
Features:
- Version-aware telemetry for current and historical TraceAAD experiments
- Seamless queued runs detection from scheduler batch manifests
- LLM vs Eval latency breakdown and parent/frontier improvement rates
- File modification & size guards (zero redundant disk I/O / JSON deserialization)
- Sparse step curve compression
- Real-time progress across all 15 runs (5 tasks x 3 repeats)
- Dynamic tmux session detection with windowed velocity blending
- Individual run inspector (code, implementation summaries, lineages, recent event stream)
- Multi-version switcher with V10.10 as the default

Usage:
    uv run python -m experiments.traceaad_v10_6.monitor [--port 8765] [--host 0.0.0.0]
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent / "results"
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


def _version_dir_pattern(version_id: str) -> str | None:
    """Run-dir filter separating batches that share one results root."""
    return KNOWN_VERSIONS.get(version_id, {}).get("dir_pattern")

KNOWN_VERSIONS = {
    "v10_10": {
        "id": "v10_10", "name": "TraceAAD V10.10（有界错误修复）",
        "badge": "V10.10", "default_prefix": "v1010",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_10" / "results",
        "is_latest": True, "dir_pattern": r"_v1010_rep\d+$",
    },
    "v10_9": {
        "id": "v10_9", "name": "TraceAAD V10.9（迁移与精炼）",
        "badge": "V10.9", "default_prefix": "v109",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_9" / "results",
        "is_latest": False, "dir_pattern": r"_v109_rep\d+$",
    },
    "v10_8": {
        "id": "v10_8", "name": "TraceAAD V10.8（形成轨迹）",
        "badge": "V10.8", "default_prefix": "v108",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_8" / "results",
        "is_latest": False,
        "dir_pattern": r"(?<!_[ABCD])_v108_rep\d+$",
    },
    "v10_7r": {
        "id": "v10_7r",
        "name": "TraceAAD V10.7R（任务证据正式）",
        "badge": "V10.7R",
        "default_prefix": "v107r",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_7" / "results",
        "is_latest": False,
        # Only the V10.7R batch dirs (…_v107r_repN); the frozen V10.7
        # batch (…_v107_repN) lives in the same results root.
        "dir_pattern": r"_v107r_rep\d+$",
    },
    "v10_7": {
        "id": "v10_7",
        "name": "TraceAAD V10.7（冻结对照）",
        "badge": "V10.7",
        "default_prefix": "v107",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_7" / "results",
        "is_latest": False,
        "dir_pattern": r"_v107_rep\d+$",
    },
    "v10_6": {
        "id": "v10_6",
        "name": "TraceAAD V10.6",
        "badge": "V10.6",
        "default_prefix": "v106",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_6" / "results",
        "is_latest": False,
    },
    "v10_5": {
        "id": "v10_5",
        "name": "TraceAAD V10.5",
        "badge": "V10.5",
        "default_prefix": "v105",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_5" / "results",
        "is_latest": False,
    },
    "v10_4": {
        "id": "v10_4",
        "name": "TraceAAD V10.4",
        "badge": "V10.4",
        "default_prefix": "v104",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_4" / "results",
        "is_latest": False,
    },
    "v10_3": {
        "id": "v10_3",
        "name": "TraceAAD V10.3",
        "badge": "V10.3",
        "default_prefix": "v103",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_3" / "results",
        "is_latest": False,
    },
    "v10_2": {
        "id": "v10_2",
        "name": "TraceAAD V10.2",
        "badge": "V10.2",
        "default_prefix": "v102",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_2" / "results",
        "is_latest": False,
    },
    "v10_1": {
        "id": "v10_1",
        "name": "TraceAAD V10.1",
        "badge": "V10.1",
        "default_prefix": "v101",
        "path": REPO_ROOT / "experiments" / "traceaad_v10_1" / "results",
        "is_latest": False,
    },
}


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


def _is_run_session_alive(base_session: str, active_tmux: set[str]) -> tuple[bool, str]:
    if base_session in active_tmux:
        return True, base_session
    prefix = f"{base_session}_r"
    for s in active_tmux:
        if s.startswith(prefix):
            return True, s
    return False, base_session


class MonitorDataEngine:
    def __init__(
        self,
        results_root: Path | None = None,
        default_version: str = "v10_6",
        default_session_prefix: str = "v106",
    ):
        self.default_results_root = results_root or KNOWN_VERSIONS.get(
            default_version, {}
        ).get("path", DEFAULT_RESULTS_ROOT)
        self.default_version = default_version
        self.default_session_prefix = default_session_prefix

        # Cache indexed by version_id
        self._cache_overview: dict[str, dict[str, Any]] = {}
        self._cache_overview_ts: dict[str, float] = {}
        self._cache_runs: dict[str, dict[str, dict[str, Any]]] = {}
        self._cache_summaries: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get_available_versions(self) -> list[dict[str, Any]]:
        versions: list[dict[str, Any]] = []
        for vid, info in KNOWN_VERSIONS.items():
            if info["path"].is_dir():
                versions.append(
                    {
                        "id": vid,
                        "name": info["name"],
                        "badge": info["badge"],
                        "is_latest": info["is_latest"],
                    }
                )
        if not versions:
            versions.append(
                {
                    "id": self.default_version,
                    "name": f"TraceAAD {self.default_version.upper()}",
                    "badge": self.default_version.upper(),
                    "is_latest": True,
                }
            )
        return versions

    def _resolve_version_meta(
        self, version: str | None
    ) -> tuple[str, Path, str, str]:
        vid = version or self.default_version
        if vid in KNOWN_VERSIONS:
            vmeta = KNOWN_VERSIONS[vid]
            # Use explicit root if specified and matches default
            path = self.default_results_root if (vid == self.default_version) else vmeta["path"]
            return vid, path, vmeta["default_prefix"], vmeta["badge"]
        return vid, self.default_results_root, self.default_session_prefix, vid.upper()

    def get_overview(
        self, version: str | None = None, max_age_sec: float = 3.0
    ) -> dict[str, Any]:
        vid, root_dir, prefix, badge = self._resolve_version_meta(version)
        now_ts = time.time()
        with self._lock:
            cached = self._cache_overview.get(vid)
            cached_ts = self._cache_overview_ts.get(vid, 0.0)
            if cached is not None and (now_ts - cached_ts) < max_age_sec:
                return cached
        overview = self._scan_overview(root_dir, vid, prefix, badge)
        with self._lock:
            self._cache_overview[vid] = overview
            self._cache_overview_ts[vid] = time.time()
            return overview

    def get_run_detail(
        self, task: str, run_name: str, version: str | None = None
    ) -> dict[str, Any] | None:
        vid, root_dir, prefix, _ = self._resolve_version_meta(version)
        run_dir = root_dir / task / run_name
        if not run_dir.is_dir():
            # Check if this is a queued run described in manifest
            manifest_run = self._find_queued_run_in_manifest(
                root_dir, task, run_name, prefix, vid)
            if manifest_run:
                return {
                    "summary": manifest_run,
                    "best_node": None,
                    "nodes": [],
                    "recent_events": [],
                    "scatter_points": [],
                }
            return None
        with self._lock:
            if vid not in self._cache_runs:
                self._cache_runs[vid] = {}
            cached = self._cache_runs[vid].get(run_name)
            stamp = self._get_run_mtime(run_dir)
            if cached and cached.get("_stamp") == stamp:
                data = cached["data"]
            else:
                data = self._parse_run_detail(run_dir, task, run_name, prefix)
            if data:
                # Session and queue state can change without a new run artifact.
                data['summary'] = self._parse_run_summary_cached(
                    run_dir, TASK_MAP.get(task, {'key': task, 'label': task, 'unit': 'fitness',
                                              'direction': 'max', 'short': task}), data['summary']['rep'],
                    _get_active_tmux_sessions(), datetime.now(), vid, prefix)
                self._cache_runs[vid][run_name] = {"_stamp": stamp, "data": data}
            return data

    def get_node_detail(
        self, task: str, run_name: str, node_id: int, version: str | None = None
    ) -> dict[str, Any] | None:
        _, root_dir, _, _ = self._resolve_version_meta(version)
        run_dir = root_dir / task / run_name
        tree_p = run_dir / "tree_state.json"
        if not tree_p.exists():
            return None
        try:
            tree_data = json.loads(tree_p.read_text(encoding="utf-8"))
            nodes_by_id = {n["id"]: n for n in tree_data.get("nodes", [])}
            target = nodes_by_id.get(node_id)
            if not target:
                return None

            if not target.get("operator") and target.get("origin_operator"):
                target["operator"] = target["origin_operator"]
            if target.get("evaluation_id") is None:
                events_p = run_dir / "events.jsonl"
                if events_p.exists():
                    try:
                        eval_c = 0
                        for line in events_p.read_text(encoding="utf-8").splitlines():
                            if not line.strip():
                                continue
                            ev = json.loads(line)
                            if ev.get("slot_consumed"):
                                eval_c += 1
                            if ev.get("node_id") == node_id:
                                target["evaluation_id"] = ev.get("evaluation_id") or eval_c
                                break
                    except Exception:
                        pass
                if target.get("evaluation_id") is None:
                    target["evaluation_id"] = target["id"] + 1

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
                        "operator": curr.get("operator") or curr.get("origin_operator") or "Init",
                        "fitness": curr.get("fitness"),
                        "idea": (curr.get("idea") or "")[:120],
                    }
                )

            return {
                "node": target,
                "ancestors": ancestors,
            }
        except Exception:
            return None

    def refresh(self, version: str | None = None) -> None:
        if version is not None:
            vids = [version]
        else:
            vids = [v["id"] for v in self.get_available_versions()]
            if self.default_version not in vids:
                vids.append(self.default_version)

        for vid in vids:
            try:
                v_id, root_dir, prefix, badge = self._resolve_version_meta(vid)
                if not root_dir.is_dir():
                    continue
                overview = self._scan_overview(root_dir, v_id, prefix, badge)
                with self._lock:
                    self._cache_overview[v_id] = overview
                    self._cache_overview_ts[v_id] = time.time()
            except Exception as e:
                print(f"[monitor] Failed to refresh version {vid}: {e}", flush=True)

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
            _stat(run_dir / "logs" / "run_summary.json"),
        )

    def _load_latest_batch_manifest(
        self, root_dir: Path, version_id: str | None = None
    ) -> dict[str, Any] | None:
        """Find the most recent manifest belonging to this version's batch.

        Batches that share one results root are separated by matching plan
        run names against the version's dir pattern; falls back to the
        latest manifest when the version has no pattern.
        """
        manifests = sorted(root_dir.glob("batch_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        pattern = _version_dir_pattern(version_id) if version_id else None
        fallback = None
        for m in manifests:
            try:
                data = json.loads(m.read_text(encoding="utf-8"))
                if not (isinstance(data, dict) and "plan" in data):
                    continue
                if data.get('status') in ('excluded_startup_diagnostic', 'superseded'):
                    continue
            except Exception:
                continue
            if fallback is None:
                fallback = data
            if pattern is None:
                return data
            if any(re.search(pattern, item.get("run_name") or "")
                   for item in data.get("plan", [])):
                return data
        return fallback

    def _find_queued_run_in_manifest(
        self, root_dir: Path, task: str, run_name: str, default_prefix: str,
        version_id: str | None = None,
    ) -> dict[str, Any] | None:
        manifest = self._load_latest_batch_manifest(root_dir, version_id)
        if not manifest:
            return None
        task_info = TASK_MAP.get(task, {"key": task, "label": task, "unit": "fitness", "direction": "max", "short": task})
        for item in manifest.get("plan", []):
            if item.get("task") == task and item.get("run_name") == run_name:
                rep = item.get("repeat", 1)
                return self._create_queued_run_placeholder(item, task_info, rep, default_prefix)
        return None

    def _create_queued_run_placeholder(
        self, item: dict[str, Any], task_info: dict[str, Any], rep: int, default_prefix: str
    ) -> dict[str, Any]:
        session = item.get("session") or f"{default_prefix}_{task_info['short']}_r{rep}"
        telemetry_placeholder = {
            "requested_operator_counts": {"Init": 0, "Refine": 0, "Tune": 0, "Pivot": 0, "Fuse": 0},
            "executed_operator_counts": {"Init": 0, "Refine": 0, "Tune": 0, "Pivot": 0, "Fuse": 0},
            "fuse_fallbacks": 0,
            "parent_routes": {},
            "pivot_routes": {"quality": 0, "uniform": 0},
            "parent_improved_count": 0,
            "frontier_improved_count": 0,
            "avg_llm_seconds": None,
            "avg_eval_seconds": None,
            "avg_prompt_tokens": None,
            "avg_summary_tokens": None,
            "summary_present_count": 0,
        }
        return {
            "name": item.get("run_name") or f"queued_{task_info['short']}_rep{rep}",
            "task": task_info["key"],
            "rep": rep,
            "backend": item.get("backend") or "排队分配中",
            "method": default_prefix,
            "expected_session": session,
            "actual_session": None,
            "status": "queued",
            "has_finished_summary": False,
            "budget": 1000,
            "budget_used": 0,
            "pct": 0.0,
            "node_count": 0,
            "started_at": "–",
            "last_active_at": "–",
            "elapsed_seconds": None,
            "elapsed_formatted": "–",
            "sec_per_eval": None,
            "speed_formatted": "–",
            "eta_seconds": None,
            "eta_formatted": "排队中",
            "eta_finish_time": "–",
            "best_fitness": None,
            "best_display": "–",
            "operator_counts": {"Init": 0, "Refine": 0, "Tune": 0, "Pivot": 0, "Fuse": 0},
            "status_counts": {"ok": 0, "eval_failed": 0, "invalid_output": 0},
            "curve": [],
            "breakthroughs": [],
            "v106_telemetry": telemetry_placeholder,
            "v105_telemetry": telemetry_placeholder,
        }

    def _scan_overview(
        self, root_dir: Path, version_id: str, default_prefix: str, badge: str
    ) -> dict[str, Any]:
        active_tmux = _get_active_tmux_sessions()
        now = datetime.now()
        manifest = self._load_latest_batch_manifest(root_dir, version_id)

        # Pre-group manifest plan items by (task, rep)
        planned_names = {item.get('run_name') for item in manifest.get('plan', [])} if manifest else set()
        manifest_by_task_rep: dict[tuple[str, int], dict[str, Any]] = {}
        dir_pattern = _version_dir_pattern(version_id)
        if manifest:
            for item in manifest.get("plan", []):
                if dir_pattern and not re.search(dir_pattern, item.get('run_name', '')):
                    continue
                t_k = item.get("task")
                r_num = item.get("repeat")
                if t_k and r_num:
                    manifest_by_task_rep[(t_k, r_num)] = item

        tasks_data: list[dict[str, Any]] = []
        total_evals = 0
        total_budget = 0
        running_count = 0
        finished_count = 0
        stalled_count = 0
        queued_count = 0
        all_etas: list[float] = []
        all_speeds: list[float] = []

        dir_pattern = _version_dir_pattern(version_id)
        for task_info in TASKS_METADATA:
            task_key = task_info["key"]
            task_dir = root_dir / task_key
            runs_by_rep: dict[int, dict[str, Any]] = {}

            # 1. Scan existing directories on disk
            if task_dir.is_dir():
                for run_dir in sorted(task_dir.iterdir()):
                    if not run_dir.is_dir() or not (run_dir / "run_config.json").exists():
                        continue
                    if dir_pattern and not re.search(dir_pattern, run_dir.name):
                        continue
                    if default_prefix in ('v108alloc', 'v109', 'v1010') and manifest and run_dir.name not in planned_names:
                        continue
                    rep_match = re.search(r"_rep(\d+)$", run_dir.name)
                    if not rep_match:
                        continue
                    rep = int(rep_match.group(1))
                    run_summary = self._parse_run_summary_cached(
                        run_dir, task_info, rep, active_tmux, now, version_id, default_prefix
                    )
                    runs_by_rep[rep] = run_summary

            # 2. Reconcile with batch manifest for planned repeats (e.g. rep 1, 2, 3)
            for rep in (1, 2, 3):
                if rep not in runs_by_rep:
                    plan_item = manifest_by_task_rep.get((task_key, rep))
                    if plan_item:
                        placeholder = self._create_queued_run_placeholder(
                            plan_item, task_info, rep, default_prefix
                        )
                        runs_by_rep[rep] = placeholder

            runs_data = [runs_by_rep[r] for r in sorted(runs_by_rep.keys())]

            for run_summary in runs_data:
                total_evals += run_summary["budget_used"]
                total_budget += run_summary["budget"]

                st = run_summary["status"]
                if st == "running":
                    running_count += 1
                    if run_summary["eta_seconds"] is not None:
                        all_etas.append(run_summary["eta_seconds"])
                    if run_summary["sec_per_eval"] is not None:
                        all_speeds.append(run_summary["sec_per_eval"])
                elif st == "finished":
                    finished_count += 1
                elif st == "queued":
                    queued_count += 1
                else:
                    stalled_count += 1

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

        sched_session = None
        for candidate in (f"{default_prefix}_launcher", f"{default_prefix}_sched"):
            if candidate in active_tmux:
                sched_session = candidate
                break
        scheduler_active = sched_session is not None

        return {
            "version": badge,
            "version_id": version_id,
            "updated_at": now.isoformat(timespec="seconds"),
            "scheduler": {
                "active": scheduler_active,
                "session": sched_session or f"{default_prefix}_launcher",
                "batch": manifest.get("batch") if manifest else None,
            },
            "global_summary": {
                "total_runs": sum(len(t["runs"]) for t in tasks_data),
                "running_runs": running_count,
                "finished_runs": finished_count,
                "queued_runs": queued_count,
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

    def _apply_waiting_queue(self, summary, run_dir, version_id):
        if summary['status'] != 'stalled':
            return summary
        manifest = self._load_latest_batch_manifest(run_dir.parent.parent, version_id)
        if manifest and manifest.get('waiting_for') and any(
                row.get('run_name') == run_dir.name and row.get('status') in ('queued', 'paused')
                for row in manifest.get('plan', [])):
            summary.update(status='queued', eta_formatted='排队中（保留进度）', eta_seconds=None,
                           eta_finish_time='–')
        return summary

    def _parse_run_summary_cached(
        self,
        run_dir: Path,
        task_info: dict[str, Any],
        rep: int,
        active_tmux: set[str],
        now: datetime,
        version_id: str,
        default_prefix: str,
    ) -> dict[str, Any]:
        stamp = self._get_run_mtime(run_dir)
        if version_id not in self._cache_summaries:
            self._cache_summaries[version_id] = {}
        cached_entry = self._cache_summaries[version_id].get(run_dir.name)

        if cached_entry and cached_entry.get("_stamp") == stamp:
            base = dict(cached_entry["summary"])
            budget = base["budget"]
            budget_used = base["budget_used"]
            sec_per_eval = base.get("sec_per_eval")
            expected_session = base.get("expected_session") or f"{base.get('method') or default_prefix}_{task_info['short']}_r{rep}"
            is_in_tmux, actual_session = _is_run_session_alive(expected_session, active_tmux)

            if base.get("has_finished_summary") or budget_used >= budget:
                status = "finished"
                eta_sec = 0.0
            elif is_in_tmux:
                status = "running"
                rem_evals = max(0, budget - budget_used)
                eta_sec = (rem_evals * sec_per_eval) if sec_per_eval else None
            else:
                status = "stalled"
                eta_sec = None

            base["status"] = status
            base["actual_session"] = actual_session
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
            return self._apply_waiting_queue(base, run_dir, version_id)

        summary = self._parse_run_summary_raw(run_dir, task_info, rep, active_tmux, now, default_prefix)
        self._cache_summaries[version_id][run_dir.name] = {"_stamp": stamp, "summary": summary}
        return self._apply_waiting_queue(summary, run_dir, version_id)

    def _parse_run_summary_raw(
        self,
        run_dir: Path,
        task_info: dict[str, Any],
        rep: int,
        active_tmux: set[str],
        now: datetime,
        default_prefix: str,
    ) -> dict[str, Any]:
        cfg_p = run_dir / "run_config.json"
        cfg = json.loads(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
        budget = cfg.get("method_params", {}).get("budget", 1000)
        backend = cfg.get("backend", "unknown")
        method = cfg.get("method") or default_prefix

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

        expected_session = f"{method}_{task_info['short']}_r{rep}"
        # Allocation arms share a task and seed; their sessions are in the manifest.
        if cfg.get('method_params', {}).get('allocation_arm') and default_prefix == 'v108alloc':
            arm = cfg['method_params']['allocation_arm']
            expected_session = f'{default_prefix}_{arm}_r{rep}'
        is_in_tmux, actual_session = _is_run_session_alive(expected_session, active_tmux)

        events_p = run_dir / "events.jsonl"
        last_event_ts = None
        op_counts = {"Init": 0, "Refine": 0, "Tune": 0, "Pivot": 0, "Fuse": 0}
        req_op_counts = {"Init": 0, "Refine": 0, "Tune": 0, "Pivot": 0, "Fuse": 0}
        fuse_fallbacks = 0
        parent_routes: dict[str, int] = {}
        pivot_routes = {"quality": 0, "uniform": 0}
        parent_improved_count = 0
        frontier_improved_count = 0
        llm_times: list[float] = []
        eval_times: list[float] = []
        prompt_tokens_list: list[int] = []
        summary_tokens_list: list[int] = []
        summary_present_count = 0

        status_counts = {"ok": 0, "eval_failed": 0, "invalid_output": 0}
        recent_timestamps: list[datetime] = []

        node_to_eval: dict[int, int] = {}
        eval_counter = 0
        fit_by_id: dict[int, float] = {}
        running_best_fitness: float | None = None

        if events_p.exists():
            try:
                for line in events_p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        if ev.get("slot_consumed"):
                            eval_counter += 1

                        nid = ev.get("node_id")
                        if nid is not None and nid not in node_to_eval:
                            node_to_eval[nid] = ev.get("evaluation_id") or (eval_counter if eval_counter > 0 else (nid + 1))

                        op = ev.get("operator") or ev.get("origin_operator")
                        req_op = ev.get("requested_operator") or op
                        if op in op_counts:
                            op_counts[op] += 1
                        if req_op in req_op_counts:
                            req_op_counts[req_op] += 1
                        if req_op == "Fuse" and op == "Refine":
                            fuse_fallbacks += 1

                        p_route = ev.get("selection", {}).get("parent_route")
                        if p_route:
                            parent_routes[p_route] = parent_routes.get(p_route, 0) + 1
                        if op == "Pivot" and p_route in pivot_routes:
                            pivot_routes[p_route] += 1

                        fit = ev.get("fitness")
                        pid = ev.get("parent_id")
                        p_imp = ev.get("parent_improved")
                        f_imp = ev.get("frontier_improved")

                        if fit is not None:
                            if f_imp is None:
                                f_imp = (running_best_fitness is None or fit > running_best_fitness)
                            if running_best_fitness is None or fit > running_best_fitness:
                                running_best_fitness = fit
                            if p_imp is None and pid is not None and pid in fit_by_id:
                                p_imp = (fit > fit_by_id[pid])
                            if nid is not None:
                                fit_by_id[nid] = fit

                        if p_imp:
                            parent_improved_count += 1
                        if f_imp:
                            frontier_improved_count += 1

                        if ev.get("llm_seconds") is not None:
                            llm_times.append(float(ev["llm_seconds"]))
                        if ev.get("eval_seconds") is not None:
                            eval_times.append(float(ev["eval_seconds"]))
                        if ev.get("prompt_tokens") is not None:
                            prompt_tokens_list.append(int(ev["prompt_tokens"]))
                        if ev.get("summary_tokens") is not None:
                            summary_tokens_list.append(int(ev["summary_tokens"]))
                        if ev.get("summary_status") == "present":
                            summary_present_count += 1

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

        has_finished_summary = False
        for s_name in ("run_summary.json", "summary.json"):
            summary_p = run_dir / "logs" / s_name
            if summary_p.exists():
                try:
                    s = json.loads(summary_p.read_text(encoding="utf-8"))
                    if s.get("status") == "finished":
                        has_finished_summary = True
                        break
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

        for n in nodes:
            nid = n.get("id")
            if n.get("evaluation_id") is None:
                n["evaluation_id"] = node_to_eval.get(nid) or ((nid + 1) if nid is not None else 1)
            if not n.get("operator") and n.get("origin_operator"):
                n["operator"] = n["origin_operator"]

        best_fitness = None
        breakthroughs: list[dict[str, Any]] = []

        sorted_nodes = sorted(
            nodes, key=lambda n: n.get("evaluation_id") or 0
        )
        seen_codes = set()
        for n in sorted_nodes:
            eid = n.get("evaluation_id")
            fit = n.get("fitness")
            if eid is None or fit is None:
                continue
            # Archived V10.8 runs retain their original scoring convention.
            if method == "v108" and tree_data.get("version", 1080) == 1080:
                if n.get("code") in seen_codes:
                    continue
                seen_codes.add(n.get("code"))
            if best_fitness is None or fit > best_fitness:
                best_fitness = fit
                breakthroughs.append(
                    {
                        "eid": eid,
                        "fitness": fit,
                        "display": self._format_metric(fit, task_info),
                        "node_id": n.get("id"),
                        "operator": n.get("operator") or n.get("origin_operator") or "Init",
                        "idea": (n.get("idea") or "")[:120],
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

        avg_llm = round(sum(llm_times) / len(llm_times), 1) if llm_times else None
        avg_eval = round(sum(eval_times) / len(eval_times), 2) if eval_times else None
        avg_tokens = int(sum(prompt_tokens_list) / len(prompt_tokens_list)) if prompt_tokens_list else None
        avg_summary_tokens = int(sum(summary_tokens_list) / len(summary_tokens_list)) if summary_tokens_list else None

        telemetry_payload = {
            "requested_operator_counts": req_op_counts,
            "executed_operator_counts": op_counts,
            "fuse_fallbacks": fuse_fallbacks,
            "parent_routes": parent_routes,
            "pivot_routes": pivot_routes,
            "parent_improved_count": parent_improved_count,
            "frontier_improved_count": frontier_improved_count,
            "avg_llm_seconds": avg_llm,
            "avg_eval_seconds": avg_eval,
            "avg_prompt_tokens": avg_tokens,
            "avg_summary_tokens": avg_summary_tokens,
            "summary_present_count": summary_present_count,
        }

        return {
            "name": run_dir.name,
            "task": task_info["key"],
            "rep": rep,
            "backend": backend,
            "method": method,
            "expected_session": expected_session,
            "actual_session": actual_session,
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
            "telemetry": telemetry_payload,
            "v108_telemetry": telemetry_payload,
            "v106_telemetry": telemetry_payload,
            "v105_telemetry": telemetry_payload,
        }

    def _parse_run_detail(
        self, run_dir: Path, task: str, run_name: str, default_prefix: str
    ) -> dict[str, Any]:
        task_info = TASK_MAP.get(task, {"key": task, "label": task, "unit": "fitness", "direction": "max", "short": task})
        rep_match = re.search(r"_rep(\d+)$", run_name)
        rep = int(rep_match.group(1)) if rep_match else 0
        active_tmux = _get_active_tmux_sessions()
        now = datetime.now()

        summary = self._parse_run_summary_raw(run_dir, task_info, rep, active_tmux, now, default_prefix)

        tree_p = run_dir / "tree_state.json"
        tree_data: dict[str, Any] = {}
        if tree_p.exists():
            try:
                tree_data = json.loads(tree_p.read_text(encoding="utf-8"))
            except Exception:
                pass

        nodes = tree_data.get("nodes", [])

        events_p = run_dir / "events.jsonl"
        recent_events: list[dict[str, Any]] = []
        scatter_points: list[dict[str, Any]] = []
        node_to_eval: dict[int, int] = {}
        fit_by_id: dict[int, float] = {}
        running_best_fitness: float | None = None

        if events_p.exists():
            try:
                lines = events_p.read_text(encoding="utf-8").splitlines()
                eval_counter = 0

                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        if ev.get("slot_consumed"):
                            eval_counter += 1

                        fit = ev.get("fitness")
                        nid = ev.get("node_id")
                        pid = ev.get("parent_id")
                        p_imp = ev.get("parent_improved")
                        f_imp = ev.get("frontier_improved")

                        if nid is not None and nid not in node_to_eval:
                            node_to_eval[nid] = ev.get("evaluation_id") or (eval_counter if eval_counter > 0 else (nid + 1))

                        if fit is not None:
                            if f_imp is None:
                                f_imp = (running_best_fitness is None or fit > running_best_fitness)
                            if running_best_fitness is None or fit > running_best_fitness:
                                running_best_fitness = fit
                            if p_imp is None and pid is not None and pid in fit_by_id:
                                p_imp = (fit > fit_by_id[pid])
                            if nid is not None:
                                fit_by_id[nid] = fit

                            step_val = ev.get("evaluation_id")
                            if step_val is None:
                                step_val = ev.get("step")
                            if step_val is None:
                                step_val = eval_counter if eval_counter > 0 else i

                            op_val = ev.get("operator") or ev.get("origin_operator") or "Init"
                            scatter_points.append(
                                {
                                    "step": step_val,
                                    "fitness": fit,
                                    "operator": op_val,
                                    "status": ev.get("status", "ok"),
                                    "node_id": nid,
                                    "parent_improved": p_imp,
                                    "frontier_improved": f_imp,
                                }
                            )
                    except Exception:
                        pass

                for line in lines[-40:]:
                    if not line.strip():
                        continue
                    try:
                        raw_ev = json.loads(line)
                        p_route = raw_ev.get("selection", {}).get("parent_route")
                        op_val = raw_ev.get("operator") or raw_ev.get("origin_operator") or "Init"
                        req_op_val = raw_ev.get("requested_operator") or op_val

                        p_imp = raw_ev.get("parent_improved")
                        f_imp = raw_ev.get("frontier_improved")
                        fit = raw_ev.get("fitness")
                        pid = raw_ev.get("parent_id")
                        nid = raw_ev.get("node_id")

                        if fit is not None and p_imp is None and pid is not None and pid in fit_by_id:
                            p_imp = (fit > fit_by_id[pid])

                        recent_events.append(
                            {
                                "candidate_id": raw_ev.get("candidate_id"),
                                "step": raw_ev.get("step"),
                                "operator": op_val,
                                "requested_operator": req_op_val,
                                "parent_route": p_route,
                                "status": raw_ev.get("status", "ok"),
                                "fitness": fit,
                                "node_id": nid,
                                "ts": raw_ev.get("ts"),
                                "llm_seconds": raw_ev.get("llm_seconds"),
                                "eval_seconds": raw_ev.get("eval_seconds"),
                                "prompt_tokens": raw_ev.get("prompt_tokens"),
                                "summary_tokens": raw_ev.get("summary_tokens"),
                                "summary_status": raw_ev.get("summary_status"),
                                "parent_improved": p_imp,
                                "frontier_improved": f_imp,
                                "reason": raw_ev.get("reason"),
                            }
                        )
                    except Exception:
                        pass
                recent_events.reverse()
            except Exception:
                pass

        for n in nodes:
            nid = n.get("id")
            if n.get("evaluation_id") is None:
                n["evaluation_id"] = node_to_eval.get(nid) or ((nid + 1) if nid is not None else 1)
            if not n.get("operator") and n.get("origin_operator"):
                n["operator"] = n["origin_operator"]

        best_node = None
        if nodes:
            milestones = summary.get("breakthroughs", [])
            best_node = (next((n for n in nodes if n.get("id") == milestones[-1]["node_id"]), None)
                         if milestones else None)
            if not best_node:
                best_node = max(nodes, key=lambda n: n.get("fitness") or float("-inf"))
            if best_node:
                if not best_node.get("operator") and best_node.get("origin_operator"):
                    best_node["operator"] = best_node["origin_operator"]
                if best_node.get("evaluation_id") is None:
                    nid = best_node.get("id")
                    best_node["evaluation_id"] = node_to_eval.get(nid) or ((nid + 1) if nid is not None else 1)

        nodes_compact = []
        for n in nodes:
            nodes_compact.append(
                {
                    "id": n.get("id"),
                    "evaluation_id": n.get("evaluation_id"),
                    "operator": n.get("operator") or n.get("origin_operator") or "Init",
                    "fitness": n.get("fitness"),
                    "parent_id": n.get("parent_id"),
                    "donor_id": n.get("donor_id"),
                    "idea": (n.get("idea") or "")[:160],
                }
            )

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
            version = params.get("version", [None])[0]

            if path == "/" or path == "/index.html":
                return self._serve_file(HTML_FILE, "text/html; charset=utf-8")

            if path == "/api/versions":
                versions = engine.get_available_versions()
                return self._send_json(
                    {
                        "current": version or engine.default_version,
                        "versions": versions,
                    }
                )

            if path == "/api/state" or path == "/api/overview":
                data = engine.get_overview(version=version)
                return self._send_json(data)

            if path == "/api/run":
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                if not task or not name:
                    return self.send_error(400, "Missing task or name parameter")
                detail = engine.get_run_detail(task, name, version=version)
                if not detail:
                    return self.send_error(404, "Run not found")
                return self._send_json(detail)

            if path == "/api/node":
                task = params.get("task", [""])[0]
                name = params.get("name", [""])[0]
                node_id_str = params.get("id", [""])[0]
                if not task or not name or not node_id_str:
                    return self.send_error(400, "Missing task, name or id")
                try:
                    nid = int(node_id_str)
                except ValueError:
                    return self.send_error(400, "Invalid node id")
                node_detail = engine.get_node_detail(task, name, nid, version=version)
                if not node_detail:
                    return self.send_error(404, "Node not found")
                return self._send_json(node_detail)

            return self.send_error(404, "Endpoint not found")

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if path in ("/", "/index.html", "/api/versions", "/api/state", "/api/overview", "/api/run", "/api/node"):
                self.send_response(200)
                if path.endswith(".html") or path == "/":
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                else:
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
            else:
                self.send_error(404)

        def _serve_file(self, file_path: Path, content_type: str) -> None:
            if not file_path.exists():
                return self.send_error(404, f"File {file_path.name} not found")
            try:
                content = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(500, f"Error serving file: {e}")

        def _send_json(self, data: Any) -> None:
            try:
                payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache, must-revalidate")
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(500, f"JSON encoding error: {e}")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            # Suppress default request logging to avoid terminal clutter
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceAAD V10.6 Live Monitor")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=8765, help="HTTP server port (default: 8765)")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Path to results directory (defaults to the selected version)",
    )
    parser.add_argument(
        "--version",
        default="v10_10",
        help="Default experiment version (default: v10_10)",
    )
    parser.add_argument(
        "--session-prefix",
        default="v1010",
        help="Tmux session prefix (default: v1010)",
    )
    args = parser.parse_args()

    engine = MonitorDataEngine(
        results_root=args.results_dir,
        default_version=args.version,
        default_session_prefix=args.session_prefix,
    )

    def background_polling() -> None:
        while True:
            try:
                engine.refresh()
            except Exception as e:
                print(f"[monitor-bg] Polling error: {e}", flush=True)
            time.sleep(2.0)

    bg_thread = threading.Thread(target=background_polling, daemon=True)
    bg_thread.start()

    server_address = (args.host, args.port)
    handler_class = make_request_handler(engine)
    server = ThreadingHTTPServer(server_address, handler_class)

    print("===========================================================", flush=True)
    print(f"🚀 TraceAAD {engine._resolve_version_meta(None)[3]} 训练实验可视化监控已启动", flush=True)
    print(f"📡 本地访问地址: http://127.0.0.1:{args.port}", flush=True)
    print(f"🌐 远程访问地址: http://{args.host}:{args.port}", flush=True)
    print("===========================================================", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[monitor] Shutting down gracefully...", flush=True)
        server.server_close()


if __name__ == "__main__":
    main()
