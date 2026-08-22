"""Lightweight live dashboard for TraceAAD search batches.

Reads the compact CSV artifacts the runner already maintains per run
(``evaluations.csv``, ``best_curve.csv``, ``logs/summary.json``,
``run_config.json``) instead of tailing the multi-hundred-MB ``events.jsonl``,
so refreshing dozens of runs parses a few MB at most. Two CSV generations are
supported: older ones with ``child_fitness``/``best_fitness`` columns and a
``best_curve.csv``, and the V9.14+ minimal format whose ``fitness`` column is
folded into a running best here (V9.15 rows additionally carry p_E /
protection-hit diagnostics rendered as a mechanism chart). Older artifact
generations without ``evaluations.csv`` fall back to
``artifacts/candidates.jsonl`` (budget axis = rows with ``evaluator_called``).
When one version directory holds several batches (e.g. ``traceaad_v9_7``
holds the 0814 canonical and the qwen38 model-swap batch), tabs carry a batch
suffix so protocols never merge into one chart. Versions whose runs are all
finished collapse into an archived group; the default tab is the version
with unfinished runs.

    uv run python -m experiments.plotting.monitor [--port 8765] \
        [--pattern '*/traceaad_v9_9/v9_9_[0-9]*' ...]

Runs are grouped by method version (from the method directory name); the page
shows one tab per version with evaluation progress and best-so-far evolution.
Curves are plotted against evaluator count (the budget axis), smoke run
directories are skipped, and ETAs use a recent sample window so outage gaps
around a resume do not deflate the rate. Read-only with respect to run
directories; safe to start while a batch runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
# Any run directory that speaks one of the supported artifact formats. The
# traceaad patterns catch the versioned batches (v*_) and extension batches
# (tm_); the baseline patterns catch five-method comparison batches whose
# run names start with a date (20260822_142500_...).
DEFAULT_PATTERNS = (
    "*/traceaad_*/v*_[0-9]*",
    "*/traceaad_*/tm_[0-9]*",
    "*/eoh/[0-9]*_*",
    "*/reevo/[0-9]*_*",
    "*/mcts_ahd/[0-9]*_*",
    "*/pathwise/[0-9]*_*",
    "*/calm/[0-9]*_*",
)

# Baseline method directories get their paper name as the tab label; traceaad
# versions keep the V9.x parsing.
METHOD_LABELS = {
    "eoh": "EoH",
    "reevo": "ReEvo",
    "mcts_ahd": "MCTS-AHD",
    "pathwise": "PathWise",
    "calm": "CALM",
    "shinka_evo": "ShinkaEvo",
}

# 历史中间版本：结果已凝练进 docs/experiments/归档实验结果.md，
# 不再进入监控面板（连同其子批次，如 V9.5 的两个日期批次）。
HIDDEN_VERSIONS = {"V9.1", "V9.2", "V9.3", "V9.4", "V9.5", "V9.6"}

TASK_ORDER = {
    "tsp_construct": 0,
    "cvrp_aco": 1,
    "op_aco": 2,
    "online_bin_packing": 3,
    "vrptw_construct": 4,
}

# evaluations.csv status -> code shared with the page (0 ok, 1 eval_failed,
# 2 parse_failed)
STATUS_CODE = {"ok": 0, "eval_failed": 1, "parse_failed": 2}

# Run becomes "stalled" when its evaluations.csv has not been touched for
# this long (the runner flushes after every row). A single evaluation can
# take minutes, so the threshold is generous.
STALE_AFTER_S = 30 * 60

POLL_INTERVAL_S = 10.0
# Rate window for ETAs: (n_evals gain) / (wall time) over recent monitor
# samples, so outage gaps before a resume do not deflate the rate. Falls back
# to the full-span average until the window holds this much span.
RATE_WINDOW_S = 3600.0
RATE_MIN_SPAN_S = 600.0


def _f(value) -> float | None:
    """Pass through finite floats only, so the payload is strict JSON."""
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if value != value or value in (float("inf"), float("-inf")) else value


def _epoch(ts: str | None) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


class RunState:
    """View of one run rebuilt from its compact CSV artifacts.

    The files top out at a few hundred KB, so poll() simply re-reads them:
    no incremental offsets whose state could go stale after a resume.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.task = run_dir.parent.parent.name
        parent_name = run_dir.parent.name
        if parent_name.startswith("traceaad_"):
            # traceaad_v9_8 -> V9.8
            self.base_version = (
                parent_name.removeprefix("traceaad_").replace("_", ".").upper()
            )
        else:
            self.base_version = METHOD_LABELS.get(parent_name, parent_name.upper())
        self.version = self.base_version
        # Batches sharing a version directory get a tab suffix (V9.7·0814);
        # qwen38 is the model-swap batch and keeps its explicit name.
        if "qwen38" in run_dir.name:
            self.batch = "qwen38"
        else:
            m = re.search(r"_(\d{8})(?:_|$)", run_dir.name)
            self.batch = m.group(1)[4:] if m else ""
        self.name = run_dir.name
        rep = None
        for part in self.name.split("_"):
            if part.startswith("rep") and part[3:].isdigit():
                rep = int(part[3:])
        self.rep = rep if rep is not None else 0
        self.budget = self._read_budget()
        self.pts: list[list] = []  # [eval_count, fitness, status_code]
        self.n_evals = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.best: float | None = None
        # V9.14+/V9.15 CSV: [eval_count, p_explore, protected_selected]
        self.mech: list[list] = []
        self.n_explore: int | None = None
        self.n_protected: int | None = None
        self.samples: list[list] = []  # [timestamp, n_evals] for the rate window

    def _read_budget(self) -> int:
        try:
            config = json.loads(
                (self.run_dir / "run_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            self._config = {}
            return 1000
        self._config = config
        params = config.get("method_params", {})
        for key in ("budget", "max_sample_nums"):
            budget = params.get(key)
            if isinstance(budget, int) and budget > 0:
                return budget
        return 1000

    def _poll_evaluations_csv(
        self,
    ) -> tuple[list[list] | None, int, float | None, list[list], int | None, int | None]:
        eval_csv = self.run_dir / "evaluations.csv"
        pts: list[list] = []
        n_evals = 0
        best = None
        mech: list[list] = []
        n_explore: int | None = None
        n_protected = 0
        try:
            with eval_csv.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        count = int(row.get("eval_count") or 0)
                    except ValueError:
                        continue
                    # Older generations write child_fitness; V9.14+ write fitness.
                    fitness = _f(row.get("child_fitness"))
                    if fitness is None:
                        fitness = _f(row.get("fitness"))
                    pts.append(
                        [
                            count,
                            fitness,
                            STATUS_CODE.get(row.get("status") or "ok", 0),
                        ]
                    )
                    n_evals = max(n_evals, count)
                    row_best = _f(row.get("best_fitness"))
                    if row_best is not None:
                        best = row_best
                    elif fitness is not None and (best is None or fitness > best):
                        best = fitness
                    # V9.15 mechanism diagnostics; absent in older versions.
                    p_explore = _f(row.get("p_explore"))
                    if p_explore is not None:
                        bonus = _f(row.get("parent_bonus")) or 0.0
                        protected = 1 if bonus > 0 else 0
                        n_protected += protected
                        mech.append([count, p_explore, protected])
                        if n_explore is None:
                            n_explore = 0
                        if (row.get("intent") or "") == "explore":
                            n_explore += 1
        except OSError:
            return None, 0, None, [], None, None
        return pts, n_evals, best, mech, n_explore, n_protected

    def _poll_baseline_samples(
        self,
    ) -> tuple[list[list] | None, int, float | None]:
        """Baseline five-method runs: logs/samples/samples_*.json shards.

        Curve points use (sample_order, score), the same axis the docs
        search-curve figures use. Budget progress comes from
        method_events.jsonl: sample_registered rows with counts_budget=True
        (EoH's initial pool is evaluated but outside the formal budget);
        methods without that event (MCTS-AHD counts expansions, CALM reports
        per-epoch sample_count) fall back to their own counters.
        """
        shards = sorted((self.run_dir / "logs" / "samples").glob("samples_*.json"))
        if not shards:
            return None, 0, None
        pts: list[list] = []
        best = None
        max_order = 0
        for shard in shards:
            try:
                rows = json.loads(shard.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for row in rows:
                order = row.get("sample_order")
                if not isinstance(order, int):
                    continue
                score = _f(row.get("score"))
                pts.append([order, score, 0 if score is not None else 1])
                max_order = max(max_order, order)
                if score is not None and (best is None or score > best):
                    best = score
        pts.sort(key=lambda p: p[0])
        n_evals = self._baseline_n_evals(max_order)
        return pts, n_evals, best

    def _baseline_n_evals(self, fallback: int) -> int:
        path = self.run_dir / "logs" / "method_events.jsonl"
        budgeted = 0
        has_registered = False
        epoch_count = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("event") == "sample_registered":
                        has_registered = True
                        if row.get("counts_budget"):
                            budgeted += 1
                    elif row.get("event") == "epoch":
                        count = row.get("sample_count")
                        if isinstance(count, int):
                            epoch_count = max(epoch_count, count)
        except OSError:
            return fallback
        if has_registered:
            return budgeted
        return epoch_count or fallback

    def _poll_candidates_jsonl(self) -> tuple[list[list], int, float | None]:
        # Older artifact generations: no evaluations.csv, one JSON line per
        # attempt. Budget axis = rows that actually called the evaluator
        # (ok + eval_failed), matching summary.evaluator_call_count.
        path = self.run_dir / "artifacts" / "candidates.jsonl"
        pts: list[list] = []
        n_evals = 0
        best = None
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    count = row.get("order")
                    if not isinstance(count, int):
                        continue
                    fitness = _f(row.get("child_fitness"))
                    status = row.get("status") or "ok"
                    pts.append([count, fitness, STATUS_CODE.get(status, 0)])
                    if row.get("evaluator_called"):
                        n_evals += 1
                    if status == "ok" and fitness is not None and (best is None or fitness > best):
                        best = fitness
        except OSError:
            pass
        return pts, n_evals, best

    def poll(self) -> None:
        pts, n_evals, best, mech, n_explore, n_protected = (
            self._poll_evaluations_csv()
        )
        if pts is None:
            pts, n_evals, best = self._poll_baseline_samples()
            mech, n_explore, n_protected = [], None, None
        if pts is None:
            pts, n_evals, best = self._poll_candidates_jsonl()
            mech, n_explore, n_protected = [], None, None
        self.pts = pts
        self.n_evals = n_evals
        self.best = best
        self.mech = mech
        self.n_explore = n_explore
        self.n_protected = n_protected if n_explore is not None else None

        # The first best-curve row marks the first evaluation; the runner
        # flushes evaluations.csv after every row, so its mtime marks the
        # latest activity (both interpreted in the server's local timezone).
        # V9.14+ runs have no best_curve.csv: fall back to run_config's
        # creation time so rate/elapsed estimates keep working.
        self.first_ts = None
        try:
            with (self.run_dir / "best_curve.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                for row in csv.DictReader(handle):
                    self.first_ts = _epoch(row.get("timestamp"))
                    break
        except OSError:
            pass
        if self.first_ts is None:
            self.first_ts = _epoch(
                (self._config or {}).get("created_at")
            )
        try:
            self.last_ts = (self.run_dir / "evaluations.csv").stat().st_mtime
        except OSError:
            shards = sorted((self.run_dir / "logs" / "samples").glob("samples_*.json"))
            if shards:
                self.last_ts = shards[-1].stat().st_mtime
            else:
                try:
                    self.last_ts = (
                        self.run_dir / "artifacts" / "candidates.jsonl"
                    ).stat().st_mtime
                except OSError:
                    self.last_ts = None
        now = time.time()
        self.samples = [s for s in self.samples if now - s[0] <= RATE_WINDOW_S]
        self.samples.append([now, self.n_evals])

    def _rate_per_hour(self) -> float | None:
        """Eval rate from recent samples when available, full span otherwise."""
        now = time.time()
        window = [s for s in self.samples if now - s[0] <= RATE_WINDOW_S]
        if len(window) >= 2 and window[-1][0] - window[0][0] >= RATE_MIN_SPAN_S:
            span_h = (window[-1][0] - window[0][0]) / 3600
            if span_h > 0:
                return (window[-1][1] - window[0][1]) / span_h
            return None
        if self.first_ts and self.last_ts and self.n_evals >= 2:
            span_h = (self.last_ts - self.first_ts) / 3600
            if span_h > 0:
                return self.n_evals / span_h
        return None

    def _finished(self) -> bool:
        for name in ("summary.json", "run_summary.json"):
            try:
                summary = json.loads(
                    (self.run_dir / "logs" / name).read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                continue
            if summary.get("status") == "finished":
                return True
        return False

    def payload(self) -> dict:
        finished = self._finished()
        age = self.last_ts - time.time() if self.last_ts is not None else None
        return {
            "version": self.version,
            "task": self.task,
            "name": self.name,
            "rep": self.rep,
            "budget": self.budget,
            "finished": finished,
            "stalled": (
                not finished and age is not None and -age > STALE_AFTER_S
            ),
            "age_s": -age if age is not None else None,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "n_evals": self.n_evals,
            "rate_per_hour": self._rate_per_hour(),
            "best": self.best,
            "pts": self.pts,
            "mech": self.mech,
            "n_explore": self.n_explore,
            "n_protected": self.n_protected,
        }


class Monitor:
    def __init__(self, patterns: tuple[str, ...]) -> None:
        self.patterns = patterns
        self.states: dict[Path, RunState] = {}

    def poll(self) -> list[RunState]:
        for pattern in self.patterns:
            for run_dir in sorted(EXPERIMENTS_ROOT.glob(pattern)):
                if (
                    run_dir.is_dir()
                    and "smoke" not in run_dir.name
                    and (
                        (run_dir / "evaluations.csv").is_file()
                        or (run_dir / "artifacts" / "candidates.jsonl").is_file()
                        or any(
                            (run_dir / "logs" / "samples").glob("samples_*.json")
                        )
                    )
                    and run_dir.parent.name.removeprefix("traceaad_")
                        .replace("_", ".")
                        .upper()
                        not in HIDDEN_VERSIONS
                    and run_dir not in self.states
                ):
                    self.states[run_dir] = RunState(run_dir)
        states = list(self.states.values())
        for state in states:
            state.poll()
        # The canonical date batch keeps the plain version name; suffixes
        # appear only when several date batches share the directory (V9.5)
        # or for explicitly labeled batches (qwen38 model swap).
        date_batches: dict[str, set[str]] = {}
        for state in states:
            if state.batch and state.batch != "qwen38":
                date_batches.setdefault(state.base_version, set()).add(state.batch)
        for state in states:
            state.version = state.base_version
            if not state.batch:
                continue
            if state.batch == "qwen38" or len(date_batches.get(state.base_version, ())) > 1:
                state.version = f"{state.base_version}·{state.batch}"
        states.sort(key=lambda s: (TASK_ORDER.get(s.task, 99), s.rep, s.name))
        return states

    def payload(self) -> dict:
        return {
            "updated": time.time(),
            "budget": 1000,
            "runs": [state.payload() for state in self.poll()],
        }


PAGE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraceAAD 运行监控</title>
<style>
  :root{
    --bg:#f4f6fb; --card:#ffffff; --ink:#182036; --sub:#5b6478; --faint:#98a1b6;
    --line:#e8ebf3; --accent:#4f46e5;
    --run:#0e9f6e; --run-soft:#e2f5ec; --fin:#8a93a8; --fin-soft:#eef0f4;
    --stall:#d83a3a; --stall-soft:#fdeaea;
    --shadow:0 1px 2px rgba(24,32,54,.05),0 10px 28px -14px rgba(24,32,54,.22);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.55 -apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif}
  header{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.88);backdrop-filter:blur(10px);
         border-bottom:1px solid var(--line);padding:13px 26px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  header h1{font-size:16.5px;margin:0;font-weight:700;letter-spacing:.2px}
  .dot{width:8px;height:8px;border-radius:50%;background:var(--run);box-shadow:0 0 0 4px var(--run-soft)}
  .dot.err{background:var(--stall);box-shadow:0 0 0 4px var(--stall-soft)}
  #meta{color:var(--sub);font-size:12.5px;display:flex;gap:16px;flex-wrap:wrap;margin-left:auto}
  #meta b{color:var(--ink);font-weight:600}
  main{max-width:1180px;margin:0 auto;padding:20px 26px 60px}
  .versions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:18px}
  .chip{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:5px 14px;
        font-size:13px;color:var(--sub);cursor:pointer;transition:.15s;font-family:inherit}
  .chip:hover{border-color:#c6cde2;color:var(--ink)}
  .chip.active{background:var(--ink);border-color:var(--ink);color:#fff}
  .chip b{font-weight:700;margin-left:5px}
  .chip .fin{color:var(--fin)} .chip.active .fin{color:#c9d0de}
  .divider{width:1px;height:22px;background:var(--line);margin:0 4px}
  details.arch{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  details.arch summary{list-style:none;cursor:pointer;color:var(--faint);font-size:12.5px;
        border:1px dashed #d3d9e7;border-radius:999px;padding:4px 13px;user-select:none;white-space:nowrap}
  details.arch summary::-webkit-details-marker{display:none}
  details.arch summary:hover{color:var(--sub);border-color:#b9c1d6}
  details.arch .chip{padding:3px 11px;font-size:12px}
  .strip{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:16px}
  .stat{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 16px;box-shadow:var(--shadow)}
  .stat .k{font-size:11.5px;color:var(--faint);margin-bottom:3px;letter-spacing:.3px}
  .stat .v{font-size:17px;font-weight:700}
  .stat .v small{font-size:12px;color:var(--faint);font-weight:500;margin-left:2px}
  .tasks{display:grid;grid-template-columns:1fr;gap:16px}
  .task{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);overflow:hidden}
  .task h2{margin:0;padding:13px 16px 8px;font-size:13.5px;font-weight:700;display:flex;align-items:center;gap:8px}
  .task h2 .tag{margin-left:auto;font-size:11.5px;font-weight:600;color:var(--faint)}
  .task table{margin:0 6px 4px}
  table{border-collapse:collapse;width:100%;font-size:12.5px}
  th{color:var(--faint);font-weight:500;text-align:right;padding:2px 10px 6px}
  th:first-child{text-align:left}
  td{padding:5px 10px;text-align:right;border-top:1px solid #f2f4f9;white-space:nowrap}
  td:first-child{text-align:left}
  .repchip{display:inline-flex;align-items:center;gap:6px;font-weight:600}
  .repchip i{width:8px;height:8px;border-radius:3px;display:inline-block}
  .pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:11px;font-weight:600}
  .pill.run{background:var(--run-soft);color:var(--run)}
  .pill.fin{background:var(--fin-soft);color:var(--fin)}
  .pill.stall{background:var(--stall-soft);color:var(--stall)}
  .bar{width:104px;height:6px;background:#edf0f6;border-radius:99px;display:inline-block;vertical-align:middle;overflow:hidden}
  .bar>i{display:block;height:100%;background:var(--accent);border-radius:99px}
  .bar>i.done{background:#c3c9d8}
  .legend{display:flex;gap:12px;flex-wrap:wrap;padding:8px 16px 0}
  .legend span{display:inline-flex;align-items:center;gap:5px;font-size:11.5px;color:var(--sub)}
  .legend i{width:14px;height:3.5px;border-radius:2px;display:inline-block}
  .chart{padding:2px 10px 10px}
  svg text{font:10.5px -apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;fill:var(--faint)}
  .hint{color:#b9c0d2;text-align:center;padding:22px 0;font-size:12.5px}
</style>
</head>
<body>
<header>
  <span class="dot" id="live"></span>
  <h1>TraceAAD 运行监控</h1>
  <div id="meta">加载中…</div>
</header>
<main>
  <div class="versions" id="tabs"></div>
  <div class="strip" id="strip"></div>
  <div class="tasks" id="tasks"></div>
</main>
<script>
const REFRESH_MS = 30000;
const BUDGET = 1000;
const REP_COLORS = ["#4f46e5","#0ea5e9","#10b981","#f59e0b","#ef4444","#8b5cf6"];
const TASK_LABEL = {tsp_construct:"TSP 构造", cvrp_aco:"CVRP-ACO", op_aco:"OP-ACO", online_bin_packing:"在线装箱", vrptw_construct:"VRPTW 构造"};
const TASK_ORDER = ["tsp_construct","cvrp_aco","op_aco","online_bin_packing","vrptw_construct"];

function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));}
function fmt(v){return v==null?"–":String(+v.toPrecision(4));}
function fmtDur(h){
  if(h==null)return "–";
  if(h<1)return Math.round(h*60)+" 分钟";
  if(h<48)return h.toFixed(1)+" 小时";
  return (h/24).toFixed(1)+" 天";
}
function yTick(v){const a=Math.abs(v);return a>=1000?(v/1000).toFixed(1)+"k":a>=100?v.toFixed(0):a>=10?v.toFixed(1):v.toFixed(2);}
function repColor(r){return REP_COLORS[(r-1)%REP_COLORS.length];}

function versionKey(v){const m=/^V(\d+)\.(\d+)/.exec(v);return m?[+m[1],+m[2]]:[0,0];}
function cmpVersion(a,b){const ka=versionKey(a),kb=versionKey(b);return ka[0]-kb[0]||ka[1]-kb[1]||a.localeCompare(b);}

function ratePerHour(run){
  if(run.rate_per_hour!=null)return run.rate_per_hour;
  if(!run.first_ts||!run.last_ts||run.n_evals<2)return null;
  const h=(run.last_ts-run.first_ts)/3600;
  return h>0?run.n_evals/h:null;
}
function etaHours(run){
  const rate=ratePerHour(run),budget=run.budget||BUDGET;
  return rate&&run.n_evals<budget?(budget-run.n_evals)/rate:null;
}
function bestSeries(run){
  const best=[];let b=null;
  for(const p of run.pts){
    if(p[2]===0&&p[1]!=null&&(b==null||p[1]>b))b=p[1];
    best.push(b);
  }
  return best;
}
// 突破点：best-so-far 序列发生跳变的位置（该次评价刷新了全局最好）
function breakthroughs(run){
  const b=bestSeries(run);const out=[];let prev=null;
  b.forEach((v,i)=>{
    if(v==null)return;
    if(prev===null||v!==prev){out.push({x:run.pts[i][0],v});prev=v;}
  });
  return out;
}
function pct5(sorted){
  const pos=(sorted.length-1)*.05,i=Math.floor(pos),j=Math.ceil(pos);
  return i===j?sorted[i]:sorted[i]+(sorted[j]-sorted[i])*(pos-i);
}
function niceStep(raw){
  const p=Math.pow(10,Math.floor(Math.log10(raw)));
  for(const m of [1,2,2.5,5,10]) if(raw<=m*p) return m*p;
  return 10*p;
}

let activeVersion=null,lastState=null;
function setVersion(v){activeVersion=v;if(lastState)render(lastState);}

function statusPill(run){
  const cls=run.finished?"fin":(run.stalled?"stall":"run");
  const label=run.finished?"完成":(run.stalled?"停滞":"运行中");
  return `<span class="pill ${cls}">${label}</span>`;
}

function taskCard(task,runs){
  const fin=runs.filter(r=>r.finished).length;
  const hasMech=runs.some(r=>r.n_explore!=null);
  let rows=runs.map(r=>{
    const budget=r.budget||BUDGET;
    const pct=Math.min(100,100*r.n_evals/budget);
    const eta=r.finished
      ?(r.first_ts&&r.last_ts?fmtDur((r.last_ts-r.first_ts)/3600):"–")
      :fmtDur(etaHours(r));
    const mechCells=hasMech
      ?`<td>${r.n_explore==null?"–":(100*r.n_explore/Math.max(1,r.n_evals)).toFixed(1)+"%"}</td>`+
       `<td>${r.n_protected==null?"–":r.n_protected}</td>`
      :"";
    return `<tr>
      <td><span class="repchip"><i style="background:${repColor(r.rep)}"></i>rep${r.rep}</span></td>
      <td>${statusPill(r)}</td>
      <td><span class="bar"><i class="${r.finished?"done":""}" style="width:${pct}%"></i></span></td>
      <td>${r.n_evals}/${budget}</td>
      <td><b>${fmt(r.best)}</b></td>
      ${mechCells}
      <td>${eta}</td></tr>`;
  }).join("");
  const byRep=new Map(runs.map(r=>[r.rep,r]));
  const legend=[...byRep.keys()].sort((a,b)=>a-b).map(rep=>{
    const r=byRep.get(rep);
    return `<span><i style="background:${repColor(rep)}"></i>rep${rep} · best ${fmt(r.best)} · 突破 ${breakthroughs(r).length} 次</span>`;
  }).join("");
  return `<section class="task">
    <h2>${esc(TASK_LABEL[task]||task)}<span class="tag">${fin}/${runs.length} 完成</span></h2>
    <table><tr><th>运行</th><th>状态</th><th></th><th>评价</th><th>best</th>${hasMech?"<th>Exp%</th><th>保护</th>":""}<th>剩余 / 用时</th></tr>${rows}</table>
    <div class="legend">${legend}</div>
    <div class="chart">${overlayChart(runs)}</div>
    ${hasMech?`<div class="legend"><span><i style="background:#98a1b6"></i>p_E 探索率（0.20 基线 → 0.50 上限）</span><span>圆点 = 受保护 Explore 子节点被选中</span></div><div class="chart">${mechChart(runs)}</div>`:""}
  </section>`;
}

// V9.15 机制图：每次决策的 p_E 探索率 + 保护命中位置；y 轴固定 0.14–0.52
function mechChart(runs){
  const w=1160,h=140,padL=64,padR=20,padT=10,padB=24;
  const lo=0.14,hi=0.52;
  const iw=w-padL-padR,ih=h-padT-padB;
  const sx=v=>padL+Math.min(v,BUDGET)/BUDGET*iw;
  const sy=v=>padT+(1-(v-lo)/(hi-lo))*ih;
  let g="";
  for(const v of [0.2,0.35,0.5]){
    const y=sy(v).toFixed(1);
    g+=`<line x1="${padL}" y1="${y}" x2="${w-padR}" y2="${y}" stroke="#f1f3f9"/>`+
       `<text x="${padL-7}" y="${+y+3.5}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  for(const t of [0,250,500,750,1000]){
    g+=`<text x="${sx(t)}" y="${h-4}" text-anchor="middle">${t}</text>`;
  }
  for(const r of runs){
    if(!r.mech||!r.mech.length)continue;
    const color=repColor(r.rep);
    let d="",started=false;
    for(const m of r.mech){
      const x=sx(Math.min(m[0],BUDGET)).toFixed(1),y=sy(Math.max(lo,Math.min(hi,m[1]))).toFixed(1);
      d+=(started?"L":"M")+x+","+y;started=true;
    }
    g+=`<path d="${d}" fill="none" stroke="${color}" stroke-width="1.4" stroke-linejoin="round" opacity=".85"><title>rep${r.rep} p_E</title></path>`;
    for(const m of r.mech){
      if(!m[2])continue;
      g+=`<circle cx="${sx(Math.min(m[0],BUDGET)).toFixed(1)}" cy="${sy(Math.max(lo,Math.min(hi,m[1]))).toFixed(1)}" r="2.1" fill="${color}"><title>rep${r.rep} 保护命中 · eval ${m[0]} · p_E ${m[1].toFixed(3)}</title></circle>`;
    }
  }
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%">${g}</svg>`;
}

function overlayChart(runs){
  const w=1160,h=320,padL=64,padR=20,padT=14,padB=32;
  const series=runs.map(r=>({r,b:bestSeries(r)}));
  const all=[];
  for(const {b} of series)for(const v of b)if(v!=null)all.push(v);
  if(!all.length)return `<div class="hint">尚无有效评价</div>`;
  all.sort((x,y)=>x-y);
  // 与 plot_search_curves 相同的下界裁剪：warmup 首评可能差一个量级
  let lo=pct5(all),hi=all[all.length-1];
  const pad=Math.max((hi-lo)*.07,.1);lo-=pad;hi+=pad;
  const iw=w-padL-padR,ih=h-padT-padB;
  const sx=v=>padL+Math.min(v,BUDGET)/BUDGET*iw;
  const sy=v=>padT+(1-Math.max(0,Math.min(1,(v-lo)/(hi-lo))))*ih;
  let g="";
  const step=niceStep((hi-lo)/4);
  for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step){
    const y=sy(v).toFixed(1);
    g+=`<line x1="${padL}" y1="${y}" x2="${w-padR}" y2="${y}" stroke="#f1f3f9"/>`+
       `<text x="${padL-7}" y="${+y+3.5}" text-anchor="end">${yTick(v)}</text>`;
  }
  for(const t of [0,250,500,750,1000]){
    g+=`<line x1="${sx(t)}" y1="${padT}" x2="${sx(t)}" y2="${padT+ih}" stroke="#f1f3f9"/>`+
       `<text x="${sx(t)}" y="${h-6}" text-anchor="middle">${t}</text>`;
  }
  for(const {r,b} of series){
    const color=repColor(r.rep);
    let d="",last=null,started=false;
    r.pts.forEach((p,i)=>{
      if(b[i]==null)return;
      const x=sx(Math.min(p[0],BUDGET)).toFixed(1),y=sy(b[i]).toFixed(1);
      d+=(started?"L":"M")+x+","+y;started=true;last=[x,y];
    });
    g+=`<path d="${d}" fill="none" stroke="${color}" stroke-width="7" stroke-opacity="0"><title>rep${r.rep} · best ${fmt(r.best)} · ${r.n_evals} evals</title></path>`;
    g+=`<path d="${d}" fill="none" stroke="${color}" stroke-width="1.9" stroke-linejoin="round"/>`;
    // 突破点：空心圆标记该 rep 每一次刷新全局最好
    for(const bt of breakthroughs(r)){
      const x=sx(Math.min(bt.x,BUDGET)).toFixed(1),y=sy(bt.v).toFixed(1);
      g+=`<circle cx="${x}" cy="${y}" r="3.4" fill="#fff" stroke="${color}" stroke-width="1.7"><title>rep${r.rep} 突破 · eval ${bt.x} · ${fmt(bt.v)}</title></circle>`;
    }
    if(last)g+=`<circle cx="${last[0]}" cy="${last[1]}" r="3" fill="${color}"/>`;
  }
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%">${g}</svg>`;
}

function render(state){
  lastState=state;
  const byVersion={};
  for(const r of state.runs)(byVersion[r.version]=byVersion[r.version]||[]).push(r);
  const versions=Object.keys(byVersion).sort(cmpVersion);
  if(!versions.length){
    document.getElementById("meta").textContent="没有匹配的 run";
    document.getElementById("tabs").innerHTML="";
    document.getElementById("strip").innerHTML="";
    document.getElementById("tasks").innerHTML="";
    return;
  }
  // 活跃 = 存在未完成且未停滞的 run；早已停止的旧批次（如 V9.5 的
  // 10/12）不再占据活跃分组，自动落入归档。
  const active=versions.filter(v=>byVersion[v].some(r=>!r.finished&&!r.stalled));
  const archived=versions.filter(v=>!active.includes(v));
  if(!activeVersion||!byVersion[activeVersion]){
    activeVersion=active.length?active[active.length-1]:versions[versions.length-1];
  }

  let tabs=active.map(v=>{
    const rs=byVersion[v],fin=rs.filter(r=>r.finished).length,run=rs.length-fin;
    return `<button class="chip ${v===activeVersion?"active":""}" onclick="setVersion('${v}')">${esc(v)}<b>${run?`运行 ${run}`:`<span class="fin">${fin}/${rs.length}</span>`}</b></button>`;
  }).join("");
  if(archived.length){
    tabs+=`<span class="divider"></span><details class="arch" ${active.some(v=>archived.includes(v))?"":""}><summary>已归档版本 · ${archived.length}</summary>`+
      archived.map(v=>{
        const rs=byVersion[v],fin=rs.filter(r=>r.finished).length;
        return `<button class="chip ${v===activeVersion?"active":""}" onclick="setVersion('${v}')">${esc(v)}<b><span class="fin">${fin}/${rs.length}</span></b></button>`;
      }).join("")+`</details>`;
  }
  document.getElementById("tabs").innerHTML=tabs;

  const runs=byVersion[activeVersion];
  const fin=runs.filter(r=>r.finished).length;
  const unfinished=runs.filter(r=>!r.finished);
  const etas=unfinished.map(etaHours).filter(h=>h!=null);
  const remain=etas.length?Math.max(...etas):null;
  const avgPct=Math.round(runs.reduce((s,r)=>s+Math.min(1,r.n_evals/(r.budget||BUDGET)),0)/runs.length*100);
  const totalRate=unfinished.reduce((s,r)=>s+(ratePerHour(r)||0),0);
  document.getElementById("strip").innerHTML=
    `<div class="stat"><div class="k">完成运行</div><div class="v">${fin}<small>/ ${runs.length}</small></div></div>`+
    `<div class="stat"><div class="k">平均评价进度</div><div class="v">${avgPct}<small>%</small></div></div>`+
    `<div class="stat"><div class="k">活跃速率合计</div><div class="v">${unfinished.length?Math.round(totalRate):"–"}<small>eval/h</small></div></div>`+
    `<div class="stat"><div class="k">${fin<runs.length?"整批预计还需":"整批用时"}</div><div class="v">${fin<runs.length?fmtDur(remain):fmtDur(Math.max(...runs.map(r=>r.first_ts&&r.last_ts?(r.last_ts-r.first_ts)/3600:0)))}</div></div>`;

  document.getElementById("meta").innerHTML=
    `<span>更新于 <b>${new Date(state.updated*1000).toLocaleTimeString()}</b></span>`+
    `<span>运行中 <b>${state.runs.filter(r=>!r.finished).length}</b> / ${state.runs.length}</span>`+
    `<span>30 秒自动刷新</span>`;

  const byTask=new Map();
  for(const r of runs){
    if(!byTask.has(r.task))byTask.set(r.task,[]);
    byTask.get(r.task).push(r);
  }
  let html="";
  for(const task of TASK_ORDER){
    if(byTask.has(task))html+=taskCard(task,byTask.get(task));
  }
  document.getElementById("tasks").innerHTML=html;
}

window.onerror=function(msg){
  const m=document.getElementById("meta");
  if(m){m.textContent="脚本错误: "+msg;m.classList.add("err");}
};
async function fetchState(){
  const ctrl=new AbortController();
  const timer=setTimeout(()=>ctrl.abort(),8000);
  try{
    const res=await fetch("/state",{signal:ctrl.signal});
    return await res.json();
  }finally{clearTimeout(timer);}
}
async function tick(){
  try{
    render(await fetchState());
    document.getElementById("live").classList.remove("err");
  }catch(e){
    const live=document.getElementById("live");
    live.classList.add("err");
    document.getElementById("meta").textContent=
      e&&e.name==="AbortError"?"拉取超时（8 秒），稍后自动重试":"连接失败: "+e;
  }
}
tick();
setInterval(tick,REFRESH_MS);
</script>
</body>
</html>

"""


def make_handler(cache: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/state":
                payload = cache["payload"]
                if payload is None:  # first poll still running
                    payload = {"updated": time.time(), "budget": 1000, "runs": []}
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, *args) -> None:  # silence per-request logs
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        action="append",
        dest="patterns",
        metavar="GLOB",
        help="run-dir glob relative to experiments/, repeatable "
        f"(default: {' and '.join(DEFAULT_PATTERNS)})",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    monitor = Monitor(tuple(args.patterns or ()) or DEFAULT_PATTERNS)
    cache: dict = {"payload": None}

    def poll_forever() -> None:
        while True:
            started = time.time()
            try:
                cache["payload"] = monitor.payload()
            except Exception as exc:  # keep serving the last good payload
                print("poll failed:", exc, flush=True)
            time.sleep(max(1.0, POLL_INTERVAL_S - (time.time() - started)))

    threading.Thread(target=poll_forever, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cache))
    print(
        f"serving http://{args.host}:{args.port}  patterns:",
        *[f"experiments/{p}" for p in monitor.patterns],
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
