"""Lightweight live dashboard for TraceAAD search batches.

Reads the compact CSV artifacts the runner already maintains per run
(``evaluations.csv``, ``best_curve.csv``, ``logs/summary.json``,
``run_config.json``) instead of tailing the multi-hundred-MB ``events.jsonl``,
so refreshing dozens of runs parses a few MB at most. A background thread
polls on a fixed interval and ``/state`` serves the cached payload; requests
never touch the filesystem (stdlib only, no dependencies):

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
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
# Any traceaad run directory that speaks the CSV artifact format; runs
# without evaluations.csv (older artifact generations) are ignored.
DEFAULT_PATTERNS = ("*/traceaad_*/v*_[0-9]*",)

TASK_ORDER = {
    "tsp_construct": 0,
    "cvrp_aco": 1,
    "op_aco": 2,
    "online_bin_packing": 3,
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
        # traceaad_v9_8 -> V9.8
        self.version = (
            run_dir.parent.name.removeprefix("traceaad_").replace("_", ".").upper()
        )
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
        self.samples: list[list] = []  # [timestamp, n_evals] for the rate window

    def _read_budget(self) -> int:
        try:
            config = json.loads(
                (self.run_dir / "run_config.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return 1000
        budget = config.get("method_params", {}).get("budget")
        return budget if isinstance(budget, int) and budget > 0 else 1000

    def poll(self) -> None:
        eval_csv = self.run_dir / "evaluations.csv"
        pts: list[list] = []
        n_evals = 0
        best = None
        try:
            with eval_csv.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        count = int(row.get("eval_count") or 0)
                    except ValueError:
                        continue
                    pts.append(
                        [
                            count,
                            _f(row.get("child_fitness")),
                            STATUS_CODE.get(row.get("status") or "ok", 0),
                        ]
                    )
                    n_evals = max(n_evals, count)
                    row_best = _f(row.get("best_fitness"))
                    if row_best is not None:
                        best = row_best
        except OSError:
            pass
        self.pts = pts
        self.n_evals = n_evals or len(pts)
        self.best = best

        # The first best-curve row marks the first evaluation; the runner
        # flushes evaluations.csv after every row, so its mtime marks the
        # latest activity (both interpreted in the server's local timezone).
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
        try:
            self.last_ts = eval_csv.stat().st_mtime
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
        try:
            summary = json.loads(
                (self.run_dir / "logs" / "summary.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False
        return summary.get("status") == "finished"

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
                    and (run_dir / "evaluations.csv").is_file()
                    and run_dir not in self.states
                ):
                    self.states[run_dir] = RunState(run_dir)
        states = list(self.states.values())
        for state in states:
            state.poll()
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
<title>TraceAAD 运行监控</title>
<style>
  :root { color-scheme: light; }
  body { font: 14px/1.5 -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
         margin: 0; padding: 22px 26px; background: #fcfcfc; color: #222; }
  h1 { font-size: 18px; margin: 0 0 3px; font-weight: 600; }
  h2 { font-size: 13.5px; margin: 28px 0 6px; font-weight: 600; color: #444; }
  #meta { color: #888; font-size: 12.5px; margin-bottom: 16px; }
  #meta b { color: #333; font-weight: 600; }
  table { border-collapse: collapse; font-size: 13px; background: #fff; }
  th, td { padding: 5px 12px; border-bottom: 1px solid #f0f0f0; text-align: right;
           white-space: nowrap; }
  th { color: #999; font-weight: 500; }
  td.l, th.l { text-align: left; }
  .pill { display: inline-block; padding: 1px 9px; border-radius: 9px; font-size: 11px; }
  .run  { background: #e5f2e5; color: #23632a; }
  .fin  { background: #ececec; color: #777; }
  .stall{ background: #f6e0e0; color: #9c2f2f; }
  .bar { width: 120px; height: 7px; background: #ededed; border-radius: 4px;
         overflow: hidden; display: inline-block; vertical-align: middle; }
  .bar > div { height: 100%; background: #4e79a7; }
  .bar > div.done { background: #b9b9b9; }
  .tabs { display: flex; gap: 8px; margin: 2px 0 14px; }
  .tabs button { font-size: 13px; font-family: inherit; padding: 4px 16px;
                 border: 1px solid #e2e2e2; background: #fff; color: #888;
                 border-radius: 16px; cursor: pointer; }
  .tabs button.active { background: #333; border-color: #333; color: #fff; }
  .tabs button b { font-weight: 600; }
  .chart { background: #fff; border: 1px solid #f0f0f0; border-radius: 8px;
           padding: 10px 12px 6px; max-width: 980px; }
  svg text { font: 10.5px "Segoe UI", "Noto Sans CJK SC", sans-serif; fill: #aaa; }
  .err { color: #b00; }
</style>
</head>
<body>
<h1>TraceAAD 运行监控</h1>
<div id="meta">loading…</div>
<div id="tabs"></div>
<div id="table"></div>
<div id="tasks"></div>
<script>
const REFRESH_MS = 30000;
const BUDGET = 1000;
const REP_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"];

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}
function fmt(v) { return v == null ? "–" : String(+v.toPrecision(4)); }
function fmtDur(h) {
  if (h == null) return "–";
  if (h < 1) return Math.round(h * 60) + " 分钟";
  if (h < 48) return h.toFixed(1) + " 小时";
  return (h / 24).toFixed(1) + " 天";
}
function yTick(v) {
  const a = Math.abs(v);
  return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
}

function bestSeries(run) {
  const best = []; let b = null;
  for (const p of run.pts) {
    if (p[2] === 0 && p[1] != null && (b == null || p[1] > b)) b = p[1];
    best.push(b);
  }
  return best;
}
function versionKey(v) {
  const m = /^V(\d+)\.(\d+)$/.exec(v);
  return m ? [+m[1], +m[2]] : [0, 0];
}
function cmpVersion(a, b) {
  const ka = versionKey(a), kb = versionKey(b);
  return ka[0] - kb[0] || ka[1] - kb[1] || a.localeCompare(b);
}
function ratePerHour(run) {
  if (run.rate_per_hour != null) return run.rate_per_hour;
  if (!run.first_ts || !run.last_ts || run.n_evals < 2) return null;
  const hours = (run.last_ts - run.first_ts) / 3600;
  return hours > 0 ? run.n_evals / hours : null;
}
function etaHours(run) {
  const rate = ratePerHour(run);
  const budget = run.budget || BUDGET;
  return rate && run.n_evals < budget ? (budget - run.n_evals) / rate : null;
}

function overviewRow(run) {
  const cls = run.finished ? "fin" : (run.stalled ? "stall" : "run");
  const label = run.finished ? "完成" : (run.stalled ? "停滞" : "运行中");
  const budget = run.budget || BUDGET;
  const pct = Math.min(100, 100 * run.n_evals / budget);
  const eta = run.finished
    ? (run.first_ts && run.last_ts
        ? `用时 ${fmtDur((run.last_ts - run.first_ts) / 3600)}` : "–")
    : fmtDur(etaHours(run));
  return "<tr>" +
    `<td class="l">${esc(run.task)}</td><td>${run.rep}</td>` +
    `<td class="l"><span class="pill ${cls}">${label}</span></td>` +
    `<td><div class="bar"><div class="${run.finished ? "done" : ""}" style="width:${pct}%"></div></div>` +
      `&nbsp; ${run.n_evals} / ${budget}</td>` +
    `<td><b>${fmt(run.best)}</b></td>` +
    `<td>${eta}</td></tr>`;
}

function overlayChart(runs) {
  const w = 940, h = 230, padL = 60, padR = 14, padT = 12, padB = 24;
  let lo = Infinity, hi = -Infinity;
  const series = runs.map(r => {
    const b = bestSeries(r);
    for (const v of b) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return { r, b };
  });
  if (!isFinite(lo)) return "<div style='color:#bbb;padding:26px;text-align:center'>尚无有效评价</div>";
  const pad = (hi - lo) * 0.07 || 0.1; lo -= pad; hi += pad;
  const iw = w - padL - padR, ih = h - padT - padB;
  const sx = v => padL + v / BUDGET * iw;
  const sy = v => padT + (1 - (v - lo) / (hi - lo)) * ih;
  let g = "";
  for (const t of [0, 250, 500, 750, 1000]) {
    g += `<line x1="${sx(t)}" y1="${padT}" x2="${sx(t)}" y2="${padT + ih}" stroke="#f2f2f2"/>` +
         `<text x="${sx(t)}" y="${h - 7}" text-anchor="middle">${t}</text>`;
  }
  for (let i = 0; i <= 2; i++) {
    const v = lo + (hi - lo) * (1 - i / 2), y = padT + ih * i / 2;
    g += `<line x1="${padL}" y1="${y}" x2="${w - padR}" y2="${y}" stroke="#f2f2f2"/>` +
         `<text x="${padL - 6}" y="${y + 3.5}" text-anchor="end">${yTick(v)}</text>`;
  }
  for (const { r, b } of series) {
    let d = "", started = false;
    r.pts.forEach((p, i) => {
      if (b[i] == null) return;
      const x = sx(Math.min(p[0], BUDGET)).toFixed(1), y = sy(b[i]).toFixed(1);
      d += (started ? "L" : "M") + x + "," + y;
      started = true;
    });
    const color = REP_COLORS[(r.rep - 1) % REP_COLORS.length];
    g += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
  }
  let legend = "";
  series.forEach(({ r }, i) => {
    const color = REP_COLORS[(r.rep - 1) % REP_COLORS.length];
    const x = padL + 10 + i * 118;
    legend += `<line x1="${x}" y1="2" x2="${x + 20}" y2="2" stroke="${color}" stroke-width="2"/>` +
              `<text x="${x + 25}" y="5" fill="#666">rep${r.rep} · ${fmt(r.best)}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%">${g}${legend}</svg>`;
}

let activeVersion = null;
let lastState = null;

function setVersion(v) {
  activeVersion = v;
  if (lastState) render(lastState);
}

function render(state) {
  lastState = state;
  const byVersion = {};
  for (const r of state.runs)
    (byVersion[r.version] = byVersion[r.version] || []).push(r);
  const versions = Object.keys(byVersion).sort(cmpVersion);
  if (!versions.length) {
    document.getElementById("meta").textContent = "没有匹配的 run";
    return;
  }
  if (!activeVersion || !byVersion[activeVersion]) {
    activeVersion = [...versions].reverse()
      .find(v => byVersion[v].some(r => !r.finished)) || versions[versions.length - 1];
  }

  document.getElementById("tabs").innerHTML = versions.map(v => {
    const runs = byVersion[v];
    const fin = runs.filter(r => r.finished).length;
    return `<button class="${v === activeVersion ? "active" : ""}" ` +
           `onclick="setVersion('${v}')">${v} <b>${fin}/${runs.length}</b></button>`;
  }).join("");

  const runs = byVersion[activeVersion];
  const fin = runs.filter(r => r.finished).length;
  const remain = Math.max(0, ...runs.filter(r => !r.finished).map(etaHours));
  document.getElementById("meta").innerHTML =
    `<b>${esc(activeVersion)}</b> · 更新于 ${new Date(state.updated * 1000).toLocaleTimeString()} · ` +
    `完成 <b>${fin}/${runs.length}</b>` +
    (fin < runs.length ? ` · 整批预计还需 <b>${fmtDur(remain)}</b>（按最慢 run）` : " · 全部完成");

  document.getElementById("table").innerHTML =
    `<table><tr><th class="l">任务</th><th>rep</th><th class="l">状态</th>` +
    `<th>评价进度</th><th>best</th><th>剩余 / 用时</th></tr>` +
    runs.map(overviewRow).join("") + `</table>`;

  const byTask = new Map();
  for (const r of runs) {
    if (!byTask.has(r.task)) byTask.set(r.task, []);
    byTask.get(r.task).push(r);
  }
  let html = "";
  for (const [task, truns] of byTask) {
    html += `<h2>${esc(task)} · best-so-far</h2>` +
            `<div class="chart">${overlayChart(truns)}</div>`;
  }
  document.getElementById("tasks").innerHTML = html;
}

window.onerror = function (msg) {
  const meta = document.getElementById("meta");
  if (meta) { meta.textContent = "脚本错误: " + msg; meta.classList.add("err"); }
};

async function fetchState() {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 8000);
  try {
    const res = await fetch("/state", { signal: ctrl.signal });
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

async function tick() {
  try {
    render(await fetchState());
    document.getElementById("meta").classList.remove("err");
  } catch (e) {
    const meta = document.getElementById("meta");
    const hint = e && e.name === "AbortError"
      ? "拉取超时（8 秒），稍后自动重试"
      : "连接失败: " + e;
    meta.textContent = hint;
    meta.classList.add("err");
  }
}
tick();
setInterval(tick, REFRESH_MS);
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
