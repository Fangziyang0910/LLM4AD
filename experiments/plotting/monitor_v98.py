"""Live web dashboard for TraceAAD search batches.

Incrementally tails ``logs/events.jsonl`` of every run matching the patterns
and serves a self-contained auto-refreshing page (stdlib only, no
dependencies):

    uv run python -m experiments.plotting.monitor_v98 [--port 8765] \
        [--pattern '*/traceaad_v9_9/v9_9_[0-9]*' ...]

Runs are grouped by method version (from the method directory name); the page
shows one tab per version with evaluation progress and best-so-far evolution.
Read-only with respect to run directories; safe to start while a batch runs.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_PATTERNS = (
    "*/traceaad_v9_8/v9_8_[0-9]*",
    "*/traceaad_v9_9/v9_9_[0-9]*",
)

TASK_ORDER = {
    "tsp_construct": 0,
    "cvrp_aco": 1,
    "op_aco": 2,
    "online_bin_packing": 3,
}

# status -> code shared with the page (0 ok, 1 eval_failed, 2 parse_failed)
STATUS_CODE = {"ok": 0, "eval_failed": 1, "parse_failed": 2}

# Run becomes "stalled" when no event arrived for this long (seconds). A single
# evaluation can take minutes, so the threshold is generous.
STALE_AFTER_S = 30 * 60


def _f(value) -> float | None:
    """Pass through finite floats only, so the payload is strict JSON."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return None if value != value or value in (float("inf"), float("-inf")) else value


def _epoch(ts: str | None) -> float | None:
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


class RunState:
    """Incrementally parsed view of one run's events.jsonl."""

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
        self.offset = 0
        self.pts: list[list] = []  # [order, fitness, status_code]
        self.n_evals = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.best: float | None = None

    def _reset(self) -> None:
        self.offset = 0
        self.pts.clear()
        self.n_evals = 0
        self.first_ts = None
        self.last_ts = None
        self.best = None

    def _consume(self, event: dict) -> None:
        if event.get("event") != "response_finalized":
            return
        order = event.get("order")
        order = order if isinstance(order, int) else len(self.pts) + 1
        fit = _f(event.get("child_fitness"))
        scode = STATUS_CODE.get(event.get("status"), 0)
        self.pts.append([order, fit, scode])
        if scode == 0 and fit is not None and (self.best is None or fit > self.best):
            self.best = fit
        eval_count = event.get("eval_count")
        if isinstance(eval_count, int):
            self.n_evals = max(self.n_evals, eval_count)
        ts = _epoch(event.get("timestamp"))
        if ts is not None:
            self.first_ts = ts if self.first_ts is None else self.first_ts
            self.last_ts = ts

    def poll(self) -> None:
        path = self.run_dir / "logs" / "events.jsonl"
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self.offset:  # file truncated: a fresh attempt started
            self._reset()
        if size == self.offset:
            return
        with path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        chunks = data.split(b"\n")
        tail = chunks.pop()  # bytes after the last newline (may be partial)
        for chunk in chunks:
            self.offset += len(chunk) + 1
            text = chunk.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                self._consume(event)
        # bytes after the last newline were never added to offset, so the
        # partial tail is reread on the next poll
        if self.n_evals == 0:
            self.n_evals = len(self.pts)

    def payload(self) -> dict:
        summary = self.run_dir / "logs" / "summary.json"
        finished = False
        if summary.is_file():
            try:
                finished = (
                    json.loads(summary.read_text(encoding="utf-8")).get("status")
                    == "finished"
                )
            except (json.JSONDecodeError, OSError):
                pass
        now = time.time()
        age = self.last_ts - now if self.last_ts is not None else None
        return {
            "version": self.version,
            "task": self.task,
            "name": self.name,
            "rep": self.rep,
            "finished": finished,
            "stalled": (
                not finished and age is not None and -age > STALE_AFTER_S
            ),
            "age_s": -age if age is not None else None,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "n_evals": self.n_evals,
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
                if run_dir.is_dir() and run_dir not in self.states:
                    self.states[run_dir] = RunState(run_dir)
        states = list(self.states.values())
        for state in states:
            state.poll()
        states.sort(
            key=lambda s: (TASK_ORDER.get(s.task, 99), s.rep, s.name)
        )
        return states

    def payload(self) -> dict:
        runs = [state.payload() for state in self.poll()]
        return {
            "updated": time.time(),
            "budget": 1000,
            "runs": runs,
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
function ratePerHour(run) {
  if (!run.first_ts || !run.last_ts || run.n_evals < 2) return null;
  const hours = (run.last_ts - run.first_ts) / 3600;
  return hours > 0 ? run.n_evals / hours : null;
}
function etaHours(run) {
  const rate = ratePerHour(run);
  return rate && run.n_evals < BUDGET ? (BUDGET - run.n_evals) / rate : null;
}

function overviewRow(run) {
  const cls = run.finished ? "fin" : (run.stalled ? "stall" : "run");
  const label = run.finished ? "完成" : (run.stalled ? "停滞" : "运行中");
  const pct = Math.min(100, 100 * run.n_evals / BUDGET);
  const eta = run.finished
    ? (run.first_ts && run.last_ts
        ? `用时 ${fmtDur((run.last_ts - run.first_ts) / 3600)}` : "–")
    : fmtDur(etaHours(run));
  return "<tr>" +
    `<td class="l">${esc(run.task)}</td><td>${run.rep}</td>` +
    `<td class="l"><span class="pill ${cls}">${label}</span></td>` +
    `<td><div class="bar"><div class="${run.finished ? "done" : ""}" style="width:${pct}%"></div></div>` +
      `&nbsp; ${run.n_evals} / ${BUDGET}</td>` +
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
  const versions = Object.keys(byVersion).sort();
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

async function tick() {
  try {
    const res = await fetch("/state");
    render(await res.json());
    document.getElementById("meta").classList.remove("err");
  } catch (e) {
    document.getElementById("meta").textContent = "连接失败: " + e + "，重试中…";
    document.getElementById("meta").classList.add("err");
  }
}
tick();
setInterval(tick, REFRESH_MS);
</script>
</body>
</html>
"""


def make_handler(monitor: Monitor) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            if self.path == "/state":
                body = json.dumps(monitor.payload()).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    monitor = Monitor(tuple(args.patterns or ()) or DEFAULT_PATTERNS)
    monitor.poll()  # initial full parse before serving
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor))
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
