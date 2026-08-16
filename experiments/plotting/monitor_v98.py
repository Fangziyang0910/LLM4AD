"""Live web dashboard for TraceAAD V9.8 batches.

Incrementally tails ``logs/events.jsonl`` of every run matching a glob and
serves a self-contained auto-refreshing page (stdlib only, no dependencies):

    uv run python -m experiments.plotting.monitor_v98 [--port 8765] \
        [--pattern '*/traceaad_v9_8/v9_8_[0-9]*']

Read-only with respect to run directories; safe to start while a batch runs.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
DEFAULT_PATTERN = "*/traceaad_v9_8/v9_8_[0-9]*"

TASK_ORDER = {
    "tsp_construct": 0,
    "cvrp_aco": 1,
    "op_aco": 2,
    "online_bin_packing": 3,
}

# status -> code shared with the page (0 ok, 1 eval_failed, 2 parse_failed)
STATUS_CODE = {"ok": 0, "eval_failed": 1, "parse_failed": 2}
# intent -> code (0 refine, 1 explore, 2 other/root)
INTENT_CODE = {"refine": 0, "explore": 1}

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
        self.name = run_dir.name
        rep = None
        for part in self.name.split("_"):
            if part.startswith("rep") and part[3:].isdigit():
                rep = int(part[3:])
        self.rep = rep if rep is not None else 0
        self.offset = 0
        self.pts: list[list] = []  # [order, fitness, intent_code, status_code]
        self.frontier: list[list] = []  # [order, added, removed]
        self.n_evals = 0
        self.first_ts: float | None = None
        self.last_ts: float | None = None
        self.counts = {
            "ok": 0,
            "eval_failed": 0,
            "parse_failed": 0,
            "refine": 0,
            "explore": 0,
            "new_hyp": 0,
        }
        self.best: float | None = None

    def _reset(self) -> None:
        self.offset = 0
        self.pts.clear()
        self.frontier.clear()
        self.n_evals = 0
        self.first_ts = None
        self.last_ts = None
        self.counts = {k: 0 for k in self.counts}
        self.best = None

    def _consume(self, event: dict) -> None:
        if event.get("event") != "response_finalized":
            return
        status = event.get("status")
        scode = STATUS_CODE.get(status, 0)
        key = "ok" if scode == 0 else str(status)
        self.counts[key] = self.counts.get(key, 0) + 1
        intent = event.get("intent")
        icode = INTENT_CODE.get(intent, 2)
        if icode == 0:
            self.counts["refine"] += 1
        elif icode == 1:
            self.counts["explore"] += 1
        if event.get("kind") == "new_hypothesis":
            self.counts["new_hyp"] += 1
        order = event.get("order")
        order = order if isinstance(order, int) else len(self.pts) + 1
        fit = _f(event.get("child_fitness"))
        self.pts.append([order, fit, icode, scode])
        if scode == 0 and fit is not None and (self.best is None or fit > self.best):
            self.best = fit
        eval_count = event.get("eval_count")
        if isinstance(eval_count, int):
            self.n_evals = max(self.n_evals, eval_count)
        added = event.get("added")
        removed = event.get("removed")
        if isinstance(added, int) or isinstance(removed, int):
            self.frontier.append(
                [order, added if isinstance(added, int) else 0,
                 removed if isinstance(removed, int) else 0]
            )
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
            "counts": self.counts,
            "pts": self.pts,
            "frontier": self.frontier,
        }


class Monitor:
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
        self.states: dict[Path, RunState] = {}

    def poll(self) -> list[RunState]:
        for run_dir in sorted(EXPERIMENTS_ROOT.glob(self.pattern)):
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
<title>TraceAAD V9.8 运行监控</title>
<style>
  :root { color-scheme: light; }
  body { font: 14px/1.45 -apple-system, "Segoe UI", "Noto Sans CJK SC", sans-serif;
         margin: 0; padding: 18px; background: #fafafa; color: #222; }
  h1 { font-size: 19px; margin: 0 0 2px; }
  h2 { font-size: 15px; margin: 26px 0 8px; }
  #meta { color: #777; font-size: 12.5px; margin-bottom: 14px; }
  table { border-collapse: collapse; font-size: 12.5px; background: #fff; }
  th, td { padding: 4px 9px; border-bottom: 1px solid #eee; text-align: right;
           white-space: nowrap; }
  th { color: #888; font-weight: 500; position: sticky; top: 0; background: #fff; }
  td.l, th.l { text-align: left; }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 9px; font-size: 11px; }
  .run  { background: #e3f0e3; color: #23632a; }
  .fin  { background: #e4e6f0; color: #3a4a8c; }
  .stall{ background: #f6e0e0; color: #9c2f2f; }
  .bar { width: 110px; height: 8px; background: #e8e8e8; border-radius: 4px;
         overflow: hidden; display: inline-block; vertical-align: middle; }
  .bar > div { height: 100%; background: #4e79a7; }
  .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(540px, 1fr));
           gap: 14px; }
  .card { background: #fff; border: 1px solid #ececec; border-radius: 8px; padding: 8px 10px; }
  .card h3 { font-size: 12px; margin: 2px 2px 6px; color: #555; font-weight: 500; }
  svg text { font: 10px "Segoe UI", "Noto Sans CJK SC", sans-serif; fill: #999; }
  .err { color: #b00; }
</style>
</head>
<body>
<h1>TraceAAD V9.8 运行监控</h1>
<div id="meta">loading…</div>
<div id="table"></div>
<div id="tasks"></div>
<script>
const REFRESH_MS = 5000;
const BUDGET = 1000;
const TASK_TITLES = {
  tsp_construct: "tsp_construct", cvrp_aco: "cvrp_aco",
  op_aco: "op_aco", online_bin_packing: "online_bin_packing",
};
const REP_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#8c564b"];
const INTENT_COLORS = { 0: "#e8a33d", 1: "#4e79a7", 2: "#b0b0b0" };
const INTENT_NAMES = { 0: "refine", 1: "explore", 2: "root/其他" };

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}
function fmt(v, d) { return v == null ? "–" : v.toFixed(d == null ? 3 : d); }
function fmtDur(h) {
  if (h == null) return "–";
  if (h < 1) return Math.round(h * 60) + " m";
  if (h < 48) return h.toFixed(1) + " h";
  return (h / 24).toFixed(1) + " d";
}

function bestSeries(run) {
  const best = []; let b = null;
  for (const p of run.pts) {
    if (p[3] === 0 && p[1] != null && (b == null || p[1] > b)) b = p[1];
    best.push(b);
  }
  return best;
}

function ratePerHour(run) {
  if (!run.first_ts || !run.last_ts || run.n_evals < 2) return null;
  const hours = (run.last_ts - run.first_ts) / 3600;
  return hours > 0 ? run.n_evals / hours : null;
}

function overviewRow(run) {
  const rate = ratePerHour(run);
  const eta = rate && run.n_evals < BUDGET ? (BUDGET - run.n_evals) / rate : null;
  const total = Math.max(1, run.counts.ok + run.counts.eval_failed + run.counts.parse_failed);
  const cls = run.finished ? "fin" : (run.stalled ? "stall" : "run");
  const label = run.finished ? "完成" : (run.stalled ? "停滞" : "运行中");
  const pct = Math.min(100, 100 * run.n_evals / BUDGET);
  return "<tr>" +
    `<td class="l">${esc(TASK_TITLES[run.task] || run.task)}</td><td>${run.rep}</td>` +
    `<td class="l"><span class="pill ${cls}">${label}</span>` +
      `<span style="color:#aaa;font-size:11px;margin-left:6px">${run.age_s != null ? fmtDur(-run.age_s / 3600).replace("-", "") + " 前有事件" : ""}</span></td>` +
    `<td><div class="bar"><div style="width:${pct}%"></div></div> ${run.n_evals}/${BUDGET}</td>` +
    `<td><b>${fmt(run.best)}</b></td>` +
    `<td>${rate == null ? "–" : rate.toFixed(0) + " /h"}</td>` +
    `<td>${run.finished ? "–" : fmtDur(eta)}</td>` +
    `<td>${(100 * run.counts.ok / total).toFixed(0)}%</td>` +
    `<td>${run.counts.eval_failed + run.counts.parse_failed}</td>` +
    `<td>${run.counts.refine}</td><td>${run.counts.explore}</td>` +
    `<td>${run.counts.new_hyp}</td></tr>`;
}

function overviewTable(runs) {
  let rows = runs.map(overviewRow).join("");
  return `<table><tr>
    <th class="l">任务</th><th>rep</th><th class="l">状态</th><th>评价进度</th>
    <th>最优</th><th>速率</th><th>预计剩余</th><th>有效</th><th>失败</th>
    <th>refine</th><th>explore</th><th>新假设</th></tr>${rows}</table>`;
}

function frame(w, h, padL, padR, padT, padB, xmin, xmax, ymin, ymax) {
  const iw = w - padL - padR, ih = h - padT - padB;
  const sx = v => padL + (v - xmin) / (xmax - xmin) * iw;
  const sy = v => padT + (1 - (v - ymin) / (ymax - ymin)) * ih;
  return { sx, sy, iw, ih };
}

function axes(g, w, h, padL, padT, ih, sx, ymin, ymax, sy, yFmt) {
  for (const t of [0, 250, 500, 750, 1000]) {
    g += `<line x1="${sx(t)}" y1="${padT}" x2="${sx(t)}" y2="${padT + ih}" stroke="#f0f0f0"/>` +
         `<text x="${sx(t)}" y="${h - 6}" text-anchor="middle">${t}</text>`;
  }
  for (let i = 0; i <= 2; i++) {
    const v = ymin + (ymax - ymin) * i / 2, y = sy(v);
    g += `<line x1="${padL}" y1="${y}" x2="${w}" y2="${y}" stroke="#f0f0f0"/>` +
         `<text x="${padL - 5}" y="${y + 3}" text-anchor="end">${yFmt(v)}</text>`;
  }
  return g;
}

function overlayChart(runs) {
  const w = 900, h = 240, padL = 56, padR = 12, padT = 10, padB = 24;
  let lo = Infinity, hi = -Infinity;
  const series = runs.map(r => {
    const b = bestSeries(r);
    for (const v of b) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return { r, b };
  });
  if (!isFinite(lo)) return "<div class='card'>尚无有效评价</div>";
  const pad = (hi - lo) * 0.06 || 0.1; lo -= pad; hi += pad;
  const { sx, sy, ih } = frame(w, h, padL, padR, padT, padB, 0, BUDGET, lo, hi);
  let g = axes("", w, h, padL, padT, ih, sx, lo, hi, sy, v => v.toFixed(2));
  for (const { r, b } of series) {
    let d = "", prev = null;
    r.pts.forEach((p, i) => {
      if (b[i] == null) return;
      const x = sx(Math.min(p[0], BUDGET)), y = sy(b[i]);
      d += (prev == null ? `M${x.toFixed(1)},${y.toFixed(1)}` : `L${x.toFixed(1)},${y.toFixed(1)}`);
      prev = p[0];
    });
    const color = REP_COLORS[(r.rep - 1) % REP_COLORS.length];
    g += `<path d="${d}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
  }
  let legend = "";
  series.forEach(({ r }, i) => {
    const color = REP_COLORS[(r.rep - 1) % REP_COLORS.length];
    const x = padL + 8 + i * 96;
    legend += `<line x1="${x}" y1="4" x2="${x + 18}" y2="4" stroke="${color}" stroke-width="2"/>` +
              `<text x="${x + 22}" y="7">rep${r.rep} · ${fmt(r.best)}</text>`;
  });
  return `<svg viewBox="0 0 ${w} ${h}" style="width:100%">${g}${legend}</svg>`;
}

function runCard(run) {
  const w = 520, h = 210, padL = 52, padR = 46, padT = 8, padB = 22;
  let lo = Infinity, hi = -Infinity;
  for (const p of run.pts)
    if (p[3] === 0 && p[1] != null) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); }
  let body = "";
  if (isFinite(lo)) {
    const pad = (hi - lo) * 0.07 || 0.1; lo -= pad; hi += pad;
    const { sx, sy, ih } = frame(w, h, padL, padR, padT, padB, 0, BUDGET, lo, hi);
    let g = axes("", w, h, padL, padT, ih, sx, lo, hi, sy, v => v.toFixed(1));
    // frontier churn on a right-side 0..cumMax scale
    if (run.frontier.length > 1) {
      let cumA = 0, cumR = 0; const ma = [], mr = [];
      for (const f of run.frontier) { cumA += f[1]; cumR += f[2]; ma.push([f[0], cumA]); mr.push([f[0], cumR]); }
      const top = Math.max(cumA, cumR, 1);
      const fy = v => padT + (1 - v / top) * ih;
      const path = arr => arr.map((p, i) => `${i ? "L" : "M"}${sx(Math.min(p[0], BUDGET)).toFixed(1)},${fy(p[1]).toFixed(1)}`).join("");
      g += `<path d="${path(ma)}" fill="none" stroke="#7fbf7f" stroke-width="1" opacity="0.75"/>` +
           `<path d="${path(mr)}" fill="none" stroke="#d98c8c" stroke-width="1" opacity="0.6" stroke-dasharray="3 2"/>` +
           `<text x="${w - 4}" y="${fy(top) + 10}" text-anchor="end" fill="#7fbf7f">frontier +${cumA}/−${cumR}</text>`;
    }
    for (const p of run.pts) {
      const x = sx(Math.min(p[0], BUDGET));
      if (p[3] !== 0) {
        const y = padT + ih - 2;
        g += `<path d="M${x - 2},${y - 2} l4,4 m0,-4 l-4,4" stroke="#c0392b" stroke-width="1" opacity="0.7"/>`;
      } else if (p[1] != null) {
        g += `<circle cx="${x.toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="1.7" fill="${INTENT_COLORS[p[2]]}" opacity="0.8"/>`;
      }
    }
    const b = bestSeries(run);
    let d = "", prev = null;
    run.pts.forEach((p, i) => {
      if (b[i] == null) return;
      const x = sx(Math.min(p[0], BUDGET)), y = sy(b[i]);
      d += (prev == null ? `M${x.toFixed(1)},${y.toFixed(1)}` : `L${x.toFixed(1)},${y.toFixed(1)}`);
      prev = p[0];
    });
    g += `<path d="${d}" fill="none" stroke="#222" stroke-width="1.4"/>`;
    body = `<svg viewBox="0 0 ${w} ${h}" style="width:100%">${g}</svg>`;
  } else {
    body = "<div style='color:#aaa;padding:30px;text-align:center'>尚无有效评价</div>";
  }
  const rate = ratePerHour(run);
  const meta = `评价 ${run.n_evals}/${BUDGET} · 最优 <b>${fmt(run.best)}</b>` +
    ` · refine ${run.counts.refine} / explore ${run.counts.explore} · 新假设 ${run.counts.new_hyp}` +
    (rate ? ` · ${rate.toFixed(0)}/h` : "");
  return `<div class="card"><h3>${esc(run.name)} — rep${run.rep}</h3>${body}` +
    `<div style="font-size:11.5px;color:#777;margin-top:4px">${meta}</div>` +
    `<div style="font-size:11px;color:#999;margin-top:2px">` +
    Object.entries(INTENT_NAMES).map(([k, v]) =>
      `<span style="color:${INTENT_COLORS[k]}">●</span> ${v}`).join(" &nbsp; ") +
    ` &nbsp; <span style="color:#c0392b">×</span> 评价失败 &nbsp; — best-so-far</div></div>`;
}

function render(state) {
  const runs = state.runs;
  const fin = runs.filter(r => r.finished).length;
  document.getElementById("meta").textContent =
    `更新于 ${new Date(state.updated * 1000).toLocaleTimeString()} · 完成 ${fin}/${runs.length} · ` +
    `${new Date().toLocaleDateString()} 每 ${REFRESH_MS / 1000} s 自动刷新`;
  document.getElementById("table").innerHTML = overviewTable(runs);
  const byTask = new Map();
  for (const r of runs) {
    if (!byTask.has(r.task)) byTask.set(r.task, []);
    byTask.get(r.task).push(r);
  }
  let html = "";
  for (const [task, truns] of byTask) {
    html += `<h2>${esc(TASK_TITLES[task] || task)} — 三次重复 best-so-far</h2>` +
      `<div class="card" style="max-width:940px">${overlayChart(truns)}</div>` +
      `<div class="cards" style="margin-top:10px">${truns.map(runCard).join("")}</div>`;
  }
  document.getElementById("tasks").innerHTML = html;
}

async function tick() {
  try {
    const res = await fetch("/state");
    render(await res.json());
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
    parser.add_argument("--pattern", default=DEFAULT_PATTERN,
                        help="run-dir glob relative to experiments/ (default: %(default)s)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    monitor = Monitor(args.pattern)
    monitor.poll()  # initial full parse before serving
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor))
    print(f"serving http://{args.host}:{args.port}  pattern=experiments/{args.pattern}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
