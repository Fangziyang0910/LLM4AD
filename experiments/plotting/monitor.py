"""Live observer for official TraceAAD V9.19 runs.

Reads the V9.19 artifacts already written by the search loop
(``evaluations.csv``, ``mechanism_events.jsonl``, ``decisions.jsonl``,
``checkpoints/latest.json``, ``checkpoints/view.json``,
``checkpoints/behave.npz``, ``best_history.jsonl``) and serves an
interactive page: batch overview, best-improvement history, formation
tree, BehaveSim landscape, formation trajectory, and per-slot P/U/T
competition.

    uv run python -m experiments.plotting.monitor [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments.plotting import v919_view as view

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
PAGE = (Path(__file__).with_name("v919_page.html")).read_text(encoding="utf-8")
POLL_INTERVAL_S = 10.0
RUN_PATH = re.compile(r"^/run/([^/]+)/([^/]+)(?:/(node|slot)/([^/]+))?$")


def make_handler(cache: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            root = Path(cache["root"])
            if self.path == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            if self.path == "/state":
                payload = cache.get("overview") or {"budget": 1000, "runs": []}
                return self._send_json(payload)
            matched = RUN_PATH.match(self.path.split("?", 1)[0])
            if matched:
                task, name, kind, ident = matched.groups()
                try:
                    if kind == "node":
                        payload = view.node_payload(root, task, name, int(ident))
                    elif kind == "slot":
                        payload = view.slot_payload(root, task, name, int(ident))
                    else:
                        payload = _cached_run(cache, root, task, name)
                except (KeyError, ValueError):
                    return self.send_error(404)
                return self._send_json(payload)
            self.send_error(404)

        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, payload: object) -> None:
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json", body)

        def log_message(self, *args) -> None:
            pass

    return Handler


def _cached_run(cache: dict, root: Path, task: str, name: str) -> dict:
    run_dir = root / task / "traceaad_v9_19" / name
    stamp = (
        _mtime(run_dir / "evaluations.csv"),
        _mtime(run_dir / "checkpoints" / "latest.json"),
        _mtime(run_dir / "checkpoints" / "view.json"),
        _mtime(run_dir / "checkpoints" / "behave.npz"),
        _mtime(run_dir / "mechanism_events.jsonl"),
        _mtime(run_dir / "best_history.jsonl"),
    )
    bucket: dict = cache.setdefault("runs", {})
    hit = bucket.get((task, name))
    if hit and hit["stamp"] == stamp:
        return hit["payload"]
    payload = view.run_payload(root, task, name)
    bucket[(task, name)] = {"stamp": stamp, "payload": payload}
    return payload


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=EXPERIMENTS_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    cache: dict = {"root": root, "overview": None, "runs": {}}

    def store() -> None:
        cache["overview"] = view.overview_payload(root)
        cache["overview"]["updated"] = time.time()

    def poll_forever() -> None:
        while True:
            started = time.time()
            try:
                store()
            except Exception as exc:
                print("poll failed:", exc, flush=True)
            time.sleep(max(1.0, POLL_INTERVAL_S - (time.time() - started)))

    print("scanning V9.19 runs…", flush=True)
    store()
    threading.Thread(target=poll_forever, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(cache))
    n_runs = len((cache["overview"] or {}).get("runs") or [])
    print(f"serving http://{args.host}:{args.port}  {n_runs} V9.19 runs", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
