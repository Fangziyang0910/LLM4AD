"""Append-only run artifacts for TraceAAD V9.7.

Files under one run directory:

- ``artifacts/llm_calls.jsonl``  — one row per model call, including transport failures
- ``artifacts/candidates.jsonl`` — one row per finalized attempt; the single
  per-attempt record, carrying the anchor-to-child relation alongside the result
- ``artifacts/decisions.jsonl``  — ``route_selected``, ``anchor_selected``,
  ``history_built``, ``best_updated``
- ``logs/errors.jsonl``          — run-level errors
- ``logs/summary.json``          — written once when the run ends

Only raw facts are recorded; derived metrics are computed offline.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

_ERROR_TRUNCATE = 1000
_TRACEBACK_TRUNCATE = 4000


def _now() -> datetime:
    return datetime.now().astimezone()


class RunArtifacts:
    def __init__(self, run_dir: str | Path) -> None:
        run = Path(run_dir)
        self._started_at = _now()
        self._summary_path = run / "logs" / "summary.json"
        self._files: dict[str, TextIO] = {}
        for name, relative in (
            ("llm_calls", "artifacts/llm_calls.jsonl"),
            ("candidates", "artifacts/candidates.jsonl"),
            ("decisions", "artifacts/decisions.jsonl"),
            ("errors", "logs/errors.jsonl"),
        ):
            path = run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            self._files[name] = path.open("a", encoding="utf-8")

    def record_llm_call(self, **row: Any) -> None:
        self._append("llm_calls", row)

    def record_candidate(self, **row: Any) -> None:
        self._append("candidates", row)

    def record_decision(self, event: str, **payload: Any) -> None:
        self._append("decisions", {"event": event, **payload})

    def record_error(self, scope: str, exc: BaseException) -> None:
        self._append(
            "errors",
            {
                "scope": scope,
                "ts": _now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:_ERROR_TRUNCATE],
                "traceback": "".join(
                    traceback.format_exception(exc)
                )[:_TRACEBACK_TRUNCATE],
            },
        )

    def write_summary(self, **payload: Any) -> None:
        finished_at = _now()
        summary = {
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - self._started_at).total_seconds(),
            **payload,
        }
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def finish(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()

    def _append(self, name: str, row: dict[str, Any]) -> None:
        handle = self._files.get(name)
        if handle is None:
            return
        json.dump(row, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()


__all__ = ["RunArtifacts"]
