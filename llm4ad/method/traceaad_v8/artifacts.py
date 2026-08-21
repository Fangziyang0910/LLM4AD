"""Append-only run I/O for this TraceAAD version.

Files under one run directory:

- ``artifacts/llm_calls.jsonl``
- ``artifacts/candidates.jsonl``
- ``artifacts/edges.jsonl``
- ``artifacts/decisions.jsonl``
- ``logs/summary.json``

Only raw facts are recorded; derived metrics are computed offline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


class RunArtifacts:
    def __init__(self, run_dir: str | Path) -> None:
        root = Path(run_dir)
        self._summary_path = root / "logs" / "summary.json"
        self._candidates_path = root / "artifacts" / "candidates.jsonl"
        self._edges_path = root / "artifacts" / "edges.jsonl"
        self._llm_calls_path = root / "artifacts" / "llm_calls.jsonl"
        self._decisions_path = root / "artifacts" / "decisions.jsonl"
        self._candidates_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)

    def write_summary(self, **payload: Any) -> None:
        self._summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )

    def record_candidate(self, **payload: Any) -> None:
        self._append_jsonl(self._candidates_path, payload)

    def record_edge(self, **payload: Any) -> None:
        self._append_jsonl(self._edges_path, payload)

    def record_llm_call(self, **payload: Any) -> None:
        self._append_jsonl(self._llm_calls_path, payload)

    def record_decision(self, event: str, **payload: Any) -> None:
        self._append_jsonl(self._decisions_path, {"event": event, **payload})

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=_json_default)
            handle.write("\n")


__all__ = ["RunArtifacts"]
