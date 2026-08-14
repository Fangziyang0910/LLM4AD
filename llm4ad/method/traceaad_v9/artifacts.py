"""Append-only run I/O for this TraceAAD version.

Files under one run directory:

- ``artifacts/llm_calls.jsonl``
- ``artifacts/candidates.jsonl``
- ``artifacts/edges.jsonl``
- ``artifacts/decisions.jsonl``
- ``logs/progress.log``
- ``logs/errors.jsonl``
- ``logs/summary.json``

Only raw facts are recorded; derived metrics are computed offline.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

_RESPONSE_TRUNCATE = 2000
_ERROR_TRUNCATE = 1000


def _now() -> datetime:
    return datetime.now().astimezone()


def _truncate(text: str | None, limit: int) -> str | None:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    return str(value)


class RunArtifacts:
    def __init__(self, run_dir: str | Path) -> None:
        root = Path(run_dir)
        self._log_dir = root / "logs"
        self._artifacts_dir = root / "artifacts"
        self._started_at = _now()
        self._finished = False
        self._num_samples = 0
        self._evaluate_success = 0
        self._evaluate_failed = 0
        self._error_count = 0
        self._llm_call_count = 0
        self._candidate_count = 0
        self._edge_count = 0
        self._decision_count = 0
        self._best_score: float | None = None
        self._best_sample_order: int | None = None

        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._progress_path = self._log_dir / "progress.log"
        self._errors_path = self._log_dir / "errors.jsonl"
        self._summary_path = self._log_dir / "summary.json"
        self._candidates_path = self._artifacts_dir / "candidates.jsonl"
        self._edges_path = self._artifacts_dir / "edges.jsonl"
        self._llm_calls_path = self._artifacts_dir / "llm_calls.jsonl"
        self._decisions_path = self._artifacts_dir / "decisions.jsonl"
        self._progress_file = self._progress_path.open("a", encoding="utf-8", buffering=1)
        self._restore_existing_artifacts()

    def log_progress(self, message: str) -> None:
        self._progress_file.write(f"{_now().isoformat(timespec='seconds')} {message}\n")

    def record_error(self, stage: str, exc: BaseException | None = None, **payload: Any) -> None:
        record: dict[str, Any] = {
            "stage": stage,
            "ts": _now().isoformat(timespec="seconds"),
        }
        if exc is not None:
            record["error_type"] = type(exc).__name__
            record["error"] = _truncate(str(exc), _ERROR_TRUNCATE)
            record["traceback"] = _truncate(
                "".join(traceback.format_exception(exc)),
                4000,
            )
        for key, value in payload.items():
            if key in {"prompt", "messages", "response", "traceback"}:
                continue
            record[key] = value
        self._append_jsonl(self._errors_path, record)
        self._error_count += 1

    def write_summary(self, **payload: Any) -> None:
        if self._finished:
            return
        finished_at = _now()
        summary = {
            "status": payload.pop("status", "finished"),
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": (finished_at - self._started_at).total_seconds(),
            "num_samples": self._num_samples,
            "evaluate_success": self._evaluate_success,
            "evaluate_failed": self._evaluate_failed,
            "best_sample_order": self._best_sample_order,
            "best_score": self._best_score,
            "error_count": self._error_count,
            "llm_call_count": self._llm_call_count,
            "candidate_count": self._candidate_count,
            "edge_count": self._edge_count,
            "decision_count": self._decision_count,
        }
        summary.update(payload)
        self._summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )
        self._finished = True

    def finish(self) -> None:
        if not self._finished:
            self.write_summary(status="finished")
        self._progress_file.close()

    def record_candidate(self, **payload: Any) -> None:
        sample_order = payload.get("sample_order")
        score = payload.get("score")
        if payload.get("status") is None:
            payload["status"] = "ok" if score is not None else "eval_failed"
        self._append_jsonl(self._candidates_path, payload)
        self._candidate_count += 1
        if isinstance(sample_order, int):
            self._num_samples = max(self._num_samples, sample_order)
        if score is None:
            self._evaluate_failed += 1
        else:
            self._evaluate_success += 1
            if self._best_score is None or float(score) > float(self._best_score):
                self._best_score = float(score)
                self._best_sample_order = sample_order
            self.log_progress(
                f"sample={sample_order} score={score} best={self._best_score}"
            )

    def record_edge(self, **payload: Any) -> None:
        self._append_jsonl(self._edges_path, payload)
        self._edge_count += 1

    def record_llm_call(self, **payload: Any) -> None:
        self._append_jsonl(self._llm_calls_path, self._llm_meta(payload))
        self._llm_call_count += 1

    def record_decision(self, event: str, **payload: Any) -> None:
        self._append_jsonl(self._decisions_path, {"event": event, **payload})
        self._decision_count += 1

    def sync_after_resume(
        self,
        *,
        total_samples: int,
        best_score: Any = None,
        best_sample_order: int | None = None,
    ) -> None:
        self._restore_existing_artifacts()
        self._num_samples = max(self._num_samples, int(total_samples))
        if best_score is not None:
            self._best_score = float(best_score)
            self._best_sample_order = best_sample_order

    def _llm_meta(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        status = payload.get("status", "ok")
        record: dict[str, Any] = {
            "stage": payload.get("stage"),
            "operator": payload.get("operator"),
            "sample_order": payload.get("sample_order"),
            "iteration": payload.get("iteration"),
            "seq": payload.get("seq"),
            "prompt_tokens": payload.get("prompt_tokens"),
            "response_tokens": payload.get("response_tokens"),
            "sample_time": payload.get("sample_time"),
            "token_count_mode": payload.get("token_count_mode"),
            "status": status,
        }
        if "n_actions" in payload:
            record["n_actions"] = payload["n_actions"]
        elif payload.get("parsed_actions") is not None:
            record["n_actions"] = len(payload["parsed_actions"])
        if "program_parse_success" in payload:
            record["program_parse_success"] = payload["program_parse_success"]
        if payload.get("store_prompt") and payload.get("prompt") is not None:
            record["prompt"] = str(payload["prompt"])
        if status not in (None, "ok"):
            response = payload.get("response")
            if response is not None:
                record["response"] = _truncate(str(response), _RESPONSE_TRUNCATE)
            if payload.get("parse_errors"):
                record["parse_errors"] = payload["parse_errors"]
            for key in (
                "failure_kind",
                "error_type",
                "error",
                "counts_budget",
                "consecutive_failures",
            ):
                if payload.get(key) is not None:
                    value = payload[key]
                    record[key] = (
                        _truncate(str(value), _ERROR_TRUNCATE)
                        if key == "error"
                        else value
                    )
        return {key: value for key, value in record.items() if value is not None}

    def _restore_existing_artifacts(self) -> None:
        if self._summary_path.is_file():
            try:
                summary = json.loads(self._summary_path.read_text(encoding="utf-8"))
                started_at = summary.get("started_at")
                if isinstance(started_at, str):
                    self._started_at = datetime.fromisoformat(started_at)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass

        candidate_rows = self._read_jsonl(self._candidates_path)
        edge_rows = self._read_jsonl(self._edges_path)
        llm_rows = self._read_jsonl(self._llm_calls_path)
        decision_rows = self._read_jsonl(self._decisions_path)
        error_rows = self._read_jsonl(self._errors_path)
        self._candidate_count = len(candidate_rows)
        self._edge_count = len(edge_rows)
        self._llm_call_count = len(llm_rows)
        self._decision_count = len(decision_rows)
        self._error_count = len(error_rows)

        successful = [
            row
            for row in candidate_rows
            if row.get("status") == "ok" or row.get("score") is not None
        ]
        self._evaluate_success = len(successful)
        self._evaluate_failed = max(0, len(candidate_rows) - len(successful))
        sample_orders = [
            int(row["sample_order"])
            for row in candidate_rows
            if isinstance(row.get("sample_order"), int)
        ]
        if sample_orders:
            self._num_samples = max(self._num_samples, max(sample_orders))

        scores = []
        for row in successful:
            try:
                scores.append((float(row["score"]), row.get("sample_order")))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if scores and self._best_score is None:
            self._best_score, sample_order = max(scores, key=lambda item: item[0])
            self._best_sample_order = (
                sample_order if isinstance(sample_order, int) else None
            )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return rows
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    @staticmethod
    def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, default=_json_default)
            handle.write("\n")


__all__ = ["RunArtifacts", "_RESPONSE_TRUNCATE"]
