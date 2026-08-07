"""Shared TraceAAD run I/O: monitor logs, analysis artifacts, summary.

Three channels under a run directory (checkpoints stay in each version's
``checkpoint.py``):

- ``logs/`` — progress text, errors, end summary (monitoring)
- ``artifacts/`` — candidates, edges, llm call metadata, decisions (analysis)

This module is not part of the search mechanism. Versions only report raw
facts; derived metrics (LRR, PCD, novelty, entropy, deltas) are offline.
"""

from __future__ import annotations

import json
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pytz

_RESPONSE_TRUNCATE = 2000
_ERROR_TRUNCATE = 1000


def _shanghai_now() -> datetime:
    return datetime.now(pytz.timezone("Asia/Shanghai"))


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


class TraceAADArtifacts:
    """Minimal I/O sink shared by TraceAAD versions."""

    def __init__(
        self,
        *,
        run_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
        # Accepted for drop-in replacement of ProfilerBase construction sites.
        log_style: str | None = None,
        create_random_path: bool = False,
        **_ignored: Any,
    ) -> None:
        del log_style, create_random_path, _ignored
        if run_dir is not None:
            root = Path(run_dir)
            self._log_dir = root / "logs"
            self._artifacts_dir = root / "artifacts"
        elif log_dir is not None:
            self._log_dir = Path(log_dir)
            parent = self._log_dir.parent
            if self._log_dir.name == "logs":
                self._artifacts_dir = (
                    Path(artifacts_dir)
                    if artifacts_dir is not None
                    else parent / "artifacts"
                )
            else:
                self._artifacts_dir = (
                    Path(artifacts_dir)
                    if artifacts_dir is not None
                    else self._log_dir / "artifacts"
                )
        else:
            self._log_dir = None
            self._artifacts_dir = None if artifacts_dir is None else Path(artifacts_dir)

        self._lock = threading.Lock()
        self._started_at = _shanghai_now()
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
        self._progress_file = None

        if self._log_dir is not None:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._progress_path = self._log_dir / "progress.log"
            self._errors_path = self._log_dir / "errors.jsonl"
            self._summary_path = self._log_dir / "summary.json"
            self._progress_file = open(
                self._progress_path, "a", encoding="utf-8", buffering=1
            )
        else:
            self._progress_path = None
            self._errors_path = None
            self._summary_path = None

        if self._artifacts_dir is not None:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
            self._candidates_path = self._artifacts_dir / "candidates.jsonl"
            self._edges_path = self._artifacts_dir / "edges.jsonl"
            self._llm_calls_path = self._artifacts_dir / "llm_calls.jsonl"
            self._decisions_path = self._artifacts_dir / "decisions.jsonl"
        else:
            self._candidates_path = None
            self._edges_path = None
            self._llm_calls_path = None
            self._decisions_path = None

        self._restore_existing_artifacts()

    # --- monitor ---------------------------------------------------------

    def log_progress(self, message: str) -> None:
        line = f"{_shanghai_now().isoformat(timespec='seconds')} {message}"
        if self._progress_file is not None:
            self._progress_file.write(line + "\n")
        else:
            print(line)

    def record_error(
        self,
        stage: str,
        exc: BaseException | None = None,
        **payload: Any,
    ) -> None:
        record: dict[str, Any] = {
            "stage": stage,
            "ts": _shanghai_now().isoformat(timespec="seconds"),
        }
        if exc is not None:
            record["error_type"] = type(exc).__name__
            record["error"] = _truncate(str(exc), _ERROR_TRUNCATE)
            record["traceback"] = _truncate(
                "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
                4000,
            )
        for key, value in payload.items():
            if key in {"prompt", "messages", "response", "traceback"}:
                continue
            record[key] = value
        self._append_jsonl(self._errors_path, record)
        self._error_count += 1

    def write_summary(self, **payload: Any) -> None:
        with self._lock:
            if self._summary_path is None or self._finished:
                return
            finished_at = _shanghai_now()
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
            self._summary_path.parent.mkdir(parents=True, exist_ok=True)
            self._summary_path.write_text(
                json.dumps(
                    summary, indent=2, ensure_ascii=False, default=_json_default
                )
                + "\n",
                encoding="utf-8",
            )
            self._finished = True

    def finish(self) -> None:
        if not self._finished:
            self.write_summary(status="finished")
        if self._progress_file is not None:
            self._progress_file.close()
            self._progress_file = None

    # --- analysis raw data -----------------------------------------------

    def record_candidate(self, **payload: Any) -> None:
        sample_order = payload.get("sample_order")
        score = payload.get("score")
        status = payload.get("status")
        if status is None:
            status = "ok" if score is not None else "eval_failed"
            payload["status"] = status
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
        record = self._llm_meta(payload)
        self._append_jsonl(self._llm_calls_path, record)
        self._llm_call_count += 1

    def record_decision(self, event: str, **payload: Any) -> None:
        record = {"event": event, **payload}
        self._append_jsonl(self._decisions_path, record)
        self._decision_count += 1

    def sync_after_resume(self, *, total_samples: int, best_score: Any = None, best_sample_order: int | None = None) -> None:
        """Align counters after loading a search checkpoint."""
        self._restore_existing_artifacts()
        self._num_samples = max(self._num_samples, int(total_samples))
        if best_score is not None:
            self._best_score = float(best_score)
            self._best_sample_order = best_sample_order

    # --- duck-types used by shared failure helpers -----------------------

    def record_parameters(self, llm: Any, prob: Any, method: Any) -> None:
        del llm, prob, method

    def register_function(self, function: Any, program: str = "", *, resume_mode=False):
        del function, program, resume_mode

    def log_llm_call(self, **payload: Any) -> None:
        self.record_llm_call(**payload)

    def log_method_event(self, event: str | None = None, **payload: Any) -> None:
        name = event or payload.get("event")
        if name == "search_aborted":
            self.record_decision("search_aborted", **payload)
            self.log_progress(
                "search_aborted "
                f"reason={payload.get('reason')} "
                f"failures={payload.get('consecutive_failures')}"
            )

    def log_error(self, stage: str, exc: Exception | None = None, **payload: Any) -> None:
        self.record_error(stage, exc, **payload)

    def write_run_summary(self, **payload: Any) -> None:
        self.write_summary(**payload)

    def get_logger(self):
        return None

    # --- internals -------------------------------------------------------

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
        elif "parsed_actions" in payload and payload["parsed_actions"] is not None:
            record["n_actions"] = len(payload["parsed_actions"])
        if "program_parse_success" in payload:
            record["program_parse_success"] = payload["program_parse_success"]
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
        """Rebuild cumulative counters when a run directory is resumed."""
        summary_path = self._summary_path
        if summary_path is not None and summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
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
                score = float(row["score"])
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            scores.append((score, row.get("sample_order")))
        if scores and self._best_score is None:
            self._best_score, sample_order = max(scores, key=lambda item: item[0])
            self._best_sample_order = (
                sample_order if isinstance(sample_order, int) else None
            )

    @staticmethod
    def _read_jsonl(path: Path | None) -> list[dict[str, Any]]:
        if path is None or not path.is_file():
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

    def _append_jsonl(self, path: Path | None, payload: Mapping[str, Any]) -> None:
        if path is None:
            return
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, default=_json_default)
                handle.write("\n")


# Compatibility aliases.
TraceAADProfiler = TraceAADArtifacts
TraceAADV5Artifacts = TraceAADArtifacts

__all__ = [
    "TraceAADArtifacts",
    "TraceAADProfiler",
    "TraceAADV5Artifacts",
    "_RESPONSE_TRUNCATE",
]
