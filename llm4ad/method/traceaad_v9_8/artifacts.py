"""Auditable streaming artifacts for TraceAAD V9.8."""

from __future__ import annotations

import csv
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

_ERROR_TRUNCATE = 1000
_TRACEBACK_TRUNCATE = 4000

EVALUATIONS_CSV_HEADER = [
    "eval_count",
    "response_order",
    "response_id",
    "stage",
    "iteration",
    "operator",
    "source_hypothesis_id",
    "child_hypothesis_id",
    "anchor_id",
    "child_id",
    "program_id",
    "kind",
    "outcome",
    "parent_fitness",
    "child_fitness",
    "parent_q",
    "child_q",
    "dq",
    "realized_gain",
    "is_new_best",
    "best_fitness",
    "status",
    "error",
]

BEST_CURVE_CSV_HEADER = [
    "eval_count",
    "response_order",
    "response_id",
    "iteration",
    "best_fitness",
    "best_q",
    "program_id",
    "timestamp",
]

# Full table snapshots from selection are written into population files
# every N evaluations instead of embedding the tables in every event (see
# _distill_selection).
POPULATION_SNAPSHOT_EVERY = 50

# selection keys holding per-row tables that grow with the search, mapped to
# (row id field, selection id field, distilled event field, snapshot file).
_SELECTION_TABLES = {
    "anchors": (
        "anchor_id",
        "selected_anchor_id",
        "selected_anchor",
        "population.csv",
    ),
    "hypotheses": (
        "hypothesis_id",
        "selected_hypothesis_id",
        "selected_hypothesis",
        "population_hypotheses.csv",
    ),
}

# response_finalized fields stored elsewhere: programs/{program_id}.py keeps
# the code, llm_calls.jsonl keeps the raw response, and the diff is derivable
# from the two program files.
_EVENT_REDUNDANT_FIELDS = ("program", "raw_response", "diff")


def _now() -> datetime:
    return datetime.now().astimezone()


def _json_default(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _format_fitness(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.12g}"
    return str(val)


def _distill_selection(selection: dict[str, Any]) -> dict[str, Any]:
    """Per-decision selection context without the growing row tables.

    The tables grow linearly with the search, so embedding them in every
    finalized event costs O(n^2) bytes per run; their full evolution is
    recorded in population.csv and population_hypotheses.csv instead.
    """
    distilled = dict(selection)
    for table, (row_id_field, selected_id_field, distilled_event_field, _) in _SELECTION_TABLES.items():
        rows = distilled.pop(table, None)
        selected_id = distilled.get(selected_id_field)
        if isinstance(rows, list) and selected_id is not None:
            distilled[distilled_event_field] = next(
                (row for row in rows if isinstance(row, dict) and row.get(row_id_field) == selected_id),
                None,
            )
    return distilled


class RunArtifacts:
    def __init__(self, run_dir: str | Path, *, console_output: bool = True) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        self._started_at = _now()
        self._summary_path = self._run_dir / "logs" / "summary.json"
        self._best_program_path = self._run_dir / "best_program.py"
        self._files: dict[str, TextIO] = {}
        self._writers: dict[str, Any] = {}
        self._population_writers: dict[str, Any] = {}
        self._event_ids: set[str] = set()
        self._responses: dict[str, str] = {}
        self._evaluation_response_ids: set[str] = set()
        self._best_response_ids: set[str] = set()

        self._open_csv("evaluations", self._run_dir / "evaluations.csv", EVALUATIONS_CSV_HEADER)
        self._open_csv("best_curve", self._run_dir / "best_curve.csv", BEST_CURVE_CSV_HEADER)
        self._open_jsonl("events", self._run_dir / "logs" / "events.jsonl")
        self._open_jsonl("llm_calls", self._run_dir / "logs" / "llm_calls.jsonl")
        self._open_jsonl("errors", self._run_dir / "logs" / "errors.jsonl")
        self._load_existing_ids()

    def _open_csv(self, name: str, path: Path, header: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        handle = path.open("a", encoding="utf-8", newline="")
        self._files[name] = handle
        self._writers[name] = csv.writer(handle)
        if not exists:
            self._writers[name].writerow(header)
            handle.flush()

    def _open_jsonl(self, name: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._files[name] = path.open("a", encoding="utf-8")

    def _load_existing_ids(self) -> None:
        self._load_column_ids(self._run_dir / "evaluations.csv", "response_id", self._evaluation_response_ids)
        self._load_column_ids(self._run_dir / "best_curve.csv", "response_id", self._best_response_ids)
        events_path = self._run_dir / "logs" / "events.jsonl"
        if events_path.exists():
            with events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    event_id = payload.get("event_id")
                    if event_id is not None:
                        self._event_ids.add(str(event_id))
        llm_calls_path = self._run_dir / "logs" / "llm_calls.jsonl"
        if llm_calls_path.exists():
            with llm_calls_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    response_id = payload.get("response_id")
                    if response_id and payload.get("status") == "ok" and payload.get("raw_response"):
                        self._responses[str(response_id)] = str(payload["raw_response"])

    def _load_column_ids(self, path: Path, field: str, target: set[str]) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get(field):
                    target.add(row[field])

    def _append_jsonl(self, name: str, payload: dict[str, Any]) -> None:
        event_id = payload.get("event_id")
        if event_id is not None and str(event_id) in self._event_ids:
            return
        handle = self._files[name]
        json.dump(payload, handle, ensure_ascii=False, default=_json_default)
        handle.write("\n")
        handle.flush()
        if event_id is not None:
            self._event_ids.add(str(event_id))

    def record_request(self, *, response_id: str, **payload: Any) -> None:
        self._append_jsonl(
            "events",
            {
                "event_id": f"request:{response_id}",
                "event": "request_prepared",
                "timestamp": _now().isoformat(),
                "response_id": response_id,
                **payload,
            },
        )

    def record_decision(
        self, event: str, *, event_key: str | None = None, **payload: Any
    ) -> None:
        response_id = payload.get("response_id", "unknown")
        suffix = response_id if event_key is None else f"{response_id}:{event_key}"
        self._append_jsonl(
            "events",
            {
                "event_id": f"decision:{event}:{suffix}",
                "event": event,
                "timestamp": _now().isoformat(),
                **payload,
            },
        )

    def record_llm_call(self, **row: Any) -> None:
        response_id = str(row.get("response_id", "unknown"))
        transport_attempt = int(row.get("transport_attempt", 1))
        status = str(row.get("status", "ok"))
        self._append_jsonl(
            "llm_calls",
            {
                "event_id": f"llm:{response_id}:{status}:{transport_attempt}",
                "timestamp": _now().isoformat(),
                **row,
            },
        )
        if status == "ok" and row.get("raw_response") is not None:
            self._responses[response_id] = str(row["raw_response"])
        if status == "transport" and self._console_output:
            print(
                f"[Warning] {response_id} transport retry #{transport_attempt}: "
                f"{row.get('error', 'unknown error')}"
            )

    def recovered_response(self, response_id: str) -> str | None:
        """Return a durably logged response for idempotent checkpoint recovery."""
        return self._responses.get(response_id)

    def record_program(
        self, *, program_id: int, code: str, fitness: float, q: float, order: int
    ) -> None:
        path = self._run_dir / "programs" / f"{program_id}.py"
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Program ID: {program_id}\n"
            f"# Fitness: {fitness:.12g} | Directed quality: {q:.12g}\n"
            f"# First response order: {order}\n\n"
        )
        path.write_text(header + code.rstrip() + "\n", encoding="utf-8")

    def _snapshot_population(self, row: dict[str, Any]) -> None:
        selection = row.get("selection")
        if not isinstance(selection, dict):
            return
        eval_count = row.get("eval_count")
        if not isinstance(eval_count, int):
            return
        if eval_count <= 0 or eval_count % POPULATION_SNAPSHOT_EVERY:
            return
        stamp = _now().isoformat()
        for table, (_, _, _, filename) in _SELECTION_TABLES.items():
            rows = selection.get(table)
            if not isinstance(rows, list) or not rows:
                continue
            keys = sorted(
                {key for row_ in rows if isinstance(row_, dict) for key in row_}
            )
            writer = self._population_writers.get(table)
            if writer is None:
                path = self._run_dir / filename
                exists = path.exists() and path.stat().st_size > 0
                handle = path.open("a", encoding="utf-8", newline="")
                writer = csv.writer(handle)
                if not exists:
                    writer.writerow(["eval_count", "timestamp", *keys])
                    handle.flush()
                self._files[f"population:{table}"] = handle
                self._population_writers[table] = writer
            for entry in rows:
                if isinstance(entry, dict):
                    writer.writerow(
                        [eval_count, stamp] + [entry.get(key, "") for key in keys]
                    )
            self._files[f"population:{table}"].flush()

    def record_candidate(self, **row: Any) -> None:
        response_id = str(row["response_id"])
        outcome = row.get("outcome")
        outcome_text = outcome.value if hasattr(outcome, "value") else (outcome or "")
        event_payload = {
            "event_id": f"response:{response_id}",
            "event": "response_finalized",
            "timestamp": _now().isoformat(),
            **row,
            "outcome": outcome_text,
        }
        for field in _EVENT_REDUNDANT_FIELDS:
            event_payload.pop(field, None)
        selection = row.get("selection")
        if isinstance(selection, dict):
            event_payload["selection"] = _distill_selection(selection)
        self._append_jsonl("events", event_payload)
        self._snapshot_population(row)

        if row.get("evaluator_called") and response_id not in self._evaluation_response_ids:
            self._writers["evaluations"].writerow(
                [
                    row.get("eval_count", 0),
                    row.get("order", 0),
                    response_id,
                    row.get("stage", ""),
                    _val(row.get("iteration")),
                    row.get("intent") or "",
                    _val(row.get("source_hypothesis_id")),
                    _val(row.get("child_hypothesis_id")),
                    _val(row.get("anchor_id")),
                    _val(row.get("child_id")),
                    _val(row.get("program_id")),
                    row.get("kind", ""),
                    outcome_text,
                    _val(row.get("parent_fitness")),
                    _val(row.get("child_fitness")),
                    _val(row.get("parent_q")),
                    _val(row.get("child_q")),
                    _val(row.get("dq")),
                    _val(row.get("realized_gain")),
                    bool(row.get("is_new_best")),
                    _val(row.get("best_fitness")),
                    row.get("status", "ok"),
                    row.get("error") or "",
                ]
            )
            self._files["evaluations"].flush()
            self._evaluation_response_ids.add(response_id)
        if self._console_output:
            self._print_progress(outcome_text=outcome_text, **row)

    def record_best(
        self,
        *,
        response_id: str,
        code: str,
        fitness: float,
        q: float,
        eval_count: int,
        iteration: int | None,
        order: int,
        program_id: int,
    ) -> None:
        timestamp = _now().strftime("%Y-%m-%d %H:%M:%S")
        if response_id not in self._best_response_ids:
            self._writers["best_curve"].writerow(
                [
                    eval_count,
                    order,
                    response_id,
                    _val(iteration),
                    f"{fitness:.12g}",
                    f"{q:.12g}",
                    program_id,
                    timestamp,
                ]
            )
            self._files["best_curve"].flush()
            self._best_response_ids.add(response_id)
        header = (
            "# =============================================================================\n"
            "# Best Program Discovered by TraceAAD V9.8\n"
            f"# Fitness: {fitness:.12g} | Directed quality: {q:.12g}\n"
            f"# Evaluator Count: {eval_count} | Response Order: {order}\n"
            f"# Program ID: {program_id} | Timestamp: {timestamp}\n"
            "# =============================================================================\n\n"
        )
        self._best_program_path.write_text(
            header + code.rstrip() + "\n", encoding="utf-8"
        )

    def _print_progress(self, *, outcome_text: str, **row: Any) -> None:
        eval_count = int(row.get("eval_count", 0))
        order = int(row.get("order", 0))
        budget = int(row.get("budget", 1000))
        stage = str(row.get("stage", ""))
        intent = row.get("intent") or "root"
        hypothesis = row.get("source_hypothesis_id")
        prefix = f"[Resp {order:04d} | Eval {eval_count:04d}/{budget}]"
        if row.get("status") != "ok":
            result = f"{row.get('status')}: {row.get('error') or row.get('kind')}"
        elif row.get("evaluator_called"):
            result = (
                f"{_format_fitness(row.get('parent_fitness'))} -> "
                f"{_format_fitness(row.get('child_fitness'))} "
                f"[{outcome_text.upper() or row.get('kind', '').upper()}]"
            )
        else:
            result = str(row.get("kind", ""))
        best = _format_fitness(row.get("best_fitness"))
        marker = " NEW BEST" if row.get("is_new_best") else ""
        print(
            f"{prefix} {stage}:{intent} H{hypothesis if hypothesis is not None else '-'} "
            f"| {result} | Best {best}{marker}"
        )

    def record_error(self, scope: str, exc: BaseException) -> None:
        payload = {
            "scope": scope,
            "timestamp": _now().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc)[:_ERROR_TRUNCATE],
            "traceback": "".join(traceback.format_exception(exc))[:_TRACEBACK_TRUNCATE],
        }
        handle = self._files["errors"]
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        if self._console_output:
            print(f"[Error in {scope}] {type(exc).__name__}: {exc}")

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
            json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default)
            + "\n",
            encoding="utf-8",
        )

    def finish(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._writers.clear()


__all__ = ["BEST_CURVE_CSV_HEADER", "EVALUATIONS_CSV_HEADER", "RunArtifacts"]
