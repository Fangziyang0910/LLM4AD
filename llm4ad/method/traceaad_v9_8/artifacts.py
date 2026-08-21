"""Streaming artifacts for TraceAAD V9.8."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

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

# Per-decision selection events record the selected hypothesis/anchor rows; the
# full score tables grow linearly with the search and are not embedded.
_SELECTION_TABLES = {
    "anchors": ("anchor_id", "selected_anchor_id", "selected_anchor"),
    "hypotheses": ("hypothesis_id", "selected_hypothesis_id", "selected_hypothesis"),
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _fit(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.12g}"
    return str(val)


def _distill_selection(selection: dict[str, Any]) -> dict[str, Any]:
    distilled = dict(selection)
    for table, (row_id_field, selected_id_field, distilled_field) in _SELECTION_TABLES.items():
        rows = distilled.pop(table, None)
        selected_id = distilled.get(selected_id_field)
        if isinstance(rows, list) and selected_id is not None:
            distilled[distilled_field] = next(
                (
                    row
                    for row in rows
                    if isinstance(row, dict) and row.get(row_id_field) == selected_id
                ),
                None,
            )
    return distilled


def _with_distilled_selection(event: dict[str, Any]) -> dict[str, Any]:
    selection = event.get("selection")
    if isinstance(selection, dict):
        event["selection"] = _distill_selection(selection)
    return event


class RunArtifacts:
    def __init__(self, run_dir: str | Path, *, console_output: bool = True) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        self._best_program_path = self._run_dir / "best_program.py"
        self._files: dict[str, TextIO] = {}
        self._writers: dict[str, Any] = {}
        self._event_ids: set[str] = set()
        self._evaluation_response_ids: set[str] = set()

        self._open_csv("evaluations", self._run_dir / "evaluations.csv", EVALUATIONS_CSV_HEADER)
        self._open_jsonl("events", self._run_dir / "logs" / "events.jsonl")
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
        self._load_column_ids(
            self._run_dir / "evaluations.csv", "response_id", self._evaluation_response_ids
        )
        events_path = self._run_dir / "logs" / "events.jsonl"
        if events_path.exists():
            with events_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    event_id = json.loads(line).get("event_id")
                    if event_id is not None:
                        self._event_ids.add(str(event_id))

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
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        if event_id is not None:
            self._event_ids.add(str(event_id))

    def record_request(self, *, response_id: str, **payload: Any) -> None:
        self._append_jsonl(
            "events",
            _with_distilled_selection(
                {
                    "event_id": f"request:{response_id}",
                    "event": "request_prepared",
                    "timestamp": _now().isoformat(),
                    "response_id": response_id,
                    **payload,
                }
            ),
        )

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

    def record_candidate(self, **row: Any) -> None:
        response_id = str(row["response_id"])
        outcome = row.get("outcome")
        outcome_text = outcome.value if hasattr(outcome, "value") else (outcome or "")
        self._append_jsonl(
            "events",
            _with_distilled_selection(
                {
                    "event_id": f"response:{response_id}",
                    "event": "response_finalized",
                    "timestamp": _now().isoformat(),
                    **row,
                    "outcome": outcome_text,
                }
            ),
        )

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
            self._print_progress(**row)

    def record_best(self, *, code: str, fitness: float) -> None:
        self._best_program_path.write_text(
            f"# Fitness: {fitness:.6g}\n\n{code.rstrip()}\n", encoding="utf-8"
        )

    def _print_progress(self, **row: Any) -> None:
        eval_count = int(row.get("eval_count", 0))
        budget = int(row.get("budget", 1000))
        intent = row.get("intent") or "root"
        kind = str(row.get("kind", ""))
        if row.get("status") != "ok":
            result = f"{row.get('status')} ({row.get('error') or kind})"
        elif row.get("child_fitness") is None:
            result = kind
        else:
            result = f"{kind}, fitness={row['child_fitness']:.6g}"
        tag = "New Best" if row.get("is_new_best") else "Best"
        print(
            f"[Eval {eval_count:03d}/{budget}] {intent} {result} "
            f"| {tag}: {_fit(row.get('best_fitness'))}"
        )

    def write_summary(self, **payload: Any) -> None:
        path = self._run_dir / "logs" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def finish(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._writers.clear()


__all__ = ["RunArtifacts"]
