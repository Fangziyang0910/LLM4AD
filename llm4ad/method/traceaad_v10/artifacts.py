"""Auditable files written by TraceAAD V10."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EVALUATIONS_HEADER = [
    "slot",
    "round",
    "operator",
    "idea",
    "start_id",
    "reference_id",
    "node_id",
    "outcome",
    "fitness",
    "start_fitness",
    "q_origin",
    "created_thread",
    "attempt",
    "attempt_kind",
    "elapsed_seconds",
    "error_type",
    "error",
]


class RunArtifacts:
    """Append-only event files plus the current best program."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        evaluations = self.run_dir / "evaluations.csv"
        exists = evaluations.is_file() and evaluations.stat().st_size > 0
        self._evaluations_file = evaluations.open("a", encoding="utf-8", newline="")
        self._evaluations = csv.writer(self._evaluations_file)
        if not exists:
            self._evaluations.writerow(EVALUATIONS_HEADER)
            self._evaluations_file.flush()
        self._events_file = (self.run_dir / "mechanism_events.jsonl").open("a", encoding="utf-8")
        self._decisions_file = (self.run_dir / "decisions.jsonl").open("a", encoding="utf-8")
        self._threads_file = (self.run_dir / "threads.jsonl").open("a", encoding="utf-8")
        self._memory_file = (self.run_dir / "global_memory.jsonl").open("a", encoding="utf-8")
        self._best_file = (self.run_dir / "best_history.jsonl").open("a", encoding="utf-8")
        self._closed = False

    def record_evaluation(self, row: list[Any]) -> None:
        self._evaluations.writerow(row)
        self._evaluations_file.flush()

    def record_event(self, event: str, **payload: Any) -> None:
        self._events_file.write(json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n")
        self._events_file.flush()

    def record_decision(self, payload: dict[str, Any]) -> None:
        self._decisions_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._decisions_file.flush()

    def record_thread(self, payload: dict[str, Any]) -> None:
        self._threads_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._threads_file.flush()

    def record_memory(self, payload: dict[str, Any]) -> None:
        self._memory_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._memory_file.flush()

    def record_best(
        self,
        *,
        code: str,
        fitness: float,
        slot: int,
        node_id: int,
        idea: str | None,
        thread_id: int | None,
    ) -> None:
        (self.run_dir / "best_program.py").write_text(
            f"# Fitness: {fitness:.8g}\n\n{code.rstrip()}\n", encoding="utf-8"
        )
        self._best_file.write(
            json.dumps(
                {
                    "slot": slot,
                    "fitness": fitness,
                    "node_id": node_id,
                    "idea": idea,
                    "thread_id": thread_id,
                    "program": code,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._best_file.flush()

    def write_summary(self, **payload: Any) -> None:
        path = self.run_dir / "logs" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        for handle in (
            self._evaluations_file,
            self._events_file,
            self._decisions_file,
            self._threads_file,
            self._memory_file,
            self._best_file,
        ):
            handle.close()


__all__ = ["EVALUATIONS_HEADER", "RunArtifacts"]
