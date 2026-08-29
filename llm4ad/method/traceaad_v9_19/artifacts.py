"""Auditable run outputs for TraceAAD V9.19."""

from __future__ import annotations

import csv
import json
from pathlib import Path

EVALUATIONS_CSV_HEADER = [
    "slot",
    "parent_id",
    "child_id",
    "action",
    "mode",
    "outcome",
    "status",
    "fitness",
    "parent_fitness",
    "error",
    "p_explore",
    "t_response",
    "beta",
    "ess",
    "pool_size",
    "neighborhood_size",
    "attempt",
    "attempt_kind",
    "request_seed",
    "elapsed_seconds",
    "error_type",
    "reference_id",
]


class RunArtifacts:
    def __init__(self, run_dir: str | Path, *, console_output: bool = True) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        path = self._run_dir / "evaluations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        self._evaluations_file = path.open("a", encoding="utf-8", newline="")
        self._evaluations = csv.writer(self._evaluations_file)
        if not exists:
            self._evaluations.writerow(EVALUATIONS_CSV_HEADER)
            self._evaluations_file.flush()
        self._events_file = (self._run_dir / "mechanism_events.jsonl").open(
            "a", encoding="utf-8"
        )
        self._decisions_file = (self._run_dir / "decisions.jsonl").open(
            "a", encoding="utf-8"
        )
        self._best_history_file = (self._run_dir / "best_history.jsonl").open(
            "a", encoding="utf-8"
        )

    def record_evaluation(self, row: list[object]) -> None:
        self._evaluations.writerow(row)
        self._evaluations_file.flush()

    def record_event(self, event: str, **payload: object) -> None:
        self._events_file.write(
            json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"
        )
        self._events_file.flush()

    def record_decision(self, payload: dict[str, object]) -> None:
        self._decisions_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._decisions_file.flush()

    def record_best(
        self,
        *,
        code: str,
        fitness: float,
        slot: int,
        child_id: int,
        idea: str | None = None,
        action: str | None = None,
        novelty: float | None = None,
        behavior: str | None = None,
        t_response: float | None = None,
    ) -> None:
        self._run_dir.joinpath("best_program.py").write_text(
            f"# Fitness: {fitness:.6g}\n\n{code.rstrip()}\n", encoding="utf-8"
        )
        self._best_history_file.write(
            json.dumps(
                {
                    "slot": slot,
                    "fitness": fitness,
                    "child_id": child_id,
                    "idea": idea,
                    "action": action,
                    "novelty": novelty,
                    "behavior": behavior,
                    "t_response": t_response,
                    "program": code,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._best_history_file.flush()

    def write_summary(self, **payload: object) -> None:
        path = self._run_dir / "logs" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def finish(self) -> None:
        self._evaluations_file.close()
        self._events_file.close()
        self._decisions_file.close()
        self._best_history_file.close()


__all__ = ["EVALUATIONS_CSV_HEADER", "RunArtifacts"]
