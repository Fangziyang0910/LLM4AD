"""Process artifacts for TraceAAD V9.17."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

EVALUATIONS_CSV_HEADER = [
    "eval_count",
    "llm_call_count",
    "evaluator_call_count",
    "parent_id",
    "child_id",
    "intent",
    "mode",
    "phase",
    "hypothesis_id",
    "cycle",
    "sweep",
    "block_id",
    "block_kind",
    "block_step",
    "status",
    "fitness",
    "error",
    "attempt",
    "attempt_kind",
    "elapsed_seconds",
    "preflight_error",
    "candidate_hash",
    "error_type",
    "s_r",
    "active_ids",
    "competition_line",
]


class RunArtifacts:
    def __init__(self, run_dir: str | Path, *, console_output: bool = True) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        path = self._run_dir / "evaluations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        self._evaluations_file: TextIO = path.open("a", encoding="utf-8", newline="")
        self._evaluations = csv.DictWriter(
            self._evaluations_file, fieldnames=EVALUATIONS_CSV_HEADER
        )
        if not exists:
            self._evaluations.writeheader()
            self._evaluations_file.flush()
        self._events_file: TextIO = (
            self._run_dir / "mechanism_events.jsonl"
        ).open("a", encoding="utf-8")

    def record_evaluation(self, **payload: object) -> None:
        row = {key: payload.get(key, "") for key in EVALUATIONS_CSV_HEADER}
        row["active_ids"] = json.dumps(payload.get("active_ids", []))
        self._evaluations.writerow(row)
        self._evaluations_file.flush()
        if self._console_output:
            fitness = payload.get("fitness")
            result = payload.get("status")
            if fitness not in {None, ""}:
                result = f"{result}, fitness={float(fitness):.6g}"
            print(
                f"[Slot {int(payload['eval_count']):03d}] "
                f"{payload.get('mode')}: {result}",
                flush=True,
            )

    def record_event(self, event: str, **payload: object) -> None:
        self._events_file.write(
            json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"
        )
        self._events_file.flush()

    def record_best(self, *, code: str, fitness: float) -> None:
        self._run_dir.joinpath("best_program.py").write_text(
            f"# Fitness: {fitness:.6g}\n\n{code.rstrip()}\n", encoding="utf-8"
        )

    def write_summary(self, **payload: object) -> None:
        path = self._run_dir / "logs" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def finish(self) -> None:
        self._evaluations_file.close()
        self._events_file.close()


__all__ = ["EVALUATIONS_CSV_HEADER", "RunArtifacts"]
