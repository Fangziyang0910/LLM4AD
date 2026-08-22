"""Run outputs for TraceAAD V9.16."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .selection import Decision

EVALUATIONS_CSV_HEADER = [
    "eval_count", "parent_id", "child_id", "intent", "mode", "entry_id",
    "landing_id", "landing_step", "status", "fitness", "error", "n_stag",
    "p_explore", "beta", "ess", "n_valid", "parent_q", "attempt",
    "attempt_kind", "elapsed_seconds", "preflight_error", "candidate_hash",
    "error_type",
]


class RunArtifacts:
    def __init__(self, run_dir: str | Path, *, console_output: bool = True) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        path = self._run_dir / "evaluations.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        exists = path.exists() and path.stat().st_size > 0
        self._evaluations_file: TextIO = path.open("a", encoding="utf-8", newline="")
        self._evaluations = csv.writer(self._evaluations_file)
        if not exists:
            self._evaluations.writerow(EVALUATIONS_CSV_HEADER)
            self._evaluations_file.flush()
        self._landing_events_file: TextIO = (
            self._run_dir / "landing_events.jsonl"
        ).open("a", encoding="utf-8")

    def record_evaluation(
        self, *, eval_count: int, parent_id: int, child_id: int | None,
        intent: str | None, mode: str, entry_id: int | None,
        landing_id: int | None, landing_step: int | None, status: str,
        fitness: float | None, error: str | None, decision: Decision | None,
        n_stag: int, attempt: int = 1, attempt_kind: str = "initial",
        elapsed_seconds: float | None = None, preflight_error: str | None = None,
        candidate_hash: str | None = None, error_type: str | None = None,
    ) -> None:
        self._evaluations.writerow([
            eval_count, parent_id, "" if child_id is None else child_id,
            intent or "", mode, "" if entry_id is None else entry_id,
            "" if landing_id is None else landing_id,
            "" if landing_step is None else landing_step, status,
            "" if fitness is None else fitness, error or "", n_stag,
            "" if decision is None else f"{decision.p_explore:.6g}",
            "" if decision is None else f"{decision.beta:.6g}",
            "" if decision is None else f"{decision.ess:.6g}",
            "" if decision is None else decision.n_valid,
            "" if decision is None else f"{decision.parent_q:.6g}", attempt,
            attempt_kind, "" if elapsed_seconds is None else f"{elapsed_seconds:.6f}",
            preflight_error or "", candidate_hash or "", error_type or "",
        ])
        self._evaluations_file.flush()
        if self._console_output:
            result = status if fitness is None else f"{status}, fitness={fitness:.6g}"
            extra = "" if decision is None else f", intent={decision.intent.value}"
            print(f"[Eval {eval_count:03d}] {mode}: {result}{extra}")

    def record_landing_event(self, event: str, **payload: object) -> None:
        self._landing_events_file.write(
            json.dumps({"event": event, **payload}, ensure_ascii=False) + "\n"
        )
        self._landing_events_file.flush()

    def record_best(self, *, code: str, fitness: float) -> None:
        self._run_dir.joinpath("best_program.py").write_text(
            f"# Fitness: {fitness:.6g}\n\n{code.rstrip()}\n", encoding="utf-8"
        )

    def write_summary(self, **payload: object) -> None:
        path = self._run_dir / "logs" / "summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def finish(self) -> None:
        self._evaluations_file.close()
        self._landing_events_file.close()


__all__ = ["EVALUATIONS_CSV_HEADER", "RunArtifacts"]
