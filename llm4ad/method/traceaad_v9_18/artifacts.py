"""Auditable run outputs for TraceAAD V9.18-R0."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .selection import Decision

EVALUATIONS_CSV_HEADER = [
    "eval_count", "parent_id", "child_id", "intent", "mode", "entry_id",
    "status", "fitness", "error", "n_stag", "p_explore", "beta", "ess",
    "n_valid", "parent_q", "sigma_q", "allocation_mode", "selected_score",
    "opportunity", "decision_index", "operator_draw", "request_seed",
    "prompt_hash", "prompt_chars", "facts_hash", "facts_omitted", "diagnosis",
    "attempt", "attempt_kind", "elapsed_seconds", "preflight_error",
    "candidate_hash", "duplicate", "error_type",
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
        self._events_file: TextIO = (
            self._run_dir / "mechanism_events.jsonl"
        ).open("a", encoding="utf-8")

    def record_evaluation(
        self,
        *,
        eval_count: int,
        parent_id: int,
        child_id: int | None,
        intent: str | None,
        mode: str,
        entry_id: int | None,
        status: str,
        fitness: float | None,
        error: str | None,
        decision: Decision | None,
        n_stag: int,
        request_seed: int | None = None,
        prompt_hash: str | None = None,
        prompt_chars: int | None = None,
        facts_hash: str | None = None,
        facts_omitted: bool = False,
        diagnosis: str | None = None,
        attempt: int = 1,
        attempt_kind: str = "initial",
        elapsed_seconds: float | None = None,
        preflight_error: str | None = None,
        candidate_hash: str | None = None,
        duplicate: bool = False,
        error_type: str | None = None,
    ) -> None:
        self._evaluations.writerow([
            eval_count,
            parent_id,
            "" if child_id is None else child_id,
            intent or "",
            mode,
            "" if entry_id is None else entry_id,
            status,
            "" if fitness is None else fitness,
            error or "",
            n_stag,
            "" if decision is None else f"{decision.p_explore:.6g}",
            "" if decision is None else f"{decision.beta:.6g}",
            "" if decision is None else f"{decision.ess:.6g}",
            "" if decision is None else decision.n_valid,
            "" if decision is None else f"{decision.parent_q:.6g}",
            "" if decision is None else f"{decision.sigma_q:.6g}",
            "" if decision is None else decision.allocation_mode,
            "" if decision is None else f"{decision.selected_score:.6g}",
            "" if decision is None else f"{decision.opportunity:.6g}",
            "" if decision is None else decision.decision_index,
            "" if decision is None else f"{decision.operator_draw:.6g}",
            "" if request_seed is None else request_seed,
            prompt_hash or "",
            "" if prompt_chars is None else prompt_chars,
            facts_hash or "",
            int(facts_omitted),
            diagnosis or "",
            attempt,
            attempt_kind,
            "" if elapsed_seconds is None else f"{elapsed_seconds:.6f}",
            preflight_error or "",
            candidate_hash or "",
            int(duplicate),
            error_type or "",
        ])
        self._evaluations_file.flush()
        if self._console_output:
            result = status if fitness is None else f"{status}, fitness={fitness:.6g}"
            extra = "" if decision is None else f", intent={decision.intent.value}"
            print(f"[Eval {eval_count:03d}] {mode}: {result}{extra}")

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
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def finish(self) -> None:
        self._evaluations_file.close()
        self._events_file.close()


__all__ = ["EVALUATIONS_CSV_HEADER", "RunArtifacts"]
