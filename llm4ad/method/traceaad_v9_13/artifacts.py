"""Run artifacts for TraceAAD V9.13.

Outputs under the run directory:
- ``best_program.py``   — Best algorithm discovered so far.
- ``evaluations.csv``    — Every evaluation and candidate attempt.
- ``logs/summary.json``  — Overall summary of run statistics.
- ``logs/events.jsonl``  — Frontier-build prompt-audit decisions.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

EVALUATIONS_CSV_HEADER = [
    "eval_count",
    "sample_order",
    "stage",
    "iteration",
    "route_id",
    "anchor_id",
    "child_id",
    "program_id",
    "intent",
    "treatment",
    "kind",
    "outcome",
    "parent_fitness",
    "child_fitness",
    "dq",
    "is_new_best",
    "best_fitness",
    "status",
    "error",
]


def _now() -> datetime:
    return datetime.now().astimezone()


def _fit(val: float | None) -> str:
    return "N/A" if val is None else f"{val:.6g}"


def _val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


class RunArtifacts:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        console_output: bool = True,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        self._best_program_path = self._run_dir / "best_program.py"
        self._files: dict[str, TextIO] = {}
        self._writers: dict[str, Any] = {}

        eval_path = self._run_dir / "evaluations.csv"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_exists = eval_path.exists() and eval_path.stat().st_size > 0
        eval_file = eval_path.open("a", encoding="utf-8", newline="")
        self._files["evaluations"] = eval_file
        self._writers["evaluations"] = csv.writer(eval_file)
        if not eval_exists:
            self._writers["evaluations"].writerow(EVALUATIONS_CSV_HEADER)
            eval_file.flush()

        event_path = self._run_dir / "logs" / "events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self._files["events"] = event_path.open("a", encoding="utf-8")

    def record_candidate(
        self,
        *,
        order: int,
        stage: str,
        iteration: int | None,
        anchor_id: int | None,
        child_id: int | None,
        program_id: int | None,
        intent: str | None,
        treatment: str = "pp",
        kind: str,
        outcome: Any = None,
        status: str = "ok",
        parent_fitness: float | None = None,
        child_fitness: float | None = None,
        dq: float | None = None,
        error: str | None = None,
        eval_count: int = 0,
        route_id: int | None = None,
        best_fitness: float | None = None,
        is_new_best: bool = False,
        budget: int = 1000,
    ) -> None:
        """Record an evaluated candidate or search attempt to evaluations.csv."""
        outcome_str = (
            outcome.value
            if hasattr(outcome, "value")
            else (str(outcome) if outcome else "")
        )
        self._writers["evaluations"].writerow(
            [
                eval_count,
                order,
                stage,
                _val(iteration),
                _val(route_id),
                _val(anchor_id),
                _val(child_id),
                _val(program_id),
                intent or "",
                treatment,
                kind,
                outcome_str,
                _val(parent_fitness),
                _val(child_fitness),
                _val(dq),
                is_new_best,
                _val(best_fitness),
                status,
                error or "",
            ]
        )
        self._files["evaluations"].flush()
        if self._console_output:
            if status != "ok":
                result = f"{status} ({error or kind})"
            elif child_fitness is None:
                result = kind
            else:
                result = f"{kind}, fitness={child_fitness:.6g}"
            tag = "New Best" if is_new_best else "Best"
            print(
                f"[Eval {eval_count:03d}/{budget}] {intent or 'root'} {result} "
                f"| {tag}: {_fit(best_fitness)}"
            )

    def record_best(self, *, code: str, fitness: float) -> None:
        self._best_program_path.write_text(
            f"# Fitness: {fitness:.6g}\n\n{code.rstrip()}\n", encoding="utf-8"
        )

    def record_decision(self, event: str, **payload: Any) -> None:
        handle = self._files["events"]
        json.dump(
            {
                "event": event,
                "timestamp": _now().isoformat(timespec="milliseconds"),
                **payload,
            },
            handle,
            ensure_ascii=False,
        )
        handle.write("\n")
        handle.flush()

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
