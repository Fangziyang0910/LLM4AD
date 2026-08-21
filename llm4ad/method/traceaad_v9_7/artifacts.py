"""Run outputs for TraceAAD V9.7."""

from __future__ import annotations

import csv
import json
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


def _val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def _fit(val: float | None) -> str:
    return "N/A" if val is None else f"{val:.6g}"


class RunArtifacts:
    def __init__(
        self,
        run_dir: str | Path,
        *,
        console_output: bool = True,
    ) -> None:
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

    def record_candidate(
        self,
        *,
        order: int,
        stage: str,
        iteration: int | None,
        route_id: int | None,
        anchor_id: int | None,
        child_id: int | None,
        program_id: int | None,
        intent: str | None,
        kind: str,
        outcome: Any = None,
        parent_fitness: float | None = None,
        child_fitness: float | None = None,
        dq: float | None = None,
        is_new_best: bool = False,
        best_fitness: float | None = None,
        status: str = "ok",
        error: str | None = None,
        eval_count: int = 0,
        budget: int = 1000,
    ) -> None:
        """Record one candidate attempt to evaluations.csv and log progress."""
        outcome_str = (
            outcome.value if hasattr(outcome, "value") else (str(outcome) if outcome else "")
        )
        self._evaluations.writerow(
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
        self._evaluations_file.flush()
        if self._console_output:
            if status != "ok":
                result = f"{status} ({error or kind})"
            elif child_fitness is None:
                result = kind
            else:
                result = f"{kind}, fitness={child_fitness:.6g}"
            tag = "New Best" if is_new_best else "Best"
            print(f"[Eval {eval_count:03d}/{budget}] {result} | {tag}: {_fit(best_fitness)}")

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


__all__ = ["RunArtifacts"]
