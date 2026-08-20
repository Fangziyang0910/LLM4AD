"""Run outputs for TraceAAD V9.14."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TextIO

EVALUATIONS_CSV_HEADER = [
    "eval_count",
    "parent_id",
    "child_id",
    "intent",
    "status",
    "fitness",
    "error",
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

    def record_evaluation(
        self,
        *,
        eval_count: int,
        parent_id: int,
        child_id: int | None,
        intent: str | None,
        status: str,
        fitness: float | None,
        error: str | None,
    ) -> None:
        self._evaluations.writerow(
            [
                eval_count,
                parent_id,
                "" if child_id is None else child_id,
                intent or "",
                status,
                "" if fitness is None else fitness,
                error or "",
            ]
        )
        self._evaluations_file.flush()
        if self._console_output:
            result = status if fitness is None else f"{status}, fitness={fitness:.6g}"
            print(f"[Eval {eval_count:03d}] {result}")

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
