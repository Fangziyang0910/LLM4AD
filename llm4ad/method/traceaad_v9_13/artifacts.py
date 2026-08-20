"""Run artifacts and progress logging for TraceAAD V9.13.

Outputs under the run directory:
- ``best_program.py``    — The complete Python code of the best algorithm discovered so far.
- ``evaluations.csv``    — Concise tabular log of every evaluation and candidate attempt.
- ``best_curve.csv``     — Convergence trajectory of best fitness over evaluation budget.
- ``logs/events.jsonl``  — Route/anchor selection, prompt build, and frontier decisions.
- ``logs/llm_calls.jsonl`` — Full prompts and responses for prompt-level audits.
- ``logs/summary.json``  — Overall summary of run statistics.
- ``logs/errors.jsonl``  — Errors encountered during runtime.
"""

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

BEST_CURVE_CSV_HEADER = [
    "eval_count",
    "sample_order",
    "iteration",
    "best_fitness",
    "program_id",
    "timestamp",
]


def _now() -> datetime:
    return datetime.now().astimezone()


def _format_fitness(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.6g}"


def _val(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


class RunArtifacts:
    """Convenient, human-readable results saving and live progress logging."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        console_output: bool = True,
    ) -> None:
        self._run_dir = Path(run_dir)
        self._console_output = console_output
        self._started_at = _now()
        self._summary_path = self._run_dir / "logs" / "summary.json"
        self._best_program_path = self._run_dir / "best_program.py"

        self._files: dict[str, TextIO] = {}
        self._writers: dict[str, Any] = {}

        # 1. evaluations.csv
        eval_path = self._run_dir / "evaluations.csv"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_exists = eval_path.exists() and eval_path.stat().st_size > 0
        eval_file = eval_path.open("a", encoding="utf-8", newline="")
        self._files["evaluations"] = eval_file
        self._writers["evaluations"] = csv.writer(eval_file)
        if not eval_exists:
            self._writers["evaluations"].writerow(EVALUATIONS_CSV_HEADER)
            eval_file.flush()

        # 2. best_curve.csv
        curve_path = self._run_dir / "best_curve.csv"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve_exists = curve_path.exists() and curve_path.stat().st_size > 0
        curve_file = curve_path.open("a", encoding="utf-8", newline="")
        self._files["best_curve"] = curve_file
        self._writers["best_curve"] = csv.writer(curve_file)
        if not curve_exists:
            self._writers["best_curve"].writerow(BEST_CURVE_CSV_HEADER)
            curve_file.flush()

        # 3. logs/events.jsonl
        event_path = self._run_dir / "logs" / "events.jsonl"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self._files["events"] = event_path.open("a", encoding="utf-8")

        # 4. logs/llm_calls.jsonl
        call_path = self._run_dir / "logs" / "llm_calls.jsonl"
        call_path.parent.mkdir(parents=True, exist_ok=True)
        self._files["llm_calls"] = call_path.open("a", encoding="utf-8")

        # 5. logs/errors.jsonl
        err_path = self._run_dir / "logs" / "errors.jsonl"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        self._files["errors"] = err_path.open("a", encoding="utf-8")

    def record_candidate(
        self,
        *,
        attempt_id: int,
        order: int,
        stage: str,
        iteration: int | None,
        anchor_id: int | None,
        child_id: int | None,
        program_id: int | None,
        intent: str | None,
        treatment: str = "pp",
        idea: str | None = None,
        kind: str,
        outcome: Any = None,
        status: str = "ok",
        parent_fitness: float | None = None,
        child_fitness: float | None = None,
        dq: float | None = None,
        added: int = 0,
        removed: int = 0,
        diff: str | None = None,
        program: str = "",
        raw_response: str = "",
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
        writer = self._writers.get("evaluations")
        file = self._files.get("evaluations")
        if writer and file:
            writer.writerow(
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
            file.flush()

        if self._console_output:
            self._print_progress(
                stage=stage,
                iteration=iteration,
                eval_count=eval_count,
                budget=budget,
                route_id=route_id,
                intent=intent,
                treatment=treatment,
                kind=kind,
                outcome_str=outcome_str,
                parent_fitness=parent_fitness,
                child_fitness=child_fitness,
                best_fitness=best_fitness,
                is_new_best=is_new_best,
                status=status,
                error=error,
            )

    def record_best(
        self,
        *,
        code: str,
        fitness: float,
        eval_count: int,
        iteration: int | None,
        order: int,
        program_id: int,
    ) -> None:
        """Update best_program.py and append a point to best_curve.csv."""
        ts = _now().strftime("%Y-%m-%d %H:%M:%S")
        writer = self._writers.get("best_curve")
        file = self._files.get("best_curve")
        if writer and file:
            writer.writerow(
                [
                    eval_count,
                    order,
                    _val(iteration),
                    f"{fitness:.6g}",
                    program_id,
                    ts,
                ]
            )
            file.flush()

        header = (
            "# ==============================================================================\n"
            "# Best Program Discovered by TraceAAD V9.13\n"
            f"# Fitness: {fitness:.6g}\n"
            f"# Evaluator Count: {eval_count} | Iteration: {iteration} | Sample Order: {order}\n"
            f"# Program ID: {program_id} | Timestamp: {ts}\n"
            "# ==============================================================================\n\n"
        )
        self._best_program_path.parent.mkdir(parents=True, exist_ok=True)
        self._best_program_path.write_text(
            header + code.rstrip() + "\n", encoding="utf-8"
        )

    def _print_progress(
        self,
        *,
        stage: str,
        iteration: int | None,
        eval_count: int,
        budget: int,
        route_id: int | None,
        intent: str | None,
        treatment: str,
        kind: str,
        outcome_str: str,
        parent_fitness: float | None,
        child_fitness: float | None,
        best_fitness: float | None,
        is_new_best: bool,
        status: str,
        error: str | None,
    ) -> None:
        prefix = f"[Eval {eval_count:03d}/{budget}]"
        best_tag = f"Best: {_format_fitness(best_fitness)}"
        if is_new_best:
            best_tag = f"★ New Best: {_format_fitness(best_fitness)}"

        if stage == "root_generation":
            fit_str = (
                _format_fitness(child_fitness) if status == "ok" else f"Failed ({status})"
            )
            print(f"[Init:Root] {fit_str} | {best_tag}")
        elif stage == "bootstrap":
            route_tag = f"Root {route_id}" if route_id is not None else "Root"
            fit_change = (
                f"{_format_fitness(parent_fitness)} -> {_format_fitness(child_fitness)} ({outcome_str})"
                if status == "ok" and outcome_str
                else f"Status: {kind}"
            )
            print(f"[Init:Bootstrap] {route_tag} | {fit_change} | {best_tag}")
        else:
            route_tag = f"R{route_id}" if route_id is not None else ""
            intent_tag = f"({intent}/{treatment})" if intent else ""
            ctx = f"Iter {iteration:03d} {route_tag}{intent_tag}".strip()
            if status != "ok":
                res = f"Failed ({status}: {error or kind})"
            elif kind in (
                "no_op",
                "ancestral_return",
                "repeated_duplicate",
                "root_duplicate",
            ):
                res = f"Cached ({kind})"
            elif kind == "cached":
                res = f"Cached -> {_format_fitness(child_fitness)}"
            else:
                p_str = _format_fitness(parent_fitness)
                c_str = _format_fitness(child_fitness)
                res = f"{p_str} -> {c_str} [{outcome_str.upper()}]"
            print(f"{prefix} {ctx:<28} | {res:<32} | {best_tag}")

    def record_llm_call(self, **row: Any) -> None:
        handle = self._files.get("llm_calls")
        if handle is not None:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
        if row.get("status") == "transport" and self._console_output:
            attempt = row.get("transport_attempt", 1)
            err = row.get("error", "unknown error")
            print(f"[Warning] LLM transport retry #{attempt}: {err}")

    def record_decision(self, event: str, **payload: Any) -> None:
        handle = self._files.get("events")
        if handle is None:
            return
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

    def record_error(self, scope: str, exc: BaseException) -> None:
        handle = self._files.get("errors")
        if handle:
            err_dict = {
                "scope": scope,
                "ts": _now().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:_ERROR_TRUNCATE],
                "traceback": "".join(traceback.format_exception(exc))[
                    :_TRACEBACK_TRUNCATE
                ],
            }
            json.dump(err_dict, handle, ensure_ascii=False)
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
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if self._console_output:
            duration = summary.get("duration_seconds", 0)
            score = summary.get("best_score")
            evals = summary.get("evaluator_call_count", 0)
            print("=" * 60)
            print(
                f"TraceAAD V9.13 Finished in {duration:.1f}s | Evals: {evals} | "
                f"Best Score: {_format_fitness(score)}"
            )
            print(f"Best program saved to: {self._best_program_path}")
            print(f"Evaluations log: {self._run_dir / 'evaluations.csv'}")
            print("=" * 60)

    def finish(self) -> None:
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self._writers.clear()


__all__ = ["RunArtifacts"]
