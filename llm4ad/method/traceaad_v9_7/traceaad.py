"""TraceAAD V9.7: route-then-anchor allocation with Refine/Explore intents."""

from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

from ...base import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .forest import Forest, is_better
from .history import one_line, parent_path, render_path
from .prompt import (
    build_generation_prompt,
    build_root_prompt,
    parse_program_response,
)
from .schema import (
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    REFINE_PROBABILITY,
    Anchor,
    Attempt,
    Intent,
    Outcome,
    Pending,
    Program,
)
from .selection import select
from .source import code_diff

ERROR_MAX_CHARS = 360


def draw_intent(seed: int | None, iteration: int) -> Intent:
    """Deterministic fixed-mixture intent: no RNG state, resume-safe."""
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(f"{token}:{iteration}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return Intent.REFINE if value < REFINE_PROBABILITY else Intent.EXPLORE


class TraceAADV97:
    """Run the complete V9.7 search forest lifecycle."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
        budget: int = 1000,
        *,
        n_roots: int = INITIAL_ROOT_COUNT,
        maximize: bool = True,
        max_tokens: int = 8192,
        max_history: int = MAX_HISTORY_EVENTS,
        seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        if budget <= 0 or n_roots <= 0 or max_tokens <= 0 or max_history <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.7 requires one evolvable template function")

        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation)
        self._log = artifacts
        self._task = evaluation.task_description
        self._function: Function = template.functions[0]
        self._budget = budget
        self._n_roots = n_roots
        self._maximize = maximize
        self._max_tokens = max_tokens
        self._max_history = max_history
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

        self._forest = Forest(maximize=self._maximize)
        self._pending: Pending | None = None
        self._n_candidates = 0
        self._n_eval = 0
        self._iteration = 0
        self._initialization_complete = False
        self._bootstrapped: set[int] = set()
        self._bootstrap_deltas: list[float] = []
        self._s: float | None = None

        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self) -> Program | None:
        """当前全局最优程序，按 (q, -length, -order) 现算。"""
        return self._forest.best()

    def run(self) -> None:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            if self._pending is not None:
                self._process_pending()
            if not self._initialization_complete:
                self._initialize()
            if not self._initialization_complete:
                status = "initialization_failure"
                stop_reason = "evaluator_budget_exhausted_during_initialization"
                return

            while self._has_budget():
                assert self._s is not None
                choice = select(self._forest, self._s)
                intent = draw_intent(self._seed, self._iteration)
                self._generate(
                    self._prompt(choice.anchor_id, intent),
                    anchor_id=choice.anchor_id,
                    stage="search",
                    iteration=self._iteration,
                    intent=intent.value,
                )

            status = "finished"
            stop_reason = "evaluator_budget_exhausted"
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            raise
        finally:
            save_checkpoint(self)
            if self._log is not None:
                best = self.best
                self._log.write_summary(
                    status=status,
                    stop_reason=stop_reason,
                    best_program_id=None if best is None else best.id,
                    best_score=None if best is None else best.fitness,
                    best_sample_order=None if best is None else best.order,
                    method_sample_count=self._n_candidates,
                    evaluator_call_count=self._n_eval,
                    n_programs=len(self._forest.programs()),
                    n_anchors=len(self._forest.anchors()),
                    n_roots=len(self._forest.root_ids),
                    n_attempts=len(self._forest.attempts()),
                    n_iterations=self._iteration,
                    initialization_complete=self._initialization_complete,
                    bootstrapped=sorted(self._bootstrapped),
                    bootstrap_deltas=self._bootstrap_deltas,
                    s=self._s,
                    pending_id=None if self._pending is None else self._pending.id,
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _initialize(self) -> None:
        while len(self._forest.root_ids) < self._n_roots and self._has_budget():
            self._generate(
                build_root_prompt(
                    task_description=self._task,
                    template_function=self._function,
                    maximize=self._maximize,
                ),
                anchor_id=None,
                stage="root_generation",
                iteration=None,
                intent=None,
            )

        for root_id in tuple(self._forest.root_ids):
            if not self._has_budget():
                break
            if root_id in self._bootstrapped:
                continue
            self._generate(
                self._prompt(root_id, Intent.REFINE),
                anchor_id=root_id,
                stage="bootstrap",
                iteration=None,
                intent=Intent.REFINE.value,
            )

        complete = (
            len(self._forest.root_ids) == self._n_roots
            and self._bootstrapped == set(self._forest.root_ids)
        )
        if not complete:
            save_checkpoint(self)
            return
        self._s = (
            float(statistics.median(self._bootstrap_deltas))
            if self._bootstrap_deltas
            else 0.0
        )
        self._initialization_complete = True
        save_checkpoint(self)

    def _prompt(self, anchor_id: int, intent: Intent) -> str:
        program = self._forest.get_program(
            self._forest.get_anchor(anchor_id).program_id
        )
        return build_generation_prompt(
            task_description=self._task,
            code=program.code,
            fitness=program.fitness,
            history_text=render_path(
                self._forest, parent_path(self._forest, anchor_id, max_events=self._max_history)
            ),
            intent=intent,
            maximize=self._maximize,
        )

    def _generate(
        self,
        prompt: str,
        *,
        anchor_id: int | None,
        stage: str,
        iteration: int | None,
        intent: str | None,
    ) -> Attempt:
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if self._seed is not None:
            kwargs["seed"] = self._seed + self._n_candidates + 1
        response = self._llm.draw_sample(prompt, **kwargs)
        self._n_candidates += 1
        if anchor_id is not None:
            self._forest.get_anchor(anchor_id).n += 1
        self._pending = Pending(
            id=self._forest.next_attempt_id(),
            anchor_id=anchor_id,
            stage=stage,
            iteration=iteration,
            order=self._n_candidates,
            intent=intent,
            response=response,
        )
        save_checkpoint(self)
        return self._process_pending()

    def _process_pending(self) -> Attempt:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending candidate to process")

        parsed = parse_program_response(pending.response)
        idea = parsed.declared_idea
        code = parsed.code
        diff: str | None = None
        added = 0
        removed = 0
        if pending.anchor_id is not None:
            parent = self._forest.get_program(
                self._forest.get_anchor(pending.anchor_id).program_id
            )
            diff, added, removed = code_diff(parent.code, code)
        existing = self._forest.program_for_code(code)
        fitness = None
        error = None
        status = "ok"
        if existing is None:
            outcome, _elapsed = self._evaluator.evaluate_program_record_time_with_details(
                code
            )
            self._n_eval += 1
            if outcome.failure_kind == "prepare_error":
                raise RuntimeError(
                    one_line(
                        outcome.error
                        or "evaluator preparation failed without an error message",
                        ERROR_MAX_CHARS,
                    )
                )
            score = getattr(outcome.result, "fitness", outcome.result)
            try:
                fitness = float(score)
            except (TypeError, ValueError, OverflowError):
                fitness = math.nan
            if not math.isfinite(fitness):
                fitness = None
                error = one_line(
                    outcome.error
                    or f"evaluator returned non-finite or non-numeric fitness: {score!r}",
                    ERROR_MAX_CHARS,
                )
                status = outcome.failure_kind or "invalid_result"
        return self._finalize(
            idea=idea,
            code=code,
            diff=diff,
            added=added,
            removed=removed,
            existing=existing,
            fitness=fitness,
            error=error,
            status=status,
        )

    def _finalize(
        self,
        *,
        idea: str | None,
        code: str | None,
        diff: str | None,
        added: int,
        removed: int,
        existing: Program | None,
        fitness: float | None,
        error: str | None,
        status: str = "ok",
    ) -> Attempt:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending candidate to finalize")
        parent_anchor = (
            None
            if pending.anchor_id is None
            else self._forest.get_anchor(pending.anchor_id)
        )
        parent = (
            None
            if parent_anchor is None
            else self._forest.get_program(parent_anchor.program_id)
        )
        previous_best = self.best
        program, child, kind = self._place(
            pending=pending,
            parent_anchor=parent_anchor,
            parent=parent,
            existing=existing,
            code=code,
            fitness=fitness,
            error=error,
        )
        dq = None if parent is None or program is None else program.q - parent.q
        if parent is None:
            outcome = None
        elif dq is None:
            outcome = Outcome.INVALID
        elif dq > 0:
            outcome = Outcome.IMPROVE
        elif dq < 0:
            outcome = Outcome.REGRESS
        else:
            outcome = Outcome.PLATEAU
        attempt = Attempt(
            id=pending.id,
            anchor_id=pending.anchor_id,
            child_id=None if child is None else child.id,
            program_id=None if program is None else program.id,
            intent=pending.intent,
            idea=idea,
            diff=diff,
            added=added,
            removed=removed,
            parent_fitness=None if parent is None else parent.fitness,
            child_fitness=None if program is None else program.fitness,
            dq=dq,
            outcome=outcome,
            kind=kind,
            order=pending.order,
            stage=pending.stage,
            iteration=pending.iteration,
        )
        self._forest.add_attempt(attempt)

        if pending.stage == "bootstrap" and pending.anchor_id is not None:
            self._bootstrapped.add(pending.anchor_id)
            if child is not None and dq is not None:
                self._bootstrap_deltas.append(abs(dq))
        if pending.stage == "search" and pending.iteration is not None:
            self._iteration = max(self._iteration, pending.iteration + 1)

        best = self.best
        is_new_best = program is not None and is_better(program, previous_best)
        if is_new_best and self._log is not None:
            self._log.record_best(code=program.code, fitness=program.fitness)
        self._pending = None
        save_checkpoint(self)
        if self._log is not None:
            route_id = None
            if attempt.anchor_id is not None:
                route_id = self._forest.get_anchor(attempt.anchor_id).root_id
            elif attempt.child_id is not None:
                route_id = self._forest.get_anchor(attempt.child_id).root_id
            self._log.record_candidate(
                order=attempt.order,
                stage=attempt.stage,
                iteration=attempt.iteration,
                route_id=route_id,
                anchor_id=attempt.anchor_id,
                child_id=attempt.child_id,
                program_id=attempt.program_id,
                intent=attempt.intent,
                kind=attempt.kind,
                outcome=attempt.outcome,
                parent_fitness=attempt.parent_fitness,
                child_fitness=attempt.child_fitness,
                dq=attempt.dq,
                is_new_best=is_new_best,
                best_fitness=None if best is None else best.fitness,
                status=status,
                error=error,
                eval_count=self._n_eval,
                budget=self._budget,
            )
        return attempt

    def _place(
        self,
        *,
        pending: Pending,
        parent_anchor: Anchor | None,
        parent: Program | None,
        existing: Program | None,
        code: str | None,
        fitness: float | None,
        error: str | None,
    ) -> tuple[Program | None, Anchor | None, str]:
        if error is not None:
            return None, None, "invalid"
        if existing is not None and parent_anchor is None:
            return existing, None, "root_duplicate"
        if existing is not None and parent is not None:
            if existing.id == parent.id:
                return existing, None, "no_op"
            if existing.id in self._forest.ancestor_program_ids(parent_anchor.id):
                return existing, None, "ancestral_return"
            if self._forest.relation_exists(parent_anchor.id, existing.id):
                return existing, None, "repeated_duplicate"
            child = self._forest.add_child(
                parent_id=parent_anchor.id,
                program_id=existing.id,
                attempt_id=pending.id,
                order=pending.order,
            )
            return existing, child, "cached"
        assert code is not None and fitness is not None
        program = self._forest.add_program(
            code=code, fitness=fitness, order=pending.order
        )
        if parent_anchor is None:
            child = self._forest.add_root(program_id=program.id, order=pending.order)
            return program, child, "root_new"
        child = self._forest.add_child(
            parent_id=parent_anchor.id,
            program_id=program.id,
            attempt_id=pending.id,
            order=pending.order,
        )
        return program, child, "new"

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget


__all__ = [
    "TraceAADV97",
    "draw_intent",
]
