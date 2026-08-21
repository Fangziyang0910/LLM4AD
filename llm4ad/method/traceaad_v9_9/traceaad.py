"""TraceAAD V9.9: trajectory-first joint anchor-operator allocation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
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
    DEFAULT_MAX_RESPONSES,
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
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


class TraceAADV99:
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
        max_responses: int = DEFAULT_MAX_RESPONSES,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        if min(budget, n_roots, max_tokens, max_history) <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if max_responses <= 0:
            raise ValueError("response safety limit must be positive")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.9 requires one evolvable template function")

        self._llm = llm
        self._log = artifacts
        self._task = evaluation.task_description
        self._function: Function = template.functions[0]
        self._evaluator = SecureEvaluator(evaluation)
        self._budget = budget
        self._n_roots = n_roots
        self._maximize = maximize
        self._max_tokens = max_tokens
        self._max_history = max_history
        self._seed = seed
        self._max_responses = max_responses
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

        self._forest = Forest(maximize=maximize)
        self._pending: Pending | None = None
        self._n_candidates = 0
        self._n_eval = 0
        self._iteration = 0
        self._initialization_complete = False

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
                self._resume_pending()
            if not self._initialization_complete:
                self._initialize()
            if not self._initialization_complete:
                status = "initialization_failure"
                stop_reason = "budget_exhausted_during_initialization"
                return

            while self._has_budget() and self._can_respond():
                choice = select(
                    self._forest,
                    seed=self._seed,
                    iteration=self._iteration,
                    max_distance=self._max_history,
                )
                prompt = self._prompt(choice.anchor_id, choice.intent)
                self._request(
                    prompt,
                    anchor_id=choice.anchor_id,
                    stage="search",
                    iteration=self._iteration,
                    intent=choice.intent.value,
                    selection=choice.to_dict(),
                )
            if not self._can_respond():
                status = "aborted"
                stop_reason = "response_safety_limit"
            else:
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
                    best_q=None if best is None else best.q,
                    best_response_order=None if best is None else best.order,
                    method_response_count=self._n_candidates,
                    evaluator_call_count=self._n_eval,
                    n_programs=len(self._forest.programs()),
                    n_anchors=len(self._forest.anchors()),
                    n_roots=len(self._forest.root_ids),
                    n_attempts=len(self._forest.attempts()),
                    n_iterations=self._iteration,
                    initialization_complete=self._initialization_complete,
                    pending_response_id=(
                        None if self._pending is None else self._pending.response_id
                    ),
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _initialize(self) -> None:
        while (
            len(self._forest.root_ids) < self._n_roots
            and self._has_budget()
            and self._can_respond()
        ):
            prompt = build_root_prompt(
                task_description=self._task,
                template_function=self._function,
                maximize=self._maximize,
            )
            self._request(
                prompt,
                anchor_id=None,
                stage="root_generation",
                iteration=None,
                intent=None,
                selection=None,
            )

        if len(self._forest.root_ids) < self._n_roots:
            save_checkpoint(self)
            return
        self._initialization_complete = True
        save_checkpoint(self)

    def _prompt(self, anchor_id: int, intent: Intent) -> str:
        anchor = self._forest.get_anchor(anchor_id)
        program = self._forest.get_program(anchor.program_id)
        return build_generation_prompt(
            task_description=self._task,
            code=program.code,
            fitness=program.fitness,
            history_text=render_path(
                self._forest,
                parent_path(self._forest, anchor_id, max_events=self._max_history),
            ),
            intent=intent,
            maximize=self._maximize,
        )

    def _request(
        self,
        prompt: str,
        *,
        anchor_id: int | None,
        stage: str,
        iteration: int | None,
        intent: str | None,
        selection: dict[str, Any] | None,
    ) -> Attempt:
        if self._pending is not None:
            raise RuntimeError("cannot request a candidate while one is pending")
        if not self._can_respond():
            raise RuntimeError("response safety limit reached")
        attempt_id = self._forest.next_attempt_id()
        order = self._n_candidates + 1
        response_id = f"v99-r{order:06d}-a{attempt_id:06d}"
        generation_seed = None if self._seed is None else self._seed + order
        self._pending = Pending(
            id=attempt_id,
            response_id=response_id,
            anchor_id=anchor_id,
            stage=stage,
            iteration=iteration,
            order=order,
            intent=intent,
            prompt=prompt,
            generation_seed=generation_seed,
            selection=selection,
        )
        save_checkpoint(self)
        if self._log is not None:
            self._log.record_request(
                response_id=response_id,
                order=order,
                stage=stage,
                iteration=iteration,
                operator=intent,
                anchor_id=anchor_id,
                generation_seed=generation_seed,
                selection=selection,
            )
        return self._resume_pending()

    def _resume_pending(self) -> Attempt:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending request")
        if pending.response is None:
            kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
            if pending.generation_seed is not None:
                kwargs["seed"] = pending.generation_seed
            pending.response = self._llm.draw_sample(pending.prompt, **kwargs)
            self._n_candidates += 1
            if pending.anchor_id is not None and pending.intent is not None:
                self._forest.get_anchor(pending.anchor_id).increment(pending.intent)
            save_checkpoint(self)
        return self._process_pending()

    def _process_pending(self) -> Attempt:
        pending = self._pending
        if pending is None or pending.response is None:
            raise RuntimeError("pending response is not ready")

        parsed = parse_program_response(pending.response)
        idea = parsed.declared_idea
        code = parsed.code
        diff: str | None = None
        added = removed = 0
        if pending.anchor_id is not None:
            parent = self._forest.get_program(
                self._forest.get_anchor(pending.anchor_id).program_id
            )
            diff, added, removed = code_diff(parent.code, code)
        existing = self._forest.program_for_code(code)
        if existing is not None:
            return self._finalize(
                idea=idea,
                code=code,
                diff=diff,
                added=added,
                removed=removed,
                existing=existing,
                fitness=None,
                error=None,
                evaluated=False,
            )

        outcome, _elapsed = self._evaluator.evaluate_program_record_time_with_details(code)
        self._n_eval += 1
        if outcome.failure_kind == "prepare_error":
            raise RuntimeError(
                one_line(
                    outcome.error or "evaluator preparation failed without an error message",
                    ERROR_MAX_CHARS,
                )
            )
        score = getattr(outcome.result, "fitness", outcome.result)
        try:
            parsed_fitness = float(score)
        except (TypeError, ValueError, OverflowError):
            parsed_fitness = math.nan
        if math.isfinite(parsed_fitness):
            return self._finalize(
                idea=idea,
                code=code,
                diff=diff,
                added=added,
                removed=removed,
                existing=None,
                fitness=parsed_fitness,
                error=None,
                evaluated=True,
            )
        return self._finalize(
            idea=idea,
            code=code,
            diff=diff,
            added=added,
            removed=removed,
            existing=None,
            fitness=None,
            error=one_line(
                outcome.error
                or f"evaluator returned non-finite or non-numeric fitness: {score!r}",
                ERROR_MAX_CHARS,
            ),
            evaluated=True,
            status=outcome.failure_kind or "invalid_result",
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
        evaluated: bool,
        status: str = "ok",
    ) -> Attempt:
        pending = self._pending
        if pending is None or pending.response is None:
            raise RuntimeError("no completed response to finalize")
        parent_anchor = (
            None if pending.anchor_id is None else self._forest.get_anchor(pending.anchor_id)
        )
        parent = (
            None if parent_anchor is None else self._forest.get_program(parent_anchor.program_id)
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
        outcome = _outcome(parent is not None, invalid=kind == "invalid", dq=dq)
        attempt = Attempt(
            id=pending.id,
            response_id=pending.response_id,
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
        if pending.stage == "search" and pending.iteration is not None:
            self._iteration = max(self._iteration, pending.iteration + 1)

        best = self.best
        is_new_best = program is not None and is_better(program, previous_best)
        if program is not None and self._log is not None:
            self._log.record_program(
                program_id=program.id,
                code=program.code,
                fitness=program.fitness,
                q=program.q,
                order=program.order,
            )
        if is_new_best and program is not None and self._log is not None:
            self._log.record_best(code=program.code, fitness=program.fitness)
        selection = pending.selection
        self._pending = None
        save_checkpoint(self)
        self._record_attempt(
            attempt,
            error=error,
            evaluated=evaluated,
            status=status,
            is_new_best=is_new_best,
            best_fitness=None if best is None else best.fitness,
            selection=selection,
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
        program = self._forest.add_program(code=code, fitness=fitness, order=pending.order)
        if parent_anchor is None:
            root = self._forest.add_root(program_id=program.id, order=pending.order)
            return program, root, "root"
        child = self._forest.add_child(
            parent_id=parent_anchor.id,
            program_id=program.id,
            attempt_id=pending.id,
            order=pending.order,
        )
        return program, child, "new"

    def _record_attempt(
        self,
        attempt: Attempt,
        *,
        error: str | None,
        evaluated: bool,
        status: str,
        is_new_best: bool,
        best_fitness: float | None,
        selection: dict[str, Any] | None,
    ) -> None:
        if self._log is None:
            return
        parent_q = (
            None
            if attempt.anchor_id is None
            else self._forest.get_program(
                self._forest.get_anchor(attempt.anchor_id).program_id
            ).q
        )
        child_q = (
            None
            if attempt.program_id is None
            else self._forest.get_program(attempt.program_id).q
        )
        self._log.record_candidate(
            response_id=attempt.response_id,
            attempt_id=attempt.id,
            order=attempt.order,
            stage=attempt.stage,
            iteration=attempt.iteration,
            anchor_id=attempt.anchor_id,
            child_id=attempt.child_id,
            program_id=attempt.program_id,
            intent=attempt.intent,
            idea=attempt.idea,
            kind=attempt.kind,
            outcome=attempt.outcome,
            evaluator_called=evaluated,
            status=status,
            parent_fitness=attempt.parent_fitness,
            child_fitness=attempt.child_fitness,
            parent_q=parent_q,
            child_q=child_q,
            dq=attempt.dq,
            added=attempt.added,
            removed=attempt.removed,
            error=error,
            eval_count=self._n_eval,
            best_fitness=best_fitness,
            is_new_best=is_new_best,
            budget=self._budget,
            selection=selection,
        )

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _can_respond(self) -> bool:
        return self._n_candidates < self._max_responses


def _outcome(has_anchor: bool, *, invalid: bool, dq: float | None) -> Outcome | None:
    if not has_anchor:
        return None
    if invalid:
        return Outcome.INVALID
    assert dq is not None
    if dq > 0:
        return Outcome.IMPROVE
    if dq < 0:
        return Outcome.REGRESS
    return Outcome.PLATEAU


__all__ = ["TraceAADV99"]
