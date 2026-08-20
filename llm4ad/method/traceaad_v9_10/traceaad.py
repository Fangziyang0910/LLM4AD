"""TraceAAD V9.10: joint anchor-intent allocation with discounted Thompson sampling."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .forest import Forest, is_better
from .history import drop_oldest, one_line, parent_path, render_path
from .prompt import (
    ProgramResponseError,
    build_generation_prompt,
    build_root_prompt,
    parse_program_response,
)
from .schema import (
    CHILD_WINDOW,
    DEFAULT_MAX_CONSECUTIVE_ERRORS,
    DEFAULT_MAX_RESPONSES,
    EXPLORE_PRIOR,
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    PARENT_CHAIN_HALF_LIFE,
    PARENT_CHAIN_WINDOW,
    PRIOR_STRENGTH,
    RECENCY_HALF_LIFE,
    REFINE_PRIOR,
    Action,
    ActionStatus,
    Anchor,
    Intent,
    Outcome,
    Pending,
    Program,
)
from .selection import select, settle_pending_actions, start_quality
from .source import code_diff

TRANSPORT_RETRIES = 3
ERROR_MAX_CHARS = 360


class TraceAADV910:
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
        context_limit: int | None = None,
        max_history: int = MAX_HISTORY_EVENTS,
        seed: int | None = 0,
        max_responses: int = DEFAULT_MAX_RESPONSES,
        max_consecutive_errors: int = DEFAULT_MAX_CONSECUTIVE_ERRORS,
        child_window: int = CHILD_WINDOW,
        settlement_mode: str = "depth",
        allocation_mode: str = "thompson",
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        debug_mode: bool = False,
    ) -> None:
        if min(budget, n_roots, max_tokens, max_history) <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if context_limit is None or context_limit <= 0:
            raise ValueError("context_limit must be explicitly positive")
        if max_responses <= 0 or max_consecutive_errors <= 0:
            raise ValueError("response and error safety limits must be positive")
        if child_window < 0:
            raise ValueError("child_window must be non-negative")
        if settlement_mode not in {"depth", "response_age"}:
            raise ValueError(f"unknown settlement_mode: {settlement_mode}")
        if allocation_mode not in {"thompson", "uniform"}:
            raise ValueError(f"unknown allocation_mode: {allocation_mode}")
        if (
            evaluation.use_numba_accelerate
            or evaluation.use_protected_div
            or evaluation.random_seed is not None
        ):
            raise ValueError("V9.10 requires candidate code to be executed unchanged")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.10 requires one evolvable template function")

        self._llm = llm
        self._log = artifacts
        self._task = evaluation.task_description
        self._template = template
        self._function: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)
        self._budget = budget
        self._n_roots = n_roots
        self._maximize = maximize
        self._max_tokens = max_tokens
        self._context_limit = context_limit
        self._max_history = max_history
        self._seed = seed
        self._max_responses = max_responses
        self._max_consecutive_errors = max_consecutive_errors
        self._child_window = child_window
        self._settlement_mode = settlement_mode
        self._allocation_mode = allocation_mode
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        llm.debug_mode = debug_mode

        self._forest = Forest(maximize=maximize)
        self._pending: Pending | None = None
        self._n_candidates = 0
        self._n_eval = 0
        self._iteration = 0
        self._initialization_complete = False
        self._best_id: int | None = None
        self._consecutive_errors = 0
        self._search_aborted = False
        self._abort_reason: str | None = None

        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self) -> Program | None:
        """获取当前全局最优程序。"""
        return None if self._best_id is None else self._forest.get_program(self._best_id)

    def search_configuration(self) -> dict[str, Any]:
        return {
            "budget": self._budget,
            "n_roots": self._n_roots,
            "max_history": self._max_history,
            "maximize": self._maximize,
            "max_tokens": self._max_tokens,
            "context_limit": self._context_limit,
            "seed": self._seed,
            "prior_strength": PRIOR_STRENGTH,
            "refine_prior": REFINE_PRIOR,
            "explore_prior": EXPLORE_PRIOR,
            "recency_half_life": RECENCY_HALF_LIFE,
            "child_window": self._child_window,
            "settlement_mode": self._settlement_mode,
            "allocation_mode": self._allocation_mode,
            "parent_chain_window": PARENT_CHAIN_WINDOW,
            "parent_chain_half_life": PARENT_CHAIN_HALF_LIFE,
            "max_responses": self._max_responses,
            "max_consecutive_errors": self._max_consecutive_errors,
        }

    def run(self) -> None:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            if self._pending is not None:
                self._resume_pending()
            if not self._initialization_complete and not self._search_aborted:
                self._initialize()
            if not self._initialization_complete:
                status = "aborted" if self._search_aborted else "initialization_failure"
                stop_reason = self._abort_reason or "budget_exhausted_during_initialization"
                return

            while self._has_budget() and self._can_respond():
                choice = select(
                    self._forest,
                    seed=self._seed,
                    order=self._n_candidates + 1,
                    allocation_mode=self._allocation_mode,
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
            if self._search_aborted or not self._can_respond():
                status = "aborted"
                stop_reason = self._abort_reason or "response_safety_limit"
            else:
                status = "finished"
                stop_reason = "evaluator_budget_exhausted"
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            if self._log is not None:
                self._log.record_error("run", exc)
            raise
        finally:
            save_checkpoint(self)
            if self._log is not None:
                best = None if self._best_id is None else self._forest.get_program(self._best_id)
                actions = self._forest.actions()
                self._log.write_summary(
                    status=status,
                    stop_reason=stop_reason,
                    search_aborted=self._search_aborted,
                    abort_reason=self._abort_reason,
                    best_program_id=None if best is None else best.id,
                    best_score=None if best is None else best.fitness,
                    best_q=None if best is None else best.q,
                    best_response_order=None if best is None else best.order,
                    method_response_count=self._n_candidates,
                    evaluator_call_count=self._n_eval,
                    n_programs=len(self._forest.programs()),
                    n_anchors=len(self._forest.anchors()),
                    n_roots=len(self._forest.root_ids),
                    n_actions=len(actions),
                    n_pending=sum(
                        item.status is ActionStatus.PENDING for item in actions
                    ),
                    n_settled_success=sum(
                        item.status is ActionStatus.SETTLED and item.result == 1
                        for item in actions
                    ),
                    n_settled_failure=sum(
                        item.status is ActionStatus.SETTLED and item.result == 0
                        for item in actions
                    ),
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
            if not self._fits(prompt):
                raise RuntimeError("root prompt plus output bound exceeds context limit")
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
        if self._log is not None:
            self._log.record_decision(
                "roots_initialized",
                response_id="initialization",
                selected_program_ids=[
                    self._forest.get_anchor(root_id).program_id
                    for root_id in self._forest.root_ids
                ],
            )
        save_checkpoint(self)

    def _prompt(self, anchor_id: int, intent: Intent) -> str:
        anchor = self._forest.get_anchor(anchor_id)
        program = self._forest.get_program(anchor.program_id)
        selected = parent_path(self._forest, anchor_id, max_events=self._max_history)
        shown = selected
        while True:
            prompt = build_generation_prompt(
                task_description=self._task,
                code=program.code,
                fitness=program.fitness,
                history_text=render_path(self._forest, shown),
                intent=intent,
                maximize=self._maximize,
            )
            if self._fits(prompt):
                return prompt
            if not shown:
                raise RuntimeError(
                    "task, current code, and output budget exceed context even without history"
                )
            shown = drop_oldest(shown)

    def _request(
        self,
        prompt: str,
        *,
        anchor_id: int | None,
        stage: str,
        iteration: int | None,
        intent: str | None,
        selection: dict[str, Any] | None,
    ) -> Action | None:
        if self._pending is not None:
            raise RuntimeError("cannot request a candidate while one is pending")
        if not self._can_respond():
            raise RuntimeError("response safety limit reached")
        action_id = self._forest.next_action_id()
        order = self._n_candidates + 1
        response_id = f"v910-r{order:06d}-a{action_id:06d}"
        generation_seed = None if self._seed is None else self._seed + order
        self._pending = Pending(
            id=action_id,
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
                prompt_tokens=self._tokens(prompt),
            )
        return self._resume_pending()

    def _resume_pending(self) -> Action | None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending request")
        if pending.response is None:
            recovered = (
                None
                if self._log is None or not hasattr(self._log, "recovered_response")
                else self._log.recovered_response(pending.response_id)
            )
            response = recovered if recovered is not None else self._draw(pending)
            pending.response = response
            self._n_candidates += 1
            if pending.anchor_id is not None and pending.intent is not None:
                self._forest.get_anchor(pending.anchor_id).increment(pending.intent)
            if self._log is not None:
                self._log.record_llm_call(
                    response_id=pending.response_id,
                    stage=pending.stage,
                    iteration=pending.iteration,
                    anchor_id=pending.anchor_id,
                    intent=pending.intent,
                    order=pending.order,
                    prompt_tokens=self._tokens(pending.prompt),
                    response_tokens=self._tokens(response),
                    status="ok",
                    prompt=pending.prompt,
                    raw_response=response,
                    generation_seed=pending.generation_seed,
                )
            save_checkpoint(self)
        return self._process_pending()

    def _draw(self, pending: Pending) -> str:
        last_error: Exception | None = None
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if pending.generation_seed is not None:
            kwargs["seed"] = pending.generation_seed
        for transport_attempt in range(1, TRANSPORT_RETRIES + 2):
            try:
                return self._llm.draw_sample(pending.prompt, **kwargs)
            except Exception as exc:
                last_error = exc
                if self._log is not None:
                    self._log.record_llm_call(
                        response_id=pending.response_id,
                        stage=pending.stage,
                        iteration=pending.iteration,
                        anchor_id=pending.anchor_id,
                        intent=pending.intent,
                        order=pending.order,
                        prompt_tokens=self._tokens(pending.prompt),
                        response_tokens=0,
                        status="transport",
                        prompt=pending.prompt,
                        raw_response="",
                        generation_seed=pending.generation_seed,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        transport_attempt=transport_attempt,
                    )
                save_checkpoint(self)
        raise RuntimeError("model transport retry limit exhausted") from last_error

    def _process_pending(self) -> Action | None:
        pending = self._pending
        if pending is None or pending.response is None:
            raise RuntimeError("pending response is not ready")
        try:
            parsed = parse_program_response(
                pending.response, self._template, self._function.name
            )
        except ProgramResponseError as exc:
            return self._finalize(
                idea=exc.declared_idea,
                code=None,
                diff=None,
                added=0,
                removed=0,
                existing=None,
                fitness=None,
                error=one_line(str(exc), ERROR_MAX_CHARS),
                evaluated=False,
            )

        idea = parsed.declared_idea
        code = str(parsed.program)
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
    ) -> Action | None:
        pending = self._pending
        if pending is None or pending.response is None:
            raise RuntimeError("no completed response to finalize")
        parent_anchor = (
            None if pending.anchor_id is None else self._forest.get_anchor(pending.anchor_id)
        )
        parent = (
            None if parent_anchor is None else self._forest.get_program(parent_anchor.program_id)
        )
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
        action: Action | None = None
        if pending.anchor_id is not None:
            action = Action(
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
            self._forest.add_action(action)
        newly_settled = (
            ()
            if action is None
            else settle_pending_actions(
                self._forest,
                now_order=pending.order,
                child_window=self._child_window,
                settlement_mode=self._settlement_mode,
            )
        )
        if pending.stage == "search" and pending.iteration is not None:
            self._iteration = max(self._iteration, pending.iteration + 1)

        if kind == "invalid":
            self._consecutive_errors += 1
        else:
            self._consecutive_errors = 0
        if self._consecutive_errors >= self._max_consecutive_errors:
            self._search_aborted = True
            self._abort_reason = "consecutive_error_limit"

        is_new_best, _reason = self._update_best(program)
        if program is not None and self._log is not None:
            self._log.record_program(
                program_id=program.id,
                code=program.code,
                fitness=program.fitness,
                q=program.q,
                order=program.order,
            )
        if is_new_best and program is not None and self._log is not None:
            self._log.record_best(
                response_id=pending.response_id,
                code=program.code,
                fitness=program.fitness,
                q=program.q,
                eval_count=self._n_eval,
                iteration=pending.iteration,
                order=pending.order,
                program_id=program.id,
            )
        response = pending.response
        selection = pending.selection
        self._pending = None
        save_checkpoint(self)
        self._record_response(
            pending=pending,
            action=action,
            program=program,
            kind=kind,
            idea=idea,
            response=response,
            code=code,
            error=error,
            evaluated=evaluated,
            is_new_best=is_new_best,
            selection=selection,
        )
        if self._log is not None:
            for item in newly_settled:
                self._log.record_decision(
                    "action_settled",
                    event_key=f"a{item.id}",
                    response_id=pending.response_id,
                    action_id=item.id,
                    action_response_id=item.response_id,
                    anchor_id=item.anchor_id,
                    child_id=item.child_id,
                    intent=item.intent,
                    kind=item.kind,
                    action_order=item.order,
                    settled_result=item.result,
                    settled_order=item.settled_order,
                    pending_responses=(
                        None
                        if item.settled_order is None
                        else item.settled_order - item.order
                    ),
                    observed_depth=item.observed_depth,
                    window_best_q=item.window_best_q,
                    start_q=start_quality(self._forest, item),
                )
        return action

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
                action_id=pending.id,
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
            action_id=pending.id,
            order=pending.order,
        )
        return program, child, "new"

    def _update_best(self, program: Program | None) -> tuple[bool, str | None]:
        if program is None:
            return False, None
        incumbent = None if self._best_id is None else self._forest.get_program(self._best_id)
        if not is_better(program, incumbent):
            return False, None
        reason = "strict_fitness" if incumbent is None or program.q > incumbent.q else "tie_break"
        self._best_id = program.id
        return True, reason

    def _record_response(
        self,
        *,
        pending: Pending,
        action: Action | None,
        program: Program | None,
        kind: str,
        idea: str | None,
        response: str,
        code: str | None,
        error: str | None,
        evaluated: bool,
        is_new_best: bool,
        selection: dict[str, Any] | None,
    ) -> None:
        if self._log is None:
            return
        status = "ok"
        if kind == "invalid":
            status = "parse_failed" if not evaluated else "eval_failed"
        parent_q = (
            None
            if action is None
            else self._forest.get_program(
                self._forest.get_anchor(action.anchor_id).program_id
            ).q
        )
        child_q = None if program is None else program.q
        best = None if self._best_id is None else self._forest.get_program(self._best_id)
        self._log.record_candidate(
            response_id=pending.response_id,
            action_id=pending.id,
            order=pending.order,
            stage=pending.stage,
            iteration=pending.iteration,
            anchor_id=pending.anchor_id,
            child_id=None if action is None else action.child_id,
            program_id=None if program is None else program.id,
            intent=pending.intent,
            idea=idea if action is None else action.idea,
            kind=kind,
            outcome=None if action is None else action.outcome,
            evaluator_called=evaluated,
            status=status,
            parent_fitness=None if action is None else action.parent_fitness,
            child_fitness=None if program is None else program.fitness,
            parent_q=parent_q,
            child_q=child_q,
            dq=None if action is None else action.dq,
            added=0 if action is None else action.added,
            removed=0 if action is None else action.removed,
            diff=None if action is None else action.diff,
            program=code or "",
            raw_response=response,
            error=error,
            eval_count=self._n_eval,
            best_fitness=None if best is None else best.fitness,
            is_new_best=is_new_best,
            budget=self._budget,
            selection=selection,
            action_status=None if action is None else action.status.value,
            action_result=None if action is None else action.result,
            action_observed_depth=None if action is None else action.observed_depth,
            action_window_best_q=(
                None if action is None or action.window_best_q is None else action.window_best_q
            ),
        )

    def _fits(self, prompt: str) -> bool:
        return self._tokens(prompt) + self._max_tokens <= self._context_limit

    def _tokens(self, text: str) -> int:
        for name in ("count_prompt_tokens", "count_tokens"):
            counter = getattr(self._llm, name, None)
            if callable(counter):
                return int(counter(text))
        raise RuntimeError("model tokenizer is unavailable")

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _can_respond(self) -> bool:
        if self._search_aborted:
            return False
        if self._n_candidates >= self._max_responses:
            self._search_aborted = True
            self._abort_reason = "response_safety_limit"
            return False
        return True


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


__all__ = ["TraceAADV910"]
