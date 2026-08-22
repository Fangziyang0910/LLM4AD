"""TraceAAD V9.16 search loop with bounded Explore landings."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .history import one_line, parent_path, render_path
from .prompt import (
    build_generation_prompt,
    build_repair_prompt,
    build_root_prompt,
    format_failure_feedback,
    parse_program_response,
    preflight_code,
)
from .schema import (
    INITIAL_ROOT_COUNT,
    LANDING_HORIZON,
    LANDING_RATIO,
    MAX_HISTORY_EVENTS,
    Algorithm,
    Intent,
    LandingState,
    Pending,
)
from .selection import Decision, decide, landing_ticket
from .tree import Tree, VIRTUAL_ROOT_ID

ERROR_MAX_CHARS = 360


class TraceAADV916:
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
        error_retries: int = 2,
        error_handling: bool = True,
    ) -> None:
        if min(budget, n_roots, max_tokens, max_history) <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if n_roots > budget:
            raise ValueError("n_roots cannot exceed budget")
        if error_retries < 0:
            raise ValueError("error_retries must be non-negative")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.16 requires one evolvable template function")

        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation)
        self._log = artifacts
        self._task = evaluation.task_description
        self._function: Function = template.functions[0]
        self._budget = budget
        self._n_roots = n_roots
        self._max_tokens = max_tokens
        self._max_history = max_history
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._error_retries = error_retries
        self._error_handling = error_handling

        self._attempt_number = 1
        self._attempt_kind = "initial"
        self._last_failure: dict[str, str] | None = None
        self._candidate_hash = ""
        self._preflight_error: str | None = None
        self._attempt_elapsed = 0.0

        self._tree = Tree(maximize=maximize)
        self._pending: Pending | None = None
        self._n_eval = 0
        self._n_calls = 0
        self._n_stag = 0
        self._decision: Decision | None = None
        self._n_ordinary_decisions = 0
        self._next_entry_id = 1
        self._next_landing_id = 1
        self._landing_budget = math.floor(LANDING_RATIO * max(0, budget - n_roots))
        self._landing_slots_used = 0
        self._entry_tickets: dict[int, bool] = {}
        self._active_landing: LandingState | None = None

        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self) -> Algorithm | None:
        return self._tree.best()

    @property
    def landing_slots_remaining(self) -> int:
        return max(0, self._landing_budget - self._landing_slots_used)

    def run(self) -> None:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            if self._pending is not None:
                self._process_pending()
            self._initialize()
            if not self._initialized():
                status = "initialization_failure"
                stop_reason = "evaluator_budget_exhausted_during_initialization"
                return

            while self._has_budget():
                if self._active_landing is not None:
                    self._run_landing_step()
                    continue
                decision = decide(
                    self._tree,
                    seed=self._seed,
                    n_eval=self._n_eval,
                    decision_index=self._n_ordinary_decisions,
                )
                self._n_ordinary_decisions += 1
                self._decision = decision
                parent = decision.parent
                self._generate(
                    self._prompt(parent, decision.intent),
                    parent_id=parent.id,
                    intent=decision.intent.value,
                    mode="ordinary",
                    entry_id=(
                        parent.entry_id if decision.intent is Intent.REFINE else None
                    ),
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
                    best_algorithm_id=None if best is None else best.id,
                    best_score=None if best is None else best.fitness,
                    **self._summary_counts(),
                    n_algorithms=len(self._tree.valid_algorithms()),
                    n_roots=len(self._tree.root_algorithms()),
                    has_pending=self._pending is not None,
                    active_landing=None if self._active_landing is None else self._active_landing.id,
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _initialized(self) -> bool:
        return len(self._tree.root_algorithms()) == self._n_roots

    def _initialize(self) -> None:
        if self._initialized():
            return
        while not self._initialized() and self._has_budget():
            self._generate(
                build_root_prompt(
                    task_description=self._task,
                    template_function=self._function,
                    maximize=self._tree.maximize,
                    error_handling=self._error_handling,
                ),
                parent_id=VIRTUAL_ROOT_ID,
                intent=None,
                mode="initialization",
                entry_id=None,
            )
        self._n_stag = 0

    def _prompt(self, algorithm: Algorithm, intent: Intent) -> str:
        assert algorithm.code is not None and algorithm.fitness is not None
        path = parent_path(self._tree, algorithm.id, max_events=self._max_history)
        return build_generation_prompt(
            task_description=self._task,
            code=algorithm.code,
            fitness=algorithm.fitness,
            history_text=render_path(self._tree, path),
            intent=intent,
            maximize=self._tree.maximize,
            error_handling=self._error_handling,
        )

    def _run_landing_step(self) -> None:
        state = self._active_landing
        if state is None:
            return
        if state.completed_steps >= LANDING_HORIZON or not self._has_budget():
            self._finish_landing(state)
            return
        parent = self._tree.get_algorithm(state.latest_valid_id)
        step = state.completed_steps + 1
        self._decision = None
        self._generate(
            self._prompt(parent, Intent.REFINE),
            parent_id=parent.id,
            intent=Intent.REFINE.value,
            mode="landing",
            entry_id=state.entry_id,
            landing_id=state.id,
            landing_step=step,
        )
        if state.completed_steps >= LANDING_HORIZON or not self._has_budget():
            self._finish_landing(state)

    def _finish_landing(self, state: LandingState) -> None:
        origin_parent = self._tree.get_algorithm(state.origin_parent_id)
        latest = self._tree.get_algorithm(state.latest_valid_id)
        state.final_gain = self._tree.quality(latest) - self._tree.quality(origin_parent)
        self._event(
            "landing_finish",
            landing_id=state.id,
            entry_id=state.entry_id,
            origin_id=state.origin_id,
            valid_steps=state.valid_steps,
            completed_steps=state.completed_steps,
            strict_improvements=state.strict_improvements,
            final_gain=state.final_gain,
            max_gain=state.max_gain,
            recovered_parent=state.final_gain >= 0,
            over_parent=state.final_gain > 0,
        )
        self._active_landing = None
        save_checkpoint(self)

    def _generate(
        self,
        prompt: str,
        *,
        parent_id: int,
        intent: str | None,
        mode: str,
        entry_id: int | None,
        landing_id: int | None = None,
        landing_step: int | None = None,
    ) -> Algorithm | None:
        repair_prompt = prompt
        for attempt in range(1, self._error_retries + 2):
            self._attempt_number = attempt
            self._attempt_kind = "initial" if attempt == 1 else "repair"
            kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
            if self._seed is not None:
                kwargs["seed"] = self._seed + self._n_eval + 1
            response = self._llm.draw_sample(repair_prompt, **kwargs)
            self._pending = Pending(
                parent_id, intent, response, mode, entry_id, landing_id, landing_step
            )
            self._last_failure = None
            save_checkpoint(self)
            child = self._process_pending()
            if child is not None or self._last_failure is None:
                return child
            if not self._can_repair(attempt):
                return None
            failure = self._last_failure
            parent_code = None
            if parent_id != VIRTUAL_ROOT_ID:
                parent_code = self._tree.get_algorithm(parent_id).code
            repair_prompt = build_repair_prompt(
                task_description=self._task,
                parent_code=parent_code,
                failed_code=failure["code"],
                error=failure["error"],
                intent=None if intent is None else Intent(intent),
                maximize=self._tree.maximize,
                reliability=self._error_handling,
            )
        return None

    def _process_pending(self) -> Algorithm | None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending candidate to process")
        started = time.perf_counter()
        parsed = parse_program_response(pending.response)
        self._candidate_hash = hashlib.sha256(parsed.code.encode()).hexdigest()[:16]
        self._preflight_error = preflight_code(parsed.code, self._function.name)
        result = self._evaluator.evaluate_program_with_details(parsed.code)
        self._attempt_elapsed = time.perf_counter() - started
        self._charge_evaluation()
        if pending.parent_id != VIRTUAL_ROOT_ID:
            parent = self._tree.get_algorithm(pending.parent_id)
            parent.count += 1
            if pending.intent == Intent.REFINE.value:
                parent.refine_count += 1
            elif pending.intent == Intent.EXPLORE.value:
                parent.explore_count += 1

        if result.failure_kind == "prepare_error":
            message = one_line(result.error or "evaluator preparation failed", ERROR_MAX_CHARS)
            self._reject_pending(result.failure_kind, message, error_type=result.error_type,
                                 code=parsed.code, traceback=result.traceback)
            if self._error_retries:
                return None
            raise RuntimeError(message)

        raw_fitness = getattr(result.result, "fitness", result.result)
        try:
            fitness = float(raw_fitness)
        except (TypeError, ValueError, OverflowError):
            fitness = math.nan
        if not math.isfinite(fitness):
            return self._reject_pending(
                result.failure_kind or "invalid_result",
                one_line(result.error or f"evaluator returned invalid fitness: {raw_fitness!r}", ERROR_MAX_CHARS),
                error_type=result.error_type, code=parsed.code, traceback=result.traceback,
            )

        entry_id = pending.entry_id
        if entry_id is None:
            entry_id = self._allocate_entry()
        child = self._tree.add_algorithm(
            code=parsed.code,
            fitness=fitness,
            parent_id=pending.parent_id,
            idea=parsed.declared_idea,
            created_by=pending.intent,
            entry_id=entry_id,
        )
        self._pending = None
        if pending.mode == "landing" and pending.landing_id is not None:
            state = self._active_landing
            if state is not None and state.id == pending.landing_id:
                parent = self._tree.get_algorithm(pending.parent_id)
                state.valid_steps += 1
                gain = self._tree.quality(child) - self._tree.quality(parent)
                state.max_gain = max(state.max_gain, gain)
                if gain > 0:
                    state.strict_improvements += 1
                state.latest_valid_id = child.id
                self._mark_landing_step(pending)
        if self.best is child:
            self._n_stag = 0
        else:
            self._n_stag += 1
        self._record_evaluation(pending, child, fitness, status="ok")
        if self.best is child and self._log is not None:
            self._log.record_best(code=parsed.code, fitness=fitness)
        save_checkpoint(self)

        if pending.mode == "ordinary" and pending.intent == Intent.EXPLORE.value:
            self._maybe_start_landing(child)
        return child

    def _reject_pending(
        self, status: str, error: str, *, error_type: str | None = None,
        code: str = "", traceback: str | None = None,
    ) -> None:
        pending = self._pending
        assert pending is not None
        self._pending = None
        self._n_stag += 1
        diagnostic = error
        if self._preflight_error is not None:
            diagnostic = f"{self._preflight_error}; {error}"
        self._last_failure = {
            "code": code,
            "error": format_failure_feedback(
                error_type=error_type, error=diagnostic, traceback=traceback
            ),
        }
        self._mark_landing_step(pending)
        self._record_evaluation(pending, None, None, status=status, error=diagnostic,
                                error_type=error_type)
        save_checkpoint(self)
        return None

    def _mark_landing_step(self, pending: Pending) -> None:
        if (
            pending.mode != "landing"
            or pending.landing_id is None
            or self._attempt_kind != "initial"
        ):
            return
        state = self._active_landing
        if state is not None and state.id == pending.landing_id:
            state.completed_steps += 1

    def _record_evaluation(
        self, pending: Pending, child: Algorithm | None, fitness: float | None,
        *, status: str, error: str | None = None, error_type: str | None = None,
    ) -> None:
        if self._log is None:
            return
        self._log.record_evaluation(
            eval_count=self._n_eval,
            parent_id=pending.parent_id,
            child_id=None if child is None else child.id,
            intent=pending.intent,
            mode=pending.mode,
            entry_id=pending.entry_id if pending.entry_id is not None else (
                None if child is None else child.entry_id
            ),
            landing_id=pending.landing_id,
            landing_step=pending.landing_step,
            status=status,
            fitness=fitness,
            error=error,
            decision=self._decision,
            n_stag=self._n_stag,
            attempt=self._attempt_number,
            attempt_kind=self._attempt_kind,
            elapsed_seconds=self._attempt_elapsed,
            preflight_error=self._preflight_error,
            candidate_hash=self._candidate_hash,
            error_type=error_type,
        )

    def _maybe_start_landing(self, child: Algorithm) -> None:
        if self.landing_slots_remaining < LANDING_HORIZON:
            return
        assert child.entry_id is not None
        if child.entry_id in self._entry_tickets:
            return
        ticket = landing_ticket(seed=self._seed, entry_id=child.entry_id)
        self._entry_tickets[child.entry_id] = ticket
        self._event(
            "landing_ticket",
            entry_id=child.entry_id,
            child_id=child.id,
            parent_id=child.parent_id,
            ticket=ticket,
            eval_count=self._n_eval,
        )
        if not ticket:
            return
        assert child.parent_id is not None
        state = LandingState(
            id=self._next_landing_id,
            entry_id=child.entry_id,
            origin_id=child.id,
            origin_parent_id=child.parent_id,
            latest_valid_id=child.id,
            start_eval=self._n_eval,
        )
        self._next_landing_id += 1
        self._active_landing = state
        self._event(
            "landing_start",
            landing_id=state.id,
            entry_id=state.entry_id,
            origin_id=state.origin_id,
            origin_parent_id=state.origin_parent_id,
            start_eval=state.start_eval,
        )
        save_checkpoint(self)

    def _allocate_entry(self) -> int:
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        return entry_id

    def _event(self, event: str, **payload: object) -> None:
        if self._log is not None:
            self._log.record_landing_event(event, **payload)

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _charge_evaluation(self) -> None:
        self._n_calls += 1
        if self._attempt_kind == "initial":
            self._n_eval += 1
            if self._pending is not None and self._pending.mode == "landing":
                self._landing_slots_used += 1

    def _can_repair(self, attempt: int) -> bool:
        return attempt <= self._error_retries

    def _summary_counts(self) -> dict[str, int]:
        return {
            "evaluator_call_count": self._n_calls,
            "budget_slots": self._n_eval,
            "ordinary_decisions": self._n_ordinary_decisions,
            "landing_budget_slots": self._landing_budget,
            "landing_slots_used": self._landing_slots_used,
            "landing_tickets": sum(self._entry_tickets.values()),
            "landing_ticket_entries": len(self._entry_tickets),
            "landing_completed": self._next_landing_id - 1
            - (1 if self._active_landing is not None else 0),
        }


__all__ = ["TraceAADV916"]
