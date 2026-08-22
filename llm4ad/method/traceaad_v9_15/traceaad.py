"""TraceAAD V9.15 search loop."""

from __future__ import annotations

import math
import hashlib
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
    MAX_HISTORY_EVENTS,
    Algorithm,
    Intent,
    Pending,
)
from .selection import Decision, decide
from .tree import Tree, VIRTUAL_ROOT_ID

ERROR_MAX_CHARS = 360


class TraceAADV915:
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
        if error_retries < 0:
            raise ValueError("error_retries must be non-negative")

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.15 requires one evolvable template function")

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

        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self) -> Algorithm | None:
        return self._tree.best()

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
                decision = decide(
                    self._tree,
                    n_stag=self._n_stag,
                    seed=self._seed,
                    n_eval=self._n_eval,
                )
                self._decision = decision
                self._generate(
                    self._prompt(decision.parent, decision.intent),
                    parent_id=decision.parent.id,
                    intent=decision.intent.value,
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
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _initialized(self) -> bool:
        return len(self._tree.root_algorithms()) == self._n_roots

    def _initialize(self) -> None:
        if self._initialized():
            return  # resumed mid-search: keep the checkpointed stagnation state
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

    def _generate(
        self,
        prompt: str,
        *,
        parent_id: int,
        intent: str | None,
    ) -> Algorithm | None:
        repair_prompt = prompt
        for attempt in range(1, self._error_retries + 2):
            self._attempt_number = attempt
            self._attempt_kind = "initial" if attempt == 1 else "repair"
            kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
            if self._seed is not None:
                kwargs["seed"] = self._seed + self._n_eval + 1
            response = self._llm.draw_sample(repair_prompt, **kwargs)
            self._pending = Pending(parent_id, intent, response)
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
                parent = self._tree.get_algorithm(parent_id)
                parent_code = parent.code
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
            message = one_line(
                result.error or "evaluator preparation failed", ERROR_MAX_CHARS
            )
            self._reject_pending(
                result.failure_kind,
                message,
                error_type=result.error_type,
                code=parsed.code,
                traceback=result.traceback,
            )
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
                one_line(
                    result.error or f"evaluator returned invalid fitness: {raw_fitness!r}",
                    ERROR_MAX_CHARS,
                ),
                error_type=result.error_type,
                code=parsed.code,
                traceback=result.traceback,
            )

        child = self._tree.add_algorithm(
            code=parsed.code,
            fitness=fitness,
            parent_id=pending.parent_id,
            idea=parsed.declared_idea,
            created_by=pending.intent,
        )
        self._pending = None
        if self.best is child:
            self._n_stag = 0
        else:
            self._n_stag += 1
        save_checkpoint(self)  # initialization outcomes are reset by _initialize
        if self._log is not None:
            self._log.record_evaluation(
                eval_count=self._n_eval,
                parent_id=pending.parent_id,
                child_id=child.id,
                intent=pending.intent,
                status="ok",
                fitness=fitness,
                error=None,
                decision=self._decision,
                n_stag=self._n_stag,
                attempt=self._attempt_number,
                attempt_kind=self._attempt_kind,
                elapsed_seconds=self._attempt_elapsed,
                preflight_error=self._preflight_error,
                candidate_hash=self._candidate_hash,
            )
            if self.best is child:
                self._log.record_best(code=parsed.code, fitness=fitness)
        return child

    def _reject_pending(
        self,
        status: str,
        error: str,
        *,
        error_type: str | None = None,
        code: str = "",
        traceback: str | None = None,
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
                error_type=error_type,
                error=diagnostic,
                traceback=traceback,
            ),
        }
        save_checkpoint(self)
        if self._log is not None:
            self._log.record_evaluation(
                eval_count=self._n_eval,
                parent_id=pending.parent_id,
                child_id=None,
                intent=pending.intent,
                status=status,
                fitness=None,
                error=diagnostic,
                decision=self._decision,
                n_stag=self._n_stag,
                attempt=self._attempt_number,
                attempt_kind=self._attempt_kind,
                elapsed_seconds=self._attempt_elapsed,
                preflight_error=self._preflight_error,
                candidate_hash=self._candidate_hash,
                error_type=error_type,
            )
        return None

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _charge_evaluation(self) -> None:
        self._n_calls += 1
        if self._attempt_kind == "initial":
            self._n_eval += 1

    def _can_repair(self, attempt: int) -> bool:
        return attempt <= self._error_retries

    def _summary_counts(self) -> dict[str, int]:
        return {"evaluator_call_count": self._n_calls, "budget_slots": self._n_eval}


__all__ = ["TraceAADV915"]
