"""TraceAAD V9.18-R0 atomic search loop."""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
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
    GLOBAL_FACTS_WINDOW,
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    OPPORTUNITY_LAMBDA,
    OPPORTUNITY_TAU,
    Algorithm,
    Intent,
    Pending,
)
from .selection import Decision, decide, robust_root_scale
from .tree import Tree, VIRTUAL_ROOT_ID

ERROR_MAX_CHARS = 360
FACTS_MAX_CHARS = 900


class TraceAADV918:
    """Run the R0 atomic allocation and Global-Facts-Lite protocol."""

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
        allocation_mode: str = "q",
        explore_context: str = "legacy",
        opportunity_lambda: float = OPPORTUNITY_LAMBDA,
        opportunity_tau: float = OPPORTUNITY_TAU,
        fork_from_initialization: bool = False,
    ) -> None:
        if min(budget, n_roots, max_tokens, max_history) <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if n_roots > budget:
            raise ValueError("n_roots cannot exceed budget")
        if error_retries < 0:
            raise ValueError("error_retries must be non-negative")
        if allocation_mode not in {"q", "opportunity"}:
            raise ValueError("allocation_mode must be 'q' or 'opportunity'")
        if explore_context not in {"legacy", "facts"}:
            raise ValueError("explore_context must be 'legacy' or 'facts'")
        if opportunity_lambda < 0 or opportunity_tau <= 0:
            raise ValueError("opportunity constants are out of range")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.18 requires one evolvable template function")

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
        self._allocation_mode = allocation_mode
        self._explore_context = explore_context
        self._opportunity_lambda = opportunity_lambda
        self._opportunity_tau = opportunity_tau

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
        self._sigma_q: float | None = None
        self._last_best_eval = 0
        self._recent_outcomes: deque[str] = deque(maxlen=GLOBAL_FACTS_WINDOW)

        if resume_from is not None:
            checkpoint = load_checkpoint(
                self,
                resume_from,
                allow_protocol_mismatch=fork_from_initialization,
            )
            if fork_from_initialization:
                if self._pending is not None:
                    raise ValueError("initialization fork checkpoint cannot have pending work")
                if self._n_eval != self._n_roots:
                    raise ValueError("initialization fork must end exactly at n_roots slots")
                if self._n_ordinary_decisions != 0:
                    raise ValueError("initialization fork already contains ordinary decisions")
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self) -> Algorithm | None:
        return self._tree.best()

    @property
    def sigma_q(self) -> float | None:
        return self._sigma_q

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
            if self._sigma_q is None:
                self._sigma_q = robust_root_scale(self._tree)
                save_checkpoint(self)

            while self._has_budget():
                decision = decide(
                    self._tree,
                    seed=self._seed,
                    n_eval=self._n_eval,
                    sigma_q=self._sigma_q,
                    allocation_mode=self._allocation_mode,
                    decision_index=self._n_ordinary_decisions,
                    opportunity_lambda=self._opportunity_lambda,
                    opportunity_tau=self._opportunity_tau,
                )
                self._n_ordinary_decisions += 1
                self._decision = decision
                parent = decision.parent
                parent.n_after += 1
                if self._log is not None:
                    self._log.record_event(
                        "pre_decision",
                        eval_count=self._n_eval,
                        decision_index=decision.decision_index,
                        selected_anchor=parent.id,
                        intent=decision.intent.value,
                        operator_draw=decision.operator_draw,
                        sigma_q=decision.sigma_q,
                        allocation_mode=decision.allocation_mode,
                        parent_q=decision.parent_q,
                        selected_score=decision.selected_score,
                        opportunity=decision.opportunity,
                        n_after=parent.n_after,
                        beta=decision.beta,
                        ess=decision.ess,
                        selection_scores=dict(decision.selection_scores),
                        selection_snapshot=[
                            {
                                "algorithm_id": algorithm_id,
                                "q": q,
                                "opportunity": opportunity,
                                "n_after": n_after,
                                "score": score,
                            }
                            for algorithm_id, q, opportunity, n_after, score
                            in decision.selection_snapshot
                        ],
                    )
                save_checkpoint(self)
                prompt, facts_hash, facts_omitted = self._build_prompt(
                    parent, decision.intent
                )
                self._generate(
                    prompt,
                    parent_id=parent.id,
                    intent=decision.intent.value,
                    mode="ordinary",
                    entry_id=None,
                    facts_hash=facts_hash,
                    facts_omitted=facts_omitted,
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
                    sigma_q=self._sigma_q,
                    opportunity_lambda=self._opportunity_lambda,
                    opportunity_tau=self._opportunity_tau,
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
        if self._initialized() and self._sigma_q is None:
            self._sigma_q = robust_root_scale(self._tree)

    def _build_prompt(
        self, algorithm: Algorithm, intent: Intent
    ) -> tuple[str, str | None, bool]:
        assert algorithm.code is not None and algorithm.fitness is not None
        path = parent_path(self._tree, algorithm.id, max_events=self._max_history)
        global_facts = None
        if intent is Intent.EXPLORE and self._explore_context == "facts":
            global_facts = self._global_facts()
        prompt = build_generation_prompt(
            task_description=self._task,
            code=algorithm.code,
            fitness=algorithm.fitness,
            history_text=render_path(self._tree, path),
            intent=intent,
            maximize=self._tree.maximize,
            error_handling=self._error_handling,
            global_facts=global_facts,
        )
        facts_hash = None if global_facts is None else self._hash_text(global_facts)
        return prompt, facts_hash, False

    def _global_facts(self) -> str:
        best = self.best
        best_quality = "unavailable" if best is None else f"{self._tree.quality(best):.6g}"
        window = list(self._recent_outcomes)
        total = len(window)
        valid = sum(item in {"valid", "improve", "plateau", "regress"} for item in window)
        invalid = sum(item == "invalid" for item in window)
        duplicate = sum(item == "duplicate" for item in window)
        lines = [
            "[Global Verified Facts - R0]",
            f"Global best quality: {best_quality}",
            f"Slots since global best refresh: {max(0, self._n_eval - self._last_best_eval)}",
            (
                "Recent outcome proportions (window=32): "
                f"valid={valid / total:.3f}, invalid={invalid / total:.3f}, "
                f"duplicate={duplicate / total:.3f}"
                if total
                else "Recent outcome proportions (window=32): unavailable"
            ),
        ]
        text = "\n".join(lines)
        return text[:FACTS_MAX_CHARS]

    def _generate(
        self,
        prompt: str,
        *,
        parent_id: int,
        intent: str | None,
        mode: str,
        entry_id: int | None,
        facts_hash: str | None = None,
        facts_omitted: bool = False,
    ) -> Algorithm | None:
        repair_prompt = prompt
        for attempt in range(1, self._error_retries + 2):
            self._attempt_number = attempt
            self._attempt_kind = "initial" if attempt == 1 else "repair"
            request_seed = None if self._seed is None else self._seed + self._n_eval + 1
            kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
            if request_seed is not None:
                kwargs["seed"] = request_seed
            response = self._llm.draw_sample(repair_prompt, **kwargs)
            self._pending = Pending(
                parent_id=parent_id,
                intent=intent,
                response=response,
                mode=mode,
                entry_id=entry_id,
                decision_index=(None if self._decision is None else self._decision.decision_index),
                request_seed=request_seed,
                # A repair is a different request. Keep its hash separate so
                # audit records identify the prompt actually sent to the LLM.
                prompt_hash=self._hash_text(repair_prompt),
                prompt_chars=len(repair_prompt),
                facts_hash=facts_hash if attempt == 1 else None,
                facts_omitted=facts_omitted if attempt == 1 else False,
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
        self._candidate_hash = self._hash_text(parsed.code)[:16]
        self._preflight_error = preflight_code(parsed.code, self._function.name)
        duplicate = any(
            algorithm.code == parsed.code for algorithm in self._tree.valid_algorithms()
        )
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
                duplicate=duplicate,
                diagnosis=parsed.diagnosis,
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
                duplicate=duplicate,
                diagnosis=parsed.diagnosis,
            )

        parent_quality = None
        if pending.parent_id != VIRTUAL_ROOT_ID:
            parent_quality = self._tree.quality(self._tree.get_algorithm(pending.parent_id))
        entry_id = pending.entry_id
        is_explore_entry = (
            pending.intent == Intent.EXPLORE.value and not duplicate
        )
        if is_explore_entry:
            entry_id = self._allocate_entry()
        child = self._tree.add_algorithm(
            code=parsed.code,
            fitness=fitness,
            parent_id=pending.parent_id,
            idea=parsed.declared_idea,
            created_by=pending.intent,
            entry_id=entry_id,
            created_slot=self._n_eval,
            is_explore_entry=is_explore_entry,
        )
        self._pending = None
        outcome = "valid"
        if duplicate:
            outcome = "duplicate"
        elif parent_quality is not None:
            delta = self._tree.quality(child) - parent_quality
            outcome = "improve" if delta > 0 else "regress" if delta < 0 else "plateau"
        self._recent_outcomes.append(outcome)
        if self.best is child:
            self._n_stag = 0
            self._last_best_eval = self._n_eval
        else:
            self._n_stag += 1
        self._record_evaluation(
            pending,
            child,
            fitness,
            status="ok",
            duplicate=duplicate,
            diagnosis=parsed.diagnosis,
        )
        if self.best is child and self._log is not None:
            self._log.record_best(code=parsed.code, fitness=fitness)
        save_checkpoint(self)
        return child

    def _reject_pending(
        self,
        status: str,
        error: str,
        *,
        error_type: str | None = None,
        code: str = "",
        traceback: str | None = None,
        duplicate: bool = False,
        diagnosis: str | None = None,
    ) -> None:
        pending = self._pending
        assert pending is not None
        self._pending = None
        self._n_stag += 1
        self._recent_outcomes.append("invalid")
        diagnostic = error
        if self._preflight_error is not None:
            diagnostic = f"{self._preflight_error}; {error}"
        self._last_failure = {
            "code": code,
            "error": format_failure_feedback(
                error_type=error_type, error=diagnostic, traceback=traceback
            ),
        }
        self._record_evaluation(
            pending,
            None,
            None,
            status=status,
            error=diagnostic,
            error_type=error_type,
            duplicate=duplicate,
            diagnosis=diagnosis,
        )
        save_checkpoint(self)
        return None

    def _record_evaluation(
        self,
        pending: Pending,
        child: Algorithm | None,
        fitness: float | None,
        *,
        status: str,
        error: str | None = None,
        error_type: str | None = None,
        duplicate: bool = False,
        diagnosis: str | None = None,
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
            status=status,
            fitness=fitness,
            error=error,
            decision=self._decision,
            n_stag=self._n_stag,
            request_seed=pending.request_seed,
            prompt_hash=pending.prompt_hash,
            prompt_chars=pending.prompt_chars,
            facts_hash=pending.facts_hash,
            facts_omitted=pending.facts_omitted,
            diagnosis=diagnosis,
            attempt=self._attempt_number,
            attempt_kind=self._attempt_kind,
            elapsed_seconds=self._attempt_elapsed,
            preflight_error=self._preflight_error,
            candidate_hash=self._candidate_hash,
            duplicate=duplicate,
            error_type=error_type,
        )

    def _allocate_entry(self) -> int:
        entry_id = self._next_entry_id
        self._next_entry_id += 1
        return entry_id

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _charge_evaluation(self) -> None:
        self._n_calls += 1
        if self._attempt_kind == "initial":
            self._n_eval += 1

    def _can_repair(self, attempt: int) -> bool:
        return attempt <= self._error_retries

    def _summary_counts(self) -> dict[str, object]:
        return {
            "evaluator_call_count": self._n_calls,
            "budget_slots": self._n_eval,
            "ordinary_decisions": self._n_ordinary_decisions,
            "allocation_mode": self._allocation_mode,
            "explore_context": self._explore_context,
            "opportunity_active": self._allocation_mode == "opportunity" and bool(self._sigma_q),
            "recent_outcome_count": len(self._recent_outcomes),
        }


__all__ = ["TraceAADV918"]
