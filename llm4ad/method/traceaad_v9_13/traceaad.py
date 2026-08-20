"""TraceAAD V9.13: V9.7 search with proxy-region frontier context for Explore.

The search loop, allocation, intent mixture, parent path, and Refine prompt
are V9.7's.  The single change: once at least ``FRONTIER_ACTIVATION_EVALS``
real evaluations have completed, an Explore decision whose frozen treatment
is FP or FC appends the searched-region frontier (and for FC one
different-region frontier program) as a global-facts section of the Explore
prompt.  Refine prompts never receive global context and stay byte-identical
to V9.7.
"""

from __future__ import annotations

import copy
import hashlib
import math
import statistics
from dataclasses import dataclass
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
from .checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from .forest import Forest, is_better
from .history import drop_oldest, one_line, parent_path, render_path
from .prompt import (
    ProgramResponseError,
    build_generation_prompt,
    build_root_prompt,
    parse_program_response,
)
from .regions import (
    PROXY_RULES_VERSION,
    PROXY_TASKS,
    RegionView,
)
from .schema import (
    FRONTIER_ACTIVATION_EVALS,
    INITIAL_ROOT_COUNT,
    INTENT_SCHEDULE_ID,
    MAX_HISTORY_EVENTS,
    PROTOCOL_ID,
    REFINE_PROBABILITY,
    Anchor,
    Attempt,
    Intent,
    Outcome,
    Pending,
    Program,
    Treatment,
)
from .selection import Choice, select
from .source import code_diff

TRANSPORT_RETRIES = 3
ERROR_MAX_CHARS = 360


def draw_intent(seed: int | None, iteration: int) -> Intent:
    """Deterministic fixed-mixture intent on the inherited V9.7 schedule."""
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(
        f"{INTENT_SCHEDULE_ID}:intent:{token}:{iteration}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return Intent.REFINE if value < REFINE_PROBABILITY else Intent.EXPLORE


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One fully built prompt plus the treatment facts it carries."""

    prompt: str
    treatment: str
    frontier_rows: tuple[dict[str, Any], ...] = ()


class TraceAADV913:
    """Run the complete V9.13 search forest lifecycle."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
        budget: int = 1000,
        *,
        task_key: str,
        treatment: str = Treatment.FP.value,
        n_roots: int = INITIAL_ROOT_COUNT,
        maximize: bool = True,
        max_tokens: int = 8192,
        context_limit: int | None = None,
        max_history: int = MAX_HISTORY_EVENTS,
        seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        debug_mode: bool = False,
    ) -> None:
        if budget <= 0 or n_roots <= 0 or max_tokens <= 0 or max_history <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if context_limit is None or context_limit <= 0:
            raise ValueError("context_limit must be explicitly positive")
        if task_key not in PROXY_TASKS:
            raise ValueError(f"V9.13 has no frozen proxy rules for task: {task_key}")
        if treatment not in set(item.value for item in Treatment):
            raise ValueError(f"unknown V9.13 treatment: {treatment}")
        if (
            evaluation.use_numba_accelerate
            or evaluation.use_protected_div
            or evaluation.random_seed is not None
        ):
            raise ValueError("V9.13 requires candidate code to be executed unchanged")

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.13 requires one evolvable template function")

        self._llm = llm
        self._log = artifacts
        self._task = evaluation.task_description
        self._task_key = task_key
        self._treatment = treatment
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
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        llm.debug_mode = debug_mode

        self._forest = Forest(maximize=self._maximize)
        self._regions = RegionView(task_key)
        self._treatment_counters: dict[str, int] = {
            "explore_pp": 0,
            "explore_fp": 0,
        }
        self._pending: Pending | None = None
        self._n_candidates = 0
        self._n_eval = 0
        self._iteration = 0
        self._initialization_complete = False
        self._bootstrapped: set[int] = set()
        self._bootstrap_deltas: list[float] = []
        self._s: float | None = None
        self._best_id: int | None = None

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
            "protocol_id": PROTOCOL_ID,
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "budget": self._budget,
            "n_roots": self._n_roots,
            "max_history": self._max_history,
            "maximize": self._maximize,
            "max_tokens": self._max_tokens,
            "context_limit": self._context_limit,
            "seed": self._seed,
            "refine_probability": REFINE_PROBABILITY,
            "explore_probability": 1.0 - REFINE_PROBABILITY,
            "task_key": self._task_key,
            "treatment": self._treatment,
            "frontier_activation_evals": FRONTIER_ACTIVATION_EVALS,
            "proxy_rules_version": PROXY_RULES_VERSION,
        }

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
                self._log_choice(choice, intent)
                request = self._prompt(choice.anchor_id, intent)
                self._generate(
                    request,
                    anchor_id=choice.anchor_id,
                    stage="search",
                    iteration=self._iteration,
                    intent=intent.value,
                )

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
                best = (
                    None
                    if self._best_id is None
                    else self._forest.get_program(self._best_id)
                )
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
                    task_key=self._task_key,
                    treatment=self._treatment,
                    frontier_activation_evals=FRONTIER_ACTIVATION_EVALS,
                    visited_proxy_regions=len(self._regions.visited_families()),
                    treatment_counters=dict(self._treatment_counters),
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _initialize(self) -> None:
        while len(self._forest.root_ids) < self._n_roots and self._has_budget():
            prompt = build_root_prompt(
                task_description=self._task,
                template_function=self._function,
                maximize=self._maximize,
            )
            if not self._fits(prompt):
                raise RuntimeError("root prompt plus output bound exceeds context limit")
            self._generate(
                GenerationRequest(prompt=prompt, treatment=Treatment.PP.value),
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

    def _build_region_view(self) -> RegionView:
        view = RegionView(self._task_key)
        for program in sorted(self._forest.programs(), key=lambda item: item.id):
            view.record(program)
        return view

    def _global_context(
        self, anchor_id: int, intent: Intent
    ) -> tuple[str | None, str, tuple[dict[str, Any], ...]]:
        """Return (context text, applied treatment, frontier row facts)."""

        anchor_family = self._regions.frontier_of_code(
            self._forest.get_program(self._forest.get_anchor(anchor_id).program_id).code
        )[1]
        if (
            intent is not Intent.EXPLORE
            or self._treatment == Treatment.PP.value
            or self._n_eval < FRONTIER_ACTIVATION_EVALS
        ):
            return None, Treatment.PP.value, ()
        rows = self._region_row_facts(anchor_family)
        return self._regions.frontier_text(anchor_family), Treatment.FP.value, rows

    def _region_row_facts(self, anchor_family: str) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "family": row.family,
                "program_id": row.program_id,
                "tags": sorted(row.tags),
                "q": row.q,
                "is_anchor_region": row.family == anchor_family,
            }
            for row in self._regions.frontier_rows()
        )

    def _prompt(self, anchor_id: int, intent: Intent) -> GenerationRequest:
        program = self._forest.get_program(
            self._forest.get_anchor(anchor_id).program_id
        )
        selected = parent_path(self._forest, anchor_id, max_events=self._max_history)
        shown = selected
        global_context, treatment, row_facts = self._global_context(anchor_id, intent)
        while True:
            prompt = build_generation_prompt(
                task_description=self._task,
                code=program.code,
                fitness=program.fitness,
                history_text=render_path(self._forest, shown),
                intent=intent,
                maximize=self._maximize,
                global_context=global_context,
            )
            if self._fits(prompt):
                self._decision(
                    "history_built",
                    anchor_id=anchor_id,
                    intent=intent.value,
                    treatment=treatment,
                    selected_event_ids=selected,
                    shown_event_ids=shown,
                    dropped_for_context=len(selected) - len(shown),
                    prompt_tokens=self._tokens(prompt),
                )
                if intent is Intent.EXPLORE and self._treatment != Treatment.PP.value:
                    self._decision(
                        "frontier_built",
                        anchor_id=anchor_id,
                        iteration=self._iteration,
                        intent=intent.value,
                        requested_treatment=self._treatment,
                        applied_treatment=treatment,
                        active=self._n_eval >= FRONTIER_ACTIVATION_EVALS,
                        evals_completed=self._n_eval,
                        region_rows=list(row_facts),
                    )
                return GenerationRequest(
                    prompt=prompt,
                    treatment=treatment,
                    frontier_rows=row_facts,
                )
            if not shown:
                raise RuntimeError(
                    "task, current code, treatment context, and output budget "
                    "exceed the context window even with no history events"
                )
            shown = drop_oldest(shown)

    def _generate(
        self,
        request: GenerationRequest,
        *,
        anchor_id: int | None,
        stage: str,
        iteration: int | None,
        intent: str | None,
    ) -> Attempt:
        if self._pending is not None:
            raise RuntimeError("cannot request a candidate while one is pending")
        prompt = request.prompt
        prompt_tokens = self._tokens(prompt)
        generation_seed = (
            None if self._seed is None else self._seed + self._n_candidates + 1
        )
        if stage == "search" and intent == Intent.EXPLORE.value:
            self._treatment_counters[f"explore_{request.treatment}"] += 1
        response = self._draw(
            prompt,
            generation_seed,
            stage=stage,
            iteration=iteration,
            anchor_id=anchor_id,
            intent=intent,
            prompt_tokens=prompt_tokens,
            treatment=request.treatment,
        )
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
            treatment=request.treatment,
        )
        save_checkpoint(self)
        if self._log is not None:
            self._log.record_llm_call(
                stage=stage,
                iteration=iteration,
                anchor_id=anchor_id,
                intent=intent,
                treatment=request.treatment,
                order=self._pending.order,
                prompt_tokens=prompt_tokens,
                response_tokens=self._tokens(response),
                status="ok",
                prompt=prompt,
                generation_seed=generation_seed,
            )
        return self._process_pending()

    def _draw(
        self,
        prompt: str,
        seed: int | None,
        *,
        stage: str,
        iteration: int | None,
        anchor_id: int | None,
        intent: str | None,
        prompt_tokens: int,
        treatment: str,
    ) -> str:
        last_error: Exception | None = None
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                return self._llm.draw_sample(prompt, **kwargs)
            except Exception as exc:
                last_error = exc
                save_checkpoint(self)
                if self._log is not None:
                    self._log.record_llm_call(
                        stage=stage,
                        iteration=iteration,
                        anchor_id=anchor_id,
                        intent=intent,
                        treatment=treatment,
                        prompt_tokens=prompt_tokens,
                        response_tokens=0,
                        status="transport",
                        prompt=prompt,
                        generation_seed=seed,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        transport_attempt=attempt + 1,
                    )
        raise RuntimeError("model transport retry limit exhausted") from last_error

    def _process_pending(self) -> Attempt:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending candidate to process")

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
        added = 0
        removed = 0
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
        program, child, kind = self._place(
            pending=pending,
            parent_anchor=parent_anchor,
            parent=parent,
            existing=existing,
            code=code,
            fitness=fitness,
            error=error,
        )
        if program is not None:
            self._regions.record(program)
        dq = None if parent is None or program is None else program.q - parent.q
        outcome = _outcome(parent is not None, invalid=kind == "invalid", dq=dq)
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
            treatment=pending.treatment,
        )
        self._forest.add_attempt(attempt)

        if pending.stage == "bootstrap" and pending.anchor_id is not None:
            self._bootstrapped.add(pending.anchor_id)
            if child is not None and dq is not None:
                self._bootstrap_deltas.append(abs(dq))
        if pending.stage == "search" and pending.iteration is not None:
            self._iteration = max(self._iteration, pending.iteration + 1)

        is_new_best, _ = self._update_best(program)
        if (
            is_new_best
            and program is not None
            and self._log is not None
        ):
            self._log.record_best(
                code=program.code,
                fitness=program.fitness,
                eval_count=self._n_eval,
                iteration=pending.iteration,
                order=pending.order,
                program_id=program.id,
            )
        self._pending = None
        save_checkpoint(self)
        self._record_attempt(
            attempt,
            response=pending.response,
            code=code,
            error=error,
            evaluated=evaluated,
            is_new_best=is_new_best,
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

    def _update_best(self, program: Program | None) -> tuple[bool, str | None]:
        if program is None:
            return False, None
        incumbent = (
            None if self._best_id is None else self._forest.get_program(self._best_id)
        )
        if not is_better(program, incumbent):
            return False, None
        reason = (
            "strict_fitness" if incumbent is None or program.q > incumbent.q else "tie_break"
        )
        self._best_id = program.id
        self._decision(
            "best_updated",
            program_id=program.id,
            sample_order=program.order,
            reason=reason,
        )
        return True, reason

    def _record_attempt(
        self,
        attempt: Attempt,
        *,
        response: str,
        code: str | None,
        error: str | None,
        evaluated: bool,
        is_new_best: bool = False,
    ) -> None:
        if self._log is None:
            return
        status = "ok"
        if attempt.kind == "invalid":
            status = "parse_failed" if not evaluated else "eval_failed"

        route_id = None
        if attempt.anchor_id is not None:
            route_id = self._forest.get_anchor(attempt.anchor_id).root_id
        elif attempt.child_id is not None:
            route_id = self._forest.get_anchor(attempt.child_id).root_id

        best = (
            None
            if self._best_id is None
            else self._forest.get_program(self._best_id)
        )
        best_fitness = None if best is None else best.fitness

        self._log.record_candidate(
            attempt_id=attempt.id,
            order=attempt.order,
            stage=attempt.stage,
            iteration=attempt.iteration,
            anchor_id=attempt.anchor_id,
            child_id=attempt.child_id,
            program_id=attempt.program_id,
            intent=attempt.intent,
            treatment=attempt.treatment,
            idea=attempt.idea,
            kind=attempt.kind,
            outcome=attempt.outcome,
            evaluator_called=evaluated,
            status=status,
            parent_fitness=attempt.parent_fitness,
            child_fitness=attempt.child_fitness,
            dq=attempt.dq,
            added=attempt.added,
            removed=attempt.removed,
            diff=attempt.diff,
            program=code or "",
            raw_response=response,
            error=error,
            eval_count=self._n_eval,
            route_id=route_id,
            best_fitness=best_fitness,
            is_new_best=is_new_best,
            budget=self._budget,
        )

    def _log_choice(self, choice: Choice, intent: Intent) -> None:
        program_id = self._forest.get_anchor(choice.anchor_id).program_id
        chosen = next(item for item in choice.anchors if item.anchor_id == choice.anchor_id)
        self._decision(
            "route_selected",
            iteration=self._iteration,
            selected_root_state_id=choice.route_id,
            routes=[
                {
                    "root_state_id": item.route_id,
                    "best_q": item.q,
                    "n": item.n,
                    "optimism": item.optimism,
                    "score": item.score,
                }
                for item in choice.routes
            ],
        )
        self._decision(
            "anchor_selected",
            iteration=self._iteration,
            route_id=choice.route_id,
            selected_state_id=choice.anchor_id,
            selected_artifact_id=program_id,
            selected_score=chosen.score,
            states=[
                {
                    "state_id": item.anchor_id,
                    "artifact_id": self._forest.get_anchor(item.anchor_id).program_id,
                    "q": item.q,
                    "n": item.n,
                    "optimism": item.optimism,
                    "score": item.score,
                    "creation_order": self._forest.get_anchor(item.anchor_id).order,
                }
                for item in choice.anchors
            ],
        )

    def _decision(self, event: str, **payload: Any) -> None:
        if self._log is not None:
            self._log.record_decision(event, **payload)

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


__all__ = [
    "GenerationRequest",
    "TraceAADV913",
    "draw_intent",
]
