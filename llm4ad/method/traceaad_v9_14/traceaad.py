"""TraceAAD V9.14 search loop."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .history import one_line, parent_path, render_path
from .prompt import build_generation_prompt, build_root_prompt, parse_program_response
from .schema import (
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    REFINE_PROBABILITY,
    Algorithm,
    Intent,
    Pending,
)
from .selection import select
from .tree import Tree, VIRTUAL_ROOT_ID

ERROR_MAX_CHARS = 360


def draw_intent(seed: int | None, evaluator_count: int) -> Intent:
    value = random.Random(f"{seed}:{evaluator_count}").random()
    return Intent.REFINE if value < REFINE_PROBABILITY else Intent.EXPLORE


class TraceAADV914:
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
        if min(budget, n_roots, max_tokens, max_history) <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.14 requires one evolvable template function")

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

        self._tree = Tree(maximize=maximize)
        self._pending: Pending | None = None
        self._n_eval = 0

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
                parent = select(self._tree)
                intent = draw_intent(self._seed, self._n_eval)
                self._generate(
                    self._prompt(parent, intent),
                    parent_id=parent.id,
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
                    best_algorithm_id=None if best is None else best.id,
                    best_score=None if best is None else best.fitness,
                    evaluator_call_count=self._n_eval,
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
        while not self._initialized() and self._has_budget():
            self._generate(
                build_root_prompt(
                    task_description=self._task,
                    template_function=self._function,
                    maximize=self._tree.maximize,
                ),
                parent_id=VIRTUAL_ROOT_ID,
                intent=None,
            )

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
        )

    def _generate(
        self,
        prompt: str,
        *,
        parent_id: int,
        intent: str | None,
    ) -> Algorithm | None:
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if self._seed is not None:
            kwargs["seed"] = self._seed + self._n_eval + 1
        response = self._llm.draw_sample(prompt, **kwargs)
        self._pending = Pending(parent_id, intent, response)
        save_checkpoint(self)
        return self._process_pending()

    def _process_pending(self) -> Algorithm | None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending candidate to process")

        parsed = parse_program_response(pending.response)
        result = self._evaluator.evaluate_program_with_details(parsed.code)
        self._n_eval += 1
        if pending.parent_id != VIRTUAL_ROOT_ID:
            self._tree.get_algorithm(pending.parent_id).count += 1

        if result.failure_kind == "prepare_error":
            message = one_line(
                result.error or "evaluator preparation failed", ERROR_MAX_CHARS
            )
            self._reject_pending(result.failure_kind, message)
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
            )

        child = self._tree.add_algorithm(
            code=parsed.code,
            fitness=fitness,
            parent_id=pending.parent_id,
            idea=parsed.declared_idea,
        )
        self._pending = None
        save_checkpoint(self)
        if self._log is not None:
            self._log.record_evaluation(
                eval_count=self._n_eval,
                parent_id=pending.parent_id,
                child_id=child.id,
                intent=pending.intent,
                status="ok",
                fitness=fitness,
                error=None,
            )
            if self.best is child:
                self._log.record_best(code=parsed.code, fitness=fitness)
        return child

    def _reject_pending(self, status: str, error: str) -> None:
        pending = self._pending
        assert pending is not None
        self._pending = None
        save_checkpoint(self)
        if self._log is not None:
            self._log.record_evaluation(
                eval_count=self._n_eval,
                parent_id=pending.parent_id,
                child_id=None,
                intent=pending.intent,
                status=status,
                fitness=None,
                error=error,
            )
        return None

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget


__all__ = ["TraceAADV914", "draw_intent"]
