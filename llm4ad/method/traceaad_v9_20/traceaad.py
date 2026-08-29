"""TraceAAD V9.20: opportunity allocation plus action-matched assistance."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from . import behave
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .history import formation_path_records, one_line, render_attempt_ledger, render_path
from .landscape import Landscape, behavior_tag, region_statistics
from .prompt import build_generation_prompt, build_repair_prompt, build_root_prompt, format_failure_feedback, parse_program_response
from .schema import (
    ACTION_TEMPERATURE,
    COVERAGE_MIX,
    ESS_FRACTION,
    EXPLORE_MAX,
    EXPLORE_MIN,
    EXPLORE_NEUTRAL,
    INITIAL_ROOT_COUNT,
    MAX_REPAIRS,
    MIN_ESS_TARGET,
    Action,
    Algorithm,
    Attempt,
    Pending,
)
from .selection import decide_action, reference_utility, sample_parent
from .tree import Tree, VIRTUAL_ROOT_ID

ERROR_MAX_CHARS = 360


class TraceAADV920:
    """Search measured algorithm states under a finite primary budget."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
        budget: int = 1000,
        *,
        n_roots: int = INITIAL_ROOT_COUNT,
        max_tokens: int = 8192,
        seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        if min(budget, n_roots, max_tokens) <= 0 or n_roots > budget:
            raise ValueError("budget, n_roots, and max_tokens must be positive")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.20 requires one evolvable template function")
        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation)
        self._log = artifacts
        self._task = evaluation.task_description
        self._function: Function = template.functions[0]
        self._budget = budget
        self._n_roots = n_roots
        self._max_tokens = max_tokens
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._task_key = behave.detect_task(evaluation)
        self._protocol = behave.build_protocol(self._task_key)
        self._landscape = Landscape(task=self._task_key, protocol=self._protocol)
        self._tree = Tree()
        self._pending: Pending | None = None
        self._attempts: list[Attempt] = []
        self._n_eval = 0
        self._n_calls = 0
        self._repair_llm_calls = 0
        self._repair_eval_calls = 0
        self._n_ordinary_decisions = 0
        self._checkpoint_behave_size = 0
        if resume_from is not None:
            load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = Path(resume_from).parent

    @property
    def best(self) -> Algorithm | None:
        return self._tree.best()

    @property
    def _outcome_counts(self) -> Counter[str]:
        return Counter(attempt.outcome for attempt in self._attempts)

    @property
    def _action_counts(self) -> Counter[str]:
        return Counter(attempt.action for attempt in self._attempts if attempt.action is not None)

    def run(self) -> None:
        status, stop_reason, error = "error", None, {}
        try:
            if self._pending is not None:
                self._settle(attempt=self._pending.attempt, repairable=False)
            while not self._initialized() and self._has_budget():
                self._attempt(
                    build_root_prompt(task_description=self._task, template_function=self._function),
                    parent_id=VIRTUAL_ROOT_ID,
                    action=None,
                    mode="initialization",
                )
            if not self._initialized():
                status = "initialization_failure"
                stop_reason = "evaluator_budget_exhausted_during_initialization"
                return
            while self._has_budget():
                self._ordinary_slot()
            status, stop_reason = "finished", "evaluator_budget_exhausted"
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
                    task=self._task_key,
                    budget_slots=self._n_eval,
                    evaluator_call_count=self._n_calls,
                    repair_llm_calls=self._repair_llm_calls,
                    repair_evaluator_calls=self._repair_eval_calls,
                    ordinary_decisions=self._n_ordinary_decisions,
                    n_algorithms=len(self._tree.valid_algorithms()),
                    n_roots=len(self._tree.root_algorithms()),
                    allocation_policy="quality_continuation_plus_behavior_coverage",
                    coverage_mix=COVERAGE_MIX,
                    ess_fraction=ESS_FRACTION,
                    min_ess_target=MIN_ESS_TARGET,
                    action_temperature=ACTION_TEMPERATURE,
                    explore_probability={"neutral": EXPLORE_NEUTRAL, "min": EXPLORE_MIN, "max": EXPLORE_MAX},
                    behave_protocol_id=self._protocol["protocol_id"],
                    action_counts=dict(sorted(self._action_counts.items())),
                    outcome_counts=dict(sorted(self._outcome_counts.items())),
                    **error,
                )
                self._log.finish()

    def _initialized(self) -> bool:
        return len(self._tree.root_algorithms()) == self._n_roots

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _ordinary_slot(self) -> None:
        valid = self._tree.valid_algorithms()
        quality = {algorithm.id: self._tree.quality(algorithm) for algorithm in valid}
        stats = region_statistics(
            landscape=self._landscape,
            quality=quality,
            opportunities={algorithm.id: algorithm.opportunities for algorithm in valid},
        )
        decision_index = self._n_ordinary_decisions
        rng = random.Random(f"{self._seed}:v9_20:{decision_index}")
        parent_decision = sample_parent(tree=self._tree, stats=stats, rng=rng)
        parent = parent_decision.parent
        candidate_reference_id = None
        if len(valid) > 1:
            candidate_reference_id = self._landscape.select_crossover_reference(parent.id, quality, rng)
        candidate_reference_value = reference_utility(
            parent_id=parent.id,
            reference_id=candidate_reference_id,
            quality=quality,
            landscape=self._landscape,
        )
        action_decision = decide_action(
            algorithm=parent,
            reference_value=candidate_reference_value,
            rng=rng,
        )
        self._n_ordinary_decisions += 1
        reference_id = candidate_reference_id if action_decision.action is Action.CROSSOVER else None
        reference = None if reference_id is None else self._tree.get_algorithm(reference_id)
        selected = next(item for item in parent_decision.snapshot if item["id"] == parent.id)
        distance = None if reference is None else self._landscape.distance(parent.id, reference.id)
        if self._log is not None:
            self._log.record_event(
                "pre_decision",
                decision_index=decision_index,
                slot=self._n_eval + 1,
                parent_id=parent.id,
                beta=parent_decision.beta,
                ess=parent_decision.ess,
                marker=parent_decision.marker,
                snapshot=list(parent_decision.snapshot),
            )
            self._log.record_event(
                "action_decision",
                decision_index=decision_index,
                slot=self._n_eval + 1,
                parent_id=parent.id,
                utilities=action_decision.utilities,
                probabilities=action_decision.probabilities,
                action=action_decision.action.value,
                action_draw=action_decision.draw,
                reference_candidate_id=candidate_reference_id,
                reference_value=candidate_reference_value,
            )
            if action_decision.action is Action.CROSSOVER:
                self._log.record_event(
                    "crossover_reference",
                    decision_index=decision_index,
                    slot=self._n_eval + 1,
                    parent_id=parent.id,
                    reference_id=reference_id,
                    distance=distance,
                )
        prompt = build_generation_prompt(
            task_description=self._task,
            code=parent.code or "",
            fitness=parent.fitness if parent.fitness is not None else 0.0,
            history_text=render_path(self._tree, parent.id),
            action=action_decision.action,
            reference_code=None if reference is None else reference.code,
            reference_fitness=None if reference is None else reference.fitness,
            reference_behavior=None if reference is None else reference.behavior_tag,
            reference_distance=distance,
            context_mode=action_decision.action.value,
            attempt_summary=render_attempt_ledger(parent),
            reference_history=None if reference is None else render_path(self._tree, reference.id),
        )
        self._attempt(
            prompt,
            parent_id=parent.id,
            action=action_decision.action,
            mode=action_decision.action.value,
            reference_id=reference_id,
            audit={
                "decision_index": decision_index,
                "quality_value": float(selected["Q"]),
                "continuation_value": float(selected["C"]),
                "coverage_value": float(selected["B"]),
                "allocation_probability": float(selected["pi"]),
                "action_probabilities": action_decision.probabilities,
                "reference_value": candidate_reference_value,
            },
        )

    def _attempt(
        self,
        prompt: str,
        *,
        parent_id: int,
        action: Action | None,
        mode: str,
        reference_id: int | None = None,
        audit: dict[str, Any] | None = None,
    ) -> None:
        audit = audit or {}
        for attempt in range(1, MAX_REPAIRS + 2):
            if attempt > 1:
                self._repair_llm_calls += 1
            request_seed = None if self._seed is None else self._seed + self._n_eval + 1
            kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
            if request_seed is not None:
                kwargs["seed"] = request_seed
            self._pending = Pending(
                parent_id=parent_id,
                action=None if action is None else action.value,
                response=str(self._llm.draw_sample(prompt, **kwargs)),
                exact_prompt=prompt,
                mode=mode,
                request_seed=request_seed,
                attempt=attempt,
                decision_index=audit.get("decision_index"),
                quality_value=audit.get("quality_value"),
                continuation_value=audit.get("continuation_value"),
                coverage_value=audit.get("coverage_value"),
                allocation_probability=audit.get("allocation_probability"),
                action_probabilities=audit.get("action_probabilities"),
                reference_id=reference_id,
                reference_value=audit.get("reference_value"),
            )
            save_checkpoint(self)
            failure = self._settle(attempt=attempt, repairable=attempt <= MAX_REPAIRS)
            if failure is None:
                return
            parent_code = None if parent_id == VIRTUAL_ROOT_ID else self._tree.get_algorithm(parent_id).code
            reference_code = None if reference_id is None else self._tree.get_algorithm(reference_id).code
            reference_fitness = None if reference_id is None else self._tree.get_algorithm(reference_id).fitness
            prompt = build_repair_prompt(
                task_description=self._task,
                parent_code=parent_code,
                failed_code=failure[0],
                error=failure[1],
                action=action,
                reference_code=reference_code,
                reference_fitness=reference_fitness,
            )

    def _settle(self, *, attempt: int, repairable: bool) -> tuple[str, str] | None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("cannot settle without a pending candidate")
        parsed = parse_program_response(pending.response)
        started = time.perf_counter()
        result = self._evaluator.evaluate_program_with_details(parsed.code)
        elapsed = time.perf_counter() - started
        self._n_calls += 1
        if attempt > 1:
            self._repair_eval_calls += 1

        def reject(outcome: str, status: str, message: str) -> tuple[str, str] | None:
            self._finish(
                attempt=attempt,
                outcome=outcome,
                status=status,
                fitness=None,
                child=None,
                error=message,
                error_type=result.error_type,
                elapsed=elapsed,
                parsed=parsed,
                continuing=repairable,
            )
            if not repairable:
                return None
            return parsed.code, format_failure_feedback(
                error_type=result.error_type,
                error=message,
                traceback=result.traceback,
                task_key=self._task_key,
            )

        if result.failure_kind is not None:
            message = one_line(result.error or f"evaluation failed: {result.failure_kind}", ERROR_MAX_CHARS)
            outcome = "timeout" if result.failure_kind == "timeout" else "invalid"
            return reject(outcome, result.failure_kind, message)
        raw_fitness = getattr(result.result, "fitness", result.result)
        try:
            fitness = float(raw_fitness)
        except (TypeError, ValueError, OverflowError):
            fitness = math.nan
        if not math.isfinite(fitness):
            message = one_line(result.error or f"evaluator returned invalid fitness: {raw_fitness!r}", ERROR_MAX_CHARS)
            return reject("invalid", "invalid_result", message)
        duplicate = any(node.code == parsed.code for node in self._tree.valid_algorithms())
        if duplicate:
            self._finish(
                attempt=attempt,
                outcome="duplicate",
                status="ok",
                fitness=fitness,
                child=None,
                error=None,
                error_type=None,
                elapsed=elapsed,
                parsed=parsed,
                continuing=False,
            )
            return None
        parent = None if pending.parent_id == VIRTUAL_ROOT_ID else self._tree.get_algorithm(pending.parent_id)
        trajectories = getattr(result.result, "trajectories", None)
        if trajectories is None:
            raise TypeError("TraceAAD V9.20 requires tracked evaluation trajectories")
        if parent is None:
            novelty, distances, tag = None, None, None
        else:
            novelty, distances = self._landscape.archive_novelty(trajectories)
            tag = behavior_tag(novelty)
        previous_best = self._tree.best_quality()
        child = self._tree.add_algorithm(
            code=parsed.code,
            fitness=fitness,
            parent_id=pending.parent_id,
            idea=parsed.declared_idea,
            action=pending.action,
            created_slot=self._n_eval + (1 if attempt == 1 else 0),
            novelty=novelty,
            behavior_tag=tag,
        )
        self._landscape.add(child.id, trajectories, distances=distances)
        outcome = "plateau" if parent is None or fitness == self._tree.quality(parent) else "improve" if fitness > self._tree.quality(parent) else "regress"
        self._finish(
            attempt=attempt,
            outcome=outcome,
            status="ok",
            fitness=fitness,
            child=child,
            error=None,
            error_type=None,
            elapsed=elapsed,
            parsed=parsed,
            continuing=False,
        )
        if self._log is not None:
            self._log.record_event(
                "new_node",
                slot=self._n_eval,
                node_id=child.id,
                parent_id=pending.parent_id,
                action=pending.action,
                fitness=fitness,
                outcome=outcome,
                novelty=novelty,
                behavior=tag,
            )
            if previous_best is None or fitness > previous_best:
                self._log.record_best(
                    code=parsed.code,
                    fitness=fitness,
                    slot=self._n_eval,
                    child_id=child.id,
                    idea=parsed.declared_idea,
                    action=pending.action,
                    novelty=novelty,
                    behavior=tag,
                )
        return None

    def _finish(
        self,
        *,
        attempt: int,
        outcome: str,
        status: str,
        fitness: float | None,
        child: Algorithm | None,
        error: str | None,
        error_type: str | None,
        elapsed: float,
        parsed,
        continuing: bool,
    ) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("cannot finish without a pending candidate")
        if attempt == 1:
            self._n_eval += 1
        slot = self._n_eval
        previous = next((item for item in self._attempts if item.slot == slot), None)
        self._attempts = [item for item in self._attempts if item.slot != slot]
        self._attempts.append(Attempt(slot, pending.parent_id, pending.action, outcome, pending.reference_id))
        if pending.parent_id != VIRTUAL_ROOT_ID:
            parent = self._tree.get_algorithm(pending.parent_id)
            if previous is None:
                parent.opportunities += 1
            elif previous.outcome == "improve":
                parent.improvements = max(0, parent.improvements - 1)
            else:
                parent.failures = max(0, parent.failures - 1)
            if outcome == "improve":
                parent.improvements += 1
            else:
                parent.failures += 1
            parent.last_outcome = outcome
        parent = None if pending.parent_id == VIRTUAL_ROOT_ID else self._tree.get_algorithm(pending.parent_id)
        if self._log is not None:
            self._log.record_evaluation(
                [
                    slot,
                    pending.parent_id,
                    "" if child is None else child.id,
                    pending.action or "",
                    pending.mode,
                    outcome,
                    status,
                    "" if fitness is None else fitness,
                    "" if parent is None else parent.fitness,
                    error or "",
                    "" if pending.quality_value is None else f"{pending.quality_value:.6g}",
                    "" if pending.continuation_value is None else f"{pending.continuation_value:.6g}",
                    "" if pending.coverage_value is None else f"{pending.coverage_value:.6g}",
                    "" if pending.allocation_probability is None else f"{pending.allocation_probability:.6g}",
                    "" if pending.reference_value is None else f"{pending.reference_value:.6g}",
                    attempt,
                    "initial" if attempt == 1 else "repair",
                    "" if pending.request_seed is None else pending.request_seed,
                    f"{elapsed:.6f}",
                    error_type or "",
                    "" if pending.reference_id is None else pending.reference_id,
                ]
            )
            if pending.mode in {Action.DEVELOP.value, Action.EXPLORE.value, Action.CROSSOVER.value} and not continuing:
                if parent is None:
                    raise RuntimeError("ordinary decision parent disappeared")
                self._log.record_decision(
                    {
                        "task": self._task_key,
                        "slot": slot,
                        "decision_index": pending.decision_index,
                        "parent_id": pending.parent_id,
                        "formation_path": formation_path_records(self._tree, pending.parent_id),
                        "action": pending.action,
                        "llm_output": {"idea": parsed.declared_idea, "code": parsed.code},
                        "q_p": parent.fitness,
                        "q_c": fitness,
                        "result": outcome,
                        "behavior": None if child is None else child.behavior_tag,
                        "quality_value": pending.quality_value,
                        "continuation_value": pending.continuation_value,
                        "coverage_value": pending.coverage_value,
                        "allocation_probability": pending.allocation_probability,
                        "action_probabilities": pending.action_probabilities,
                        "reference_id": pending.reference_id,
                        "reference_value": pending.reference_value,
                        "exact_prompt": pending.exact_prompt,
                        "exact_response": pending.response,
                        "model_id": getattr(self._llm, "model", None),
                        "sampling_temperature": getattr(self._llm, "temperature", None),
                        "seed": pending.request_seed,
                    }
                )
        self._pending = None
        save_checkpoint(self)


__all__ = ["TraceAADV920"]
