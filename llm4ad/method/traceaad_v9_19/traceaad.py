"""TraceAAD V9.19 atomic search loop."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from pathlib import Path

from ...base import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from . import behave
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .history import formation_path_records, one_line, render_path
from .landscape import Landscape, behavior_tag, region_statistics
from .prompt import (
    build_generation_prompt,
    build_repair_prompt,
    build_root_prompt,
    format_failure_feedback,
    parse_program_response,
)
from .schema import (
    INITIAL_ROOT_COUNT,
    MAX_REPAIRS,
    W_PROMISE,
    W_TRAJECTORY,
    W_UNDERDEVELOPMENT,
    Action,
    Algorithm,
    Attempt,
    Pending,
)
from .selection import decide_action, sample_parent, t_response, trajectory_response
from .tree import Tree, VIRTUAL_ROOT_ID

ERROR_MAX_CHARS = 360
WEIGHTS = (W_PROMISE, W_UNDERDEVELOPMENT, W_TRAJECTORY)


class TraceAADV919:
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
        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.19 requires one evolvable template function")

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
        return Counter(
            attempt.action for attempt in self._attempts if attempt.action is not None
        )

    def run(self) -> None:
        status, stop_reason, error = "error", None, {}
        try:
            if self._pending is not None:
                self._settle(attempt=1, repairable=False)
            while not self._initialized() and self._has_budget():
                self._attempt(
                    build_root_prompt(
                        task_description=self._task,
                        template_function=self._function,
                    ),
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
                    weights=list(WEIGHTS),
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
        stats = region_statistics(
            landscape=self._landscape,
            quality={a.id: self._tree.quality(a) for a in valid},
            opportunities={a.id: a.opportunities for a in valid},
        )
        decision_index = self._n_ordinary_decisions
        rng = random.Random(f"{self._seed}:v9_19:{decision_index}")
        parent_decision = sample_parent(
            tree=self._tree,
            stats=stats,
            trajectory={a.id: trajectory_response(a) for a in valid},
            rng=rng,
            decision_index=decision_index,
        )
        selected_t = float(
            next(
                entry["T"]
                for entry in parent_decision.snapshot
                if entry["id"] == parent_decision.parent.id
            )
        )
        action_decision = decide_action(
            t_value=selected_t, rng=rng, allow_crossover=len(valid) > 1
        )
        self._n_ordinary_decisions += 1
        selected = next(
            entry
            for entry in parent_decision.snapshot
            if entry["id"] == parent_decision.parent.id
        )
        if self._log is not None:
            self._log.record_event(
                "pre_decision",
                decision_index=decision_index,
                slot=self._n_eval + 1,
                parent_id=parent_decision.parent.id,
                pool_size=stats.pool_size,
                neighborhood_size=stats.neighborhood_size,
                beta=parent_decision.beta,
                ess=parent_decision.ess,
                marker=parent_decision.marker,
                snapshot=list(parent_decision.snapshot),
            )
            self._log.record_event(
                "action_decision",
                decision_index=decision_index,
                slot=self._n_eval + 1,
                parent_id=parent_decision.parent.id,
                T=action_decision.t_response,
                p_explore=action_decision.p_explore,
                p_crossover=action_decision.p_crossover,
                p_develop=action_decision.p_develop,
                action=action_decision.action.value,
                action_draw=action_decision.draw,
            )
        parent = parent_decision.parent
        reference = None
        if action_decision.action is Action.CROSSOVER:
            reference_id = self._landscape.select_crossover_reference(
                parent.id,
                {a.id: self._tree.quality(a) for a in valid},
                rng,
            )
            if reference_id is not None:
                reference = self._tree.get_algorithm(reference_id)
            if self._log is not None:
                self._log.record_event(
                    "crossover_reference",
                    decision_index=decision_index,
                    slot=self._n_eval + 1,
                    parent_id=parent.id,
                    reference_id=reference_id,
                    distance=(
                        None
                        if reference is None
                        else self._landscape.distance(parent.id, reference.id)
                    ),
                )
        self._attempt(
            build_generation_prompt(
                task_description=self._task,
                code=parent.code,
                fitness=parent.fitness,
                history_text=render_path(self._tree, parent.id),
                action=action_decision.action,
                reference_code=None if reference is None else reference.code,
                reference_fitness=None if reference is None else reference.fitness,
                reference_behavior=None if reference is None else reference.behavior_tag,
                reference_distance=(
                    None
                    if reference is None
                    else self._landscape.distance(parent.id, reference.id)
                ),
            ),
            parent_id=parent.id,
            action=action_decision.action,
            mode="ordinary",
            reference_id=None if reference is None else reference.id,
            audit={
                "decision_index": decision_index,
                "promise": float(selected["P"]),
                "underdevelopment": float(selected["U"]),
                "t_response": float(selected["T"]),
                "p_explore": action_decision.p_explore,
                "reference_id": None if reference is None else reference.id,
                "beta": parent_decision.beta,
                "ess": parent_decision.ess,
                "pool_size": stats.pool_size,
                "neighborhood_size": stats.neighborhood_size,
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
        audit: dict[str, object] | None = None,
    ) -> None:
        audit = audit or {}
        for attempt in range(1, MAX_REPAIRS + 2):
            if attempt > 1:
                self._repair_llm_calls += 1
            request_seed = None if self._seed is None else self._seed + self._n_eval + 1
            kwargs: dict = {"max_tokens": self._max_tokens}
            if request_seed is not None:
                kwargs["seed"] = request_seed
            self._pending = Pending(
                parent_id=parent_id,
                action=None if action is None else action.value,
                response=self._llm.draw_sample(prompt, **kwargs),
                exact_prompt=prompt,
                mode=mode,
                request_seed=request_seed,
                decision_index=audit.get("decision_index"),  # type: ignore[arg-type]
                promise=audit.get("promise"),  # type: ignore[arg-type]
                underdevelopment=audit.get("underdevelopment"),  # type: ignore[arg-type]
                t_response=audit.get("t_response"),  # type: ignore[arg-type]
                p_explore=audit.get("p_explore"),  # type: ignore[arg-type]
                beta=audit.get("beta"),  # type: ignore[arg-type]
                ess=audit.get("ess"),  # type: ignore[arg-type]
                pool_size=audit.get("pool_size"),  # type: ignore[arg-type]
                neighborhood_size=audit.get("neighborhood_size"),  # type: ignore[arg-type]
                reference_id=reference_id,
            )
            save_checkpoint(self)
            failure = self._settle(attempt=attempt, repairable=attempt <= MAX_REPAIRS)
            if failure is None:
                return
            parent_code = (
                None
                if parent_id == VIRTUAL_ROOT_ID
                else self._tree.get_algorithm(parent_id).code
            )
            prompt = build_repair_prompt(
                task_description=self._task,
                parent_code=parent_code,
                failed_code=failure[0],
                error=failure[1],
                action=action,
                reference_code=(
                    None
                    if reference_id is None
                    else self._tree.get_algorithm(reference_id).code
                ),
                reference_fitness=(
                    None
                    if reference_id is None
                    else self._tree.get_algorithm(reference_id).fitness
                ),
            )

    def _settle(self, *, attempt: int, repairable: bool) -> tuple[str, str] | None:
        pending = self._pending
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
            message = one_line(
                result.error or f"evaluation failed: {result.failure_kind}",
                ERROR_MAX_CHARS,
            )
            outcome = "timeout" if result.failure_kind == "timeout" else "invalid"
            return reject(outcome, result.failure_kind, message)

        raw_fitness = getattr(result.result, "fitness", result.result)
        try:
            fitness = float(raw_fitness)
        except (TypeError, ValueError, OverflowError):
            fitness = math.nan
        if not math.isfinite(fitness):
            message = one_line(
                result.error or f"evaluator returned invalid fitness: {raw_fitness!r}",
                ERROR_MAX_CHARS,
            )
            return reject("invalid", str(result.failure_kind or "invalid_result"), message)

        # Duplicate candidates still consume a real evaluator slot.  Skipping
        # the evaluator would make ``budget_slots`` diverge from the formal
        # primary-evaluation protocol and would hide task-specific runtime or
        # validity failures behind a code-string comparison.
        duplicate = any(
            node.code == parsed.code for node in self._tree.valid_algorithms()
        )
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

        parent = (
            None
            if pending.parent_id == VIRTUAL_ROOT_ID
            else self._tree.get_algorithm(pending.parent_id)
        )
        trajectories = result.result.trajectories
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
        child.t_response = t_response(self._tree, child.id)
        self._landscape.add(child.id, trajectories, distances=distances)
        if parent is None:
            outcome = "plateau"
        elif fitness > self._tree.quality(parent):
            outcome = "improve"
        elif fitness < self._tree.quality(parent):
            outcome = "regress"
        else:
            outcome = "plateau"
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
                T=child.t_response,
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
                    t_response=child.t_response,
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
        if attempt == 1:
            self._n_eval += 1
        slot = self._n_eval
        previous = next((item for item in self._attempts if item.slot == slot), None)
        already = previous is not None
        self._attempts = [item for item in self._attempts if item.slot != slot]
        self._attempts.append(
            Attempt(
                slot=slot,
                parent_id=pending.parent_id,
                action=pending.action,
                outcome=outcome,
            )
        )
        if pending.parent_id != VIRTUAL_ROOT_ID:
            parent_for_credit = self._tree.get_algorithm(pending.parent_id)
            if not already:
                parent_for_credit.opportunities += 1
            elif previous is not None:
                if previous.outcome == "improve":
                    parent_for_credit.successful_opportunities = max(
                        0, parent_for_credit.successful_opportunities - 1
                    )
                else:
                    parent_for_credit.failed_opportunities = max(
                        0, parent_for_credit.failed_opportunities - 1
                    )
            if outcome == "improve":
                parent_for_credit.successful_opportunities += 1
            else:
                parent_for_credit.failed_opportunities += 1
        parent = (
            None
            if pending.parent_id == VIRTUAL_ROOT_ID
            else self._tree.get_algorithm(pending.parent_id)
        )
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
                    "" if pending.p_explore is None else f"{pending.p_explore:.6g}",
                    "" if pending.t_response is None else f"{pending.t_response:.6g}",
                    "" if pending.beta is None else f"{pending.beta:.6g}",
                    "" if pending.ess is None else f"{pending.ess:.6g}",
                    "" if pending.pool_size is None else pending.pool_size,
                    "" if pending.neighborhood_size is None else pending.neighborhood_size,
                    attempt,
                    "initial" if attempt == 1 else "repair",
                    "" if pending.request_seed is None else pending.request_seed,
                    f"{elapsed:.6f}",
                    error_type or "",
                    "" if pending.reference_id is None else pending.reference_id,
                ]
            )
            if pending.mode == "ordinary" and not continuing:
                parent_node = self._tree.get_algorithm(pending.parent_id)
                self._log.record_decision(
                    {
                        "task": self._task_key,
                        "slot": slot,
                        "decision_index": pending.decision_index,
                        "parent_id": pending.parent_id,
                        "current_code": parent_node.code,
                        "formation_path": formation_path_records(
                            self._tree, pending.parent_id
                        ),
                        "action": pending.action,
                        "llm_output": {
                            "idea": parsed.declared_idea,
                            "code": parsed.code,
                        },
                        "q_p": parent_node.fitness,
                        "q_c": fitness,
                        "result": outcome,
                        "nu": None if child is None else child.novelty,
                        "behavior": None if child is None else child.behavior_tag,
                        "P": pending.promise,
                        "U": pending.underdevelopment,
                        "T": pending.t_response,
                        "reference_id": pending.reference_id,
                        "exact_prompt": pending.exact_prompt,
                        "exact_response": pending.response,
                        "model_id": getattr(self._llm, "model", None),
                        "sampling_temperature": getattr(self._llm, "temperature", None),
                        "seed": pending.request_seed,
                    }
                )
        self._pending = None
        save_checkpoint(self)


__all__ = ["TraceAADV919"]
