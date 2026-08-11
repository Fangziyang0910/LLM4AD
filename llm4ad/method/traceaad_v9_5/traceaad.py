"""TraceAAD V9.5: anchor-centered evidence and optimistic continuation."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...base import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from .._observability import close_llm
from ..traceaad_artifacts import TraceAADArtifacts
from .checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from .evidence import (
    EvidenceSelection,
    remove_oldest_direct_supplement,
    remove_oldest_formation,
    render_evidence,
    select_evidence,
)
from .forest import SearchForest, is_artifact_better
from .prompt import (
    PROMPT_RENDERER_VERSION,
    ProgramResponseError,
    build_generation_prompt,
    build_root_prompt,
    parse_program_response,
    prompt_renderer_hash,
)
from .schema import (
    BUDGET_POLICY_ID,
    CANDIDATE_ACCOUNTING_POLICY_ID,
    CANDIDATE_MULTIPLICITY_POLICY_ID,
    EVIDENCE_SELECTOR_ID,
    GENERATION_OPERATOR,
    GENERATION_POLICY_ID,
    INITIALIZATION_POLICY_ID,
    NORMALIZATION_POLICY_ID,
    OPTIMISM_SCALE_POLICY_ID,
    PROTOCOL_ID,
    STATE_IDENTITY_POLICY_ID,
    STOP_POLICY_ID,
    AttemptKind,
    AttemptRecord,
    DirectOutcome,
    PendingAttempt,
    PendingStage,
    ProgramArtifact,
)
from .selection import select_anchor
from .source import actual_code_diff, text_hash

INITIAL_ROOT_COUNT = 8
MAX_EVIDENCE_ITEMS = 8
DEFAULT_DIFF_EXCERPT_CHARS = 1200
DIFF_EXCERPT_LIMITS = (1200, 600, 300, 0)
LOGICAL_MODEL_NAME = "Qwen3.6-27B"


class InfrastructureFailure(RuntimeError):
    """The model service did not return a completed response."""


class EvaluatorInfrastructureFailure(RuntimeError):
    """The evaluator could not prepare or launch candidate execution."""


class ConfigurationFailure(RuntimeError):
    """The frozen method cannot run under the supplied execution contract."""


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_artifact: ProgramArtifact | None
    n_artifacts: int
    n_states: int
    n_root_states: int
    n_attempts: int
    n_candidates: int
    n_evaluations: int
    n_llm_requests: int
    n_iterations: int
    optimism_scale: float | None
    initialization_complete: bool


def _stable_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _stable_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_identity_value(item) for item in value]
    if all(hasattr(value, name) for name in ("shape", "dtype", "tobytes")):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
            "array_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    return repr(value)


def _public_configuration(obj: Any) -> dict[str, Any]:
    return {
        key: _stable_identity_value(value)
        for key, value in sorted(vars(obj).items())
        if not key.startswith("_")
    }


def evaluation_contract_hash(evaluation: Evaluation) -> str:
    implementation_path = inspect.getsourcefile(type(evaluation))
    payload = {
        "type": f"{type(evaluation).__module__}.{type(evaluation).__qualname__}",
        "implementation_sha256": (
            None
            if implementation_path is None
            else hashlib.sha256(Path(implementation_path).read_bytes()).hexdigest()
        ),
        "template_program": str(evaluation.template_program),
        "configuration": _public_configuration(evaluation),
        "dataset": _stable_identity_value(getattr(evaluation, "_datasets", None)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _compact_error(message: str | None, limit: int = 360) -> str | None:
    if message is None:
        return None
    compact = " ".join(str(message).split())
    if not compact:
        return None
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


class TraceAADV95:
    """Run the complete V9.5 search forest lifecycle."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: TraceAADArtifacts | None = None,
        candidate_search_budget: int = 1000,
        *,
        initial_root_count: int = INITIAL_ROOT_COUNT,
        maximize: bool = True,
        code_max_tokens: int = 8192,
        context_token_limit: int | None = None,
        max_evidence_items: int = MAX_EVIDENCE_ITEMS,
        diff_excerpt_chars: int = DEFAULT_DIFF_EXCERPT_CHARS,
        transport_retry_limit: int = 3,
        generation_seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        debug_mode: bool = False,
    ) -> None:
        if candidate_search_budget <= 0:
            raise ValueError("candidate_search_budget must be positive")
        if initial_root_count <= 0:
            raise ValueError("initial_root_count must be positive")
        if context_token_limit is None or context_token_limit <= 0:
            raise ValueError("context_token_limit must be explicitly positive")
        if code_max_tokens <= 0 or max_evidence_items <= 0:
            raise ValueError("token and evidence limits must be positive")
        if diff_excerpt_chars < 0 or transport_retry_limit < 0:
            raise ValueError("diff and retry limits cannot be negative")
        if (
            evaluation.use_numba_accelerate
            or evaluation.use_protected_div
            or evaluation.random_seed is not None
        ):
            raise ConfigurationFailure(
                "V9.5 requires evaluator_input_code to be executed unchanged"
            )

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.5 requires one evolvable template function")

        self._llm = llm
        self._evaluation = evaluation
        self._artifacts = profiler
        self._profiler = profiler
        self._task_description_str = evaluation.task_description
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)
        self._candidate_search_budget = candidate_search_budget
        self._initial_root_count = initial_root_count
        self._maximize = maximize
        self._code_max_tokens = code_max_tokens
        self._context_token_limit = context_token_limit
        self._max_evidence_items = max_evidence_items
        self._diff_excerpt_chars = diff_excerpt_chars
        self._transport_retry_limit = transport_retry_limit
        self._generation_seed = generation_seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        llm.debug_mode = debug_mode

        self._evaluator_contract_hash = evaluation_contract_hash(evaluation)
        self._forest = SearchForest(
            self._evaluator_contract_hash, maximize=self._maximize
        )
        self._pending_attempt: PendingAttempt | None = None
        self._candidate_count = 0
        self._llm_request_count = 0
        self._evaluation_count = 0
        self._transport_failure_count = 0
        self._next_iteration = 0
        self._initialization_complete = False
        self._bootstrapped_root_ids: set[int] = set()
        self._bootstrap_deltas: list[float] = []
        self._optimism_scale: float | None = None
        self._best_artifact_id: int | None = None
        self._best_artifact_sample_order: int | None = None
        self._outcome_counts: dict[str, int] = {}

        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)
        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent
            self._record_decision(
                "checkpoint_loaded",
                checkpoint=str(checkpoint),
                candidate_count=self._candidate_count,
                pending_attempt_id=(
                    None
                    if self._pending_attempt is None
                    else self._pending_attempt.attempt_id
                ),
            )

    def search_configuration(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "candidate_search_budget": self._candidate_search_budget,
            "candidate_budget_unit": "completed_candidate_response",
            "initial_root_count": self._initial_root_count,
            "max_evidence_items": self._max_evidence_items,
            "diff_excerpt_chars": self._diff_excerpt_chars,
            "logical_model_name": LOGICAL_MODEL_NAME,
            "evaluator_contract_hash": self._evaluator_contract_hash,
            "deterministic_fitness_cache": True,
            "maximize": self._maximize,
            "code_max_tokens": self._code_max_tokens,
            "context_token_limit": self._context_token_limit,
            "transport_retry_limit": self._transport_retry_limit,
            "generation_seed": self._generation_seed,
            "evidence_selector_id": EVIDENCE_SELECTOR_ID,
            "generation_policy_id": GENERATION_POLICY_ID,
            "candidate_multiplicity_policy_id": CANDIDATE_MULTIPLICITY_POLICY_ID,
            "budget_policy_id": BUDGET_POLICY_ID,
            "initialization_policy_id": INITIALIZATION_POLICY_ID,
            "optimism_scale_policy_id": OPTIMISM_SCALE_POLICY_ID,
            "state_identity_policy_id": STATE_IDENTITY_POLICY_ID,
            "candidate_accounting_policy_id": CANDIDATE_ACCOUNTING_POLICY_ID,
            "stop_policy_id": STOP_POLICY_ID,
            "normalization_policy_id": NORMALIZATION_POLICY_ID,
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "prompt_renderer_hash": prompt_renderer_hash(),
        }

    def runtime_identity(self) -> dict[str, Any]:
        return {
            "task_description_sha256": text_hash(self._task_description_str),
            "template_program_sha256": text_hash(str(self._template_program)),
            "evaluation_contract_hash": self._evaluator_contract_hash,
            "logical_model_name": LOGICAL_MODEL_NAME,
            "temperature": getattr(self._llm, "temperature", None),
            "top_p": getattr(self._llm, "top_p", None),
            "top_k": getattr(self._llm, "top_k", None),
            "max_new_tokens": self._code_max_tokens,
            "sampling_seed": self._generation_seed,
            "sampling_seed_support": self._generation_seed is not None,
            "max_input_context": self._context_token_limit,
            "tokenizer_identity": getattr(self._llm, "tokenizer_identity", None),
            "tokenizer_version": getattr(self._llm, "tokenizer_version", None),
            "chat_template_hash": getattr(self._llm, "chat_template_hash", None),
            "serving_api": type(self._llm).__name__,
            "serving_api_version": getattr(self._llm, "api_version", None),
            "prompt_renderer_version": PROMPT_RENDERER_VERSION,
            "prompt_renderer_hash": prompt_renderer_hash(),
        }

    def run(self) -> TraceAADRunResult:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        result: TraceAADRunResult | None = None
        try:
            if self._pending_attempt is not None:
                self._process_pending_attempt()
            if not self._initialization_complete:
                self._initialize()
            if not self._initialization_complete:
                status = "initialization_failure"
                stop_reason = "candidate_budget_exhausted_during_initialization"
                result = self._result()
                return result

            while self._has_budget():
                assert self._optimism_scale is not None
                state_id, scores = select_anchor(self._forest, self._optimism_scale)
                state = self._forest.get_state(state_id)
                artifact = self._forest.get_artifact(state.artifact_id)
                selected_score = next(
                    item for item in scores if item.state_id == state_id
                )
                self._record_decision(
                    "anchor_selected",
                    iteration=self._next_iteration,
                    selected_state_id=state_id,
                    selected_artifact_id=artifact.artifact_id,
                    selected_score=selected_score.score,
                    states=[
                        {
                            "state_id": item.state_id,
                            "artifact_id": self._forest.get_state(
                                item.state_id
                            ).artifact_id,
                            "q": item.directed_fitness,
                            "n": item.generation_count_n,
                            "optimism": item.optimism,
                            "score": item.score,
                            "creation_order": self._forest.get_state(
                                item.state_id
                            ).creation_order,
                        }
                        for item in scores
                    ],
                )
                prompt = self._build_anchor_prompt(state_id)
                self._request_candidate(
                    prompt,
                    anchor_state_id=state_id,
                    stage="search",
                    iteration=self._next_iteration,
                )

            status = "finished"
            stop_reason = "candidate_budget_exhausted"
            result = self._result()
            return result
        except InfrastructureFailure as exc:
            status = "infrastructure_failure"
            stop_reason = "transport_retry_exhausted"
            error = {"error_type": type(exc).__name__, "error": str(exc)}
            if self._artifacts is not None:
                self._artifacts.record_error("transport", exc)
            raise
        except EvaluatorInfrastructureFailure as exc:
            status = "infrastructure_failure"
            stop_reason = "evaluator_infrastructure_failure"
            error = {"error_type": type(exc).__name__, "error": str(exc)}
            if self._artifacts is not None:
                self._artifacts.record_error("evaluator", exc)
            raise
        except ConfigurationFailure as exc:
            status = "configuration_failure"
            stop_reason = "configuration_contract_failed"
            error = {"error_type": type(exc).__name__, "error": str(exc)}
            if self._artifacts is not None:
                self._artifacts.record_error("configuration", exc)
            raise
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            if self._artifacts is not None:
                self._artifacts.record_error("run", exc)
            raise
        finally:
            save_checkpoint(self)
            if result is None:
                result = self._result()
            if self._artifacts is not None:
                best = result.best_artifact
                self._artifacts.write_summary(
                    status=status,
                    stop_reason=stop_reason,
                    best_artifact_id=(None if best is None else best.artifact_id),
                    best_score=None if best is None else best.fitness,
                    best_sample_order=self._best_artifact_sample_order,
                    method_sample_count=self._candidate_count,
                    evaluator_call_count=self._evaluation_count,
                    llm_request_count=self._llm_request_count,
                    transport_failure_count=self._transport_failure_count,
                    n_artifacts=result.n_artifacts,
                    n_states=result.n_states,
                    n_root_states=result.n_root_states,
                    n_attempts=result.n_attempts,
                    n_iterations=result.n_iterations,
                    initialization_complete=self._initialization_complete,
                    bootstrapped_root_ids=sorted(self._bootstrapped_root_ids),
                    bootstrap_deltas=self._bootstrap_deltas,
                    optimism_scale=self._optimism_scale,
                    outcome_counts=self._outcome_counts,
                    pending_attempt_id=(
                        None
                        if self._pending_attempt is None
                        else self._pending_attempt.attempt_id
                    ),
                    **error,
                )
                self._artifacts.finish()
            close_llm(self._llm)

    def _initialize(self) -> None:
        while (
            len(self._forest.root_state_ids) < self._initial_root_count
            and self._has_budget()
        ):
            prompt = build_root_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                maximize=self._maximize,
            )
            self._require_prompt_capacity(prompt, "root_generation")
            self._request_candidate(
                prompt,
                anchor_state_id=None,
                stage="root_generation",
                iteration=None,
            )

        for root_state_id in tuple(self._forest.root_state_ids):
            if not self._has_budget():
                break
            if root_state_id in self._bootstrapped_root_ids:
                continue
            prompt = self._build_anchor_prompt(root_state_id)
            self._request_candidate(
                prompt,
                anchor_state_id=root_state_id,
                stage="bootstrap",
                iteration=None,
            )

        complete = len(
            self._forest.root_state_ids
        ) == self._initial_root_count and self._bootstrapped_root_ids == set(
            self._forest.root_state_ids
        )
        if not complete:
            save_checkpoint(self)
            return
        self._optimism_scale = (
            float(statistics.median(self._bootstrap_deltas))
            if self._bootstrap_deltas
            else 0.0
        )
        self._initialization_complete = True
        save_checkpoint(self)
        self._record_decision(
            "initialization_completed",
            root_state_ids=list(self._forest.root_state_ids),
            bootstrapped_root_ids=sorted(self._bootstrapped_root_ids),
            bootstrap_deltas=list(self._bootstrap_deltas),
            optimism_scale=self._optimism_scale,
            candidate_count=self._candidate_count,
        )

    def _build_anchor_prompt(self, state_id: int) -> str:
        state = self._forest.get_state(state_id)
        artifact = self._forest.get_artifact(state.artifact_id)
        selection = select_evidence(
            self._forest, state_id, max_items=self._max_evidence_items
        )
        limits = tuple(
            limit for limit in DIFF_EXCERPT_LIMITS if limit <= self._diff_excerpt_chars
        )
        if self._diff_excerpt_chars not in limits:
            limits = (self._diff_excerpt_chars, *limits)
        if not limits:
            limits = (0,)

        current = selection
        for limit in limits:
            built = self._render_anchor_prompt(artifact, current, limit)
            if self._prompt_fits(built[0]):
                self._record_evidence(state_id, current, built[1], limit, built[0])
                return built[0]

        while current.formation_attempt_ids:
            current = remove_oldest_formation(current)
            built = self._render_anchor_prompt(artifact, current, 0)
            if self._prompt_fits(built[0]):
                self._record_evidence(state_id, current, built[1], 0, built[0])
                return built[0]

        coverage = set(current.direct_coverage_ids)
        while any(item not in coverage for item in current.direct_attempt_ids):
            current = remove_oldest_direct_supplement(current)
            built = self._render_anchor_prompt(artifact, current, 0)
            if self._prompt_fits(built[0]):
                self._record_evidence(state_id, current, built[1], 0, built[0])
                return built[0]
        raise ConfigurationFailure(
            "task, current code, required evidence, and output budget exceed context"
        )

    def _render_anchor_prompt(
        self,
        artifact: ProgramArtifact,
        selection: EvidenceSelection,
        diff_limit: int,
    ):
        rendered = render_evidence(
            self._forest, selection, diff_excerpt_chars=diff_limit
        )
        prompt = build_generation_prompt(
            task_description=self._task_description_str,
            anchor=artifact,
            evidence_text=rendered.text,
            maximize=self._maximize,
        )
        return prompt, rendered

    def _record_evidence(
        self, state_id, selection, rendered, diff_limit: int, prompt: str
    ) -> None:
        self._record_decision(
            "evidence_built",
            anchor_state_id=state_id,
            formation_pool_ids=selection.formation_pool_ids,
            direct_pool_ids=selection.direct_pool_ids,
            selected_formation_ids=selection.formation_attempt_ids,
            selected_direct_ids=selection.direct_attempt_ids,
            direct_coverage_ids=selection.direct_coverage_ids,
            folded_attempt_ids=selection.folded_attempt_ids,
            removed_reasons=selection.removed_reasons,
            diff_excerpt_chars=diff_limit,
            excerpt_hashes=rendered.excerpt_hashes,
            truncated_attempt_ids=rendered.truncated_attempt_ids,
            prompt_tokens=self._count_tokens(prompt),
        )

    def _request_candidate(
        self,
        prompt: str,
        *,
        anchor_state_id: int | None,
        stage: str,
        iteration: int | None,
    ) -> AttemptRecord:
        if self._pending_attempt is not None:
            raise RuntimeError("cannot request a candidate while one is pending")
        prompt_tokens = self._count_tokens(prompt)
        generation_seed = (
            None
            if self._generation_seed is None
            else self._generation_seed + self._candidate_count + 1
        )
        for transport_attempt in range(self._transport_retry_limit + 1):
            start = time.time()
            self._llm_request_count += 1
            try:
                kwargs: dict[str, Any] = {"max_tokens": self._code_max_tokens}
                if generation_seed is not None:
                    kwargs["seed"] = generation_seed
                response = self._llm.draw_sample(prompt, **kwargs)
                sample_time = time.time() - start
                break
            except Exception as exc:
                sample_time = time.time() - start
                self._transport_failure_count += 1
                save_checkpoint(self)
                self._record_llm_call(
                    stage=stage,
                    iteration=iteration,
                    anchor_state_id=anchor_state_id,
                    sample_order=None,
                    sample_time=sample_time,
                    prompt_tokens=prompt_tokens,
                    response_tokens=0,
                    status="transport",
                    prompt=prompt,
                    store_prompt=True,
                    failure_kind="transport",
                    error_type=type(exc).__name__,
                    error=str(exc),
                    transport_attempt=transport_attempt + 1,
                    generation_seed=generation_seed,
                )
                if transport_attempt == self._transport_retry_limit:
                    raise InfrastructureFailure(
                        "model transport retry limit exhausted"
                    ) from exc

        self._candidate_count += 1
        if anchor_state_id is not None:
            self._forest.get_state(anchor_state_id).generation_count_n += 1
        pending = PendingAttempt(
            attempt_id=self._forest.next_attempt_id(),
            anchor_state_id=anchor_state_id,
            stage_name=stage,
            iteration=iteration,
            candidate_order=self._candidate_count,
            response=response,
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            response_tokens=self._count_tokens(response),
            sample_time=sample_time,
            generation_seed=generation_seed,
        )
        self._pending_attempt = pending
        save_checkpoint(self)
        self._record_llm_call(
            stage=stage,
            iteration=iteration,
            anchor_state_id=anchor_state_id,
            sample_order=pending.candidate_order,
            sample_time=sample_time,
            prompt_tokens=prompt_tokens,
            response_tokens=pending.response_tokens,
            status="ok",
            prompt=prompt,
            store_prompt=True,
            program_parse_success=None,
            generation_seed=generation_seed,
        )
        return self._process_pending_attempt()

    def _process_pending_attempt(self) -> AttemptRecord:
        pending = self._pending_attempt
        if pending is None:
            raise RuntimeError("no pending candidate to process")
        if pending.processing_stage is PendingStage.RESPONSE_RECEIVED:
            try:
                parsed = parse_program_response(
                    pending.response,
                    self._template_program,
                    self._function_to_evolve.name,
                )
            except ProgramResponseError as exc:
                pending.declared_idea = exc.declared_idea
                pending.raw_code = exc.raw_code
                pending.raw_code_hash = (
                    None if exc.raw_code is None else text_hash(exc.raw_code)
                )
                pending.failure_category = "parse"
                pending.failure_feedback = _compact_error(str(exc))
                return self._finalize_pending_attempt()

            pending.declared_idea = parsed.declared_idea
            pending.raw_code = parsed.raw_code
            pending.raw_code_hash = text_hash(parsed.raw_code)
            pending.evaluator_input_code = str(parsed.program)
            pending.evaluator_input_hash = text_hash(pending.evaluator_input_code)
            if pending.anchor_state_id is not None:
                parent_state = self._forest.get_state(pending.anchor_state_id)
                parent = self._forest.get_artifact(parent_state.artifact_id)
                pending.actual_diff, pending.diff_statistics = actual_code_diff(
                    parent.evaluator_input_code, pending.evaluator_input_code
                )
            pending.processing_stage = PendingStage.PARSED
            save_checkpoint(self)

        if pending.processing_stage is PendingStage.PARSED:
            assert pending.evaluator_input_code is not None
            existing = self._forest.artifact_for_code(pending.evaluator_input_code)
            if existing is not None:
                return self._finalize_pending_attempt(existing_artifact=existing)

            outcome, evaluate_time = (
                self._evaluator.evaluate_program_record_time_with_details(
                    pending.evaluator_input_code
                )
            )
            self._evaluation_count += 1
            pending.evaluator_called = True
            pending.evaluate_time = evaluate_time
            if outcome.failure_kind == "prepare_error":
                raise EvaluatorInfrastructureFailure(
                    _compact_error(outcome.error)
                    or "evaluator preparation failed without an error message"
                )
            score = getattr(outcome.result, "fitness", outcome.result)
            try:
                fitness = float(score)
            except (TypeError, ValueError, OverflowError):
                fitness = math.nan
            if math.isfinite(fitness):
                pending.evaluated_fitness = fitness
            else:
                pending.failure_category = outcome.failure_kind or "invalid_result"
                pending.failure_feedback = _compact_error(
                    outcome.error
                ) or _compact_error(
                    f"evaluator returned non-finite or non-numeric fitness: {score!r}"
                )
            pending.processing_stage = PendingStage.EVALUATED
            save_checkpoint(self)

        return self._finalize_pending_attempt()

    def _finalize_pending_attempt(
        self, *, existing_artifact: ProgramArtifact | None = None
    ) -> AttemptRecord:
        pending = self._pending_attempt
        if pending is None:
            raise RuntimeError("no pending candidate to finalize")
        anchor_state = (
            None
            if pending.anchor_state_id is None
            else self._forest.get_state(pending.anchor_state_id)
        )
        parent_artifact = (
            None
            if anchor_state is None
            else self._forest.get_artifact(anchor_state.artifact_id)
        )
        artifact = existing_artifact
        child_state = None
        kind: AttemptKind

        if pending.failure_category is not None:
            kind = AttemptKind.INVALID
        elif artifact is not None and anchor_state is None:
            kind = AttemptKind.ROOT_DUPLICATE
        elif artifact is not None and parent_artifact is not None:
            if artifact.artifact_id == parent_artifact.artifact_id:
                kind = AttemptKind.NO_OP
            elif artifact.artifact_id in self._forest.ancestor_artifact_ids(
                anchor_state.state_id
            ):
                kind = AttemptKind.ANCESTRAL_RETURN
            elif self._forest.relation_exists(
                anchor_state.state_id, artifact.artifact_id
            ):
                kind = AttemptKind.REPEATED_DUPLICATE
            else:
                kind = AttemptKind.CACHED_ARTIFACT
                child_state = self._forest.add_child_state(
                    parent_state_id=anchor_state.state_id,
                    artifact_id=artifact.artifact_id,
                    attempt_id=pending.attempt_id,
                    creation_order=pending.candidate_order,
                )
        else:
            assert pending.evaluator_input_code is not None
            assert pending.evaluated_fitness is not None
            artifact = self._forest.add_artifact(
                evaluator_input_code=pending.evaluator_input_code,
                fitness=pending.evaluated_fitness,
                discovery_order=pending.candidate_order,
            )
            if anchor_state is None:
                kind = AttemptKind.ROOT_NEW
                child_state = self._forest.add_root_state(
                    artifact_id=artifact.artifact_id,
                    creation_order=pending.candidate_order,
                )
            else:
                kind = AttemptKind.NEW_ARTIFACT
                child_state = self._forest.add_child_state(
                    parent_state_id=anchor_state.state_id,
                    artifact_id=artifact.artifact_id,
                    attempt_id=pending.attempt_id,
                    creation_order=pending.candidate_order,
                )

        directed_delta = (
            None
            if parent_artifact is None or artifact is None
            else artifact.directed_fitness - parent_artifact.directed_fitness
        )
        direct_outcome = self._direct_outcome(
            anchor_state is not None,
            invalid=kind is AttemptKind.INVALID,
            directed_delta=directed_delta,
        )
        attempt = AttemptRecord(
            attempt_id=pending.attempt_id,
            status="finalized",
            anchor_state_id=pending.anchor_state_id,
            child_state_id=None if child_state is None else child_state.state_id,
            artifact_id=None if artifact is None else artifact.artifact_id,
            declared_idea=pending.declared_idea,
            raw_code_hash=pending.raw_code_hash,
            evaluator_input_hash=pending.evaluator_input_hash,
            actual_diff=pending.actual_diff,
            diff_statistics=pending.diff_statistics,
            parent_fitness=(
                None if parent_artifact is None else parent_artifact.fitness
            ),
            child_fitness=None if artifact is None else artifact.fitness,
            directed_delta=directed_delta,
            direct_outcome=direct_outcome,
            attempt_kind=kind,
            failure_category=pending.failure_category,
            failure_feedback=pending.failure_feedback,
            evaluator_called=pending.evaluator_called,
            candidate_order=pending.candidate_order,
            creation_time=datetime.now(timezone.utc).isoformat(),
            stage=pending.stage_name,
            iteration=pending.iteration,
        )
        self._forest.add_attempt(attempt)

        if pending.stage_name == "bootstrap" and pending.anchor_state_id is not None:
            self._bootstrapped_root_ids.add(pending.anchor_state_id)
            if child_state is not None and directed_delta is not None:
                self._bootstrap_deltas.append(abs(directed_delta))
        if pending.stage_name == "search" and pending.iteration is not None:
            self._next_iteration = max(self._next_iteration, pending.iteration + 1)

        best_updated, best_reason = self._update_best(artifact)
        self._outcome_counts[kind.value] = self._outcome_counts.get(kind.value, 0) + 1
        audit_pending = pending
        self._pending_attempt = None
        save_checkpoint(self)
        self._record_finalized_attempt(
            attempt,
            audit_pending,
            best_updated=best_updated,
            best_reason=best_reason,
        )
        return attempt

    @staticmethod
    def _direct_outcome(
        has_anchor: bool, *, invalid: bool, directed_delta: float | None
    ) -> DirectOutcome | None:
        if not has_anchor:
            return None
        if invalid:
            return DirectOutcome.INVALID
        assert directed_delta is not None
        if directed_delta > 0:
            return DirectOutcome.IMPROVE
        if directed_delta < 0:
            return DirectOutcome.REGRESS
        return DirectOutcome.PLATEAU

    def _update_best(self, artifact: ProgramArtifact | None) -> tuple[bool, str | None]:
        if artifact is None:
            return False, None
        incumbent = (
            None
            if self._best_artifact_id is None
            else self._forest.get_artifact(self._best_artifact_id)
        )
        if not is_artifact_better(artifact, incumbent):
            return False, None
        reason = (
            "strict_fitness"
            if incumbent is None
            or artifact.directed_fitness > incumbent.directed_fitness
            else "tie_shorter"
        )
        self._best_artifact_id = artifact.artifact_id
        self._best_artifact_sample_order = artifact.first_discovery_order
        self._record_decision(
            "best_updated",
            artifact_id=artifact.artifact_id,
            sample_order=artifact.first_discovery_order,
            reason=reason,
        )
        return True, reason

    def _record_finalized_attempt(
        self,
        attempt: AttemptRecord,
        pending: PendingAttempt,
        *,
        best_updated: bool,
        best_reason: str | None,
    ) -> None:
        status = "ok"
        if attempt.attempt_kind is AttemptKind.INVALID:
            status = (
                "parse_failed" if attempt.failure_category == "parse" else "eval_failed"
            )
        self._record_candidate(
            sample_order=attempt.candidate_order,
            score=attempt.child_fitness,
            status=status,
            failure_kind=attempt.failure_category,
            error=attempt.failure_feedback,
            program=pending.evaluator_input_code or "",
            raw_response=pending.response,
            raw_code=pending.raw_code,
            idea=attempt.declared_idea,
            raw_code_hash=attempt.raw_code_hash,
            evaluator_input_hash=attempt.evaluator_input_hash,
            evaluator_contract_hash=self._evaluator_contract_hash,
            actual_diff=attempt.actual_diff,
            diff_statistics=(
                None
                if attempt.diff_statistics is None
                else {
                    "added_lines": attempt.diff_statistics.added_lines,
                    "removed_lines": attempt.diff_statistics.removed_lines,
                    "changed_lines": attempt.diff_statistics.changed_lines,
                }
            ),
            attempt_id=attempt.attempt_id,
            attempt_kind=attempt.attempt_kind,
            direct_outcome=attempt.direct_outcome,
            evaluator_called=attempt.evaluator_called,
            evaluate_time=pending.evaluate_time,
            sample_time=pending.sample_time,
            parent_node_id=attempt.anchor_state_id,
            child_state_id=attempt.child_state_id,
            artifact_id=attempt.artifact_id,
            iteration=attempt.iteration,
            stage=attempt.stage,
        )
        if attempt.child_state_id is not None and attempt.anchor_state_id is not None:
            self._artifacts_record_edge(
                edge_id=attempt.attempt_id,
                parent_id=attempt.anchor_state_id,
                child_id=attempt.child_state_id,
                sample_order=attempt.candidate_order,
                iteration=attempt.iteration,
                stage=attempt.stage,
                operator=GENERATION_OPERATOR,
                implemented_idea=attempt.declared_idea,
                actual_diff=attempt.actual_diff,
                delta_parent=attempt.directed_delta,
                outcome=attempt.direct_outcome,
                new_global_best=best_updated,
                strict_breakthrough=best_reason == "strict_fitness",
                global_best_update_reason=best_reason,
            )
        self._record_decision(
            "attempt_finalized",
            attempt_id=attempt.attempt_id,
            candidate_order=attempt.candidate_order,
            anchor_state_id=attempt.anchor_state_id,
            child_state_id=attempt.child_state_id,
            artifact_id=attempt.artifact_id,
            attempt_kind=attempt.attempt_kind,
            direct_outcome=attempt.direct_outcome,
            evaluator_called=attempt.evaluator_called,
            cache_hit=attempt.attempt_kind
            in {
                AttemptKind.ROOT_DUPLICATE,
                AttemptKind.CACHED_ARTIFACT,
                AttemptKind.NO_OP,
                AttemptKind.REPEATED_DUPLICATE,
                AttemptKind.ANCESTRAL_RETURN,
            },
            ancestral_membership=attempt.attempt_kind is AttemptKind.ANCESTRAL_RETURN,
            relation_existed=attempt.attempt_kind is AttemptKind.REPEATED_DUPLICATE,
            raw_code_hash=attempt.raw_code_hash,
            evaluator_input_hash=attempt.evaluator_input_hash,
            actual_diff_hash=(
                None if attempt.actual_diff is None else text_hash(attempt.actual_diff)
            ),
            failure_category=attempt.failure_category,
            failure_feedback=attempt.failure_feedback,
        )

    def _require_prompt_capacity(self, prompt: str, stage: str) -> None:
        if not self._prompt_fits(prompt):
            raise ConfigurationFailure(
                f"V9.5 {stage} prompt plus output bound exceeds context limit"
            )

    def _prompt_fits(self, prompt: str) -> bool:
        return (
            self._count_tokens(prompt) + self._code_max_tokens
            <= self._context_token_limit
        )

    def _count_tokens(self, text: str) -> int:
        counter = getattr(self._llm, "count_tokens", None)
        return int(counter(text)) if callable(counter) else len(text.encode("utf-8"))

    @property
    def _token_count_mode(self) -> str:
        mode = getattr(self._llm, "token_count_mode", None)
        if isinstance(mode, str):
            return mode
        return (
            "llm_count_tokens"
            if callable(getattr(self._llm, "count_tokens", None))
            else "utf8_byte_upper_bound"
        )

    def _record_candidate(self, **payload: Any) -> None:
        if self._artifacts is not None:
            self._artifacts.record_candidate(operator=GENERATION_OPERATOR, **payload)

    def _record_llm_call(self, *, anchor_state_id: int | None, **payload: Any) -> None:
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                operator=GENERATION_OPERATOR,
                parent_node_id=anchor_state_id,
                token_count_mode=self._token_count_mode,
                **payload,
            )

    def _artifacts_record_edge(self, **payload: Any) -> None:
        if self._artifacts is not None:
            self._artifacts.record_edge(**payload)

    def _record_decision(self, event: str, **payload: Any) -> None:
        if self._artifacts is not None:
            self._artifacts.record_decision(event, **payload)

    def _has_budget(self) -> bool:
        return self._candidate_count < self._candidate_search_budget

    def _result(self) -> TraceAADRunResult:
        best = (
            None
            if self._best_artifact_id is None
            else self._forest.get_artifact(self._best_artifact_id)
        )
        return TraceAADRunResult(
            best_artifact=best,
            n_artifacts=len(self._forest.artifacts()),
            n_states=len(self._forest.states()),
            n_root_states=len(self._forest.root_state_ids),
            n_attempts=len(self._forest.attempts()),
            n_candidates=self._candidate_count,
            n_evaluations=self._evaluation_count,
            n_llm_requests=self._llm_request_count,
            n_iterations=self._next_iteration,
            optimism_scale=self._optimism_scale,
            initialization_complete=self._initialization_complete,
        )


__all__ = [
    "DEFAULT_DIFF_EXCERPT_CHARS",
    "INITIAL_ROOT_COUNT",
    "LOGICAL_MODEL_NAME",
    "MAX_EVIDENCE_ITEMS",
    "ConfigurationFailure",
    "EvaluatorInfrastructureFailure",
    "InfrastructureFailure",
    "TraceAADRunResult",
    "TraceAADV95",
    "evaluation_contract_hash",
]
