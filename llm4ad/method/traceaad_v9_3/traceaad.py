"""TraceAAD V9.3: allocate evaluator budget as short trajectory rollouts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from ...base import (
    Evaluation,
    Function,
    LLM,
    Program,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from .._observability import (
    close_llm,
    init_observability,
    is_search_aborted,
    record_sample_failure,
    reset_sample_failures,
)
from ..traceaad_artifacts import TraceAADArtifacts
from .checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from .complexity import code_hash, nonempty_loc
from .context import canonical_window
from .prompt import (
    IDEA_MAX_CHARS,
    ParsedProgram,
    build_code_implementation_prompt,
    build_strategy_plan_prompt,
    build_strategy_root_prompt,
    build_trajectory_decision_prompt,
    extract_idea,
    parse_code_response,
    parse_strategy_plan,
    parse_program_response,
)
from .schema import GENERATION_OPERATOR, PROTOCOL_ID, GenerationEvent, ProgramNode
from .tree import FactGraph, is_node_better
from .value import anchor_rank_key, select_anchor

WINDOW_SIZE = 8
FORMATION_QUOTA = 4
DOWNSTREAM_QUOTA = 4
DOWNSTREAM_DEPTH = 3
QUALITY_POOL_SIZE = 10
INITIAL_ROUTE_POOL_SIZE = 8
INITIAL_ANCHOR_COUNT = 6
STRATEGY_PLAN_MAX_TOKENS = 2048
STRATEGY_PLANNING_OPERATOR = "initial_strategy_planning"
TRAJECTORY_DECISION_MAX_TOKENS = 1024
TRAJECTORY_DECISION_OPERATOR = "trajectory_decision"
ROLLOUT_LENGTH = 3


def _stable_identity_value(value):
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


def _configuration_digest(obj) -> str:
    payload = {
        key: _stable_identity_value(value)
        for key, value in sorted(vars(obj).items())
        if not key.startswith("_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _Attempt:
    idea: str
    parsed: ParsedProgram | None
    fitness: float | None
    failure_kind: str | None
    budget_order: int
    sample_time: float


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_root_children: int
    n_eligible_nodes: int
    n_events: int
    n_samples: int
    n_evaluations: int
    n_iterations: int


class TraceAADV93:
    """Select an anchor, develop a short trajectory, then update global credit."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: TraceAADArtifacts | None = None,
        max_sample_nums: int | None = 1000,
        *,
        initial_route_pool_size: int = INITIAL_ROUTE_POOL_SIZE,
        initial_anchor_count: int = INITIAL_ANCHOR_COUNT,
        maximize: bool = True,
        code_max_tokens: int = 8192,
        context_token_limit: int | None = None,
        debug_mode: bool = False,
        max_consecutive_sample_failures: int = 20,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        resume_from: str | Path | None = None,
    ) -> None:
        if max_sample_nums is not None and max_sample_nums <= 0:
            raise ValueError("max_sample_nums must be positive or None")
        if initial_route_pool_size <= 0:
            raise ValueError("initial_route_pool_size must be positive")
        if not 0 < initial_anchor_count <= initial_route_pool_size:
            raise ValueError(
                "initial_anchor_count must be positive and no larger than "
                "initial_route_pool_size"
            )
        if (
            max_sample_nums is not None
            and max_sample_nums < (1 + ROLLOUT_LENGTH) * initial_route_pool_size
        ):
            raise ValueError(
                "max_sample_nums must cover one root and one complete short rollout "
                "per initial route"
            )
        if context_token_limit is None or context_token_limit <= 0:
            raise ValueError("context_token_limit must be explicitly positive")
        if code_max_tokens <= 0 or checkpoint_interval <= 0:
            raise ValueError("token and checkpoint limits must be positive")

        self._llm = llm
        self._evaluation = evaluation
        self._evaluation_configuration_sha256 = _configuration_digest(evaluation)
        self._artifacts = profiler
        self._profiler = profiler
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._initial_route_pool_size = initial_route_pool_size
        self._initial_anchor_count = initial_anchor_count
        self._maximize = maximize
        self._code_max_tokens = code_max_tokens
        self._context_token_limit = context_token_limit
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = checkpoint_interval
        self._last_checkpoint_budget = -1
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.3 requires one evolvable template function")
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._graph = FactGraph()
        self._best_node: ProgramNode | None = None
        self._best_node_sample_order: int | None = None
        self._tot_sample_nums = 0
        self._evaluation_count = 0
        self._next_iteration = 0
        self._initialization_complete = False
        self._initial_strategy_cards: tuple[str, ...] = ()
        self._root_strategy_cards: dict[int, str] = {}
        self._strategy_planning_calls = 0
        self._bootstrapped_root_ids: set[int] = set()
        self._eligible_node_ids: set[int] = set()
        self._next_rollout_id = 0
        self._active_rollout: dict | None = None
        init_observability(self, max_consecutive_sample_failures)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)
        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent
            self._record_decision(
                "checkpoint_loaded",
                checkpoint=str(checkpoint),
                budget_order=self._tot_sample_nums,
            )

    def runtime_identity(self) -> dict[str, str | None]:
        def setting(obj, name: str) -> str | None:
            value = getattr(obj, name, None)
            return None if value is None else repr(value)

        return {
            "task_description_sha256": hashlib.sha256(
                self._task_description_str.encode("utf-8")
            ).hexdigest(),
            "template_program_sha256": hashlib.sha256(
                str(self._template_program).encode("utf-8")
            ).hexdigest(),
            "evaluation_type": f"{type(self._evaluation).__module__}.{type(self._evaluation).__qualname__}",
            "evaluation_configuration_sha256": self._evaluation_configuration_sha256,
            "evaluation_random_seed": setting(self._evaluation, "random_seed"),
            "llm_type": f"{type(self._llm).__module__}.{type(self._llm).__qualname__}",
            "llm_model": setting(self._llm, "model"),
            "llm_base_url": setting(self._llm, "base_url"),
        }

    def search_configuration(self) -> dict:
        return {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "max_sample_nums": self._max_sample_nums,
            "initialization_protocol": "strategy_short_rollout_curation",
            "initial_route_pool_size": self._initial_route_pool_size,
            "initial_anchor_count": self._initial_anchor_count,
            "initial_route_length": 1 + ROLLOUT_LENGTH,
            "initial_route_selection": "best_rollout_representative_by_route_value",
            "generation_protocol": "trajectory_decision_then_code",
            "generation_operator": GENERATION_OPERATOR,
            "trajectory_decision_operator": TRAJECTORY_DECISION_OPERATOR,
            "rollout_length": ROLLOUT_LENGTH,
            "code_representation": "comment_and_docstring_free_ast_canonical",
            "window_protocol": "canonical_formation4_downstream4_depth3",
            "window_size": WINDOW_SIZE,
            "formation_quota": FORMATION_QUOTA,
            "downstream_quota": DOWNSTREAM_QUOTA,
            "downstream_depth": DOWNSTREAM_DEPTH,
            "quality_pool_size": QUALITY_POOL_SIZE,
            "quality_policy": "anchor_initialized_mean_rollout_best_absolute_quality",
            "budget_policy": "top10_unverified_first_then_highest_q",
            "invalid_outcome": "rollout_start_anchor_directed_fitness",
            "credit_scope": "selected_rollout_start_anchor_only",
            "eligible_policy": "best_program_per_completed_rollout",
            "ancestor_backup": False,
            "maximize": self._maximize,
            "code_max_tokens": self._code_max_tokens,
            "context_token_limit": self._context_token_limit,
            "max_consecutive_sample_failures": self._max_consecutive_sample_failures,
            "checkpoint_interval": self._checkpoint_interval,
        }

    def run(self) -> TraceAADRunResult:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        result: TraceAADRunResult | None = None
        try:
            if not self._initialization_complete:
                self._initialize()
                save_checkpoint(self)
            while (
                self._initialization_complete
                and self._has_budget()
                and not is_search_aborted(self)
            ):
                if self._active_rollout is None:
                    selection = select_anchor(
                        self._graph,
                        eligible_node_ids=self._eligible_node_ids,
                        pool_size=QUALITY_POOL_SIZE,
                    )
                    anchor = self._graph.get_node(selection.selected_node_id)
                    self._record_decision(
                        "anchor_selected",
                        iteration=self._next_iteration,
                        selected_node_id=anchor.id,
                        quality_pool_ids=selection.quality_pool_ids,
                        quality_rank=selection.quality_rank,
                        allocation_mode=selection.mode,
                        anchor_fitness=anchor.fitness,
                        anchor_directed_fitness=anchor.directed_fitness,
                        budget_value=anchor.budget_value,
                        budget_event_count=anchor.budget_event_count,
                    )
                else:
                    anchor = self._graph.get_node(
                        int(self._active_rollout["start_anchor_id"])
                    )
                consumed = self._perform_anchor_rollout(
                    anchor, stage="search", iteration=self._next_iteration
                )
                if consumed:
                    self._next_iteration += 1
                    self._save_checkpoint_if_due()
            result = self._result()
            if is_search_aborted(self):
                status = "aborted"
                stop_reason = "max_consecutive_sample_failures"
            else:
                status = "finished"
                stop_reason = (
                    "budget_exhausted"
                    if self._initialization_complete
                    else "budget_exhausted_during_initialization"
                )
            return result
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            if self._artifacts is not None:
                self._artifacts.record_error("run", exc)
            raise
        finally:
            if result is None:
                result = self._result()
            save_checkpoint(self)
            if self._artifacts is not None:
                self._artifacts.write_summary(
                    status=status,
                    best_node_id=None
                    if result.best_node is None
                    else result.best_node.id,
                    best_score=None
                    if result.best_node is None
                    else result.best_node.fitness,
                    best_sample_order=self._best_node_sample_order,
                    method_sample_count=self._tot_sample_nums,
                    evaluator_call_count=self._evaluation_count,
                    n_total_nodes=result.n_total_nodes,
                    n_valid_nodes=result.n_valid_nodes,
                    n_root_children=result.n_root_children,
                    n_eligible_nodes=result.n_eligible_nodes,
                    n_events=result.n_events,
                    n_iterations=result.n_iterations,
                    initial_route_target=self._initial_route_pool_size,
                    initial_route_actual=result.n_root_children,
                    initial_anchor_target=self._initial_anchor_count,
                    initial_anchor_actual=(
                        len(self._eligible_node_ids)
                        if self._initialization_complete
                        else 0
                    ),
                    strategy_planning_calls=self._strategy_planning_calls,
                    initialization_complete=self._initialization_complete,
                    search_aborted=is_search_aborted(self),
                    stop_reason=stop_reason,
                    **error,
                )
                self._artifacts.finish()
            close_llm(self._llm)

    def _initialize(self) -> None:
        self._plan_initial_strategies()
        while (
            len(self._graph.root.child_ids) < self._initial_route_pool_size
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            strategy_index = len(self._graph.root.child_ids) + 1
            strategy = self._initial_strategy_cards[strategy_index - 1]
            prompt = build_strategy_root_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                maximize=self._maximize,
                strategy_index=strategy_index,
                strategy=strategy,
            )
            self._require_context(prompt, "initialization")
            attempt = self._attempt_program(
                prompt,
                stage="root_generation",
                iteration=None,
                anchor=None,
                template_program=self._template_program,
            )
            if attempt is None or attempt.fitness is None or attempt.parsed is None:
                continue
            root = self._graph.add_root(
                code=str(attempt.parsed.program),
                idea=attempt.parsed.idea,
                fitness=attempt.fitness,
                maximize=self._maximize,
                creation_order=attempt.budget_order,
            )
            self._root_strategy_cards[root.id] = strategy
            self._update_best(root)
            self._record_decision(
                "initial_anchor_created",
                node_id=root.id,
                budget_order=attempt.budget_order,
                root_count=len(self._graph.root.child_ids),
                strategy_index=strategy_index,
                strategy=strategy,
            )
            self._save_checkpoint_if_due()

        for root_id in tuple(self._graph.root.child_ids):
            root = self._graph.get_node(root_id)
            while (
                root_id not in self._bootstrapped_root_ids
                and self._has_budget()
                and not is_search_aborted(self)
            ):
                consumed = self._perform_anchor_rollout(
                    root,
                    stage="trajectory_bootstrap_rollout",
                    iteration=None,
                    initial_strategy=self._root_strategy_cards[root.id],
                )
                if consumed:
                    self._bootstrapped_root_ids.add(root.id)
                    self._save_checkpoint_if_due()

        routes_complete = len(
            self._graph.root.child_ids
        ) == self._initial_route_pool_size and self._bootstrapped_root_ids == set(
            self._graph.root.child_ids
        )
        if routes_complete and not self._eligible_node_ids:
            self._curate_initial_routes()
        self._initialization_complete = (
            routes_complete
            and len(self._eligible_node_ids) == self._initial_anchor_count
        )
        if self._initialization_complete:
            self._record_decision(
                "initialization_completed",
                root_ids=self._graph.root.child_ids,
                bootstrapped_root_ids=sorted(self._bootstrapped_root_ids),
                eligible_node_ids=sorted(self._eligible_node_ids),
                budget_order=self._tot_sample_nums,
            )

    def _plan_initial_strategies(self) -> None:
        while not self._initial_strategy_cards and not is_search_aborted(self):
            prompt = build_strategy_plan_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                maximize=self._maximize,
                strategy_count=self._initial_route_pool_size,
            )
            self._require_context(prompt, "initial_strategy_planning")
            prompt_tokens = self._count_tokens(prompt)
            start = time.time()
            self._strategy_planning_calls += 1
            try:
                response = self._llm.draw_sample(
                    prompt, max_tokens=STRATEGY_PLAN_MAX_TOKENS
                )
                sample_time = time.time() - start
            except Exception as exc:
                sample_time = time.time() - start
                self._record_llm_call(
                    operator=STRATEGY_PLANNING_OPERATOR,
                    stage="initial_strategy_planning",
                    iteration=None,
                    anchor=None,
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
                )
                record_sample_failure(
                    self,
                    exc,
                    stage="initial_strategy_planning",
                    operator=STRATEGY_PLANNING_OPERATOR,
                    sample_order=self._tot_sample_nums + 1,
                    counts_budget=False,
                    iteration=None,
                    parent_node_id=None,
                )
                continue
            strategies = parse_strategy_plan(response, self._initial_route_pool_size)
            self._record_llm_call(
                operator=STRATEGY_PLANNING_OPERATOR,
                stage="initial_strategy_planning",
                iteration=None,
                anchor=None,
                sample_order=None,
                sample_time=sample_time,
                prompt_tokens=prompt_tokens,
                response_tokens=self._count_tokens(response),
                status="ok" if strategies is not None else "parse_failed",
                prompt=prompt,
                store_prompt=True,
                response=response,
            )
            if strategies is None:
                record_sample_failure(
                    self,
                    ValueError("invalid initial strategy plan"),
                    stage="initial_strategy_planning",
                    operator=STRATEGY_PLANNING_OPERATOR,
                    sample_order=self._tot_sample_nums + 1,
                    counts_budget=False,
                    iteration=None,
                    parent_node_id=None,
                )
                continue
            reset_sample_failures(self)
            self._initial_strategy_cards = strategies
            self._record_decision(
                "initial_strategies_planned",
                strategies=strategies,
                planning_call=self._strategy_planning_calls,
            )
            save_checkpoint(self)

    def _curate_initial_routes(self) -> None:
        routes: list[tuple[ProgramNode, ProgramNode, tuple[GenerationEvent, ...]]] = []
        for root_id in self._graph.root.child_ids:
            root = self._graph.get_node(root_id)
            events = tuple(
                event
                for event in self._graph.events()
                if event.rollout_start_anchor_id == root_id
                and event.stage == "trajectory_bootstrap_rollout"
            )
            representative = root
            for event in events:
                if event.child_id is None:
                    continue
                child = self._graph.get_node(event.child_id)
                if is_node_better(child, representative):
                    representative = child
            routes.append((representative, root, events))
        routes.sort(
            key=lambda item: (
                -item[1].budget_value,
                *anchor_rank_key(item[0]),
            )
        )
        selected = routes[: self._initial_anchor_count]
        self._eligible_node_ids = {
            representative.id for representative, _, _ in selected
        }
        selected_ids = self._eligible_node_ids
        self._record_decision(
            "initial_routes_curated",
            selected_anchor_ids=[item[0].id for item in selected],
            routes=[
                {
                    "root_id": root.id,
                    "representative_anchor_id": representative.id,
                    "selected": representative.id in selected_ids,
                    "strategy": self._root_strategy_cards[root.id],
                    "root_fitness": root.fitness,
                    "bootstrap_outcomes": [event.outcome for event in events],
                    "bootstrap_event_ids": [event.id for event in events],
                    "root_rollout_value": root.budget_value,
                    "representative_fitness": representative.fitness,
                    "representative_budget_value": representative.budget_value,
                }
                for representative, root, events in routes
            ],
        )
        save_checkpoint(self)

    def _perform_anchor_rollout(
        self,
        anchor: ProgramNode,
        *,
        stage: str,
        iteration: int | None,
        initial_strategy: str | None = None,
    ) -> bool:
        if self._active_rollout is None:
            rollout_id = self._next_rollout_id
            self._next_rollout_id += 1
            self._active_rollout = {
                "rollout_id": rollout_id,
                "start_anchor_id": anchor.id,
                "current_anchor_id": anchor.id,
                "representative_node_id": anchor.id,
                "completed_steps": 0,
                "stage": stage,
                "iteration": iteration,
                "initial_strategy": initial_strategy,
                "event_ids": [],
                "last_budget_order": None,
                "budget_value_before": anchor.budget_value,
            }
            self._record_decision(
                "trajectory_rollout_started",
                rollout_id=rollout_id,
                stage=stage,
                iteration=iteration,
                start_anchor_id=anchor.id,
                rollout_length=ROLLOUT_LENGTH,
                budget_value_before=anchor.budget_value,
            )
        active = self._active_rollout
        if (
            int(active["start_anchor_id"]) != anchor.id
            or str(active["stage"]) != stage
            or active["iteration"] != iteration
        ):
            raise ValueError(
                "active trajectory rollout does not match requested anchor"
            )

        while (
            int(active["completed_steps"]) < ROLLOUT_LENGTH
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            current = self._graph.get_node(int(active["current_anchor_id"]))
            rollout_step = int(active["completed_steps"]) + 1
            window = canonical_window(
                self._graph,
                current.id,
                max_events=WINDOW_SIZE,
                formation_quota=FORMATION_QUOTA,
                downstream_quota=DOWNSTREAM_QUOTA,
                downstream_depth=DOWNSTREAM_DEPTH,
            )
            idea = self._decide_trajectory_step(
                current,
                window.text,
                stage=stage,
                iteration=iteration,
                rollout_id=int(active["rollout_id"]),
                rollout_step=rollout_step,
                initial_strategy=initial_strategy,
            )
            if idea is None:
                continue
            prompt = build_code_implementation_prompt(
                task_description=self._task_description_str,
                anchor=current,
                idea=idea,
                maximize=self._maximize,
            )
            self._require_context(prompt, f"{stage}_code")
            current_program = TextFunctionProgramConverter.text_to_program(current.code)
            if current_program is None:
                raise ValueError(f"anchor {current.id} is not parseable")
            attempt = self._attempt_program(
                prompt,
                stage=stage,
                iteration=iteration,
                anchor=current,
                template_program=current_program,
                idea=idea,
                rollout_id=int(active["rollout_id"]),
                rollout_step=rollout_step,
                rollout_start_anchor_id=anchor.id,
            )
            if attempt is None:
                continue
            if attempt.parsed is None or attempt.fitness is None:
                event = self._graph.add_invalid_event(
                    anchor_id=current.id,
                    idea=attempt.idea,
                    code=(
                        None if attempt.parsed is None else str(attempt.parsed.program)
                    ),
                    failure_kind=attempt.failure_kind or "invalid_result",
                    stage=stage,
                    iteration=iteration,
                    budget_order=attempt.budget_order,
                    rollout_id=int(active["rollout_id"]),
                    rollout_step=rollout_step,
                    rollout_start_anchor_id=anchor.id,
                )
                child = None
            else:
                program_text = str(attempt.parsed.program)
                program_loc = nonempty_loc(program_text)
                new_best, reason = self._candidate_best_status(
                    attempt.fitness, program_loc, attempt.budget_order
                )
                child, event = self._graph.add_valid_event(
                    anchor_id=current.id,
                    code=program_text,
                    idea=attempt.parsed.idea,
                    fitness=attempt.fitness,
                    maximize=self._maximize,
                    stage=stage,
                    iteration=iteration,
                    budget_order=attempt.budget_order,
                    rollout_id=int(active["rollout_id"]),
                    rollout_step=rollout_step,
                    rollout_start_anchor_id=anchor.id,
                    new_global_best=new_best,
                    global_best_update_reason=reason,
                )
                self._update_best(child)
                active["current_anchor_id"] = child.id
                representative = self._graph.get_node(
                    int(active["representative_node_id"])
                )
                if is_node_better(child, representative):
                    active["representative_node_id"] = child.id
            self._record_edge(event)
            active["completed_steps"] = rollout_step
            active["event_ids"].append(event.id)
            active["last_budget_order"] = attempt.budget_order
            self._record_decision(
                "trajectory_rollout_step_completed",
                rollout_id=int(active["rollout_id"]),
                rollout_step=rollout_step,
                stage=stage,
                iteration=iteration,
                start_anchor_id=anchor.id,
                step_anchor_id=current.id,
                child_id=None if child is None else child.id,
                event_id=event.id,
                status=event.status,
                failure_kind=event.failure_kind,
                formation_event_ids=window.formation_event_ids,
                downstream_event_ids=window.downstream_event_ids,
            )
            self._save_checkpoint_if_due()

        completed_steps = int(active["completed_steps"])
        if completed_steps == 0:
            return False
        if completed_steps < ROLLOUT_LENGTH and self._has_budget():
            return False

        representative = self._graph.get_node(int(active["representative_node_id"]))
        last_budget_order = int(active["last_budget_order"])
        self._graph.record_rollout_outcome(
            anchor.id, representative.directed_fitness, last_budget_order
        )
        if stage == "search" and representative.id != anchor.id:
            self._eligible_node_ids.add(representative.id)
        self._record_decision(
            "trajectory_rollout_completed",
            rollout_id=int(active["rollout_id"]),
            stage=stage,
            iteration=iteration,
            start_anchor_id=anchor.id,
            event_ids=active["event_ids"],
            completed_steps=completed_steps,
            completion_reason=(
                "target_length"
                if completed_steps == ROLLOUT_LENGTH
                else "budget_exhausted"
            ),
            representative_node_id=representative.id,
            representative_fitness=representative.fitness,
            rollout_credit=representative.directed_fitness,
            budget_value_before=float(active["budget_value_before"]),
            budget_value_after=anchor.budget_value,
            budget_event_count_after=anchor.budget_event_count,
            representative_became_eligible=(
                stage == "search" and representative.id != anchor.id
            ),
        )
        self._active_rollout = None
        return True

    def _decide_trajectory_step(
        self,
        anchor: ProgramNode,
        window_text: str,
        *,
        stage: str,
        iteration: int | None,
        rollout_id: int,
        rollout_step: int,
        initial_strategy: str | None,
    ) -> str | None:
        prompt = build_trajectory_decision_prompt(
            task_description=self._task_description_str,
            anchor=anchor,
            window_text=window_text,
            maximize=self._maximize,
            initial_strategy=initial_strategy,
        )
        self._require_context(prompt, f"{stage}_decision")
        prompt_tokens = self._count_tokens(prompt)
        start = time.time()
        try:
            response = self._llm.draw_sample(
                prompt, max_tokens=TRAJECTORY_DECISION_MAX_TOKENS
            )
            sample_time = time.time() - start
        except Exception as exc:
            sample_time = time.time() - start
            self._record_llm_call(
                operator=TRAJECTORY_DECISION_OPERATOR,
                stage=f"{stage}_decision",
                iteration=iteration,
                anchor=anchor,
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
                rollout_id=rollout_id,
                rollout_step=rollout_step,
            )
            record_sample_failure(
                self,
                exc,
                stage=f"{stage}_decision",
                operator=TRAJECTORY_DECISION_OPERATOR,
                sample_order=self._tot_sample_nums + 1,
                counts_budget=False,
                iteration=iteration,
                parent_node_id=anchor.id,
            )
            return None
        idea = extract_idea(response)
        self._record_llm_call(
            operator=TRAJECTORY_DECISION_OPERATOR,
            stage=f"{stage}_decision",
            iteration=iteration,
            anchor=anchor,
            sample_order=None,
            sample_time=sample_time,
            prompt_tokens=prompt_tokens,
            response_tokens=self._count_tokens(response),
            status="ok" if idea is not None else "parse_failed",
            prompt=prompt,
            store_prompt=True,
            response=response,
            rollout_id=rollout_id,
            rollout_step=rollout_step,
        )
        if idea is None:
            record_sample_failure(
                self,
                ValueError("invalid trajectory decision"),
                stage=f"{stage}_decision",
                operator=TRAJECTORY_DECISION_OPERATOR,
                sample_order=self._tot_sample_nums + 1,
                counts_budget=False,
                iteration=iteration,
                parent_node_id=anchor.id,
            )
            return None
        reset_sample_failures(self)
        return idea[:IDEA_MAX_CHARS]

    def _attempt_program(
        self,
        prompt: str,
        *,
        stage: str,
        iteration: int | None,
        anchor: ProgramNode | None,
        template_program: Program,
        idea: str | None = None,
        rollout_id: int | None = None,
        rollout_step: int | None = None,
        rollout_start_anchor_id: int | None = None,
    ) -> _Attempt | None:
        llm_stage = stage if idea is None else f"{stage}_code"
        prompt_tokens = self._count_tokens(prompt)
        start = time.time()
        try:
            response = self._llm.draw_sample(prompt, max_tokens=self._code_max_tokens)
            sample_time = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
            sample_time = time.time() - start
            self._record_llm_call(
                stage=llm_stage,
                iteration=iteration,
                anchor=anchor,
                sample_order=self._tot_sample_nums + 1,
                sample_time=sample_time,
                prompt_tokens=prompt_tokens,
                response_tokens=0,
                status="transport",
                prompt=prompt,
                store_prompt=True,
                failure_kind="transport",
                error_type=type(exc).__name__,
                error=str(exc),
                rollout_id=rollout_id,
                rollout_step=rollout_step,
                rollout_start_anchor_id=rollout_start_anchor_id,
            )
            record_sample_failure(
                self,
                exc,
                stage=llm_stage,
                operator=GENERATION_OPERATOR,
                sample_order=self._tot_sample_nums + 1,
                counts_budget=False,
                iteration=iteration,
                parent_node_id=None if anchor is None else anchor.id,
            )
            return None

        self._tot_sample_nums += 1
        budget_order = self._tot_sample_nums
        parsed = (
            parse_program_response(
                response,
                template_program,
                self._function_to_evolve.name,
                signature_template=self._template_program,
            )
            if idea is None
            else parse_code_response(
                response,
                idea,
                template_program,
                self._function_to_evolve.name,
                signature_template=self._template_program,
            )
        )
        self._record_llm_call(
            stage=llm_stage,
            iteration=iteration,
            anchor=anchor,
            sample_order=budget_order,
            sample_time=sample_time,
            prompt_tokens=prompt_tokens,
            response_tokens=self._count_tokens(response),
            status="ok" if parsed is not None else "parse_failed",
            prompt=prompt,
            store_prompt=True,
            program_parse_success=parsed is not None,
            response=response,
            rollout_id=rollout_id,
            rollout_step=rollout_step,
            rollout_start_anchor_id=rollout_start_anchor_id,
        )
        if parsed is None:
            self._record_candidate(
                sample_order=budget_order,
                score=None,
                status="parse_failed",
                failure_kind="parse",
                program="",
                idea=idea or "",
                sample_time=sample_time,
                parent_node_id=None if anchor is None else anchor.id,
                iteration=iteration,
                stage=stage,
                rollout_id=rollout_id,
                rollout_step=rollout_step,
                rollout_start_anchor_id=rollout_start_anchor_id,
            )
            return _Attempt(
                idea or extract_idea(response) or "unavailable",
                None,
                None,
                "parse",
                budget_order,
                sample_time,
            )

        outcome, evaluate_time = (
            self._evaluator.evaluate_program_record_time_with_details(parsed.program)
        )
        self._evaluation_count += 1
        result = outcome.result
        score = getattr(result, "fitness", result)
        fitness: float | None = None
        if score is not None:
            try:
                candidate = float(score)
            except (TypeError, ValueError, OverflowError):
                candidate = math.nan
            if math.isfinite(candidate):
                fitness = candidate
        failure_kind = (
            None if fitness is not None else outcome.failure_kind or "invalid_result"
        )
        program_text = str(parsed.program)
        self._record_candidate(
            sample_order=budget_order,
            score=fitness,
            status="ok" if fitness is not None else "eval_failed",
            failure_kind=failure_kind,
            error_type=outcome.error_type,
            error=outcome.error,
            program=program_text,
            idea=parsed.idea,
            code_hash=code_hash(program_text),
            program_loc=nonempty_loc(program_text),
            evaluate_time=evaluate_time,
            sample_time=sample_time,
            parent_node_id=None if anchor is None else anchor.id,
            iteration=iteration,
            stage=stage,
            rollout_id=rollout_id,
            rollout_step=rollout_step,
            rollout_start_anchor_id=rollout_start_anchor_id,
        )
        return _Attempt(
            parsed.idea,
            parsed,
            fitness,
            failure_kind,
            budget_order,
            sample_time,
        )

    def _candidate_best_status(
        self, fitness: float, program_loc: int, creation_order: int
    ) -> tuple[bool, str | None]:
        if self._best_node is None:
            return True, "strict_fitness"
        directed = fitness if self._maximize else -fitness
        candidate_key = (directed, -program_loc, -creation_order)
        incumbent_key = (
            self._best_node.directed_fitness,
            -self._best_node.program_loc,
            -self._best_node.creation_order,
        )
        if candidate_key <= incumbent_key:
            return False, None
        return True, (
            "strict_fitness"
            if directed > self._best_node.directed_fitness
            else "tie_shorter"
        )

    def _update_best(self, candidate: ProgramNode) -> None:
        if not is_node_better(candidate, self._best_node):
            return
        old = self._best_node
        self._best_node = candidate
        self._best_node_sample_order = candidate.creation_order
        reason = (
            "strict_fitness"
            if old is None or candidate.directed_fitness > old.directed_fitness
            else "tie_shorter"
        )
        self._record_decision(
            "best_updated",
            node_id=candidate.id,
            sample_order=candidate.creation_order,
            reason=reason,
        )

    def _record_edge(self, event: GenerationEvent) -> None:
        if self._artifacts is None:
            return
        self._artifacts.record_edge(
            edge_id=event.id,
            parent_id=event.anchor_id,
            child_id=event.child_id,
            sample_order=event.budget_order,
            iteration=event.iteration,
            stage=event.stage,
            operator=GENERATION_OPERATOR,
            implemented_idea=event.idea,
            delta_parent=event.delta_parent,
            outcome=event.outcome,
            delta_loc=event.delta_loc,
            code_change_ratio=event.code_change_ratio,
            new_global_best=event.new_global_best,
            global_best_update_reason=event.global_best_update_reason,
            rollout_id=event.rollout_id,
            rollout_step=event.rollout_step,
            rollout_start_anchor_id=event.rollout_start_anchor_id,
        )

    def _record_candidate(self, **payload) -> None:
        if self._artifacts is not None:
            self._artifacts.record_candidate(operator=GENERATION_OPERATOR, **payload)

    def _record_llm_call(
        self,
        *,
        anchor: ProgramNode | None,
        operator: str = GENERATION_OPERATOR,
        **payload,
    ) -> None:
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                operator=operator,
                parent_node_id=None if anchor is None else anchor.id,
                token_count_mode=self._token_count_mode,
                **payload,
            )

    def _record_decision(self, event: str, **payload) -> None:
        if self._artifacts is not None:
            self._artifacts.record_decision(event, **payload)

    def _require_context(self, prompt: str, stage: str) -> None:
        tokens = self._count_tokens(prompt)
        if tokens > self._context_token_limit:
            raise ValueError(
                f"V9.3 {stage} context requires {tokens} tokens; "
                f"limit is {self._context_token_limit}"
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

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._tot_sample_nums - self._last_checkpoint_budget
            >= self._checkpoint_interval
        ):
            save_checkpoint(self)

    def _has_budget(self) -> bool:
        return (
            self._max_sample_nums is None
            or self._tot_sample_nums < self._max_sample_nums
        )

    def _result(self) -> TraceAADRunResult:
        nodes = self._graph.nodes()
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=len(nodes),
            n_valid_nodes=len(nodes),
            n_root_children=len(self._graph.root.child_ids),
            n_eligible_nodes=len(self._eligible_node_ids),
            n_events=len(self._graph.events()),
            n_samples=self._tot_sample_nums,
            n_evaluations=self._evaluation_count,
            n_iterations=self._next_iteration,
        )


__all__ = [
    "INITIAL_ANCHOR_COUNT",
    "INITIAL_ROUTE_POOL_SIZE",
    "TraceAADRunResult",
    "TraceAADV93",
]
