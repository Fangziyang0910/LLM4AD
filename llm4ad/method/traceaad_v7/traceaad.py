"""Independent implementation of the complete TraceAAD v7 mechanism."""

from __future__ import annotations

import copy
import hashlib
import math
import random
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
from ...tools.profiler import ProfilerBase
from .._observability import (
    close_llm,
    finish_profiler,
    init_observability,
    is_search_aborted,
    log_event,
    log_llm_call,
    record_sample_failure,
    reset_sample_failures,
)
from .checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from .complexity import code_change_ratio
from .context import (
    build_action_prompt,
    trajectory_history,
)
from .derivation_graph import DerivationGraph
from .operators import (
    DEFAULT_OPERATORS,
    Operator,
    classify_outcome,
    is_dual_operator,
    select_operator,
)
from .prompt import (
    build_code_prompt,
    build_initial_prompt,
    parse_actions,
    parse_program_response,
)
from .schema import PROTOCOL_ID, OperatorName, ProgramNode, Trajectory
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    compact_best_node,
    directed_delta,
    is_program_better,
    program_quality_key,
    reference_sampling_distribution,
    score_active_trajectories,
    search_value_survivor_sample,
    trajectory_sampling_distribution,
    weighted_choice,
)


@dataclass(frozen=True, slots=True)
class _GeneratedProgram:
    idea: str
    program: Program
    sample_time: float


@dataclass(frozen=True, slots=True)
class _PromptContext:
    prompt: str
    token_count: int
    primary_history: str
    reference_history: str
    primary_edge_ids: tuple[int, ...]
    reference_edge_ids: tuple[int, ...]
    used_dual: bool


@dataclass(frozen=True, slots=True)
class _EvaluatedCandidate:
    action: str
    seq: int
    generated: _GeneratedProgram
    fitness: float | None
    sample_order: int | None


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


class TraceAADV7:
    """Trace-guided trajectory evolution with executable-state routing.

    Every generated node and edge remains an auditable fact, while only one
    active trajectory may end in a given executable program state.
    """

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase = None,
        max_sample_nums: int | None = 100,
        *,
        n_init: int = 30,
        actions_per_iteration: int = 2,
        max_trajectory_length: int = 8,
        max_active_trajectories: int = 30,
        elite_count: int | None = None,
        softmax_temperature: float = 0.2,
        maximize: bool = True,
        value_weights: ValueWeights | None = None,
        operators: tuple[Operator, ...] = DEFAULT_OPERATORS,
        action_max_tokens: int = 1024,
        code_max_tokens: int = 8192,
        context_token_limit: int | None = None,
        random_seed: int | None = None,
        debug_mode: bool = False,
        max_consecutive_sample_failures: int = 20,
        max_stalled_iterations: int = 20,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        resume_from: str | Path | None = None,
    ) -> None:
        if n_init < 0:
            raise ValueError("n_init must be non-negative")
        if actions_per_iteration <= 0:
            raise ValueError("actions_per_iteration must be positive")
        if max_active_trajectories <= 0:
            raise ValueError("max_active_trajectories must be positive")
        if max_trajectory_length < 1:
            raise ValueError("max_trajectory_length must be positive")
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if context_token_limit is None or context_token_limit <= 0:
            raise ValueError(
                "context_token_limit must be explicitly set to a positive input limit"
            )
        self._llm = llm
        self._evaluation = evaluation
        self._profiler = profiler
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = int(n_init)
        self._actions_per_iteration = int(actions_per_iteration)
        self._max_active_trajectories = int(max_active_trajectories)
        self._management_threshold = 2 * self._max_active_trajectories
        self._elite_count = (
            max(2, math.ceil(0.1 * self._max_active_trajectories))
            if elite_count is None
            else max(1, int(elite_count))
        )
        self._softmax_temperature = float(softmax_temperature)
        self._maximize = bool(maximize)
        self._value_weights = value_weights or ValueWeights()
        self._debug_mode = debug_mode
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_batch = -1
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD v7 requires exactly one evolvable template function."
            )
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._operators = tuple(operators)
        if not self._operators:
            raise ValueError("at least one TraceAAD v7 operator is required")
        if not any(not is_dual_operator(operator.name) for operator in self._operators):
            raise ValueError(
                "TraceAAD v7 requires at least one single-trajectory operator"
            )
        self._action_max_tokens = int(action_max_tokens)
        self._code_max_tokens = int(code_max_tokens)
        if self._action_max_tokens <= 0 or self._code_max_tokens <= 0:
            raise ValueError("token limits must be positive")
        self._context_token_limit = int(context_token_limit)
        self._best_node: ProgramNode | None = None
        self._best_trajectory_id: int | None = None
        self._best_node_sample_order: int | None = None
        self._tot_sample_nums = 0
        self._next_attempt_id = 0
        self._batch_count = 0
        self._stalled_iterations = 0
        self._initialization_complete = False
        init_observability(self, max_consecutive_sample_failures)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)
        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent
            log_event(
                self,
                event="checkpoint_loaded",
                status="ok",
                checkpoint=str(checkpoint),
                sample_order=self._tot_sample_nums,
                next_attempt_id=self._next_attempt_id,
                batch_count=self._batch_count,
            )

    def runtime_identity(self) -> dict[str, str | None]:
        """Return the non-secret task/model identity required for safe resume."""

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
            "evaluation_type": (
                f"{type(self._evaluation).__module__}."
                f"{type(self._evaluation).__qualname__}"
            ),
            "evaluation_random_seed": setting(self._evaluation, "random_seed"),
            "evaluation_timeout_seconds": setting(self._evaluation, "timeout_seconds"),
            "evaluation_safe_evaluate": setting(self._evaluation, "safe_evaluate"),
            "evaluation_exec_code": setting(self._evaluation, "exec_code"),
            "evaluation_fork_proc": setting(self._evaluation, "fork_proc"),
            "evaluation_use_numba": setting(self._evaluation, "use_numba_accelerate"),
            "evaluation_use_protected_div": setting(
                self._evaluation, "use_protected_div"
            ),
            "llm_type": (
                f"{type(self._llm).__module__}.{type(self._llm).__qualname__}"
            ),
            "llm_model": (
                None
                if getattr(self._llm, "model", None) is None
                else str(self._llm.model)
            ),
            "llm_base_url": (
                None
                if getattr(self._llm, "base_url", None) is None
                else str(self._llm.base_url)
            ),
            "llm_max_tokens": setting(self._llm, "max_tokens"),
            "llm_temperature": setting(self._llm, "temperature"),
            "llm_enable_thinking": setting(self._llm, "enable_thinking"),
            "llm_do_auto_trim": setting(self._llm, "do_auto_trim"),
        }

    def search_configuration(self) -> dict:
        return {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "max_sample_nums": self._max_sample_nums,
            "n_init": self._n_init,
            "actions_per_iteration": self._actions_per_iteration,
            "max_trajectory_length": self._memory.max_trajectory_length,
            "max_active_trajectories": self._max_active_trajectories,
            "management_threshold": self._management_threshold,
            "elite_count": self._elite_count,
            "softmax_temperature": self._softmax_temperature,
            "maximize": self._maximize,
            "value_weights": {
                "endpoint_quality": self._value_weights.endpoint_quality,
                "best_quality": self._value_weights.best_quality,
                "search_quality": self._value_weights.search_quality,
                "search_trend": self._value_weights.search_trend,
                "ucb_c": self._value_weights.ucb_c,
                "discount": self._value_weights.discount,
                "positive_threshold": self._value_weights.positive_threshold,
            },
            "action_max_tokens": self._action_max_tokens,
            "code_max_tokens": self._code_max_tokens,
            "context_token_limit": self._context_token_limit,
            "max_consecutive_sample_failures": self._max_consecutive_sample_failures,
            "max_stalled_iterations": self._max_stalled_iterations,
            "checkpoint_interval": self._checkpoint_interval,
            "random_seed": self._random_seed,
            "operators": [str(operator.name) for operator in self._operators],
        }

    def run(self) -> TraceAADRunResult:
        attempt_id = self._next_attempt_id
        status = "error"
        error: dict[str, str] = {}
        result: TraceAADRunResult | None = None
        try:
            if not self._initialization_complete:
                self._initialize()
                self._initialization_complete = True
                save_checkpoint(self)
            while self._has_budget() and not is_search_aborted(self):
                if not self._memory.active():
                    log_event(
                        self, event="search_stopped", status="no_active_trajectory"
                    )
                    break
                valid_nodes_before = len(self._graph.nodes())
                self._run_iteration(attempt_id)
                stop_for_stall = False
                if len(self._graph.nodes()) == valid_nodes_before:
                    self._stalled_iterations += 1
                    if self._stalled_iterations >= self._max_stalled_iterations:
                        log_event(
                            self,
                            event="search_stopped",
                            status="stalled_generation",
                            attempt_id=attempt_id,
                        )
                        stop_for_stall = True
                else:
                    self._stalled_iterations = 0
                if len(self._memory.active()) >= self._management_threshold:
                    self._manage_population()
                attempt_id += 1
                self._next_attempt_id = attempt_id
                self._save_checkpoint_if_due()
                if stop_for_stall:
                    break
            if len(self._memory.active()) >= self._management_threshold:
                self._manage_population()
            result = self._result()
            status = "aborted" if is_search_aborted(self) else "finished"
            return result
        except Exception as exc:
            error = {
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
            raise
        finally:
            if result is None:
                result = self._result()
            save_checkpoint(self)
            finish_profiler(
                self,
                status=status,
                best_node_id=(
                    None if result.best_node is None else result.best_node.id
                ),
                best_score=(
                    None if result.best_node is None else result.best_node.fitness
                ),
                best_sample_order=self._best_node_sample_order,
                n_total_nodes=result.n_total_nodes,
                n_valid_nodes=result.n_valid_nodes,
                n_edges=result.n_edges,
                n_trajectories=result.n_trajectories,
                **error,
            )
            close_llm(self._llm)

    def _initialize(self) -> None:
        if len(self._memory.active()) >= self._n_init:
            return
        prompt = build_initial_prompt(
            task_description=self._task_description_str,
            template_function=self._function_to_evolve,
        )
        prompt_tokens = self._count_tokens(prompt)
        if prompt_tokens > self._context_token_limit:
            log_event(
                self,
                event="context_overflow",
                status="stopped",
                stage="init",
                prompt_tokens=prompt_tokens,
                context_token_limit=self._context_token_limit,
            )
            return

        stalled_draws = 0
        draw_seq = 0
        while (
            len(self._memory.active()) < self._n_init
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            active_count_before = len(self._memory.active())
            generated = self._draw_program(
                prompt,
                stage="init",
                iteration=None,
                seq=draw_seq,
                operator="init",
                max_tokens=self._code_max_tokens,
                prompt_tokens=prompt_tokens,
            )
            draw_seq += 1
            if generated is not None:
                evaluated, sample_order = self._evaluate_detailed(
                    generated.program,
                    idea=generated.idea,
                    operator="init",
                    sample_time=generated.sample_time,
                )
                if evaluated is not None:
                    node = self._graph.add_node(
                        code=str(generated.program),
                        idea=generated.idea,
                        fitness=evaluated,
                    )
                    route = self._memory.create_initial(node_id=node.id)
                    route_keepers = self._deduplicate_active_endpoints()
                    kept_route_id = route_keepers.get(route.id, route.id)
                    self._update_best(
                        node,
                        trajectory_id=kept_route_id,
                        iteration=None,
                        operator="init",
                        sample_order=sample_order,
                    )
                    log_event(
                        self,
                        event="trajectory_created",
                        status="ok",
                        stage="init",
                        node_id=node.id,
                        trajectory_id=route.id,
                        active=(
                            self._memory.get_trajectory(route.id).status.value
                            == "active"
                        ),
                        program_loc=node.program_loc,
                        code_hash=node.code_hash,
                    )
            if len(self._memory.active()) > active_count_before:
                stalled_draws = 0
            else:
                stalled_draws += 1
                if stalled_draws >= self._max_stalled_iterations:
                    log_event(
                        self,
                        event="initialization_stopped",
                        status="stalled_unique_population",
                        active_count=len(self._memory.active()),
                        target_count=self._n_init,
                    )
                    break
        if self._memory.active():
            score_active_trajectories(
                memory=self._memory,
                graph=self._graph,
                maximize=self._maximize,
                w=self._value_weights,
            )

    def _run_iteration(self, attempt_id: int) -> None:
        # Freeze the state used for all children in this batch.
        global_best_before = self._best_node

        selected, active_before = self._select_trajectory(
            attempt_id, batch_count=self._batch_count
        )
        self._batch_count += 1
        visited = self._memory.record_visit(selected.id)

        anchor_id, anchor_role = self._select_anchor(selected)
        decision = select_operator(
            operators=self._operators,
            allow_dual=len(active_before) > 1,
            rng=self._rng,
        )
        operator = decision.operator
        reference_route: Trajectory | None = None
        reference_node: ProgramNode | None = None
        if decision.use_dual:
            reference_distribution = reference_sampling_distribution(
                primary=selected,
                active=active_before,
                temperature=self._softmax_temperature,
            )
            if reference_distribution:
                reference_route = weighted_choice(
                    [route for route, _, _ in reference_distribution],
                    [probability for _, _, probability in reference_distribution],
                    self._rng,
                )
                reference_node = self._graph.get_node(reference_route.compact_best_id)
            else:
                decision = select_operator(
                    operators=self._operators,
                    allow_dual=False,
                    rng=self._rng,
                )
                operator = decision.operator

        context = self._build_action_context(
            selected=selected,
            anchor_id=anchor_id,
            operator=operator,
            reference_route=reference_route,
            reference_node=reference_node,
        )
        if context is None and decision.use_dual:
            previous_operator = operator.name
            decision = select_operator(
                operators=self._operators,
                allow_dual=False,
                rng=self._rng,
            )
            operator = decision.operator
            reference_route = None
            reference_node = None
            context = self._build_action_context(
                selected=selected,
                anchor_id=anchor_id,
                operator=operator,
                reference_route=None,
                reference_node=None,
            )
            if context is not None:
                log_event(
                    self,
                    event="operator_fallback",
                    status="ok",
                    attempt_id=attempt_id,
                    from_operator=previous_operator,
                    to_operator=operator.name,
                    reason="dual_action_context_overflow",
                )
        if context is None:
            log_event(
                self,
                event="context_overflow",
                status="skipped",
                stage="action",
                attempt_id=attempt_id,
                selected_trajectory_id=selected.id,
                base_node_id=anchor_id,
                context_token_limit=self._context_token_limit,
            )
            return

        log_event(
            self,
            event="operator_selection",
            status="ok",
            attempt_id=attempt_id,
            selected_operator=operator.name,
            selected_trajectory_id=selected.id,
            base_node_id=anchor_id,
            anchor_role=anchor_role,
            selection_reason=decision.reason,
            used_dual=context.used_dual,
            reference_trajectory_id=(
                None if reference_route is None else reference_route.id
            ),
            reference_program_id=(
                None if reference_node is None else reference_node.id
            ),
            parent_visit_count=visited.visit_count,
            batch_count=self._batch_count,
        )

        actions = self._generate_actions(
            context,
            operator=operator,
            iteration=attempt_id,
        )
        base_node = self._graph.get_node(anchor_id)
        route_best_before = compact_best_node(selected, self._graph, self._maximize)

        def prepare_code_requests(
            action_items: list[str],
        ) -> list[tuple[int, str, str, int]]:
            requests: list[tuple[int, str, str, int]] = []
            for seq, action in enumerate(action_items):
                prompt = build_code_prompt(
                    current_node=base_node,
                    action=action,
                    task_description=self._task_description_str,
                    template_function=self._function_to_evolve,
                    history=context.primary_history,
                    reference_node=reference_node,
                    reference_history=context.reference_history,
                )
                requests.append((seq, action, prompt, self._count_tokens(prompt)))
            return requests

        code_requests = prepare_code_requests(actions)
        if context.used_dual and any(
            tokens > self._context_token_limit for _, _, _, tokens in code_requests
        ):
            previous_operator = operator.name
            decision = select_operator(
                operators=self._operators,
                allow_dual=False,
                rng=self._rng,
            )
            operator = decision.operator
            reference_route = None
            reference_node = None
            single_context = self._build_action_context(
                selected=selected,
                anchor_id=anchor_id,
                operator=operator,
                reference_route=None,
                reference_node=None,
            )
            if single_context is None:
                log_event(
                    self,
                    event="context_overflow",
                    status="skipped",
                    stage="code_fallback",
                    attempt_id=attempt_id,
                    selected_trajectory_id=selected.id,
                    base_node_id=anchor_id,
                    context_token_limit=self._context_token_limit,
                )
                return
            context = single_context
            log_event(
                self,
                event="operator_fallback",
                status="ok",
                attempt_id=attempt_id,
                from_operator=previous_operator,
                to_operator=operator.name,
                reason="dual_code_context_overflow",
            )
            actions = self._generate_actions(
                context,
                operator=operator,
                iteration=attempt_id,
            )
            code_requests = prepare_code_requests(actions)

        evaluated_candidates: list[_EvaluatedCandidate] = []
        for seq, action, code_prompt, code_prompt_tokens in code_requests:
            if not self._has_budget() or is_search_aborted(self):
                break
            if code_prompt_tokens > self._context_token_limit:
                log_event(
                    self,
                    event="context_overflow",
                    status="skipped",
                    stage="code",
                    attempt_id=attempt_id,
                    seq=seq,
                    operator=operator.name,
                    action=action,
                    prompt_tokens=code_prompt_tokens,
                    context_token_limit=self._context_token_limit,
                )
                continue
            generated = self._draw_program(
                code_prompt,
                stage="code",
                iteration=attempt_id,
                seq=seq,
                operator=operator.name,
                action=action,
                max_tokens=self._code_max_tokens,
                prompt_tokens=code_prompt_tokens,
            )
            if generated is None:
                continue
            fitness, sample_order = self._evaluate_detailed(
                generated.program,
                idea=generated.idea,
                operator=operator.name,
                sample_time=generated.sample_time,
            )
            evaluated_candidates.append(
                _EvaluatedCandidate(
                    action=action,
                    seq=seq,
                    generated=generated,
                    fitness=fitness,
                    sample_order=sample_order,
                )
            )

        created: list[tuple[_EvaluatedCandidate, ProgramNode]] = []
        for item in evaluated_candidates:
            if item.fitness is None:
                continue
            child = self._graph.add_node(
                code=str(item.generated.program),
                idea=item.generated.idea,
                fitness=item.fitness,
            )
            created.append((item, child))

        # Choose one batch winner from the frozen global-best snapshot.  This
        # keeps equal action order from changing best-program labels.
        global_candidates = [
            (item, child)
            for item, child in created
            if is_program_better(child, global_best_before, self._maximize)
        ]
        global_winner: tuple[_EvaluatedCandidate, ProgramNode] | None = None
        if global_candidates:
            best_key = max(
                program_quality_key(child, self._maximize)
                for _, child in global_candidates
            )
            tied = [
                pair
                for pair in global_candidates
                if program_quality_key(pair[1], self._maximize) == best_key
            ]
            global_winner = min(
                tied,
                key=lambda pair: (
                    pair[1].code_hash,
                    pair[1].idea,
                    pair[0].action,
                ),
            )
        global_winner_id = None if global_winner is None else global_winner[1].id
        global_winner_reason = (
            None
            if global_winner is None
            else self._best_update_decision(global_winner[1], global_best_before)[1]
        )

        accepted_count = 0
        child_routes: dict[int, Trajectory] = {}
        for item, child in created:
            delta_parent = directed_delta(
                base_node.fitness, child.fitness, self._maximize
            )
            delta_route = directed_delta(
                route_best_before.fitness, child.fitness, self._maximize
            )
            delta_global = directed_delta(
                None if global_best_before is None else global_best_before.fitness,
                child.fitness,
                self._maximize,
            )
            outcome = classify_outcome(
                delta_parent, self._value_weights.positive_threshold
            )
            new_global_best = child.id == global_winner_id
            global_update_reason = global_winner_reason if new_global_best else None
            new_compact = self._compact_best_for_child(
                selected=selected,
                anchor_id=anchor_id,
                child=child,
            )
            edge = self._graph.add_edge(
                parent_id=anchor_id,
                child_id=child.id,
                action=item.action,
                operator=operator.name,
                anchor_role=anchor_role,
                primary_trajectory_id=selected.id,
                reference_trajectory_id=(
                    None if reference_route is None else reference_route.id
                ),
                reference_program_id=(
                    None if reference_node is None else reference_node.id
                ),
                delta_parent=delta_parent,
                delta_route_best=delta_route,
                delta_global_best=delta_global,
                delta_loc=child.program_loc - base_node.program_loc,
                code_change_ratio=code_change_ratio(base_node.code, child.code),
                outcome=outcome,
                iteration=attempt_id,
                new_global_best=new_global_best,
                global_best_update_reason=global_update_reason,
            )
            child_route = self._memory.branch_from(
                trajectory_id=selected.id,
                base_node_id=anchor_id,
                child_id=child.id,
                edge_id=edge.id,
                compact_best_id=new_compact.id,
            )
            child_routes[child.id] = child_route
            log_event(
                self,
                event="child_accepted",
                status="ok",
                iteration=attempt_id,
                seq=item.seq,
                operator=operator.name,
                child_id=child.id,
                trajectory_id=child_route.id,
                compact_best_id=child_route.compact_best_id,
                score=child.fitness,
                parent_id=anchor_id,
                edge_id=edge.id,
                action=item.action,
                delta_parent=delta_parent,
                delta_route_best=delta_route,
                delta_global_best=delta_global,
                outcome=outcome,
                anchor_role=anchor_role,
                reference_trajectory_id=(
                    None if reference_route is None else reference_route.id
                ),
                program_loc=child.program_loc,
                delta_loc=child.program_loc - base_node.program_loc,
                code_hash=child.code_hash,
                global_best_update_reason=global_update_reason,
            )
            accepted_count += 1

        # Duplicates remain in the derivation graph but cannot consume more
        # than one active reproduction slot.
        route_keepers = self._deduplicate_active_endpoints()
        for child_id, child_route in tuple(child_routes.items()):
            kept_id = route_keepers.get(child_route.id, child_route.id)
            child_routes[child_id] = self._memory.get_trajectory(kept_id)

        if global_winner is not None:
            winner_item, winner_node = global_winner
            self._update_best(
                winner_node,
                trajectory_id=child_routes[winner_node.id].id,
                iteration=attempt_id,
                operator=operator.name,
                sample_order=winner_item.sample_order,
            )

        if self._memory.active():
            score_active_trajectories(
                memory=self._memory,
                graph=self._graph,
                maximize=self._maximize,
                w=self._value_weights,
            )
        log_event(
            self,
            event="operator_batch",
            status="ok" if accepted_count else "empty",
            attempt_id=attempt_id,
            operator=operator.name,
            candidate_count=accepted_count,
            active_trajectories=len(self._memory.active()),
            parent_visit_count=visited.visit_count,
            batch_count=self._batch_count,
        )

    def _select_trajectory(
        self, attempt_id: int, *, batch_count: int
    ) -> tuple[Trajectory, tuple[Trajectory, ...]]:
        distribution = trajectory_sampling_distribution(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
            remaining_budget_ratio=self._remaining_budget_ratio(),
            selection_count=batch_count,
        )
        routes = [item[0] for item in distribution]
        probs = [item[2] for item in distribution]
        selected = weighted_choice(routes, probs, self._rng)
        selected_adjusted, selected_prob = next(
            (adjusted, probability)
            for route, adjusted, probability in distribution
            if route.id == selected.id
        )
        sorted_probs = sorted(probs, reverse=True)
        log_event(
            self,
            event="trajectory_selection",
            status="ok",
            attempt_id=attempt_id,
            selected_trajectory_id=selected.id,
            active_count=len(routes),
            selected_probability=selected_prob,
            selected_adjusted_score=selected_adjusted,
            selected_quality=(
                None if selected.value is None else selected.value.quality
            ),
            selected_visit_count=selected.visit_count,
            batch_count=batch_count,
            remaining_budget_ratio=self._remaining_budget_ratio(),
            max_probability=sorted_probs[0],
            top5_probability_mass=sum(sorted_probs[:5]),
            effective_candidate_count=1.0 / sum(prob * prob for prob in probs),
        )
        return selected, tuple(route for route, _, _ in distribution)

    def _select_anchor(self, trajectory: Trajectory) -> tuple[int, str]:
        endpoint = trajectory.endpoint_id
        compact = trajectory.compact_best_id
        if endpoint == compact:
            return endpoint, "endpoint_compact_best"
        if self._rng.randrange(2) == 0:
            return endpoint, "endpoint"
        return compact, "compact_best"

    def _build_action_context(
        self,
        *,
        selected: Trajectory,
        anchor_id: int,
        operator: Operator,
        reference_route: Trajectory | None,
        reference_node: ProgramNode | None,
    ) -> _PromptContext | None:
        use_reference = reference_route is not None and reference_node is not None
        max_steps = self._memory.max_trajectory_length
        for history_steps in range(max_steps, -1, -1):
            primary_history = trajectory_history(
                self._graph,
                selected,
                base_node_id=anchor_id,
                max_steps=history_steps,
            )
            ref_history = (
                trajectory_history(
                    self._graph,
                    reference_route,
                    base_node_id=reference_node.id,
                    max_steps=history_steps,
                )
                if use_reference
                else None
            )
            prompt = build_action_prompt(
                base_node=self._graph.get_node(anchor_id),
                primary_history=primary_history,
                operator_constraint=operator.prompt_constraint,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                action_count=self._actions_per_iteration,
                maximize=self._maximize,
                reference_node=reference_node if use_reference else None,
                reference_history=ref_history,
            )
            token_count = self._count_tokens(prompt)
            if token_count > self._context_token_limit:
                continue
            return _PromptContext(
                prompt=prompt,
                token_count=token_count,
                primary_history=primary_history.text,
                reference_history=("" if ref_history is None else ref_history.text),
                primary_edge_ids=primary_history.edge_ids,
                reference_edge_ids=(
                    () if ref_history is None else ref_history.edge_ids
                ),
                used_dual=use_reference,
            )
        return None

    def _count_tokens(self, prompt: str) -> int:
        counter = getattr(self._llm, "count_tokens", None)
        if callable(counter):
            return int(counter(prompt))
        return len(prompt.encode("utf-8"))

    @property
    def _token_count_mode(self) -> str:
        reported_mode = getattr(self._llm, "token_count_mode", None)
        if isinstance(reported_mode, str):
            return reported_mode
        return (
            "llm_count_tokens"
            if callable(getattr(self._llm, "count_tokens", None))
            else "utf8_byte_upper_bound"
        )

    def _generate_actions(
        self,
        context: _PromptContext,
        *,
        operator: Operator,
        iteration: int,
    ) -> list[str]:
        start = time.time()
        try:
            response = self._llm.draw_sample(
                context.prompt,
                max_tokens=self._action_max_tokens,
            )
            sample_time = time.time() - start
        except Exception as exc:
            sample_time = time.time() - start
            record_sample_failure(
                self,
                exc,
                stage="action",
                operator=operator.name,
                sample_order=self._tot_sample_nums + 1,
                prompt=context.prompt,
                counts_budget=False,
                iteration=iteration,
            )
            log_llm_call(
                self,
                stage="action",
                operator=operator.name,
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=0,
                prompt=context.prompt,
                response=None,
                sample_time=sample_time,
                prompt_tokens=context.token_count,
                response_tokens=0,
                token_count_mode=self._token_count_mode,
                status="llm_error",
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
            return []
        actions, errors = parse_actions(
            response,
            expected_count=self._actions_per_iteration,
        )
        if actions:
            reset_sample_failures(self)
        else:
            self._record_generation_failure(
                stage="action_parse",
                iteration=iteration,
                operator=operator.name,
            )
        log_llm_call(
            self,
            stage="action",
            operator=operator.name,
            sample_order=self._tot_sample_nums + 1,
            iteration=iteration,
            seq=0,
            prompt=context.prompt,
            response=response,
            sample_time=sample_time,
            prompt_tokens=context.token_count,
            response_tokens=self._count_tokens(response),
            token_count_mode=self._token_count_mode,
            parsed_actions=actions,
            parse_errors=errors,
            primary_edge_ids=list(context.primary_edge_ids),
            reference_edge_ids=list(context.reference_edge_ids),
            status="ok" if actions else "parse_failed",
        )
        return actions

    def _compact_best_for_child(
        self,
        *,
        selected: Trajectory,
        anchor_id: int,
        child: ProgramNode,
    ) -> ProgramNode:
        anchor_index = selected.node_ids.index(anchor_id)
        candidates = [
            self._graph.get_node(node_id)
            for node_id in selected.node_ids[: anchor_index + 1]
        ]
        candidates.append(child)
        candidates = candidates[-self._memory.max_trajectory_length :]
        best = candidates[0]
        for candidate in candidates[1:]:
            if is_program_better(candidate, best, self._maximize):
                best = candidate
        return best

    def _best_update_decision(
        self,
        candidate: ProgramNode,
        incumbent: ProgramNode | None,
    ) -> tuple[bool, str | None]:
        if not is_program_better(candidate, incumbent, self._maximize):
            return False, None
        if (
            incumbent is None
            or directed_delta(incumbent.fitness, candidate.fitness, self._maximize) > 0
        ):
            return True, "strict_fitness"
        return True, "tie_shorter"

    def _update_best(
        self,
        node: ProgramNode,
        *,
        trajectory_id: int | None,
        iteration: int | None,
        operator: str | OperatorName,
        sample_order: int | None = None,
    ) -> None:
        update, reason = self._best_update_decision(node, self._best_node)
        if not update:
            return
        previous = self._best_node
        self._best_node = node
        actual_sample_order = (
            self._tot_sample_nums if sample_order is None else int(sample_order)
        )
        self._best_node_sample_order = actual_sample_order
        if trajectory_id is not None:
            self._best_trajectory_id = trajectory_id
        log_event(
            self,
            event="best_updated",
            status="ok",
            iteration=iteration,
            sample_order=actual_sample_order,
            previous_best_node_id=None if previous is None else previous.id,
            new_best_node_id=node.id,
            operator=operator,
            delta_to_previous_best=directed_delta(
                None if previous is None else previous.fitness,
                node.fitness,
                self._maximize,
            ),
            update_reason=reason,
            program_loc=node.program_loc,
        )

    def _manage_population(self) -> None:
        ranked = list(
            score_active_trajectories(
                memory=self._memory,
                graph=self._graph,
                maximize=self._maximize,
                w=self._value_weights,
            )
        )
        if len(ranked) < self._management_threshold:
            return
        best_route = next(
            (route for route in ranked if route.id == self._best_trajectory_id),
            None,
        )
        if best_route is None and self._best_node is not None:
            best_route = next(
                (route for route in ranked if self._best_node.id in route.node_ids),
                None,
            )
        if best_route is None and self._best_node is not None:
            best_route = next(
                (
                    route
                    for route in ranked
                    if self._graph.get_node(route.endpoint_id).code_hash
                    == self._best_node.code_hash
                ),
                None,
            )
        if best_route is None:
            best_route = ranked[0]
        quality_ranked = sorted(
            (route for route in ranked if route.id != best_route.id),
            key=lambda route: (
                -(route.value.quality if route.value is not None else 0.0),
                -(route.scalar_value if route.scalar_value is not None else 0.0),
                route.id,
            ),
        )
        elites = quality_ranked[
            : min(self._elite_count, self._max_active_trajectories - 1)
        ]
        elite_ids = {route.id for route in elites}
        remaining = [route for route in quality_ranked if route.id not in elite_ids]
        sampled = search_value_survivor_sample(
            remaining,
            max(0, self._max_active_trajectories - 1 - len(elites)),
            temperature=self._softmax_temperature,
            rng=self._rng,
        )
        keep_ids = {
            best_route.id,
            *(route.id for route in elites),
            *(route.id for route in sampled),
        }
        decisions = []
        for route in ranked:
            keep = route.id in keep_ids
            if not keep:
                self._memory.archive(route.id)
            recorded_route = self._memory.get_trajectory(route.id)
            decisions.append(
                {
                    "trajectory_id": route.id,
                    "decision": "keep" if keep else "archive",
                    "roles": (
                        ["global_best"]
                        if route.id == best_route.id
                        else ["quality_elite"]
                        if route.id in elite_ids
                        else ["qp_softmax"]
                        if keep
                        else []
                    ),
                    "quality": (
                        None
                        if recorded_route.value is None
                        else recorded_route.value.quality
                    ),
                }
            )
        log_event(
            self,
            event="population_management",
            status="ok",
            management_threshold=self._management_threshold,
            before=len(ranked),
            after=len(self._memory.active()),
            decisions=decisions,
            selection_mode="global_best_quality_elite_qp_softmax",
        )
        save_checkpoint(self)

    def _deduplicate_active_endpoints(self) -> dict[int, int]:
        """Archive duplicate active endpoints and return each route's keeper id."""

        groups: dict[str, list[Trajectory]] = {}
        for route in self._memory.active():
            endpoint = self._graph.get_node(route.endpoint_id)
            groups.setdefault(endpoint.code_hash, []).append(route)

        keepers: dict[int, int] = {}
        for endpoint_hash, routes in groups.items():
            keeper = min(
                routes,
                key=lambda route: (
                    0 if route.id == self._best_trajectory_id else 1,
                    route.id,
                ),
            )
            for route in routes:
                keepers[route.id] = keeper.id
                if route.id == keeper.id:
                    continue
                archived = self._memory.archive(route.id)
                log_event(
                    self,
                    event="trajectory_archived_duplicate_endpoint",
                    status="ok",
                    trajectory_id=archived.id,
                    kept_trajectory_id=keeper.id,
                    endpoint_id=archived.endpoint_id,
                    endpoint_code_hash=endpoint_hash,
                )
        return keepers

    def _draw_program(
        self,
        prompt: str,
        *,
        stage: str,
        iteration: int | None,
        seq: int,
        operator: str | OperatorName,
        action: str | None = None,
        max_tokens: int | None = None,
        prompt_tokens: int,
    ) -> _GeneratedProgram | None:
        sample_order = self._tot_sample_nums + 1
        start = time.time()
        try:
            if max_tokens is None:
                response = self._llm.draw_sample(prompt)
            else:
                response = self._llm.draw_sample(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start
        except Exception as exc:
            elapsed = time.time() - start
            record_sample_failure(
                self,
                exc,
                stage=stage,
                operator=operator,
                sample_order=sample_order,
                prompt=prompt,
                counts_budget=False,
                iteration=iteration,
                seq=seq,
                action=action,
            )
            log_llm_call(
                self,
                stage=stage,
                operator=operator,
                sample_order=sample_order,
                iteration=iteration,
                seq=seq,
                sample_time=elapsed,
                prompt_tokens=prompt_tokens,
                response_tokens=0,
                token_count_mode=self._token_count_mode,
                prompt=prompt,
                response=None,
                action=action,
                status="llm_error",
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
            return None
        generated = parse_program_response(
            response, self._template_program, self._function_to_evolve.name
        )
        call_record = {
            "stage": stage,
            "operator": operator,
            "sample_order": sample_order,
            "iteration": iteration,
            "seq": seq,
            "sample_time": elapsed,
            "prompt_tokens": prompt_tokens,
            "response_tokens": self._count_tokens(response),
            "token_count_mode": self._token_count_mode,
            "program_parse_success": generated is not None,
            "status": "ok" if generated is not None else "parse_failed",
            "prompt": prompt,
            "response": response,
            "action": action,
        }
        log_llm_call(self, **call_record)
        if generated is None:
            self._record_generation_failure(
                stage=f"{stage}_parse",
                iteration=iteration,
                operator=operator,
            )
            return None
        reset_sample_failures(self)
        return _GeneratedProgram(
            idea=generated.idea,
            program=generated.program,
            sample_time=elapsed,
        )

    def _evaluate_detailed(
        self,
        program: Program,
        *,
        idea: str,
        operator: str | OperatorName,
        sample_time: float,
    ) -> tuple[float | None, int | None]:
        if not self._has_budget():
            return None, None
        outcome, eval_time = self._evaluator.evaluate_program_record_time_with_details(
            program
        )
        self._tot_sample_nums += 1
        result = outcome.result
        raw_score = getattr(result, "fitness", result)
        score: float | None = None
        failure_kind = outcome.failure_kind
        failure_error_type = outcome.error_type
        failure_error = outcome.error
        if raw_score is not None:
            try:
                numeric_score = float(raw_score)
            except (TypeError, ValueError, OverflowError) as exc:
                failure_kind = "invalid_result"
                failure_error_type = type(exc).__name__
                failure_error = str(exc)
            else:
                if math.isfinite(numeric_score):
                    score = numeric_score
                else:
                    failure_kind = "invalid_result"
                    failure_error_type = "NonFiniteEvaluationResult"
                    failure_error = (
                        f"evaluator returned non-finite score: {numeric_score}"
                    )
        function = copy.deepcopy(program.get_function(self._function_to_evolve.name))
        function.algorithm = idea
        function.score = score
        function.sample_time = sample_time
        function.evaluate_time = eval_time
        function.operator = str(operator)
        if self._profiler is not None:
            self._profiler.register_function(function, program=str(program))
        failure = (
            {}
            if score is not None
            else {
                "failure_kind": failure_kind or "invalid_result",
                "error_type": failure_error_type,
                "error": failure_error,
            }
        )
        log_event(
            self,
            event="program_evaluated",
            status="ok" if score is not None else "eval_failed",
            operator=operator,
            sample_order=self._tot_sample_nums,
            score=score,
            evaluate_time=eval_time,
            counts_budget=True,
            **failure,
        )
        return score, self._tot_sample_nums

    def _record_generation_failure(
        self,
        *,
        stage: str,
        iteration: int | None,
        operator: str | OperatorName,
    ) -> None:
        self._consecutive_sample_failures += 1
        log_event(
            self,
            event="generation_failure",
            status="parse_failed",
            stage=stage,
            iteration=iteration,
            operator=operator,
            consecutive_failures=self._consecutive_sample_failures,
            max_consecutive_failures=self._max_consecutive_sample_failures,
        )
        if self._consecutive_sample_failures >= self._max_consecutive_sample_failures:
            self._search_aborted = True
            log_event(
                self,
                event="search_aborted",
                status="aborted",
                reason="max_consecutive_sample_failures",
                consecutive_failures=self._consecutive_sample_failures,
                max_consecutive_failures=self._max_consecutive_sample_failures,
            )

    def _save_checkpoint_if_due(self) -> None:
        if self._checkpoint_dir is None:
            return
        if self._batch_count - self._last_checkpoint_batch >= self._checkpoint_interval:
            save_checkpoint(self)

    def active_trajectories(self) -> tuple[Trajectory, ...]:
        return self._memory.active()

    def _has_budget(self) -> bool:
        return (
            self._max_sample_nums is None
            or self._tot_sample_nums < self._max_sample_nums
        )

    def _remaining_budget_ratio(self) -> float:
        if self._max_sample_nums is None:
            return 1.0
        if self._max_sample_nums <= 0:
            return 0.0
        remaining = self._max_sample_nums - self._tot_sample_nums
        return min(1.0, max(0.0, remaining / self._max_sample_nums))

    def _result(self) -> TraceAADRunResult:
        nodes = self._graph.nodes()
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=len(nodes),
            n_valid_nodes=len(nodes),
            n_trajectories=len(self._memory.trajectories()),
            n_edges=len(self._graph.edges()),
            n_samples=self._tot_sample_nums,
        )


__all__ = ["TraceAADRunResult", "TraceAADV7"]
