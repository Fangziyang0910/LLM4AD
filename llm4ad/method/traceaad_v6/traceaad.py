"""Independent implementation of the complete TraceAAD v6 mechanism."""

from __future__ import annotations

import copy
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
from .attempts import AttemptMemory
from .checkpoint import load_checkpoint, save_checkpoint
from .complexity import code_change_ratio
from .context import (
    build_action_prompt,
    reference_history,
    render_tested_attempts,
    trajectory_history,
)
from .derivation_graph import DerivationGraph
from .operators import (
    DEFAULT_OPERATORS,
    Operator,
    classify_outcome,
    recent_route_progress,
    select_operator,
)
from .prompt import (
    build_code_prompt,
    build_initial_prompt,
    parse_actions,
    parse_program_response,
)
from .schema import AnchorAttempt, OperatorName, ProgramNode, Trajectory
from .similarity import route_difference
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    compact_best_node,
    deduplicate_by_endpoint_hash,
    directed_delta,
    edge_credit,
    is_mature_trajectory,
    is_program_better,
    meaningful_improvement,
    program_quality_key,
    qualified_reference_candidates,
    reference_sampling_distribution,
    score_active_trajectories,
    select_diverse_trajectories,
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
    failure_kind: str | None
    sample_order: int | None


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


class TraceAADV6:
    """Quality-gated trajectory evolution with route-credit parent selection."""

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
        max_tested_attempts: int = 6,
        elite_count: int | None = None,
        diversity_count: int | None = None,
        softmax_temperature: float = 0.2,
        dual_probability: float = 0.25,
        maximize: bool = True,
        value_weights: ValueWeights | None = None,
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
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
        if not 0.0 <= dual_probability <= 1.0:
            raise ValueError("dual_probability must be in [0, 1]")
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
        self._max_tested_attempts = max(1, int(max_tested_attempts))
        self._elite_count = (
            max(2, math.ceil(0.1 * self._max_active_trajectories))
            if elite_count is None
            else max(1, int(elite_count))
        )
        self._diversity_count = (
            max(2, math.ceil(0.1 * self._max_active_trajectories))
            if diversity_count is None
            else max(0, int(diversity_count))
        )
        self._softmax_temperature = float(softmax_temperature)
        self._dual_probability = float(dual_probability)
        self._maximize = bool(maximize)
        self._value_weights = value_weights or ValueWeights()
        self._debug_mode = debug_mode
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_sample = -1
        self._last_checkpoint_batch = -1
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD v6 requires exactly one evolvable template function."
            )
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._attempts = AttemptMemory()
        self._operators = tuple(operator_type() for operator_type in operators)
        if not self._operators:
            raise ValueError("at least one TraceAAD v6 operator is required")
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

    def search_configuration(self) -> dict:
        return {
            "max_sample_nums": self._max_sample_nums,
            "n_init": self._n_init,
            "actions_per_iteration": self._actions_per_iteration,
            "max_trajectory_length": self._memory.max_trajectory_length,
            "max_active_trajectories": self._max_active_trajectories,
            "management_threshold": self._management_threshold,
            "max_tested_attempts": self._max_tested_attempts,
            "elite_count": self._elite_count,
            "diversity_count": self._diversity_count,
            "softmax_temperature": self._softmax_temperature,
            "dual_probability": self._dual_probability,
            "maximize": self._maximize,
            "value_weights": {
                "endpoint_quality": self._value_weights.endpoint_quality,
                "best_quality": self._value_weights.best_quality,
                "search_quality": self._value_weights.search_quality,
                "search_credit": self._value_weights.search_credit,
                "ucb_c": self._value_weights.ucb_c,
                "discount": self._value_weights.discount,
                "positive_threshold": self._value_weights.positive_threshold,
                "mature_quantile": self._value_weights.mature_quantile,
                "mature_min_edges": self._value_weights.mature_min_edges,
            },
            "action_max_tokens": self._action_max_tokens,
            "code_max_tokens": self._code_max_tokens,
            "context_token_limit": self._context_token_limit,
            "max_consecutive_sample_failures": self._max_consecutive_sample_failures,
            "max_stalled_iterations": self._max_stalled_iterations,
            "checkpoint_interval": self._checkpoint_interval,
            "random_seed": self._random_seed,
            "operators": [operator.name for operator in self._operators],
        }

    def run(self) -> TraceAADRunResult:
        attempt_id = self._next_attempt_id
        status = "error"
        error: dict[str, str] = {}
        result: TraceAADRunResult | None = None
        try:
            if not self._initialization_complete:
                self._initialize()
                if len(self._memory.trajectories()) >= self._n_init:
                    self._initialization_complete = True
                save_checkpoint(self)
            while self._has_budget() and not is_search_aborted(self):
                if not self._memory.active():
                    log_event(
                        self, event="search_stopped", status="no_active_trajectory"
                    )
                    break
                before = self._tot_sample_nums
                self._run_iteration(attempt_id)
                stop_for_stall = False
                if self._tot_sample_nums == before:
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
        stalled_draws = 0
        draw_seq = 0
        while (
            len(self._memory.trajectories()) < self._n_init
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=self._init_diversity_hint(
                    len(self._memory.trajectories())
                ),
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
                break
            generated, _ = self._draw_program(
                prompt,
                stage="init",
                iteration=None,
                seq=draw_seq,
                operator="init",
                max_tokens=self._code_max_tokens,
            )
            draw_seq += 1
            if generated is None:
                stalled_draws += 1
                if stalled_draws >= self._max_stalled_iterations:
                    break
                continue
            evaluated = self._evaluate(
                generated.program,
                idea=generated.idea,
                operator="init",
                sample_time=generated.sample_time,
            )
            if evaluated is None:
                continue
            stalled_draws = 0
            node = self._graph.add_node(
                code=str(generated.program),
                idea=generated.idea,
                fitness=evaluated,
            )
            route = self._memory.create_initial(node_id=node.id)
            self._update_best(
                node,
                trajectory_id=route.id,
                iteration=None,
                operator="init",
            )
            log_event(
                self,
                event="trajectory_created",
                status="ok",
                stage="init",
                node_id=node.id,
                trajectory_id=route.id,
                program_loc=node.program_loc,
                code_hash=node.code_hash,
            )
        if self._memory.active():
            score_active_trajectories(
                memory=self._memory,
                graph=self._graph,
                maximize=self._maximize,
                w=self._value_weights,
            )

    def _init_diversity_hint(self, slot: int) -> str:
        if slot == 0:
            return "Provide a simple, complete, and valid algorithm."
        ideas = [
            node.idea.strip()
            for node in self._graph.nodes()
            if node.idea and node.idea.strip()
        ][-6:]
        if not ideas:
            return "Use a clearly different algorithmic idea from a trivial baseline."
        return "Use an idea clearly different from: " + "; ".join(
            f"'{idea[:100]}'" for idea in ideas
        )

    def _run_iteration(self, attempt_id: int) -> None:
        # Freeze batch-before snapshot for statistics and credit.
        active_before = score_active_trajectories(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
        )
        batch_best_nodes = tuple(
            compact_best_node(route, self._graph, self._maximize)
            for route in active_before
        )
        global_best_before = self._best_node

        selected = self._select_trajectory(attempt_id, batch_count=self._batch_count)
        self._batch_count += 1
        visited = self._memory.record_visit(selected.id)

        anchor_id, anchor_role = self._select_anchor(selected)
        mature = is_mature_trajectory(
            selected, active=active_before, graph=self._graph, w=self._value_weights
        )
        progress = recent_route_progress(
            edge_route_improvements=tuple(
                meaningful_improvement(
                    self._graph.get_edge(edge_id).delta_route_best,
                    self._value_weights.positive_threshold,
                )
                for edge_id in selected.edge_ids
            )
        )
        prefer_trim = self._prefer_trim_refine(selected, anchor_id)
        qualified = qualified_reference_candidates(
            primary=selected,
            anchor_id=anchor_id,
            active=active_before,
            graph=self._graph,
            w=self._value_weights,
        )
        decision = select_operator(
            operators=self._operators,
            mature=mature,
            has_qualified_reference=bool(qualified),
            anchor_role=anchor_role,
            recent_progress=progress,
            prefer_trim_refine=prefer_trim,
            rng=self._rng,
            dual_probability=self._dual_probability,
        )
        operator = decision.operator
        reference_route: Trajectory | None = None
        reference_node: ProgramNode | None = None
        reference_distribution = ()
        if decision.use_dual and qualified:
            reference_distribution = reference_sampling_distribution(
                primary=selected,
                candidates=qualified,
                temperature=self._softmax_temperature,
            )
            reference_route = weighted_choice(
                [route for route, _, _ in reference_distribution],
                [probability for _, _, probability in reference_distribution],
                self._rng,
            )
            reference_node = self._graph.get_node(reference_route.compact_best_id)

        context = self._build_action_context(
            selected=selected,
            anchor_id=anchor_id,
            operator=operator,
            reference_route=reference_route,
            reference_node=reference_node,
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
        if decision.use_dual and not context.used_dual:
            # Dual evidence did not fit; fall back to a single-trace operator.
            decision = select_operator(
                operators=self._operators,
                mature=mature,
                has_qualified_reference=False,
                anchor_role=anchor_role,
                recent_progress=progress,
                prefer_trim_refine=prefer_trim,
                rng=self._rng,
                dual_probability=0.0,
            )
            operator = decision.operator
            reference_route = None
            reference_node = None
            reference_distribution = ()
            context = self._build_action_context(
                selected=selected,
                anchor_id=anchor_id,
                operator=operator,
                reference_route=None,
                reference_node=None,
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
            mature=mature,
            recent_progress=progress,
            selection_reason=decision.reason,
            used_dual=context.used_dual,
            reference_trajectory_id=(
                None if reference_route is None else reference_route.id
            ),
            reference_program_id=(
                None if reference_node is None else reference_node.id
            ),
            reference_difference=(
                None
                if reference_route is None
                else route_difference(
                    graph=self._graph, left=selected, right=reference_route
                )
            ),
            qualified_reference_count=len(qualified),
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

        evaluated_candidates: list[_EvaluatedCandidate] = []
        for seq, action in enumerate(actions):
            if not self._has_budget() or is_search_aborted(self):
                break
            code_prompt = build_code_prompt(
                current_node=base_node,
                action=action,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                history=context.primary_history,
                reference_node=reference_node,
                reference_history=context.reference_history,
            )
            code_prompt_tokens = self._count_tokens(code_prompt)
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
            generated, draw_failure = self._draw_program(
                code_prompt,
                stage="code",
                iteration=attempt_id,
                seq=seq,
                operator=operator.name,
                action=action,
                max_tokens=self._code_max_tokens,
            )
            if generated is None:
                if draw_failure == "parse_failed":
                    self._attempts.record(
                        AnchorAttempt(
                            anchor_node_id=anchor_id,
                            primary_trajectory_id=selected.id,
                            operator=str(operator.name),
                            action=action,
                            iteration=attempt_id,
                            status="parse_failed",
                            failure_kind="parse_failed",
                        )
                    )
                continue
            fitness, failure_kind, sample_order = self._evaluate_detailed(
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
                    failure_kind=failure_kind,
                    sample_order=sample_order,
                )
            )

        created: list[tuple[_EvaluatedCandidate, ProgramNode]] = []
        for item in evaluated_candidates:
            if item.fitness is None:
                self._attempts.record(
                    AnchorAttempt(
                        anchor_node_id=anchor_id,
                        primary_trajectory_id=selected.id,
                        operator=str(operator.name),
                        action=item.action,
                        iteration=attempt_id,
                        status="eval_failed",
                        idea=item.generated.idea,
                        failure_kind=item.failure_kind or "eval_failed",
                    )
                )
                continue
            child = self._graph.add_node(
                code=str(item.generated.program),
                idea=item.generated.idea,
                fitness=item.fitness,
            )
            created.append((item, child))

        batch_keys = [
            program_quality_key(node, self._maximize) for node in batch_best_nodes
        ]
        batch_keys.extend(
            program_quality_key(child, self._maximize) for _, child in created
        )

        # Determine the one batch winner against the frozen global-best snapshot.
        # The stable textual tie-break only chooses a representative among programs
        # that are exactly equal under the documented fitness/LOC ordering.
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
            route_improved = meaningful_improvement(
                delta_route, self._value_weights.positive_threshold
            )
            credit = edge_credit(
                child=child,
                batch_keys=batch_keys,
                route_improved=route_improved,
                maximize=self._maximize,
            )
            new_global_best = child.id == global_winner_id
            global_update_reason = (
                global_winner_reason if new_global_best else None
            )
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
                edge_credit=credit,
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
            self._attempts.record(
                AnchorAttempt(
                    anchor_node_id=anchor_id,
                    primary_trajectory_id=selected.id,
                    operator=str(operator.name),
                    action=item.action,
                    iteration=attempt_id,
                    status="valid",
                    idea=child.idea,
                    fitness=child.fitness,
                    delta_parent=delta_parent,
                    delta_route_best=delta_route,
                    delta_global_best=delta_global,
                    outcome=outcome,
                    edge_id=edge.id,
                    child_id=child.id,
                    program_loc=child.program_loc,
                    new_route_best=route_improved,
                    new_global_best=new_global_best,
                )
            )
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
                edge_credit=credit,
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

    def _prefer_trim_refine(self, trajectory: Trajectory, anchor_id: int) -> bool:
        if not trajectory.edge_ids:
            return False
        recent = []
        for edge_id in reversed(trajectory.edge_ids):
            edge = self._graph.get_edge(edge_id)
            if edge.parent_id != anchor_id and edge.child_id != anchor_id:
                if anchor_id not in (
                    edge.parent_id,
                    edge.child_id,
                ) and edge.parent_id not in trajectory.node_ids:
                    continue
            recent.append(edge)
            if len(recent) >= 3:
                break
        if len(recent) < 2:
            return False
        return all(
            edge.delta_loc > 0
            and not meaningful_improvement(
                edge.delta_route_best, self._value_weights.positive_threshold
            )
            for edge in recent
        )

    def _select_trajectory(self, attempt_id: int, *, batch_count: int) -> Trajectory:
        distribution = trajectory_sampling_distribution(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
            batch_count=batch_count,
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
            selected_scalar_value=selected.scalar_value,
            selected_adjusted_score=selected_adjusted,
            selected_quality=(
                None if selected.value is None else selected.value.quality
            ),
            selected_credit=(None if selected.value is None else selected.value.credit),
            selected_visit_count=selected.visit_count,
            batch_count=batch_count,
            max_probability=sorted_probs[0],
            top5_probability_mass=sum(sorted_probs[:5]),
            effective_candidate_count=1.0 / sum(prob * prob for prob in probs),
        )
        return selected

    def _select_anchor(self, trajectory: Trajectory) -> tuple[int, str]:
        endpoint = trajectory.endpoint_id
        compact = trajectory.compact_best_id
        if endpoint == compact:
            return endpoint, "endpoint_compact_best"
        if self._rng.random() < 0.5:
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
        requested_dual = reference_route is not None and reference_node is not None
        modes = (True, False) if requested_dual else (False,)
        max_steps = self._memory.max_trajectory_length
        # Keep high-priority attempts while first removing lower-priority attempts,
        # then progressively shorten the retained path. The order is fixed so the
        # same state and token counter always produce the same context.
        reductions = sorted(
            (
                (history_steps, attempt_limit)
                for history_steps in range(max_steps, -1, -1)
                for attempt_limit in range(self._max_tested_attempts, -1, -1)
            ),
            key=lambda item: (-(item[0] + item[1]), -item[0], -item[1]),
        )
        for use_reference in modes:
            ref_history = (
                reference_history(
                    self._graph,
                    reference_route,
                    base_node_id=reference_node.id,
                    max_steps=max_steps,
                    positive_threshold=self._value_weights.positive_threshold,
                )
                if use_reference
                else None
            )
            for history_steps, attempt_limit in reductions:
                primary_history = trajectory_history(
                    self._graph,
                    selected,
                    base_node_id=anchor_id,
                    max_steps=history_steps,
                    positive_threshold=self._value_weights.positive_threshold,
                )
                tested_text = render_tested_attempts(
                    self._attempts.for_anchor(anchor_id),
                    excluded_edge_ids=set(primary_history.edge_ids),
                    limit=attempt_limit,
                )
                prompt = build_action_prompt(
                    base_node=self._graph.get_node(anchor_id),
                    primary_history=primary_history,
                    tested_attempts_text=tested_text,
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
                    primary_history=primary_history.text + "\n" + tested_text,
                    reference_history=(
                        "" if ref_history is None else ref_history.text
                    ),
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
        deduped = list(
            deduplicate_by_endpoint_hash(
                routes=tuple(ranked),
                graph=self._graph,
                best_trajectory_id=self._best_trajectory_id,
            )
        )
        deduped = list(
            score_active_trajectories(
                memory=self._memory,
                graph=self._graph,
                maximize=self._maximize,
                w=self._value_weights,
                trajectories=tuple(deduped),
            )
        )
        if len(deduped) <= self._max_active_trajectories:
            keep_ids = {route.id for route in deduped}
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
                        "roles": ["exact_dedup"] if keep else [],
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
                selection_mode="exact_dedup_only",
            )
            save_checkpoint(self)
            return

        # Quality elite by Q.
        by_quality = sorted(
            deduped,
            key=lambda route: (
                -(0.0 if route.value is None else route.value.quality),
                -(route.scalar_value or 0.0),
                route.id,
            ),
        )
        elite_count = min(self._elite_count, self._max_active_trajectories, len(by_quality))
        elites = by_quality[:elite_count]
        if self._best_trajectory_id is not None:
            best_route = next(
                (route for route in by_quality if route.id == self._best_trajectory_id),
                None,
            )
            if best_route is not None and best_route.id not in {
                route.id for route in elites
            }:
                weakest = min(
                    elites,
                    key=lambda route: (
                        0.0 if route.value is None else route.value.quality,
                        route.scalar_value or 0.0,
                        -route.id,
                    ),
                )
                elites = [best_route, *[route for route in elites if route.id != weakest.id]]

        elite_ids = {route.id for route in elites}
        mature_pool = tuple(
            route
            for route in by_quality
            if route.id not in elite_ids
            and is_mature_trajectory(
                route, active=tuple(by_quality), graph=self._graph, w=self._value_weights
            )
        )
        diversity_slots = min(
            self._diversity_count,
            max(0, self._max_active_trajectories - len(elites)),
        )
        if not mature_pool:
            diversity_slots = 0
        diverse = select_diverse_trajectories(
            candidates=mature_pool,
            graph=self._graph,
            count=diversity_slots,
            reference=tuple(elites),
        )
        diverse_ids = {route.id for route in diverse}
        remaining = [
            route
            for route in by_quality
            if route.id not in elite_ids and route.id not in diverse_ids
        ]
        sampled = self._weighted_survivor_sample(
            remaining,
            self._max_active_trajectories - len(elites) - len(diverse),
        )
        roles: dict[int, list[str]] = {}
        for route in elites:
            roles.setdefault(route.id, []).append("quality_elite")
        for route in diverse:
            roles.setdefault(route.id, []).append("difference_reserve")
        for route in sampled:
            roles.setdefault(route.id, []).append("continue_develop")
        if self._best_trajectory_id is not None and self._best_trajectory_id in roles:
            roles[self._best_trajectory_id].append("best_anchor")
        keep_ids = set(roles)
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
                    "roles": roles.get(route.id, []),
                    "quality": (
                        None
                        if recorded_route.value is None
                        else recorded_route.value.quality
                    ),
                    "credit": (
                        None
                        if recorded_route.value is None
                        else recorded_route.value.credit
                    ),
                    "scalar_value": recorded_route.scalar_value,
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
            selection_mode="dedup_elite_difference_v_softmax",
        )
        save_checkpoint(self)

    def _weighted_survivor_sample(
        self, candidates: list[Trajectory], count: int
    ) -> tuple[Trajectory, ...]:
        remaining = list(candidates)
        selected: list[Trajectory] = []
        while remaining and len(selected) < count:
            scores = [float(route.scalar_value or 0.0) for route in remaining]
            maximum = max(scores)
            weights = [
                math.exp((score - maximum) / self._softmax_temperature)
                for score in scores
            ]
            choice = weighted_choice(remaining, weights, self._rng)
            remaining.remove(choice)
            selected.append(choice)
        return tuple(selected)

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
    ) -> tuple[_GeneratedProgram | None, str | None]:
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
                prompt_tokens=self._count_tokens(prompt),
                response_tokens=0,
                token_count_mode=self._token_count_mode,
                prompt=prompt,
                response=None,
                action=action,
                status="llm_error",
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
            return None, "llm_error"
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
            "prompt_tokens": self._count_tokens(prompt),
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
            return None, "parse_failed"
        reset_sample_failures(self)
        return (
            _GeneratedProgram(
                idea=generated.idea,
                program=generated.program,
                sample_time=elapsed,
            ),
            None,
        )

    def _evaluate(
        self,
        program: Program,
        *,
        idea: str,
        operator: str | OperatorName,
        sample_time: float,
    ) -> float | None:
        score, _, _ = self._evaluate_detailed(
            program, idea=idea, operator=operator, sample_time=sample_time
        )
        return score

    def _evaluate_detailed(
        self,
        program: Program,
        *,
        idea: str,
        operator: str | OperatorName,
        sample_time: float,
    ) -> tuple[float | None, str | None, int | None]:
        if not self._has_budget():
            return None, "no_budget", None
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
                    failure_error = f"evaluator returned non-finite score: {numeric_score}"
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
        return score, failure_kind, self._tot_sample_nums

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

    def _result(self) -> TraceAADRunResult:
        nodes = self._graph.nodes()
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=len(nodes),
            n_valid_nodes=sum(node.fitness is not None for node in nodes),
            n_trajectories=len(self._memory.trajectories()),
            n_edges=len(self._graph.edges()),
            n_samples=self._tot_sample_nums,
        )


__all__ = ["TraceAADRunResult", "TraceAADV6"]
