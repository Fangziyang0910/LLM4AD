"""Independent implementation of the complete TraceAAD v5 mechanism."""

from __future__ import annotations

import copy
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...base import (
    Evaluation,
    Function,
    LLM,
    Program,
    SampleTrimmer,
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
    log_state,
    record_sample_failure,
    reset_sample_failures,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .complexity import code_change_ratio
from .context import build_action_prompt, trajectory_history
from .credit import directed_delta
from .derivation_graph import DerivationGraph
from .experience import build_reflection_prompt
from .operators import DEFAULT_OPERATORS, Operator, classify_outcome
from .operators.semantic import TraceSynthesizeOp
from .prompt import IDEA_MAX_CHARS, build_code_prompt, build_initial_prompt
from .schema import (
    EvalResult,
    OperatorName,
    ProgramNode,
    Trajectory,
)
from .similarity import code_similarity
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    compact_best_node,
    is_program_better,
    reference_sampling_distribution,
    sample_reference_trajectory,
    sample_trajectory,
    score_active_trajectories,
    select_diverse_trajectories,
    trajectory_sampling_distribution,
)


@dataclass(frozen=True, slots=True)
class _GeneratedProgram:
    idea: str
    program: Program


@dataclass(frozen=True, slots=True)
class _CandidateObservation:
    node_id: int
    score: float
    reference_score: float
    outcome: str


@dataclass(frozen=True, slots=True)
class _PromptContext:
    prompt: str
    window: int
    token_count: int
    primary_edge_ids: tuple[int, ...]
    reference_edge_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int
    n_experience_updates: int


class TraceAADV5:
    """Trajectory search with local history, references, and global experience."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase = None,
        max_sample_nums: Optional[int] = 100,
        *,
        n_init: int = 30,
        actions_per_iteration: int = 2,
        max_trajectory_length: int = 8,
        max_active_trajectories: int = 30,
        elite_count: int | None = None,
        softmax_temperature: float = 0.2,
        maximize: bool = True,
        value_weights: ValueWeights | None = None,
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
        global_reflection_code_batch: int = 40,
        global_reflection_max_tokens: int = 1024,
        max_context_tokens: int = 32768,
        output_token_reserve: int = 8192,
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
        if max_context_tokens <= output_token_reserve:
            raise ValueError("context window must exceed the output reserve")

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
        self._diversity_count = max(2, math.ceil(0.1 * self._max_active_trajectories))
        self._softmax_temperature = float(softmax_temperature)
        self._maximize = bool(maximize)
        self._value_weights = value_weights or ValueWeights()
        self._global_reflection_code_batch = int(global_reflection_code_batch)
        if self._global_reflection_code_batch <= 0:
            raise ValueError("global_reflection_code_batch must be positive")
        self._global_reflection_max_tokens = int(global_reflection_max_tokens)
        if self._global_reflection_max_tokens <= 0:
            raise ValueError("global_reflection_max_tokens must be positive")
        if max_context_tokens <= self._global_reflection_max_tokens:
            raise ValueError("context window must exceed reflection output tokens")
        self._max_context_tokens = int(max_context_tokens)
        self._output_token_reserve = int(output_token_reserve)
        self._debug_mode = debug_mode
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_sample = -1
        self._rng = random.Random(random_seed)
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD v5 requires exactly one evolvable template function."
            )
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(prompt_window=max_trajectory_length)
        self._operators = tuple(operator_type() for operator_type in operators)
        if not self._operators:
            raise ValueError("at least one TraceAAD v5 operator is required")
        self._best_node: ProgramNode | None = None
        self._best_trajectory_id: int | None = None
        self._global_experience = ""
        self._pending_reflection_edge_ids: list[int] = []
        self._experience_reflection_attempts = 0
        self._experience_update_index = 0
        self._tot_sample_nums = 0
        self._next_attempt_id = 0
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
            )

    def run(self) -> TraceAADRunResult:
        stalled_attempts = 0
        attempt_id = self._next_attempt_id
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
                before = self._tot_sample_nums
                edge_count_before = len(self._graph.edges())
                self._run_iteration(attempt_id)
                round_edge_ids = tuple(
                    edge.id for edge in self._graph.edges()[edge_count_before:]
                )
                stop_for_stall = False
                if self._tot_sample_nums == before:
                    stalled_attempts += 1
                    if stalled_attempts >= self._max_stalled_iterations:
                        log_event(
                            self,
                            event="search_stopped",
                            status="stalled_generation",
                            attempt_id=attempt_id,
                        )
                        stop_for_stall = True
                else:
                    stalled_attempts = 0
                self._pending_reflection_edge_ids.extend(round_edge_ids)
                self._maybe_update_global_experience(attempt_id)
                self._maybe_manage_population()
                attempt_id += 1
                self._next_attempt_id = attempt_id
                self._save_checkpoint_if_due()
                if stop_for_stall:
                    break
            self._maybe_manage_population()
            result = self._result()
            finish_profiler(
                self,
                status="aborted" if is_search_aborted(self) else "finished",
                best_node_id=None if result.best_node is None else result.best_node.id,
                best_score=(
                    None if result.best_node is None else result.best_node.fitness
                ),
                n_total_nodes=result.n_total_nodes,
                n_valid_nodes=result.n_valid_nodes,
                n_edges=result.n_edges,
                n_trajectories=result.n_trajectories,
                n_experience_updates=result.n_experience_updates,
            )
            return result
        finally:
            save_checkpoint(self)
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
            generated = self._draw_program(
                prompt,
                stage="init",
                iteration=None,
                seq=draw_seq,
                operator="init",
            )
            draw_seq += 1
            if generated is None:
                stalled_draws += 1
                if stalled_draws >= self._max_stalled_iterations:
                    break
                continue
            evaluated = self._evaluate(
                generated.program, idea=generated.idea, operator="init"
            )
            if evaluated is None or evaluated.fitness is None:
                continue
            stalled_draws = 0
            node = self._add_node(generated, evaluated)
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
            self._score_active_pool()

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
        selected = self._select_trajectory(attempt_id)
        anchor_id, anchor_role = self._select_anchor(selected)
        reference_candidates = self._reference_candidates(selected, anchor_id)
        eligible = [
            operator
            for operator in self._operators
            if operator.name not in (OperatorName.SYNTHESIZE, OperatorName.TRANSFER)
            or reference_candidates
        ]
        if not eligible:
            self._memory.record_visit(selected.id)
            log_event(
                self,
                event="operator_selection",
                status="no_eligible_operator",
                attempt_id=attempt_id,
                selected_trajectory_id=selected.id,
            )
            return
        operator = self._rng.choice(eligible)
        reference_route: Trajectory | None = None
        reference_node: ProgramNode | None = None
        reference_distribution = ()
        if operator.name in (OperatorName.SYNTHESIZE, OperatorName.TRANSFER):
            reference_distribution = reference_sampling_distribution(
                candidates=reference_candidates,
                temperature=self._softmax_temperature,
            )
            reference_route = sample_reference_trajectory(
                candidates=reference_candidates,
                temperature=self._softmax_temperature,
                rng=self._rng,
            )
            reference_node = self._graph.get_node(reference_route.compact_best_id)
            self._memory.record_reference_use(reference_route.id)
        context = self._build_action_context(
            selected=selected,
            anchor_id=anchor_id,
            anchor_role=anchor_role,
            operator=operator,
            reference_route=reference_route,
            reference_node=reference_node,
        )
        log_event(
            self,
            event="operator_selection",
            status="ok" if context is not None else "context_overflow",
            attempt_id=attempt_id,
            selected_operator=operator.name,
            selected_trajectory_id=selected.id,
            base_node_id=anchor_id,
            anchor_role=anchor_role,
            reference_trajectory_id=(
                None if reference_route is None else reference_route.id
            ),
            reference_program_id=(
                None if reference_node is None else reference_node.id
            ),
            reference_similarity=(
                None
                if reference_node is None
                else code_similarity(
                    self._graph.get_node(anchor_id).code,
                    reference_node.code,
                )
            ),
            eligible=[candidate.name for candidate in eligible],
            operator_probabilities={
                candidate.name.value: 1.0 / len(eligible) for candidate in eligible
            },
            reference_candidate_ids=[route.id for route in reference_candidates],
            reference_candidates=[
                {
                    "trajectory_id": route.id,
                    "quality": (None if route.value is None else route.value.quality),
                    "probability": probability,
                    "code_hash": self._graph.get_node(route.compact_best_id).code_hash,
                }
                for route, probability in reference_distribution
            ],
            selection_mode="uniform_available_operator_q_ucb_primary_q_reference",
        )
        log_state(
            self,
            phase="iteration_start",
            iteration=attempt_id,
            selected_trajectory_id=selected.id,
            selected_endpoint_id=selected.endpoint_id,
            operator=operator.name,
            base_node_id=anchor_id,
            anchor_role=anchor_role,
            selected_value=selected.scalar_value,
        )
        if context is None:
            self._memory.record_visit(selected.id)
            return
        actions = self._generate_actions(
            context,
            operator=operator,
            iteration=attempt_id,
        )
        observations: list[_CandidateObservation] = []
        base_node = self._graph.get_node(anchor_id)
        route_best_before = compact_best_node(selected, self._graph, self._maximize)
        for seq, action in enumerate(actions):
            if not self._has_budget() or is_search_aborted(self):
                break
            code_prompt = build_code_prompt(
                current_node=base_node,
                action=action,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
            )
            if not self._prompt_fits(code_prompt):
                log_event(
                    self,
                    event="context_overflow",
                    status="code_prompt",
                    iteration=attempt_id,
                    seq=seq,
                    operator=operator.name,
                    token_count=self._count_tokens(code_prompt),
                )
                continue
            generated = self._draw_program(
                code_prompt,
                stage="code",
                iteration=attempt_id,
                seq=seq,
                operator=operator.name,
                action=action,
            )
            if generated is None:
                continue
            old_global = self._best_node
            evaluated = self._evaluate(
                generated.program,
                idea=generated.idea,
                operator=operator.name,
            )
            if evaluated is None or evaluated.fitness is None:
                continue
            child = self._add_node(generated, evaluated)
            delta_parent = directed_delta(
                base_node.fitness, child.fitness, self._maximize
            )
            delta_route = directed_delta(
                route_best_before.fitness, child.fitness, self._maximize
            )
            delta_global = directed_delta(
                None if old_global is None else old_global.fitness,
                child.fitness,
                self._maximize,
            )
            outcome = classify_outcome(
                delta_parent, self._value_weights.positive_threshold
            )
            best_update, best_reason = self._best_update_decision(child, old_global)
            new_compact = self._compact_best_for_child(
                selected=selected,
                anchor_id=anchor_id,
                child=child,
            )
            edge = self._graph.add_edge(
                parent_id=anchor_id,
                child_id=child.id,
                action=action,
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
                new_global_best=best_update,
                global_best_update_reason=best_reason,
            )
            child_route = self._memory.branch_from(
                trajectory_id=selected.id,
                base_node_id=anchor_id,
                child_id=child.id,
                edge_id=edge.id,
                compact_best_id=new_compact.id,
            )
            self._update_best(
                child,
                trajectory_id=child_route.id,
                iteration=attempt_id,
                operator=operator.name,
            )
            self._log_child(
                iteration=attempt_id,
                seq=seq,
                operator=operator,
                child=child,
                trajectory=child_route,
                parent_id=anchor_id,
                edge_id=edge.id,
                action=action,
                delta_parent=delta_parent,
                delta_route_best=delta_route,
                delta_global_best=delta_global,
                outcome=outcome,
                anchor_role=anchor_role,
                reference_trajectory_id=(
                    None if reference_route is None else reference_route.id
                ),
                reference_program_id=(
                    None if reference_node is None else reference_node.id
                ),
                program_loc=child.program_loc,
                delta_loc=child.program_loc - base_node.program_loc,
                code_hash=child.code_hash,
                global_best_update_reason=best_reason,
            )
            observations.append(
                _CandidateObservation(
                    node_id=child.id,
                    score=float(child.fitness),
                    reference_score=float(base_node.fitness),
                    outcome=outcome,
                )
            )
        visited = self._memory.record_visit(selected.id)
        log_event(
            self,
            event="operator_batch",
            status="ok" if observations else "empty",
            attempt_id=attempt_id,
            operator=operator.name,
            candidates=[
                {
                    "node_id": item.node_id,
                    "score": item.score,
                    "reference_score": item.reference_score,
                    "outcome": item.outcome,
                }
                for item in observations
            ],
            active_trajectories=len(self._memory.active()),
            parent_visit_count=visited.visit_count,
        )

    def _select_trajectory(self, attempt_id: int) -> Trajectory:
        distribution = trajectory_sampling_distribution(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
        )
        selected = sample_trajectory(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
            rng=self._rng,
        )
        log_event(
            self,
            event="trajectory_selection",
            status="ok",
            attempt_id=attempt_id,
            candidates=[
                {
                    "trajectory_id": route.id,
                    "quality": (None if route.value is None else route.value.quality),
                    "analysis_trend": (
                        None if route.value is None else route.value.trend
                    ),
                    "visit_count": route.visit_count,
                    "adjusted_score": adjusted,
                    "probability": probability,
                }
                for route, adjusted, probability in distribution
            ],
            selected_trajectory_id=selected.id,
            selection_mode="q_plus_ucb_softmax",
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

    def _reference_candidates(
        self,
        primary: Trajectory,
        anchor_id: int,
    ) -> tuple[Trajectory, ...]:
        candidates: list[Trajectory] = []
        feasibility_operator = TraceSynthesizeOp()
        for route in self._memory.active():
            if route.id == primary.id:
                continue
            node = self._graph.get_node(route.compact_best_id)
            context = self._build_action_context(
                selected=primary,
                anchor_id=anchor_id,
                anchor_role="candidate_check",
                operator=feasibility_operator,
                reference_route=route,
                reference_node=node,
                minimum_only=True,
            )
            if context is not None:
                candidates.append(route)
        return tuple(candidates)

    def _build_action_context(
        self,
        *,
        selected: Trajectory,
        anchor_id: int,
        anchor_role: str,
        operator: Operator,
        reference_route: Trajectory | None,
        reference_node: ProgramNode | None,
        minimum_only: bool = False,
    ) -> _PromptContext | None:
        minimum = max(
            self._minimum_history_window(selected, anchor_id),
            (1 if reference_route is not None and reference_route.edge_ids else 0),
            1,
        )
        start = minimum if minimum_only else self._memory.prompt_window
        stop = minimum
        for window in range(start, stop - 1, -1):
            primary_history = trajectory_history(
                self._graph,
                selected,
                base_node_id=anchor_id,
                max_steps=window,
            )
            reference_history = (
                None
                if reference_route is None or reference_node is None
                else trajectory_history(
                    self._graph,
                    reference_route,
                    base_node_id=reference_node.id,
                    max_steps=window,
                )
            )
            prompt = build_action_prompt(
                graph=self._graph,
                trajectory=selected,
                base_node_id=anchor_id,
                operator_constraint=operator.build_constraint(),
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                action_count=self._actions_per_iteration,
                maximize=self._maximize,
                max_steps=window,
                global_experience=self._global_experience,
                reference_trajectory=reference_route,
                reference_node_id=(
                    None if reference_node is None else reference_node.id
                ),
            )
            token_count = self._count_tokens(prompt)
            if token_count <= self._prompt_token_budget:
                displayed_primary = primary_history.edge_ids
                displayed_reference = (
                    () if reference_history is None else reference_history.edge_ids
                )
                if not minimum_only:
                    log_event(
                        self,
                        event="prompt_context",
                        status="ok",
                        operator=operator.name,
                        prompt_window=window,
                        token_count=token_count,
                        token_count_mode=self._token_count_mode,
                        primary_edge_ids=list(displayed_primary),
                        reference_edge_ids=list(displayed_reference),
                        pruned_primary_edge_ids=[
                            edge
                            for edge in selected.edge_ids
                            if edge not in displayed_primary
                        ],
                        pruned_reference_edge_ids=(
                            []
                            if reference_route is None
                            else [
                                edge
                                for edge in reference_route.edge_ids
                                if edge not in displayed_reference
                            ]
                        ),
                    )
                return _PromptContext(
                    prompt=prompt,
                    window=window,
                    token_count=token_count,
                    primary_edge_ids=displayed_primary,
                    reference_edge_ids=displayed_reference,
                )
        return None

    def _minimum_history_window(self, trajectory: Trajectory, anchor_id: int) -> int:
        if not trajectory.edge_ids:
            return 1
        index = trajectory.node_ids.index(anchor_id)
        has_before = bool(trajectory.edge_ids[:index])
        has_after = bool(trajectory.edge_ids[index:])
        return 2 if has_before and has_after else 1

    @property
    def _prompt_token_budget(self) -> int:
        return self._max_context_tokens - self._output_token_reserve

    def _count_tokens(self, prompt: str) -> int:
        counter = getattr(self._llm, "count_tokens", None)
        if callable(counter):
            return int(counter(prompt))
        # UTF-8 byte count is a conservative upper bound for ordinary BPE tokens.
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

    def _prompt_fits(
        self, prompt: str, *, output_token_reserve: int | None = None
    ) -> bool:
        reserve = (
            self._output_token_reserve
            if output_token_reserve is None
            else int(output_token_reserve)
        )
        return self._count_tokens(prompt) <= self._max_context_tokens - reserve

    def _generate_actions(
        self,
        context: _PromptContext,
        *,
        operator: Operator,
        iteration: int,
    ) -> list[str]:
        start = time.time()
        try:
            response = self._llm.draw_sample(context.prompt)
            sample_time = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
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
            return []
        actions, errors = _parse_actions(
            response,
            expected_count=self._actions_per_iteration,
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

    def _maybe_update_global_experience(self, iteration: int) -> None:
        if (
            len(self._pending_reflection_edge_ids)
            < self._global_reflection_code_batch
        ):
            return
        recent_edge_ids = tuple(
            self._pending_reflection_edge_ids[
                : self._global_reflection_code_batch
            ]
        )
        prompt = build_reflection_prompt(
            task_description=self._task_description_str,
            maximize=self._maximize,
            old_experience=self._global_experience,
            recent_edge_ids=recent_edge_ids,
            graph=self._graph,
        )
        self._experience_reflection_attempts += 1
        if not self._prompt_fits(
            prompt, output_token_reserve=self._global_reflection_max_tokens
        ):
            log_event(
                self,
                event="global_experience_update",
                status="context_overflow",
                iteration=iteration,
                reflection_attempt=self._experience_reflection_attempts,
                code_count=len(recent_edge_ids),
                recent_edge_ids=list(recent_edge_ids),
            )
            save_checkpoint(self)
            return
        start = time.time()
        try:
            response = self._llm.draw_sample(
                prompt,
                max_tokens=self._global_reflection_max_tokens,
                temperature=0.2,
            )
            elapsed = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="global_experience",
                operator="reflection",
                sample_order=self._tot_sample_nums,
                prompt=prompt,
                counts_budget=False,
                iteration=iteration,
            )
            save_checkpoint(self)
            return
        self._global_experience = str(response).strip()
        del self._pending_reflection_edge_ids[
            : self._global_reflection_code_batch
        ]
        self._experience_update_index += 1
        log_llm_call(
            self,
            stage="global_experience",
            operator="reflection",
            sample_order=self._tot_sample_nums,
            iteration=iteration,
            seq=0,
            prompt=prompt,
            response=response,
            sample_time=elapsed,
            prompt_tokens=self._count_tokens(prompt),
            response_tokens=self._count_tokens(response),
            token_count_mode=self._token_count_mode,
            reflection_attempt=self._experience_reflection_attempts,
            successful_update_index=self._experience_update_index,
            code_count=len(recent_edge_ids),
            recent_edge_ids=list(recent_edge_ids),
            status="ok",
        )
        save_checkpoint(self)

    def _add_node(
        self, generated: _GeneratedProgram, evaluated: EvalResult
    ) -> ProgramNode:
        return self._graph.add_node(
            code=str(generated.program),
            idea=generated.idea,
            fitness=evaluated.fitness,
        )

    def _update_best(
        self,
        node: ProgramNode,
        *,
        trajectory_id: int | None,
        iteration: int | None,
        operator,
    ) -> None:
        update, reason = self._best_update_decision(node, self._best_node)
        if not update:
            return
        previous = self._best_node
        self._best_node = node
        if trajectory_id is not None:
            self._best_trajectory_id = trajectory_id
        log_event(
            self,
            event="best_updated",
            status="ok",
            iteration=iteration,
            sample_order=self._tot_sample_nums,
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

    def _log_child(
        self,
        *,
        iteration: int,
        seq: int,
        operator: Operator,
        child: ProgramNode,
        trajectory: Trajectory,
        **fields,
    ) -> None:
        log_event(
            self,
            event="child_accepted",
            status="ok",
            iteration=iteration,
            seq=seq,
            operator=operator.name,
            child_id=child.id,
            trajectory_id=trajectory.id,
            compact_best_id=trajectory.compact_best_id,
            score=child.fitness,
            **fields,
        )

    def _score_active_pool(self) -> tuple[Trajectory, ...]:
        return score_active_trajectories(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
        )

    def _maybe_manage_population(self) -> None:
        if len(self._memory.active()) >= self._management_threshold:
            self._manage_population()

    def _manage_population(self) -> None:
        ranked = list(self._score_active_pool())
        if len(ranked) < self._management_threshold:
            return
        elite_count = min(self._elite_count, self._max_active_trajectories, len(ranked))
        elites = ranked[:elite_count]
        if self._best_trajectory_id is not None:
            best_route = next(
                (route for route in ranked if route.id == self._best_trajectory_id),
                None,
            )
            if best_route is not None and best_route.id not in {
                route.id for route in elites
            }:
                elites = [best_route, *elites[: max(0, elite_count - 1)]]
        diversity_count = min(
            self._diversity_count,
            max(0, self._max_active_trajectories - len(elites)),
            max(0, len(ranked) - len(elites)),
        )
        elite_ids = {route.id for route in elites}
        diverse = select_diverse_trajectories(
            candidates=tuple(route for route in ranked if route.id not in elite_ids),
            graph=self._graph,
            count=diversity_count,
            reference=tuple(elites),
        )
        diverse_ids = {route.id for route in diverse}
        remaining = [
            route
            for route in ranked
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
            roles.setdefault(route.id, []).append("diversity")
        for route in sampled:
            roles.setdefault(route.id, []).append("softmax")
        if self._best_trajectory_id is not None and self._best_trajectory_id in roles:
            roles[self._best_trajectory_id].append("best_anchor")
        keep_ids = set(roles)
        decisions = []
        for route in ranked:
            keep = route.id in keep_ids
            if not keep:
                self._memory.archive(route.id)
            decisions.append(
                {
                    "trajectory_id": route.id,
                    "decision": "keep" if keep else "archive",
                    "roles": roles.get(route.id, []),
                    "quality": (None if route.value is None else route.value.quality),
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
            selection_mode="quality_elite_diversity_q_softmax",
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
            total = sum(weights)
            needle = self._rng.random() * total if total > 0 else 0.0
            index = 0
            for index, weight in enumerate(weights):
                needle -= weight
                if needle <= 0:
                    break
            selected.append(remaining.pop(index))
        return tuple(selected)

    def _draw_program(
        self,
        prompt: str,
        *,
        stage: str,
        iteration: int | None,
        seq: int,
        operator,
        action: str | None = None,
    ) -> _GeneratedProgram | None:
        sample_order = self._tot_sample_nums + 1
        start = time.time()
        try:
            response = self._llm.draw_sample(prompt)
            elapsed = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
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
            return None
        generated = _parse_program_response(
            response, self._template_program, self._function_to_evolve.name
        )
        log_llm_call(
            self,
            stage=stage,
            operator=operator,
            sample_order=sample_order,
            iteration=iteration,
            seq=seq,
            action=action,
            prompt=prompt,
            response=response,
            sample_time=elapsed,
            prompt_tokens=self._count_tokens(prompt),
            response_tokens=self._count_tokens(response),
            token_count_mode=self._token_count_mode,
            parsed_idea=None if generated is None else generated.idea,
            program_parse_success=generated is not None,
            status="ok" if generated is not None else "parse_failed",
        )
        return generated

    def _evaluate(self, program: Program, *, idea: str, operator) -> EvalResult | None:
        if not self._has_budget():
            return None
        outcome, eval_time = (
            self._evaluator.evaluate_program_record_time_with_details(program)
        )
        self._tot_sample_nums += 1
        result = outcome.result
        score = result.fitness if isinstance(result, EvalResult) else result
        try:
            function = copy.deepcopy(
                program.get_function(self._function_to_evolve.name)
            )
        except ValueError:
            function = None
        if function is not None:
            function.algorithm = idea
            function.score = score
            function.evaluate_time = eval_time
            function.operator = str(operator)
            if self._profiler is not None:
                self._profiler.register_function(function, program=str(program))
        failure = (
            {}
            if score is not None
            else {
                "failure_kind": outcome.failure_kind,
                "error_type": outcome.error_type,
                "error": outcome.error,
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
        return None if score is None else EvalResult(fitness=float(score))

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._tot_sample_nums - self._last_checkpoint_sample
            >= self._checkpoint_interval
        ):
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
            n_experience_updates=self._experience_update_index,
        )


def _parse_actions(
    response: str,
    *,
    expected_count: int,
) -> tuple[list[str], list[str]]:
    actions: list[str] = []
    for line in str(response).strip().splitlines():
        match = re.match(
            r"^(?P<number>\d+)[\.\)]\s+(?P<action>\S.*)$",
            line.strip(),
        )
        if match is None:
            continue
        if int(match.group("number")) != len(actions) + 1:
            continue
        action = match.group("action").strip()
        actions.append(action)
    parsed_count = len(actions)
    errors = (
        []
        if parsed_count == expected_count
        else [f"expected_{expected_count}_actions_got_{parsed_count}"]
    )
    return actions[:expected_count], errors


def _parse_program_response(
    response: str,
    template_program: Program,
    function_name: str,
) -> _GeneratedProgram | None:
    idea = (
        _extract_idea(response) or _extract_boxed_text(response) or "Generated program"
    )
    idea = _short_idea(idea)
    blocks = _extract_code_blocks(response)
    candidates = reversed(blocks) if blocks else (response,)
    program = next(
        (
            parsed
            for code in candidates
            if (parsed := _parse_program_candidate(code, template_program, function_name))
            is not None
        ),
        None,
    )
    return None if program is None else _GeneratedProgram(idea=idea, program=program)


def _parse_program_candidate(
    code: str,
    template_program: Program,
    function_name: str,
) -> Program | None:
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is not None:
        completed = _complete_program(parsed, template_program, function_name)
        if completed is not None:
            return completed
    trimmed = SampleTrimmer.sample_to_program(code, template_program)
    if trimmed is None:
        return None
    return _complete_program(trimmed, template_program, function_name)


def _complete_program(
    parsed: Program,
    template_program: Program,
    function_name: str,
) -> Program | None:
    try:
        parsed.get_function(function_name)
    except ValueError:
        return None
    return Program(
        preface=_merge_prefaces(template_program.preface, parsed.preface),
        functions=parsed.functions,
    )


def _merge_prefaces(template_preface: str, generated_preface: str) -> str:
    future: list[str] = []
    ordinary: list[str] = []
    for source in (template_preface, generated_preface):
        kept: list[str] = []
        for line in source.splitlines():
            if line.lstrip().startswith("from __future__"):
                if line not in future:
                    future.append(line)
            else:
                kept.append(line)
        block = "\n".join(kept).strip()
        if block and block not in ordinary:
            ordinary.append(block)
    return "\n".join(future + ordinary)


def _extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>.+?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else match.group("idea").strip()


def _extract_boxed_text(response: str) -> str | None:
    match = re.search(
        r"(?:\\)?boxed\s*\{(?P<idea>[^{}]+)\}",
        response,
        flags=re.IGNORECASE,
    )
    return None if match is None else match.group("idea").strip()


def _short_idea(idea: str) -> str:
    compact = " ".join(str(idea).split())
    if len(compact) <= IDEA_MAX_CHARS:
        return compact
    return compact[: IDEA_MAX_CHARS - 1].rstrip() + "…"


def _extract_code_blocks(response: str) -> tuple[str, ...]:
    return tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )


__all__ = ["TraceAADRunResult", "TraceAADV5"]
