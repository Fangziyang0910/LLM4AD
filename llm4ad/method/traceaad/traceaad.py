"""TraceAAD v4：以完整算法改进轨迹驱动语义搜索。"""
from __future__ import annotations

import copy
import math
import random
import re
import statistics
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
from .context import build_action_prompt
from .credit import directed_delta
from .derivation_graph import DerivationGraph
from .experience_memory import ExperienceMemory
from .operators import DEFAULT_OPERATORS, Operator, OperatorContext, classify_outcome
from .portfolio import OperatorPortfolio, PortfolioWeights, aggregate_batch_utility, signed_utility
from .prompt import build_code_prompt, build_initial_prompt
from .schema import EvalResult, ProgramNode, Trajectory
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    sample_survivors,
    sample_trajectory,
    score_active_trajectories,
    select_diverse_trajectories,
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
    accepted: bool
    outcome: str


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


class TraceAAD:
    """TraceAAD v4 implementation.

    ``max_active_trajectories`` is the post-management population size ``M``.
    During an expansion epoch all valid child trajectories remain active and can
    be selected; management runs once the active pool reaches ``2 * M``.
    """

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
        population_growth_factor: float = 2.0,
        elite_count: int | None = None,
        softmax_temperature: float = 0.2,
        maximize: bool = True,
        value_weights: ValueWeights | None = None,
        portfolio_weights: PortfolioWeights | None = None,
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
        novelty_threshold: float | None = None,
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
        if population_growth_factor <= 1.0:
            raise ValueError("population_growth_factor must be greater than 1")
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

        self._llm = llm
        self._evaluation = evaluation
        self._profiler = profiler
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = n_init
        self._actions_per_iteration = actions_per_iteration
        self._max_active_trajectories = int(max_active_trajectories)
        self._population_growth_factor = float(population_growth_factor)
        self._management_threshold = max(
            self._max_active_trajectories + 1,
            math.ceil(self._population_growth_factor * self._max_active_trajectories),
        )
        self._elite_count = (
            max(2, math.ceil(0.1 * self._max_active_trajectories))
            if elite_count is None
            else max(1, int(elite_count))
        )
        self._diversity_count = max(2, math.ceil(0.1 * self._max_active_trajectories))
        self._softmax_temperature = float(softmax_temperature)
        self._maximize = maximize
        self._value_weights = value_weights or ValueWeights()
        self._portfolio_weights = portfolio_weights or PortfolioWeights()
        # Kept as a readable compatibility parameter; v4 deliberately does not gate
        # candidates by code or trajectory similarity.
        self._novelty_threshold = novelty_threshold
        self._debug_mode = debug_mode
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_sample = -1
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD requires an evaluation template with exactly one evolvable function."
            )
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._experience_memory = ExperienceMemory(self._graph)
        self._operators = tuple(operator_type() for operator_type in operators)
        if not self._operators:
            raise ValueError("at least one TraceAAD operator is required")
        self._portfolio = OperatorPortfolio(self._operators, self._portfolio_weights)

        self._best_node: ProgramNode | None = None
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
                    log_event(self, event="search_stopped", status="no_active_trajectory")
                    break
                samples_before = self._tot_sample_nums
                self._run_iteration(attempt_id)
                if self._tot_sample_nums == samples_before:
                    stalled_attempts += 1
                    if stalled_attempts >= self._max_stalled_iterations:
                        log_event(
                            self,
                            event="search_stopped",
                            status="stalled_generation",
                            attempt_id=attempt_id,
                        )
                        break
                else:
                    stalled_attempts = 0
                self._maybe_manage_population()
                attempt_id += 1
                self._next_attempt_id = attempt_id
                self._save_checkpoint_if_due()

            # A run may end during an expansion epoch. Preserve that temporary
            # pool; survival management is triggered only at the 2M threshold.
            self._maybe_manage_population()
            result = self._result()
            finish_profiler(
                self,
                status="aborted" if is_search_aborted(self) else "finished",
                best_node_id=None if result.best_node is None else result.best_node.id,
                best_score=None if result.best_node is None else result.best_node.fitness,
                n_total_nodes=result.n_total_nodes,
                n_valid_nodes=result.n_valid_nodes,
                n_edges=result.n_edges,
                n_trajectories=result.n_trajectories,
            )
            return result
        finally:
            save_checkpoint(self)
            close_llm(self._llm)

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._tot_sample_nums - self._last_checkpoint_sample >= self._checkpoint_interval
        ):
            save_checkpoint(self)

    def _initialize(self) -> None:
        stalled_draws = 0
        draw_seq = 0
        while self._tot_sample_nums < self._n_init and self._has_budget() and not is_search_aborted(self):
            slot = self._tot_sample_nums
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=self._init_diversity_hint(slot),
            )
            generated = self._draw_program(prompt, stage="init", iteration=None, seq=draw_seq, operator="init")
            draw_seq += 1
            if generated is None:
                stalled_draws += 1
                if stalled_draws >= self._max_stalled_iterations:
                    break
                continue
            evaluated = self._evaluate(generated.program, idea=generated.idea, operator="init")
            if evaluated is None or evaluated.fitness is None:
                continue
            stalled_draws = 0
            node = self._add_node(generated, evaluated)
            trajectory = self._memory.create_initial(node_id=node.id)
            self._update_best(node, operator="init")
            log_event(
                self,
                event="trajectory_created",
                status="ok",
                stage="init",
                node_id=node.id,
                trajectory_id=trajectory.id,
            )
        if self._memory.active():
            self._score_active_pool()

    def _init_diversity_hint(self, slot: int) -> str:
        if slot == 0:
            return "Provide a simple, complete, and valid algorithm."
        ideas = [node.idea.strip() for node in self._graph.nodes() if node.idea and node.idea.strip()][-6:]
        if not ideas:
            return "Use a clearly different algorithmic idea from a trivial baseline."
        listed = "; ".join(f"'{idea[:100]}'" for idea in ideas)
        return f"Use a clearly different algorithmic idea from: {listed}."

    def _run_iteration(self, attempt_id: int) -> None:
        selected = self._select_trajectory()
        available = [operator for operator in self._operators if operator.trigger(
            OperatorContext(
                graph=self._graph,
                memory=self._memory,
                selected=selected,
                maximize=self._maximize,
            )
        )]
        operator = random.choice(available or list(self._operators))
        secondary = None
        if getattr(operator, "requires_second", False) and len(self._memory.active()) >= 2:
            secondary = self._select_trajectory(exclude_ids={selected.id})
        primary_anchor = self._select_anchor(selected)
        secondary_anchor = None if secondary is None else self._select_anchor(secondary)
        constraint = operator.build_constraint(
            OperatorContext(
                graph=self._graph,
                memory=self._memory,
                selected=selected,
                maximize=self._maximize,
            ),
            primary_anchor,
        )
        log_event(
            self,
            event="operator_selection",
            status="ok",
            attempt_id=attempt_id,
            selected_operator=operator.name,
            selected_trajectory_id=selected.id,
            secondary_trajectory_id=None if secondary is None else secondary.id,
            base_node_id=primary_anchor,
            secondary_base_node_id=secondary_anchor,
            eligible=[candidate.name for candidate in available or self._operators],
            selection_mode="uniform_operator_softmax_trajectory",
        )
        log_state(
            self,
            phase="iteration_start",
            iteration=attempt_id,
            selected_trajectory_id=selected.id,
            selected_endpoint_id=selected.endpoint_id,
            operator=operator.name,
            base_node_id=primary_anchor,
            selected_value=selected.scalar_value,
        )
        prompt = build_action_prompt(
            graph=self._graph,
            trajectory=selected,
            base_node_id=primary_anchor,
            base_reason="endpoint_or_best",
            operator_name=operator.name,
            operator_constraint=constraint,
            experience_memory=self._experience_memory,
            task_description=self._task_description_str,
            template_function=self._function_to_evolve,
            action_count=self._actions_per_iteration,
            maximize=self._maximize,
            max_steps=self._memory.max_trajectory_length,
            secondary_trajectory=secondary,
            secondary_base_node_id=secondary_anchor,
        )
        actions = self._generate_actions(prompt, attempt_id)
        observations: list[_CandidateObservation] = []
        for seq, action in enumerate(actions):
            if not self._has_budget() or is_search_aborted(self):
                break
            base_node = self._graph.get_node(primary_anchor)
            code_prompt = build_code_prompt(
                current_node=base_node,
                action=action,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
            )
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
            evaluated = self._evaluate(generated.program, idea=generated.idea, operator=operator.name)
            if evaluated is None or evaluated.fitness is None:
                continue
            child = self._add_node(generated, evaluated)
            primary_delta = directed_delta(base_node.fitness, child.fitness, self._maximize)
            primary_outcome = classify_outcome(primary_delta, self._value_weights.positive_threshold)
            primary_edge = self._graph.add_edge(
                parent_id=primary_anchor,
                child_id=child.id,
                action=action,
                operator=operator.name,
                delta=primary_delta,
                outcome=primary_outcome,
                iteration=attempt_id,
            )
            primary_trajectory = self._memory.branch_from(
                trajectory_id=selected.id,
                base_node_id=primary_anchor,
                child_id=child.id,
                edge_id=primary_edge.id,
            )
            self._log_child(
                iteration=attempt_id,
                seq=seq,
                operator=operator,
                child=child,
                trajectory=primary_trajectory,
                accepted=True,
                parent_id=primary_anchor,
                edge_id=primary_edge.id,
                action=action,
                delta=primary_delta,
                outcome=primary_outcome,
            )
            if secondary is not None and secondary_anchor is not None:
                if secondary_anchor == primary_anchor:
                    secondary_edge = primary_edge
                else:
                    secondary_delta = directed_delta(
                        self._graph.get_node(secondary_anchor).fitness,
                        child.fitness,
                        self._maximize,
                    )
                    secondary_edge = self._graph.add_edge(
                        parent_id=secondary_anchor,
                        child_id=child.id,
                        action=action,
                        operator=operator.name,
                        delta=secondary_delta,
                        outcome=classify_outcome(
                            secondary_delta,
                            self._value_weights.positive_threshold,
                        ),
                        iteration=attempt_id,
                    )
                secondary_trajectory = self._memory.branch_from(
                    trajectory_id=secondary.id,
                    base_node_id=secondary_anchor,
                    child_id=child.id,
                    edge_id=secondary_edge.id,
                )
                log_event(
                    self,
                    event="trajectory_created",
                    status="ok",
                    stage="dual_writeback",
                    node_id=child.id,
                    trajectory_id=secondary_trajectory.id,
                    source_trajectory_id=secondary.id,
                )
            self._update_best(child, iteration=attempt_id, operator=operator.name)
            observations.append(
                _CandidateObservation(
                    node_id=child.id,
                    score=float(child.fitness),
                    reference_score=float(base_node.fitness),
                    accepted=True,
                    outcome=primary_outcome,
                )
            )
        reward = self._portfolio_reward(observations)
        self._portfolio.record(operator, reward)
        self._memory.record_visit(selected.id)
        log_event(
            self,
            event="operator_batch",
            status="ok" if observations else "empty",
            attempt_id=attempt_id,
            operator=operator.name,
            reward=reward,
            candidates=[
                {
                    "node_id": observation.node_id,
                    "score": observation.score,
                    "reference_score": observation.reference_score,
                    "accepted": observation.accepted,
                    "outcome": observation.outcome,
                }
                for observation in observations
            ],
            active_trajectories=len(self._memory.active()),
            portfolio=self._portfolio.snapshot(),
        )

    def _select_trajectory(self, exclude_ids: set[int] | None = None) -> Trajectory:
        return sample_trajectory(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
            exclude_ids=exclude_ids,
        )

    def _select_anchor(self, trajectory: Trajectory) -> int:
        endpoint = trajectory.endpoint_id
        best = endpoint
        best_fitness = self._graph.get_node(endpoint).fitness
        for node_id in trajectory.node_ids:
            fitness = self._graph.get_node(node_id).fitness
            if fitness is None:
                continue
            if best_fitness is None or _is_better(fitness, best_fitness, self._maximize):
                best, best_fitness = node_id, fitness
        return random.choice((endpoint, best))

    def _add_node(self, generated: _GeneratedProgram, evaluated: EvalResult) -> ProgramNode:
        return self._graph.add_node(
            code=str(generated.program), idea=generated.idea, fitness=evaluated.fitness
        )

    def _log_child(self, *, iteration: int, seq: int, operator: Operator,
                   child: ProgramNode, trajectory: Trajectory, accepted: bool, **fields) -> None:
        log_event(
            self,
            event="child_accepted",
            status="ok" if accepted else "rejected",
            iteration=iteration,
            seq=seq,
            operator=operator.name,
            child_id=child.id,
            trajectory_id=trajectory.id,
            score=child.fitness,
            **fields,
        )

    def _portfolio_reward(self, observations: list[_CandidateObservation]) -> float:
        if not observations:
            return -1.0
        scale = self._fitness_scale()
        utilities = []
        for observation in observations:
            delta = directed_delta(observation.reference_score, observation.score, self._maximize)
            utilities.append(signed_utility((delta or 0.0) / scale))
        return aggregate_batch_utility(utilities)

    def _score_active_pool(self) -> tuple[Trajectory, ...]:
        return score_active_trajectories(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
        )

    def _maybe_manage_population(self) -> None:
        if len(self._memory.active()) >= self._management_threshold:
            self._manage_population(force=False)

    def _manage_population(self, *, force: bool) -> None:
        active = self._memory.active()
        if not active:
            return
        if not force and len(active) < self._management_threshold:
            return
        if force and len(active) <= self._max_active_trajectories:
            self._score_active_pool()
            return
        ranked = list(self._score_active_pool())
        if len(ranked) <= self._max_active_trajectories:
            return
        elite_count = min(self._elite_count, self._max_active_trajectories, len(ranked))
        elites = ranked[:elite_count]
        diversity_count = min(
            self._diversity_count,
            max(0, self._max_active_trajectories - elite_count),
            max(0, len(ranked) - elite_count),
        )
        diverse = select_diverse_trajectories(
            candidates=tuple(ranked[elite_count:]),
            graph=self._graph,
            count=diversity_count,
        )
        diverse_ids = {trajectory.id for trajectory in diverse}
        sampled = sample_survivors(
            tuple(
                trajectory
                for trajectory in ranked[elite_count:]
                if trajectory.id not in diverse_ids
            ),
            self._max_active_trajectories - elite_count - diversity_count,
            self._softmax_temperature,
        )
        keep_ids = {trajectory.id for trajectory in elites + list(diverse) + list(sampled)}
        archived = 0
        for trajectory in ranked:
            if trajectory.id not in keep_ids:
                self._memory.archive(trajectory.id)
                archived += 1
        log_event(
            self,
            event="population_management",
            status="ok",
            management_threshold=self._management_threshold,
            before=len(ranked),
            after=len(self._memory.active()),
            elite_count=elite_count,
            diversity_count=diversity_count,
            archived=archived,
            selection_mode="elite_plus_diversity_plus_softmax_without_replacement",
        )

    def _fitness_scale(self) -> float:
        scores = [
            node.fitness
            for trajectory in self._memory.active()
            if (node := self._graph.get_node(trajectory.endpoint_id)).fitness is not None
        ]
        if len(scores) < 2:
            return 1.0
        return max(max(scores) - min(scores), 0.05 * abs(statistics.median(scores)), 1e-3)

    def _generate_actions(self, prompt: str, iteration: int) -> list[str]:
        start = time.time()
        try:
            response = self._llm.draw_sample(prompt)
            sample_time = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="action",
                operator="semantic",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                counts_budget=False,
                iteration=iteration,
            )
            return []
        actions = _parse_actions(response, expected_count=self._actions_per_iteration)
        log_llm_call(
            self,
            stage="action",
            operator="semantic",
            sample_order=self._tot_sample_nums + 1,
            iteration=iteration,
            seq=0,
            prompt=prompt,
            response=response,
            sample_time=sample_time,
            parsed_actions=actions,
            status="ok" if actions else "parse_failed",
        )
        return actions

    def _draw_program(self, prompt: str, *, stage: str, iteration: int | None, seq: int,
                      operator: str, action: str | None = None) -> _GeneratedProgram | None:
        sample_order = self._tot_sample_nums + 1
        start = time.time()
        try:
            response = self._llm.draw_sample(prompt)
            sample_time = time.time() - start
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
        generated = _parse_program_response(response, self._template_program, self._function_to_evolve.name)
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
            sample_time=sample_time,
            parsed_idea=None if generated is None else generated.idea,
            program_parse_success=generated is not None,
            status="ok" if generated is not None else "parse_failed",
        )
        return generated

    def _evaluate(self, program: Program, *, idea: str, operator: str) -> EvalResult | None:
        if not self._has_budget():
            return None
        result, eval_time = self._evaluator.evaluate_program_record_time(program)
        self._tot_sample_nums += 1
        score = result.fitness if isinstance(result, EvalResult) else result
        function = TextFunctionProgramConverter.program_to_function(program)
        if function is not None:
            function.algorithm = idea
            function.score = score
            function.evaluate_time = eval_time
            function.operator = operator
            if self._profiler is not None:
                self._profiler.register_function(function, program=str(program))
        log_event(
            self,
            event="program_evaluated",
            status="ok" if score is not None else "eval_failed",
            operator=operator,
            sample_order=self._tot_sample_nums,
            score=score,
            evaluate_time=eval_time,
            counts_budget=True,
        )
        return None if score is None else EvalResult(fitness=float(score))

    def _update_best(self, node: ProgramNode, *, iteration: int | None = None, operator: str) -> None:
        if node.fitness is None:
            return
        if self._best_node is not None and not _is_better(node.fitness, self._best_node.fitness, self._maximize):
            return
        previous = self._best_node
        self._best_node = node
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
        )

    def active_trajectories(self) -> tuple[Trajectory, ...]:
        return self._memory.active()

    def operator_portfolio_snapshot(self) -> dict[str, dict]:
        return self._portfolio.snapshot()

    def _has_budget(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

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


def _is_better(candidate: float, incumbent: float, maximize: bool) -> bool:
    return candidate > incumbent if maximize else candidate < incumbent


def _parse_program_response(response: str, template_program: Program, function_name: str) -> _GeneratedProgram | None:
    idea = _extract_idea(response) or _extract_boxed_text(response) or "Generated program"
    code = _extract_first_code_block(response) or response
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is not None and len(parsed.functions) == 1:
        function = parsed.functions[0]
        if function.name == function_name:
            program = parsed
        else:
            program = TextFunctionProgramConverter.function_to_program(function, template_program)
            if program is None:
                program = parsed
    else:
        program = SampleTrimmer.sample_to_program(code, template_program)
    return None if program is None else _GeneratedProgram(idea=idea, program=program)


def _parse_actions(response: str, *, expected_count: int) -> list[str]:
    actions: list[str] = []
    for line in response.strip().splitlines():
        match = re.match(
            r"^(?:[-*]\s*)?(?:\d+[\.\)]\s*)?(?:Action\s*\d*\s*:\s*)?(?P<action>.+)$",
            line.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            continue
        action = match.group("action").strip()
        if action and not action.startswith(("#", "`")):
            actions.append(action)
    return actions[:expected_count]


def _extract_idea(response: str) -> str | None:
    match = re.search(r"^\s*Idea\s*:\s*(?P<idea>.+?)\s*$", response, flags=re.IGNORECASE | re.MULTILINE)
    return None if match is None else match.group("idea").strip()


def _extract_boxed_text(response: str) -> str | None:
    match = re.search(r"(?:\\)?boxed\s*\{(?P<idea>[^{}]+)\}", response, flags=re.IGNORECASE)
    return None if match is None else match.group("idea").strip()


def _extract_first_code_block(response: str) -> str | None:
    match = re.search(r"```(?:python|py)?\s*(?P<code>.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    return None if match is None else match.group("code").strip()


__all__ = ["TraceAAD", "TraceAADRunResult"]
