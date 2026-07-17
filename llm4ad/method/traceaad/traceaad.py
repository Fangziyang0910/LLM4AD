"""TraceAAD —— 过程信息为一等公民的融合搜索。

三层记忆（ProgramMemory=DerivationGraph / TrajectoryMemory / PatternMemory）+ 三回路
（进化主回路 / 蒸馏回路 / 反思回路）。以有界 trajectory 为唯一搜索单位，stepwise credit，
多维 ValueVec + trajectory-UCB，因果叙事三段式 context，算子组合 + bandit portfolio，
islands + 多层多样性 + novelty gate + 对比反馈。

"""
from __future__ import annotations

import ast
import concurrent.futures
import copy
import random
import re
import time
from dataclasses import dataclass
from typing import Literal, Optional

from ...base import Evaluation, Function, LLM, Program, SampleTrimmer, SecureEvaluator, TextFunctionProgramConverter
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
    shutdown_executor,
)
from .context import build_action_prompt
from .credit import directed_delta
from .derivation_graph import DerivationGraph
from .feedback import RankingModel
from .islands import IslandsManager
from .operators import DEFAULT_OPERATORS, Operator, OperatorContext, classify_outcome, infer_mechanism_tag
from .pattern_memory import PatternMemory
from .portfolio import OperatorPortfolio, PortfolioWeights
from .prompt import build_code_prompt, build_initial_prompt
from .reflection import distill, reflect
from .schema import EvalResult, OperatorName, ProgramNode, Trajectory
from .similarity import max_similarity_to_active
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    best_by_quality,
    compute_value_vec,
    pareto_survival_order,
    robust_active_fitness_bounds,
    scalarize,
    select_trajectory,
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
    mechanism_tag: str
    complexity: int
    reference_complexity: int


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


# 初始化：强制跨机制族多样性（不是仅 thought 多样性）
_INIT_MECHANISM_HINTS = (
    "Design a constructive heuristic using an explicit nearest neighbor rank for candidate selection.",
    "Design a heuristic based on local density or neighborhood scoring of candidate nodes.",
    "Design a heuristic using row-wise normalization before comparing candidate scores.",
    "Design a heuristic that searches only a sparsified candidate list at each construction step.",
)


class TraceAAD:
    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase = None,
        max_sample_nums: Optional[int] = 100,
        *,
        n_init: int = 4,
        actions_per_iteration: int = 2,
        max_trajectory_length: int = 8,
        max_active_trajectories: int | None = None,
        n_islands: int = 4,
        max_per_island: int = 40,
        maximize: bool = True,
        sampling_strategy: Literal["trajectory_ucb", "best", "random"] = "trajectory_ucb",
        value_weights: ValueWeights | None = None,
        portfolio_weights: PortfolioWeights | None = None,
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
        novelty_threshold: float = 0.92,
        k_distill: int = 20,
        patience_reflect: int = 20,
        migration_interval: int = 20,
        min_reflect_new_edges: int = 8,
        num_evaluators: int = 1,
        resume_mode: bool = False,
        debug_mode: bool = False,
        max_consecutive_sample_failures: int = 20,
        max_stalled_iterations: int = 20,
        random_seed: int | None = 0,
        multi_thread_or_process_eval: Literal["thread", "process"] = "thread",
        **kwargs,
    ) -> None:
        if n_init < 0:
            raise ValueError("n_init must be non-negative")
        if actions_per_iteration <= 0:
            raise ValueError("actions_per_iteration must be positive")
        if n_islands <= 0:
            raise ValueError("n_islands must be positive")
        if max_per_island <= 0:
            raise ValueError("max_per_island must be positive")
        if max_active_trajectories is not None and max_active_trajectories <= 0:
            raise ValueError("max_active_trajectories must be positive")
        if num_evaluators <= 0:
            raise ValueError("num_evaluators must be positive")

        self._llm = llm
        self._evaluation = evaluation
        self._profiler = profiler
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = n_init
        self._actions_per_iteration = actions_per_iteration
        self._max_active_trajectories = (
            n_islands * max_per_island
            if max_active_trajectories is None
            else max_active_trajectories
        )
        self._n_islands = n_islands
        self._max_per_island = max_per_island
        self._maximize = maximize
        self._sampling_strategy = sampling_strategy
        self._value_weights = value_weights or ValueWeights()
        self._portfolio_weights = portfolio_weights or PortfolioWeights()
        self._novelty_threshold = novelty_threshold
        self._k_distill = max(1, int(k_distill))
        self._patience_reflect = max(1, int(patience_reflect))
        self._migration_interval = max(1, int(migration_interval))
        self._min_reflect_new_edges = max(1, int(min_reflect_new_edges))
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)
        llm.debug_mode = debug_mode

        self._template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        if self._template_program is None or len(self._template_program.functions) != 1:
            raise ValueError("TraceAAD requires an evaluation template with exactly one evolvable function.")
        self._function_to_evolve: Function = copy.deepcopy(self._template_program.functions[0])

        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        # 三层记忆
        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._pattern_memory = PatternMemory()
        # 回路辅助
        self._islands = IslandsManager(n_islands=n_islands)
        self._ranking = RankingModel()
        self._operators = tuple(op_cls() for op_cls in operators)
        self._portfolio = OperatorPortfolio(
            self._operators,
            self._portfolio_weights,
            rng=self._rng,
        )
        # 状态
        self._best_node: ProgramNode | None = None
        self._tot_sample_nums = 0
        self._batch_cost = 0.0
        self._last_reflect_edge_count = 0
        init_observability(self, max_consecutive_sample_failures)

        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_evaluators)

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    # ---------------- main loop ----------------
    def run(self) -> TraceAADRunResult:
        try:
            if not self._resume_mode:
                self._initialize()
            phase_horizon = self._planned_iterations()
            search_sample_start = self._tot_sample_nums
            last_best = self._best_node.fitness if self._best_node else None
            stagnation = 0
            stalled_iterations = 0
            completed_search_iteration = 0
            attempt_id = 0
            while self._has_budget() and not is_search_aborted(self):
                if len(self._memory.active()) == 0:
                    log_event(self, event="search_stopped", status="no_active_trajectory")
                    break
                samples_before = self._tot_sample_nums
                search_iteration = (
                    self._tot_sample_nums - search_sample_start
                ) // self._actions_per_iteration
                self._run_iteration(
                    search_iteration,
                    phase_horizon,
                    attempt_id=attempt_id,
                )

                new_completed_iteration = (
                    self._tot_sample_nums - search_sample_start
                ) // self._actions_per_iteration
                if new_completed_iteration > completed_search_iteration:
                    cur_best = self._best_node.fitness if self._best_node else None
                    if cur_best is None or last_best is None or _equal(cur_best, last_best):
                        stagnation += 1
                    else:
                        stagnation = 0
                        last_best = cur_best
                    self._stagnation = stagnation
                    self._periodic_hooks(new_completed_iteration)
                    completed_search_iteration = new_completed_iteration

                if self._tot_sample_nums == samples_before:
                    stalled_iterations += 1
                    if stalled_iterations >= self._max_stalled_iterations:
                        log_event(
                            self,
                            event="search_stopped",
                            status="stalled_generation",
                            iteration=search_iteration,
                            attempt_id=attempt_id,
                            stalled_iterations=stalled_iterations,
                        )
                        break
                else:
                    stalled_iterations = 0
                attempt_id += 1

            if self._memory.active():
                self._survive()
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
            close_llm(self._llm)
            shutdown_executor(self._evaluation_executor)

    def _initialize(self) -> None:
        initial_sample_count = self._tot_sample_nums
        draw_seq = 0
        stalled_draws = 0
        while (
            self._tot_sample_nums - initial_sample_count < self._n_init
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            slot = self._tot_sample_nums - initial_sample_count
            hint = _INIT_MECHANISM_HINTS[slot % len(_INIT_MECHANISM_HINTS)]
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=hint,
            )
            gen = self._draw_program(
                prompt,
                stage="init",
                iteration=None,
                seq=draw_seq,
                operator="init",
            )
            draw_seq += 1
            if gen is None:
                stalled_draws += 1
                if stalled_draws >= self._max_stalled_iterations:
                    log_event(
                        self,
                        event="initialization_stopped",
                        status="stalled_generation",
                        stalled_draws=stalled_draws,
                        initialized_samples=self._tot_sample_nums - initial_sample_count,
                    )
                    break
                continue
            ev = self._evaluate(gen.program, idea=gen.idea, operator="init")
            stalled_draws = 0
            hint_tag = infer_mechanism_tag(hint)
            tag = infer_mechanism_tag(
                f"{gen.idea}\n{gen.program}",
                hint=hint_tag,
            )
            if ev is None or ev.fitness is None:
                self._graph.add_node(
                    code=str(gen.program), idea=gen.idea, fitness=None, is_valid=False,
                    mechanism_tag=tag,
                )
                continue
            node = self._graph.add_node(
                code=str(gen.program), idea=gen.idea, fitness=ev.fitness, is_valid=True,
                complexity=ev.complexity, mechanism_tag=tag,
            )
            island = slot % self._n_islands
            traj = self._memory.create_initial(node_id=node.id, island_id=island)
            self._update_best(node)
            log_event(self, event="trajectory_created", status="ok", stage="init",
                       node_id=node.id, trajectory_id=traj.id, island_id=island, mechanism_tag=tag)

    def _run_iteration(
        self,
        iteration: int,
        max_iter: int,
        *,
        attempt_id: int,
    ) -> None:
        incumbent = self._best_node.fitness if self._best_node is not None else None
        reward_scale = self._fitness_scale()
        self._batch_cost = 0.0
        if self._sampling_strategy == "best":
            selected = best_by_quality(
                memory=self._memory,
                graph=self._graph,
                maximize=self._maximize,
            )
        elif self._sampling_strategy == "random":
            selected = self._rng.choice(self._memory.unique_active())
        else:
            selected = select_trajectory(
                memory=self._memory, graph=self._graph,
                maximize=self._maximize, iteration=iteration, max_iter=max_iter, w=self._value_weights,
                elite_endpoint_id=None if self._best_node is None else self._best_node.id,
                stagnation=getattr(self, "_stagnation", 0),
                rng=self._rng,
            )
        selected = self._memory.get_trajectory(selected.id)
        ctx = OperatorContext(
            graph=self._graph, memory=self._memory, pattern_memory=self._pattern_memory,
            islands=self._islands, selected=selected,
            maximize=self._maximize, positive_threshold=self._value_weights.positive_threshold,
            iteration=iteration, best_stagnation=getattr(self, "_stagnation", 0),
        )
        op = self._portfolio.choose(ctx=ctx, iteration=iteration, max_iter=max_iter)
        # 算子可 override 选题：backtrack 主动从 pool 选「endpoint 退步但前缀高 value」的 trajectory
        target = op.select_trajectory(ctx)
        if target is not None:
            ctx.selected = self._memory.get_trajectory(target.id)
        base_node_id, base_reason = op.select_base(ctx)
        op_constraint = op.build_constraint(ctx, base_node_id)

        log_state(
            self, phase="iteration_start", iteration=iteration,
            selected_trajectory_id=ctx.selected.id, selected_endpoint_id=ctx.selected.endpoint_id,
            operator=op.name, operator_role=op.role,
            base_node_id=base_node_id, base_reason=base_reason,
            n_active_trajectories=len(self._memory.active()),
            best_stagnation=getattr(self, "_stagnation", 0),
            selected_value=None if ctx.selected.value is None else ctx.selected.value.as_tuple(),
            selected_scalar_value=ctx.selected.scalar_value,
            attempt_id=attempt_id,
        )

        if base_node_id is None:
            observations = self._run_fresh_start(op, ctx, op_constraint, iteration, incumbent)
        else:
            observations = self._run_refine(
                op, ctx, op_constraint, base_node_id, base_reason, iteration, incumbent
            )

        self._update_portfolio_batch(
            op,
            iteration,
            attempt_id,
            observations,
            incumbent,
            reward_scale,
        )

        self._memory.record_visit(ctx.selected.id)

    def _run_fresh_start(
        self,
        op: Operator,
        ctx: OperatorContext,
        constraint: str,
        iteration: int,
        incumbent: float | None,
    ) -> list[_CandidateObservation]:
        """novelty：initial-style 生成若干新起点，每个 create_initial 到新 island。"""
        observations: list[_CandidateObservation] = []
        for seq in range(self._actions_per_iteration):
            if not self._has_budget() or is_search_aborted(self):
                break
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=constraint,
            )
            gen = self._draw_program(prompt, stage="novelty", iteration=iteration, seq=seq, operator=op.name)
            if gen is None:
                continue
            ev = self._evaluate(gen.program, idea=gen.idea, operator=op.name)
            if ev is None or ev.fitness is None:
                continue
            requested_tag = ctx.hints.get("mechanism_tag_hint") or infer_mechanism_tag(constraint)
            tag = infer_mechanism_tag(
                f"{gen.idea}\n{gen.program}",
                hint=requested_tag,
            )
            ctx.hints["observed_mechanism_tag"] = tag
            child = self._graph.add_node(
                code=str(gen.program), idea=gen.idea, fitness=ev.fitness, is_valid=True,
                complexity=ev.complexity, mechanism_tag=tag,
            )
            new_traj = op.insert(ctx, child.id, None, None)
            live_best = self._best_node.fitness if self._best_node is not None else None
            is_record = live_best is None or _is_better(ev.fitness, live_best, self._maximize)
            is_near_record = is_record or self._is_near_record(ev.fitness, live_best)
            accepted = self._apply_novelty_gate(new_traj, protect=is_record)
            self._update_best(child)
            reference = incumbent if incumbent is not None else ev.fitness
            outcome = classify_outcome(
                directed_delta(reference, ev.fitness, self._maximize),
                self._value_weights.positive_threshold,
            )
            self._pattern_memory.record_mechanism_outcome(
                operator=op.name,
                mechanism_tag=tag,
                support_id=child.id,
                success=is_near_record,
                iteration=iteration,
            )
            observations.append(_CandidateObservation(
                node_id=child.id,
                score=ev.fitness,
                reference_score=reference,
                accepted=accepted,
                outcome=outcome,
                mechanism_tag=tag,
                complexity=child.complexity,
                reference_complexity=child.complexity,
            ))
            log_event(self, event="child_accepted", status="ok" if accepted else "novelty_rejected",
                       iteration=iteration, seq=seq, operator=op.name, child_id=child.id,
                       trajectory_id=new_traj.id, score=ev.fitness, mechanism_tag=tag)
        return observations

    def _run_refine(self, op: Operator, ctx: OperatorContext, constraint: str,
                     base_node_id: int, base_reason: str, iteration: int,
                     incumbent: float | None) -> list[_CandidateObservation]:
        base_node = self._graph.get_node(base_node_id)
        observations: list[_CandidateObservation] = []
        contrast = self._ranking.contrast(
            graph=self._graph, memory=self._memory, maximize=self._maximize
        )
        action_prompt = build_action_prompt(
            graph=self._graph, trajectory=ctx.selected, base_node_id=base_node_id,
            base_reason=base_reason, operator_name=op.name, operator_role=op.role,
            operator_constraint=constraint, pattern_memory=self._pattern_memory,
            contrast=contrast, task_description=self._task_description_str,
            template_function=self._function_to_evolve, action_count=self._actions_per_iteration,
            maximize=self._maximize,
        )
        actions = self._generate_actions(action_prompt, iteration)
        for seq, action in enumerate(actions):
            if not self._has_budget() or is_search_aborted(self):
                break
            code_prompt = build_code_prompt(
                current_node=base_node, action=action,
                task_description=self._task_description_str, template_function=self._function_to_evolve,
            )
            gen = self._draw_program(code_prompt, stage="code", iteration=iteration, seq=seq,
                                       operator=op.name, action=action)
            if gen is None:
                continue
            ev = self._evaluate(gen.program, idea=gen.idea, operator=op.name)
            if ev is None or ev.fitness is None:
                continue
            hint = ctx.hints.get("mechanism_tag_hint") or ctx.hints.get("donor_mechanism")
            tag = infer_mechanism_tag(
                f"{action}\n{gen.idea}\n{gen.program}",
                hint=hint,
            )
            delta = directed_delta(base_node.fitness, ev.fitness, self._maximize)
            outcome = classify_outcome(delta, self._value_weights.positive_threshold)
            child = self._graph.add_node(
                code=str(gen.program), idea=gen.idea, fitness=ev.fitness, is_valid=True,
                complexity=ev.complexity, mechanism_tag=tag,
            )
            edge = self._graph.add_edge(
                parent_id=base_node_id, child_id=child.id, action=action, operator=op.name,
                mechanism_tag=tag, delta=delta, outcome=outcome,
                iteration=iteration,
            )
            new_traj = op.insert(ctx, child.id, edge.id, base_node_id)
            live_best = self._best_node.fitness if self._best_node is not None else None
            is_record = live_best is None or _is_better(ev.fitness, live_best, self._maximize)
            novel = self._apply_novelty_gate(new_traj, protect=is_record)
            if base_node.fitness is not None:
                self._ranking.update_by_fitness(
                    a=child.id, b=base_node_id, fitness_a=ev.fitness,
                    fitness_b=base_node.fitness, maximize=self._maximize,
                )
            self._update_best(child)
            self._pattern_memory.record_mechanism_outcome(
                operator=op.name,
                mechanism_tag=tag,
                support_id=edge.id,
                success=outcome == "improve",
                iteration=iteration,
            )
            observations.append(_CandidateObservation(
                node_id=child.id,
                score=ev.fitness,
                reference_score=base_node.fitness,
                accepted=novel,
                outcome=outcome,
                mechanism_tag=tag,
                complexity=child.complexity,
                reference_complexity=base_node.complexity,
            ))
            log_event(self, event="child_accepted", status="ok" if novel else "novelty_rejected",
                       iteration=iteration, seq=seq, operator=op.name, parent_id=base_node_id,
                       child_id=child.id, edge_id=edge.id, trajectory_id=new_traj.id, action=action,
                       score=ev.fitness, delta=delta, outcome=outcome, mechanism_tag=tag)
        return observations

    # ---------------- periodic hooks (蒸馏/反思/migration/survival) ----------------
    def _periodic_hooks(self, iteration: int) -> None:
        self._survive()
        if iteration > 0 and iteration % self._k_distill == 0:
            n = distill(graph=self._graph, pattern_memory=self._pattern_memory, iteration=iteration)
            if n:
                log_event(self, event="distill", status="ok", iteration=iteration, n_patterns=n)
        stagnation = getattr(self, "_stagnation", 0)
        edge_count = len(self._graph.edges())
        has_new_reflection_evidence = (
            edge_count - self._last_reflect_edge_count >= self._min_reflect_new_edges
        )
        if (
            stagnation > 0
            and stagnation % self._patience_reflect == 0
            and has_new_reflection_evidence
        ):
            contrast = reflect(memory=self._memory, graph=self._graph, pattern_memory=self._pattern_memory,
                                ranking=self._ranking, maximize=self._maximize, iteration=iteration)
            if contrast:
                log_event(self, event="reflect", status="ok", iteration=iteration)
                self._last_reflect_edge_count = edge_count
        # 停滞时促进 island 间流动
        if (
            stagnation > 0
            and iteration > 0
            and iteration % self._migration_interval == 0
        ):
            trajectories_before = len(self._memory.trajectories())
            moved = self._islands.migrate(memory=self._memory)
            if moved:
                log_event(
                    self,
                    event="migrate",
                    status="ok",
                    iteration=iteration,
                    moved=moved,
                    trajectories_before=trajectories_before,
                    trajectories_after=len(self._memory.trajectories()),
                    island_sizes={
                        island: len(self._memory.active_in_island(island))
                        for island in self._memory.island_ids()
                    },
                )

    def _survive(self) -> None:
        duplicate_count = self._memory.archive_duplicate_paths()
        if duplicate_count:
            log_event(
                self,
                event="trajectory_deduplicated",
                status="ok",
                archived_duplicates=duplicate_count,
            )
        self._score_active_pool()
        elite_paths = tuple(
            trajectory
            for trajectory in self._memory.active()
            if self._best_node is not None
            and trajectory.endpoint_id == self._best_node.id
        )
        protected = (
            {pareto_survival_order(elite_paths)[0].id}
            if elite_paths
            else set()
        )
        for island in self._memory.island_ids():
            members = self._memory.active_in_island(island)
            self._archive_to_cap(members, self._max_per_island, protected)
        actives = self._memory.active()
        self._archive_to_cap(actives, self._max_active_trajectories, protected)

    def _archive_to_cap(
        self,
        members: tuple[Trajectory, ...],
        cap: int,
        protected: set[int],
    ) -> None:
        if len(members) <= cap:
            return
        protected_ids = {trajectory.id for trajectory in members if trajectory.id in protected}
        keep_ids = set(protected_ids)
        slots = max(cap - len(keep_ids), 0)
        for trajectory in pareto_survival_order(members):
            if trajectory.id in keep_ids:
                continue
            if slots > 0:
                keep_ids.add(trajectory.id)
                slots -= 1
        for trajectory in members:
            if trajectory.id not in keep_ids:
                self._memory.archive(trajectory.id)

    # ---------------- novelty gate & scoring ----------------
    def _apply_novelty_gate(self, traj: Trajectory, *, protect: bool = False) -> bool:
        others = tuple(o for o in self._memory.active() if o.id != traj.id)
        if not others:
            self._score_trajectory(traj)
            return True
        weights = (self._value_weights.w_sim_code, self._value_weights.w_sim_mechanism,
                    self._value_weights.w_sim_trajectory)
        max_sim = max_similarity_to_active(graph=self._graph, candidate=traj, others=others, weights=weights)
        endpoint = self._graph.get_node(traj.endpoint_id)
        historical = tuple(
            other for other in self._memory.trajectories() if other.id != traj.id
        )
        behavioral_duplicate = any(
            self._graph.get_node(other.endpoint_id).mechanism_tag == endpoint.mechanism_tag
            and _equal(self._graph.get_node(other.endpoint_id).fitness, endpoint.fitness)
            for other in historical
            if self._graph.get_node(other.endpoint_id).fitness is not None
            and endpoint.fitness is not None
        )
        reject_reason = None
        if max_sim >= self._novelty_threshold:
            reject_reason = "structural_similarity"
        elif behavioral_duplicate:
            reject_reason = "behavioral_duplicate"
        if reject_reason is not None and not protect:
            self._memory.archive(traj.id)
            log_event(
                self,
                event="novelty_gate",
                status="rejected",
                trajectory_id=traj.id,
                endpoint_id=traj.endpoint_id,
                max_similarity=max_sim,
                reason=reject_reason,
            )
            return False
        self._score_trajectory(traj)
        log_event(
            self,
            event="novelty_gate",
            status="quality_override" if protect and reject_reason is not None else "accepted",
            trajectory_id=traj.id,
            endpoint_id=traj.endpoint_id,
            max_similarity=max_sim,
            reason=reject_reason,
        )
        return True

    def _score_trajectory(self, traj: Trajectory) -> None:
        actives = self._memory.active()
        fmin, fmax = robust_active_fitness_bounds(
            trajectories=actives,
            graph=self._graph,
            clip_quantile=self._value_weights.fitness_clip_quantile,
        )
        others = tuple(o for o in actives if o.id != traj.id)
        value = compute_value_vec(
            trajectory=traj, graph=self._graph,
            active_others=others, fmin=fmin, fmax=fmax, maximize=self._maximize, w=self._value_weights,
        )
        self._memory.set_value(traj.id, value, scalarize(value, self._value_weights))

    def _score_active_pool(self) -> None:
        actives = self._memory.active()
        fmin, fmax = robust_active_fitness_bounds(
            trajectories=actives,
            graph=self._graph,
            clip_quantile=self._value_weights.fitness_clip_quantile,
        )
        for trajectory in actives:
            others = tuple(other for other in actives if other.id != trajectory.id)
            value = compute_value_vec(
                trajectory=trajectory,
                graph=self._graph,
                active_others=others,
                fmin=fmin,
                fmax=fmax,
                maximize=self._maximize,
                w=self._value_weights,
            )
            self._memory.set_value(
                trajectory.id,
                value,
                scalarize(value, self._value_weights),
            )

    def _update_portfolio_batch(
        self,
        op: Operator,
        iteration: int,
        attempt_id: int,
        observations: list[_CandidateObservation],
        incumbent: float | None,
        reward_scale: float,
    ) -> None:
        if not observations:
            self._portfolio.update_batch(
                op=op,
                iteration=attempt_id,
                normalized_reward=-1.0,
                best_valid=False,
                best_novel=False,
                best_regress=True,
                total_cost=self._batch_cost,
                global_best=False,
                near_record=False,
            )
            return
        best = max(observations, key=lambda x: x.score) if self._maximize else min(
            observations, key=lambda x: x.score
        )
        gain = directed_delta(best.reference_score, best.score, self._maximize) or 0.0
        reward = max(-1.0, min(1.0, gain / reward_scale))
        if op.name == OperatorName.SIMPLIFY and reward >= 0.0 and best.reference_complexity > 0:
            complexity_gain = (
                best.reference_complexity - best.complexity
            ) / best.reference_complexity
            reward = max(-1.0, min(1.0, 0.8 * reward + 0.2 * complexity_gain))
        is_record = incumbent is None or _is_better(best.score, incumbent, self._maximize)
        is_near_record = is_record or self._is_near_record(best.score, incumbent)
        self._portfolio.update_batch(
            op=op,
            iteration=attempt_id,
            normalized_reward=reward,
            best_valid=True,
            best_novel=best.accepted,
            best_regress=reward < 0.0,
            total_cost=self._batch_cost,
            global_best=is_record,
            near_record=is_near_record,
        )
        log_event(
            self,
            event="operator_batch",
            status="ok",
            iteration=iteration,
            attempt_id=attempt_id,
            operator=op.name,
            best_child_id=best.node_id,
            best_score=best.score,
            normalized_reward=reward,
            global_best=is_record,
            near_record=is_near_record,
            total_cost=self._batch_cost,
            mechanism_tag=best.mechanism_tag,
            portfolio=self._portfolio.snapshot(),
        )

    def _fitness_scale(self) -> float:
        scores = sorted(
            node.fitness
            for node in (
                self._graph.get_node(t.endpoint_id) for t in self._memory.unique_active()
            )
            if node.is_valid and node.fitness is not None
        )
        if len(scores) < 2:
            return 1.0
        lo = scores[int(0.1 * (len(scores) - 1))]
        hi = scores[int(0.9 * (len(scores) - 1))]
        median = scores[len(scores) // 2]
        return max(abs(hi - lo), 0.05 * abs(median), 1e-3)

    def _is_near_record(
        self,
        candidate: float,
        incumbent: float | None,
    ) -> bool:
        if incumbent is None:
            return True
        shortfall = directed_delta(incumbent, candidate, self._maximize)
        if shortfall is None:
            return False
        tolerance = max(0.0, self._portfolio_weights.near_record_tolerance)
        return shortfall >= -tolerance * self._fitness_scale()

    # ---------------- LLM generation & evaluation ----------------
    def _generate_actions(self, prompt: str, iteration: int) -> list[str]:
        try:
            start = time.time()
            response = self._llm.draw_sample(prompt)
            sample_time = time.time() - start
            self._batch_cost += sample_time
            reset_sample_failures(self)
        except Exception as exc:
            self._batch_cost += time.time() - start
            record_sample_failure(self, exc, stage="action", operator="action",
                                   sample_order=self._tot_sample_nums + 1, prompt=prompt, counts_budget=False, iteration=iteration)
            return []
        actions = _parse_actions(response, expected_count=self._actions_per_iteration)
        log_llm_call(self, stage="action", operator="action", sample_order=self._tot_sample_nums + 1,
                      iteration=iteration, seq=0, prompt=prompt, response=response, sample_time=sample_time,
                      parsed_actions=actions, status="ok")
        return actions

    def _draw_program(self, prompt: str, *, stage: str, iteration: int | None, seq: int,
                       operator: str, action: str | None = None) -> _GeneratedProgram | None:
        sample_order = self._tot_sample_nums + 1
        try:
            start = time.time()
            response = self._llm.draw_sample(prompt)
            sample_time = time.time() - start
            self._batch_cost += sample_time
            reset_sample_failures(self)
        except Exception as exc:
            self._batch_cost += time.time() - start
            record_sample_failure(self, exc, stage=stage, operator=operator, sample_order=sample_order,
                                   prompt=prompt, counts_budget=False, iteration=iteration, seq=seq, action=action)
            return None
        gen = _parse_program_response(response, self._template_program, self._function_to_evolve.name)
        log_llm_call(self, stage=stage, operator=operator, sample_order=sample_order, iteration=iteration,
                      seq=seq, action=action, prompt=prompt, response=response, sample_time=sample_time,
                      parsed_idea=None if gen is None else gen.idea, program_parse_success=gen is not None,
                      status="ok" if gen is not None else "parse_failed")
        return gen

    def _evaluate(self, program: Program, *, idea: str, operator: str) -> EvalResult | None:
        if not self._has_budget():
            return None
        future = self._evaluation_executor.submit(self._evaluator.evaluate_program_record_time, program)
        result, eval_time = future.result()
        self._batch_cost += eval_time
        self._tot_sample_nums += 1
        sample_order = self._tot_sample_nums
        if isinstance(result, EvalResult):
            score = result.fitness
        else:
            score = result
        function = TextFunctionProgramConverter.program_to_function(program)
        if function is not None:
            function.algorithm = idea
            function.score = score
            function.evaluate_time = eval_time
            function.operator = operator
            if self._profiler is not None:
                self._profiler.register_function(function, program=str(program))
        log_event(self, event="program_evaluated", status="ok" if score is not None else "eval_failed",
                   operator=operator, sample_order=sample_order, score=score, evaluate_time=eval_time, counts_budget=True)
        if score is None:
            return None
        if isinstance(result, EvalResult):
            complexity = result.complexity or _ast_complexity(str(program))
            return EvalResult(fitness=score, complexity=complexity)
        complexity = _ast_complexity(str(program))
        return EvalResult(fitness=score, complexity=complexity)

    # ---------------- bookkeeping ----------------
    def _update_best(self, node: ProgramNode) -> None:
        if not node.is_valid or node.fitness is None:
            return
        if self._best_node is None or _is_better(node.fitness, self._best_node.fitness, self._maximize):
            self._best_node = node

    def active_trajectories(self) -> tuple[Trajectory, ...]:
        return self._memory.active()

    def operator_portfolio_snapshot(self) -> dict[str, dict]:
        return self._portfolio.snapshot()

    def _planned_iterations(self) -> int:
        if self._max_sample_nums is None:
            return 10**9
        remaining = max(self._max_sample_nums - self._tot_sample_nums, 0)
        return max(1, (remaining + self._actions_per_iteration - 1) // self._actions_per_iteration)

    def _has_budget(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _result(self) -> TraceAADRunResult:
        nodes = self._graph.nodes()
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=len(nodes),
            n_valid_nodes=sum(1 for n in nodes if n.is_valid),
            n_trajectories=len(self._memory.trajectories()),
            n_edges=len(self._graph.edges()),
            n_samples=self._tot_sample_nums,
        )


def _is_better(candidate: float, incumbent: float, maximize: bool) -> bool:
    return candidate > incumbent if maximize else candidate < incumbent


def _equal(a: float, b: float) -> bool:
    return abs(a - b) < 1e-12


def _ast_complexity(code: str) -> int:
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except Exception:
        return code.count("\n") + 1


def _parse_program_response(response: str, template_program: Program, function_name: str) -> _GeneratedProgram | None:
    idea = _extract_idea(response) or _extract_boxed_text(response) or "Generated program"
    code = _extract_first_code_block(response) or response
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is not None and len(parsed.functions) == 1:
        fn = parsed.functions[0]
        if fn.name == function_name:
            program = parsed
        else:
            program = TextFunctionProgramConverter.function_to_program(fn, template_program)
            if program is None:
                program = parsed
    else:
        program = SampleTrimmer.sample_to_program(code, template_program)
    if program is None:
        return None
    return _GeneratedProgram(idea=idea, program=program)


def _parse_actions(response: str, *, expected_count: int) -> list[str]:
    actions: list[str] = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(
            r"^(?:[-*]\s*)?(?:\d+[\.\)]\s*)?(?:Action\s*\d*\s*:\s*)?(?P<action>.+)$",
            line, flags=re.IGNORECASE,
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
    match = re.search(r"boxed\s*\{(?P<idea>[^{}]+)\}", response, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"\\boxed\s*\{(?P<idea>[^{}]+)\}", response, flags=re.IGNORECASE)
    return None if match is None else match.group("idea").strip()


def _extract_first_code_block(response: str) -> str | None:
    match = re.search(r"```(?:python|py)?\s*(?P<code>.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    return None if match is None else match.group("code").strip()
