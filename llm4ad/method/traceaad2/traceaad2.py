"""TraceAAD2 —— 过程信息为一等公民的融合搜索（design traceaad-fusion-design.md）。

三层记忆（ProgramMemory=DerivationGraph / TrajectoryMemory / PatternMemory）+ 三回路
（进化主回路 / 蒸馏回路 / 反思回路）。以有界 trajectory 为唯一搜索单位，stepwise+泛化信用，
多维 ValueVec + trajectory-UCB，因果叙事三段式 context，6 算子 + bandit portfolio，
islands + 多层多样性 + novelty gate + 鲁棒对比反馈。

与 v1（method/traceaad）并行存在，互不影响。
"""
from __future__ import annotations

import ast
import concurrent.futures
import copy
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
from .credit import directed_delta, normalize_fitness, step_generalization_signal
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
from .value import ValueWeights, compute_value_vec, scalarize, select_trajectory


@dataclass(frozen=True, slots=True)
class _GeneratedProgram:
    idea: str
    program: Program


@dataclass(frozen=True, slots=True)
class TraceAAD2RunResult:
    best_node: ProgramNode | None
    best_generalization_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


# 初始化：强制跨机制族多样性（不是仅 thought 多样性）
_INIT_MECHANISM_HINTS = (
    "Design a direct constructive heuristic using nearest-neighbor / greedy distance selection.",
    "Design a heuristic based on local density or neighborhood scoring of candidate nodes.",
    "Design a heuristic using ranking or probabilistic/softmax selection over candidates.",
    "Design a hybrid heuristic combining two simple scoring components.",
)


class TraceAAD2:
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
        max_active_trajectories: int = 1000,
        n_islands: int = 4,
        max_per_island: int = 40,
        maximize: bool = True,
        sampling_strategy: Literal["trajectory_ucb", "best", "random"] = "trajectory_ucb",
        value_weights: ValueWeights | None = None,
        portfolio_weights: PortfolioWeights | None = None,
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
        novelty_threshold: float = 0.92,
        k_distill: int = 10,
        patience_reflect: int = 8,
        num_evaluators: int = 1,
        resume_mode: bool = False,
        debug_mode: bool = False,
        max_consecutive_sample_failures: int = 20,
        multi_thread_or_process_eval: Literal["thread", "process"] = "thread",
        **kwargs,
    ) -> None:
        if n_init < 0:
            raise ValueError("n_init must be non-negative")
        if actions_per_iteration <= 0:
            raise ValueError("actions_per_iteration must be positive")

        self._llm = llm
        self._evaluation = evaluation
        self._profiler = profiler
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = n_init
        self._actions_per_iteration = actions_per_iteration
        self._max_active_trajectories = max_active_trajectories
        self._n_islands = n_islands
        self._max_per_island = max_per_island
        self._maximize = maximize
        self._sampling_strategy = sampling_strategy
        self._value_weights = value_weights or ValueWeights()
        self._portfolio_weights = portfolio_weights or PortfolioWeights()
        self._novelty_threshold = novelty_threshold
        self._k_distill = k_distill
        self._patience_reflect = patience_reflect
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        llm.debug_mode = debug_mode

        self._template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        if self._template_program is None or len(self._template_program.functions) != 1:
            raise ValueError("TraceAAD2 requires an evaluation template with exactly one evolvable function.")
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
        self._portfolio = OperatorPortfolio(self._operators, self._portfolio_weights)
        # 状态
        self._best_node: ProgramNode | None = None
        self._best_generalization_node: ProgramNode | None = None
        self._best_score_history: list[float] = []
        self._tot_sample_nums = 0
        init_observability(self, max_consecutive_sample_failures)

        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_evaluators)

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    # ---------------- main loop ----------------
    def run(self) -> TraceAAD2RunResult:
        try:
            if not self._resume_mode:
                self._initialize()
            max_iter = self._planned_iterations()
            last_best = self._best_node.fitness if self._best_node else None
            stagnation = 0
            for iteration in range(max_iter):
                if not self._has_budget() or is_search_aborted(self):
                    break
                if len(self._memory.active()) == 0:
                    log_event(self, event="search_stopped", status="no_active_trajectory")
                    break
                self._run_iteration(iteration, max_iter)
                self._periodic_hooks(iteration)

                cur_best = self._best_node.fitness if self._best_node else None
                if cur_best is None or last_best is None or _equal(cur_best, last_best):
                    stagnation += 1
                else:
                    stagnation = 0
                    last_best = cur_best
                self._stagnation = stagnation

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
        for seq in range(self._n_init):
            if not self._has_budget() or is_search_aborted(self):
                break
            hint = _INIT_MECHANISM_HINTS[seq % len(_INIT_MECHANISM_HINTS)]
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=hint,
            )
            gen = self._draw_program(prompt, stage="init", iteration=None, seq=seq, operator="init")
            if gen is None:
                continue
            ev = self._evaluate(gen.program, idea=gen.idea, operator="init")
            tag = infer_mechanism_tag(hint)
            if ev is None or ev.fitness is None:
                self._graph.add_node(
                    code=str(gen.program), idea=gen.idea, fitness=None, is_valid=False,
                    mechanism_tag=tag, sample_order=self._tot_sample_nums,
                )
                continue
            node = self._graph.add_node(
                code=str(gen.program), idea=gen.idea, fitness=ev.fitness, is_valid=True,
                runtime=ev.runtime, complexity=ev.complexity, robustness=ev.robustness,
                mechanism_tag=tag, sample_order=self._tot_sample_nums,
            )
            island = self._islands.assign(tag)
            traj = self._memory.create_initial(node_id=node.id, island_id=island)
            self._update_best(node)
            log_event(self, event="trajectory_created", status="ok", stage="init",
                       node_id=node.id, trajectory_id=traj.id, island_id=island, mechanism_tag=tag)

    def _run_iteration(self, iteration: int, max_iter: int) -> None:
        selected = select_trajectory(
            memory=self._memory, graph=self._graph, pattern_memory=self._pattern_memory,
            maximize=self._maximize, iteration=iteration, max_iter=max_iter, w=self._value_weights,
        )
        selected = self._memory.get_trajectory(selected.id)
        ctx = OperatorContext(
            graph=self._graph, memory=self._memory, pattern_memory=self._pattern_memory,
            ranking=self._ranking, islands=self._islands, selected=selected,
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
        )

        if base_node_id is None:
            self._run_fresh_start(op, ctx, op_constraint, iteration)
        else:
            self._run_refine(op, ctx, op_constraint, base_node_id, base_reason, iteration)

        self._memory.record_visit(ctx.selected.id)

    def _run_fresh_start(self, op: Operator, ctx: OperatorContext, constraint: str, iteration: int) -> None:
        """novelty：initial-style 生成若干新起点，每个 create_initial 到新 island。"""
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
                self._portfolio.update(op=op, gain=0.0, valid=False, novel=False, regress=False, cost=0.0)
                continue
            ev = self._evaluate(gen.program, idea=gen.idea, operator=op.name)
            if ev is None or ev.fitness is None:
                self._portfolio.update(op=op, gain=0.0, valid=False, novel=False, regress=False, cost=ev.runtime if ev else 0.0)
                continue
            tag = ctx.hints.get("mechanism_tag_hint") or infer_mechanism_tag(constraint)
            child = self._graph.add_node(
                code=str(gen.program), idea=gen.idea, fitness=ev.fitness, is_valid=True,
                runtime=ev.runtime, complexity=ev.complexity, robustness=ev.robustness,
                mechanism_tag=tag, iteration=iteration, sample_order=self._tot_sample_nums,
            )
            new_traj = op.insert(ctx, child.id, None, None)
            accepted = self._apply_novelty_gate(new_traj)
            self._update_best(child)
            self._portfolio.update(op=op, gain=self._relative_quality_gain(ev.fitness),
                                     valid=True, novel=True, regress=False, cost=ev.runtime)
            log_event(self, event="child_accepted", status="ok" if accepted else "novelty_rejected",
                       iteration=iteration, seq=seq, operator=op.name, child_id=child.id,
                       trajectory_id=new_traj.id, score=ev.fitness, mechanism_tag=tag)

    def _run_refine(self, op: Operator, ctx: OperatorContext, constraint: str,
                     base_node_id: int, base_reason: str, iteration: int) -> None:
        base_node = self._graph.get_node(base_node_id)
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
                self._portfolio.update(op=op, gain=0.0, valid=False, novel=False, regress=False, cost=0.0)
                continue
            ev = self._evaluate(gen.program, idea=gen.idea, operator=op.name)
            if ev is None or ev.fitness is None:
                self._portfolio.update(op=op, gain=0.0, valid=False, novel=False, regress=False, cost=ev.runtime if ev else 0.0)
                continue
            hint = ctx.hints.get("mechanism_tag_hint") or ctx.hints.get("donor_mechanism")
            tag = infer_mechanism_tag(action, hint=hint)
            delta = directed_delta(base_node.fitness, ev.fitness, self._maximize)
            outcome = classify_outcome(delta, self._value_weights.positive_threshold)
            gen_signal = step_generalization_signal(
                mechanism_tag=tag, delta=delta, pattern_memory=self._pattern_memory, maximize=self._maximize,
            )
            child = self._graph.add_node(
                code=str(gen.program), idea=gen.idea, fitness=ev.fitness, is_valid=True,
                runtime=ev.runtime, complexity=ev.complexity, robustness=ev.robustness,
                mechanism_tag=tag, iteration=iteration, sample_order=self._tot_sample_nums,
            )
            edge = self._graph.add_edge(
                parent_id=base_node_id, child_id=child.id, action=action, operator=op.name,
                mechanism_tag=tag, delta=delta, outcome=outcome, generalization_signal=gen_signal,
                iteration=iteration,
            )
            new_traj = op.insert(ctx, child.id, edge.id, base_node_id)
            novel = self._apply_novelty_gate(new_traj)
            if novel:
                self._score_trajectory(new_traj)
            if base_node.fitness is not None:
                self._ranking.update_by_fitness(
                    a=child.id, b=base_node_id, fitness_a=ev.fitness,
                    fitness_b=base_node.fitness, maximize=self._maximize,
                )
            self._update_best(child)
            self._portfolio.update(
                op=op, gain=delta or 0.0, valid=True, novel=novel,
                regress=(outcome == "regress"), cost=ev.runtime,
            )
            log_event(self, event="child_accepted", status="ok" if novel else "novelty_rejected",
                       iteration=iteration, seq=seq, operator=op.name, parent_id=base_node_id,
                       child_id=child.id, edge_id=edge.id, trajectory_id=new_traj.id, action=action,
                       score=ev.fitness, delta=delta, outcome=outcome, mechanism_tag=tag)

    # ---------------- periodic hooks (蒸馏/反思/migration/survival) ----------------
    def _periodic_hooks(self, iteration: int) -> None:
        self._survive()
        if iteration > 0 and iteration % self._k_distill == 0:
            n = distill(memory=self._memory, graph=self._graph, pattern_memory=self._pattern_memory,
                         maximize=self._maximize, iteration=iteration)
            if n:
                log_event(self, event="distill", status="ok", iteration=iteration, n_patterns=n)
        stagnation = getattr(self, "_stagnation", 0)
        if stagnation > 0 and stagnation % self._patience_reflect == 0:
            contrast = reflect(memory=self._memory, graph=self._graph, pattern_memory=self._pattern_memory,
                                ranking=self._ranking, maximize=self._maximize, iteration=iteration)
            if contrast:
                log_event(self, event="reflect", status="ok", iteration=iteration)
        # 停滞时促进 island 间流动
        if stagnation > 0 and iteration > 0 and iteration % 5 == 0:
            moved = self._islands.migrate(memory=self._memory)
            if moved:
                log_event(self, event="migrate", status="ok", iteration=iteration, moved=moved)

    def _survive(self) -> None:
        # 每个 island 内按 scalar_value 保留 top max_per_island
        for island in self._memory.island_ids():
            members = self._memory.active_in_island(island)
            if len(members) <= self._max_per_island:
                continue
            ranked = sorted(members, key=lambda t: t.scalar_value if t.scalar_value is not None else float("-inf"), reverse=True)
            for t in ranked[self._max_per_island:]:
                self._memory.archive(t.id)
        # 全局上限
        actives = self._memory.active()
        if len(actives) <= self._max_active_trajectories:
            return
        best_id = self._best_node.id if self._best_node else -1
        ranked = sorted(actives, key=lambda t: t.scalar_value if t.scalar_value is not None else float("-inf"), reverse=True)
        for t in ranked[self._max_active_trajectories:]:
            if t.id != best_id:
                self._memory.archive(t.id)

    # ---------------- novelty gate & scoring ----------------
    def _apply_novelty_gate(self, traj: Trajectory) -> bool:
        others = tuple(o for o in self._memory.active() if o.id != traj.id)
        if not others:
            self._score_trajectory(traj)
            return True
        weights = (self._value_weights.w_sim_code, self._value_weights.w_sim_mechanism,
                    self._value_weights.w_sim_trajectory)
        max_sim = max_similarity_to_active(graph=self._graph, candidate=traj, others=others, weights=weights)
        if max_sim >= self._novelty_threshold:
            self._memory.archive(traj.id)
            return False
        self._score_trajectory(traj)
        return True

    def _score_trajectory(self, traj: Trajectory) -> None:
        fmin, fmax = self._graph.fitness_range()
        others = tuple(o for o in self._memory.active() if o.id != traj.id)
        value = compute_value_vec(
            trajectory=traj, graph=self._graph, pattern_memory=self._pattern_memory,
            active_others=others, fmin=fmin, fmax=fmax, maximize=self._maximize, w=self._value_weights,
        )
        self._memory.set_value(traj.id, value, scalarize(value, self._value_weights))

    # ---------------- LLM generation & evaluation ----------------
    def _generate_actions(self, prompt: str, iteration: int) -> list[str]:
        try:
            start = time.time()
            response = self._llm.draw_sample(prompt)
            sample_time = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
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
            reset_sample_failures(self)
        except Exception as exc:
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
        score, eval_time = future.result()
        self._tot_sample_nums += 1
        sample_order = self._tot_sample_nums
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
        complexity = _ast_complexity(str(program))
        return EvalResult(fitness=score, runtime=eval_time, complexity=complexity,
                           robustness=1.0, confidence=1.0)

    # ---------------- bookkeeping ----------------
    def _update_best(self, node: ProgramNode) -> None:
        if not node.is_valid or node.fitness is None:
            return
        if self._best_node is None or _is_better(node.fitness, self._best_node.fitness, self._maximize):
            self._best_node = node
        if self._best_generalization_node is None or node.robustness > self._best_generalization_node.robustness:
            self._best_generalization_node = node

    def _relative_quality_gain(self, fitness: float | None) -> float:
        """novelty 等无 parent-delta 的算子用：endpoint quality 相对 best 的归一化差，
        让这类算子的真实价值进 bandit（而非恒为 0）。"""
        if fitness is None or self._best_node is None or self._best_node.fitness is None:
            return 0.0
        fmin, fmax = self._graph.fitness_range()
        if fmin is None or fmax is None or abs(fmax - fmin) < 1e-12:
            return 0.0
        return normalize_fitness(fitness, fmin, fmax, self._maximize) - normalize_fitness(
            self._best_node.fitness, fmin, fmax, self._maximize)

    def _planned_iterations(self) -> int:
        if self._max_sample_nums is None:
            return 10**9
        remaining = max(self._max_sample_nums - self._tot_sample_nums, 0)
        return max(1, (remaining + self._actions_per_iteration - 1) // self._actions_per_iteration)

    def _has_budget(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _result(self) -> TraceAAD2RunResult:
        nodes = self._graph.nodes()
        return TraceAAD2RunResult(
            best_node=self._best_node,
            best_generalization_node=self._best_generalization_node,
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
