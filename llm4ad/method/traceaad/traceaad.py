from __future__ import annotations

import concurrent.futures
import copy
import re
import time
import traceback
from dataclasses import dataclass
from typing import Literal, Optional

from .derivation_graph import DerivationGraph
from .prompt import TraceAADPrompt
from .schema import ImprovementEdge, ProgramNode, Trajectory, TrajectoryId
from .trajectory_branching import BaseNodeSelection, select_base_node
from .trajectory_library import TrajectoryLibrary
from .trajectory_sampler import (
    archive_low_score_trajectories,
    sample_best_node,
    sample_random_trajectory,
    sample_trajectory,
)
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
from ...base import Evaluation, Function, LLM, Program, SampleTrimmer, SecureEvaluator, TextFunctionProgramConverter
from ...tools.profiler import ProfilerBase


@dataclass(frozen=True, slots=True)
class _GeneratedProgram:
    idea: str
    program: Program


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


_INITIALIZATION_HINTS = {
    "constructive": "Design a direct constructive algorithm that builds a feasible solution step by step.",
    "local-search": "Design a program with a simple initial solution followed by local improvement moves.",
    "repair-based": "Design a program that first builds a fast approximate solution and then repairs weak parts.",
    "greedy-scoring": "Design a greedy program with a clear scoring rule for choosing the next decision.",
    "hybrid-simple": "Combine two or three simple algorithmic components while keeping the function easy to evaluate.",
}


class TraceAAD:
    def __init__(
            self,
            llm: LLM,
            evaluation: Evaluation,
            profiler: ProfilerBase = None,
            max_sample_nums: Optional[int] = 100,
            *,
            n_init: int = 4,
            n_iterations: Optional[int] = None,
            actions_per_iteration: int = 2,
            max_actions_in_prompt: int = 5,
            max_trajectory_length: int = 8,
            max_active_trajectories: int = 1000,
            sampling_strategy: Literal["trajectory_ucb", "best_node", "random"] = "trajectory_ucb",
            top_k: int = 5,
            temperature: float = 0.8,
            w_end: float = 0.45,
            w_path: float = 0.55,
            w_consistency: float = 0.25,
            w_downside: float = 0.5,
            discount: float = 0.8,
            positive_threshold: float = 1e-6,
            c0: float = 0.4,
            maximize: bool = True,
            initialization_strategies: tuple[str, ...] | list[str] | None = None,
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
        if max_active_trajectories <= 0:
            raise ValueError("max_active_trajectories must be positive")

        self._llm = llm
        self._evaluation = evaluation
        self._profiler = profiler
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = n_init
        self._n_iterations = n_iterations
        self._actions_per_iteration = actions_per_iteration
        self._max_actions_in_prompt = max_actions_in_prompt
        self._max_active_trajectories = max_active_trajectories
        self._sampling_strategy = sampling_strategy
        self._top_k = top_k
        self._temperature = temperature
        self._w_end = w_end
        self._w_path = w_path
        self._w_consistency = w_consistency
        self._w_downside = w_downside
        self._discount = discount
        self._positive_threshold = positive_threshold
        self._c0 = c0
        self._maximize = maximize
        self._initialization_strategies = tuple(initialization_strategies or _INITIALIZATION_HINTS.keys())
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval
        llm.debug_mode = debug_mode

        self._template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        if self._template_program is None or len(self._template_program.functions) != 1:
            raise ValueError("TraceAAD requires an evaluation template with exactly one evolvable function.")
        self._function_to_evolve: Function = copy.deepcopy(self._template_program.functions[0])
        self._function_to_evolve_name = self._function_to_evolve.name

        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self.graph = DerivationGraph()
        self.library = TrajectoryLibrary(max_trajectory_length=max_trajectory_length)
        self._best_node: ProgramNode | None = None
        self._tot_sample_nums = 0
        init_observability(self, max_consecutive_sample_failures)

        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_evaluators)

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def run(self) -> TraceAADRunResult:
        try:
            if not self._resume_mode:
                self._initialize()

            max_iterations = self._planned_iterations()
            for iteration in range(max_iterations):
                if not self._has_budget() or is_search_aborted(self):
                    break
                if len(self.library.active()) == 0:
                    log_event(self, event="search_stopped", status="no_active_trajectory")
                    break
                self._run_iteration(iteration, max_iterations)
                archived = archive_low_score_trajectories(
                    library=self.library,
                    max_active=self._max_active_trajectories,
                )
                if archived:
                    log_event(self, event="archive", status="ok", archived_count=archived)

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
            hint = self._initialization_hint(seq)
            prompt = TraceAADPrompt.build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=hint,
            )
            generated = self._draw_program(prompt, stage="init", iteration=None, seq=seq, operator="init")
            if generated is None:
                continue
            node = self._evaluate_and_add_initial(generated, seq=seq)
            if node is not None and node.is_valid:
                trajectory = self.library.create_initial(node_id=node.id)
                log_event(
                    self,
                    event="trajectory_created",
                    status="ok",
                    stage="init",
                    node_id=node.id,
                    trajectory_id=trajectory.id,
                )

    def _run_iteration(self, iteration: int, max_iterations: int) -> None:
        selected = self._sample_trajectory(iteration, max_iterations)
        selected = self.library.get_trajectory(selected.id)
        base_selection = self._select_base_node(selected)
        base_node = self.graph.get_node(base_selection.node_id)

        log_state(
            self,
            phase="iteration_start",
            iteration=iteration,
            selected_trajectory_id=selected.id,
            selected_trajectory_nodes=list(selected.node_ids),
            selected_trajectory_score=selected.score,
            selected_trajectory_visits=selected.visit_count,
            selected_endpoint_id=selected.endpoint_id,
            selected_base_node_id=base_node.id,
            selected_base_node_fitness=base_node.fitness,
            base_selection_reason=base_selection.reason,
            n_active_trajectories=len(self.library.active()),
        )

        actions = self._generate_actions(selected, base_selection, iteration)
        children: list[dict] = []
        for seq, action in enumerate(actions):
            if not self._has_budget() or is_search_aborted(self):
                break
            result = self._generate_and_evaluate_code(
                parent_node=base_node,
                action=action,
                parent_trajectory_id=selected.id,
                iteration=iteration,
                seq=seq,
            )
            children.append(self._child_event_payload(result, base_node.id, action))

        selected = self.library.record_visit(selected.id)
        log_event(
            self,
            event="iteration_end",
            status="ok",
            iteration=iteration,
            selected_trajectory_id=selected.id,
            selected_trajectory_visits=selected.visit_count,
            selected_base_node_id=base_node.id,
            base_selection_reason=base_selection.reason,
            actions=actions,
            children=children,
            n_active_trajectories=len(self.library.active()),
        )

    def _generate_actions(
            self,
            trajectory: Trajectory,
            base_selection: BaseNodeSelection,
            iteration: int,
    ) -> list[str]:
        prompt = TraceAADPrompt.build_action_prompt(
            graph=self.graph,
            trajectory=trajectory,
            task_description=self._task_description_str,
            max_actions=self._max_actions_in_prompt,
            action_count=self._actions_per_iteration,
            maximize=self._maximize,
            base_node_id=base_selection.node_id,
            base_selection_reason=base_selection.reason,
        )
        try:
            start = time.time()
            response = self._llm.draw_sample(prompt)
            sample_time = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="action",
                operator="action",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                counts_budget=False,
                iteration=iteration,
            )
            return []

        actions = self._parse_actions(response, expected_count=self._actions_per_iteration)
        log_llm_call(
            self,
            stage="action",
            operator="action",
            sample_order=self._tot_sample_nums + 1,
            iteration=iteration,
            seq=0,
            prompt=prompt,
            response=response,
            sample_time=sample_time,
            parsed_actions=actions,
            status="ok",
        )
        return actions

    def _generate_and_evaluate_code(
            self,
            parent_node: ProgramNode,
            action: str,
            parent_trajectory_id: TrajectoryId,
            iteration: int,
            seq: int,
    ) -> tuple[ProgramNode, ImprovementEdge, Trajectory] | None:
        prompt = TraceAADPrompt.build_code_prompt(
            current_node=parent_node,
            action=action,
            task_description=self._task_description_str,
            template_function=self._function_to_evolve,
        )
        generated = self._draw_program(prompt, stage="code", iteration=iteration, seq=seq, operator="code", action=action)
        if generated is None:
            return None

        score, eval_time, sample_order = self._evaluate_program(
            generated.program,
            idea=generated.idea,
            operator="code",
        )
        if score is None:
            log_event(
                self,
                event="child_rejected",
                status="eval_failed",
                iteration=iteration,
                seq=seq,
                parent_id=parent_node.id,
                trajectory_id=parent_trajectory_id,
                action=action,
                sample_order=sample_order,
                evaluate_time=eval_time,
            )
            return None

        child_node = self.graph.add_node(
            code=str(generated.program),
            idea=generated.idea,
            fitness=score,
            is_valid=True,
            iteration=iteration,
            sample_order=sample_order,
        )
        child_edge = self.graph.add_edge(
            parent_id=parent_node.id,
            child_id=child_node.id,
            action=action,
            iteration=iteration,
        )
        new_trajectory = self.library.branch_from(
            trajectory_id=parent_trajectory_id,
            base_node_id=parent_node.id,
            edge=child_edge,
        )
        self._update_best(child_node)
        log_event(
            self,
            event="child_accepted",
            status="ok",
            iteration=iteration,
            seq=seq,
            parent_id=parent_node.id,
            child_id=child_node.id,
            edge_id=child_edge.id,
            trajectory_id=new_trajectory.id,
            action=action,
            score=score,
            sample_order=sample_order,
            evaluate_time=eval_time,
        )
        return child_node, child_edge, new_trajectory

    def _evaluate_and_add_initial(self, generated: _GeneratedProgram, *, seq: int) -> ProgramNode | None:
        score, eval_time, sample_order = self._evaluate_program(generated.program, idea=generated.idea, operator="init")
        node = self.graph.add_node(
            code=str(generated.program),
            idea=generated.idea,
            fitness=score,
            is_valid=score is not None,
            iteration=None,
            sample_order=sample_order,
        )
        if node.is_valid:
            self._update_best(node)
        log_event(
            self,
            event="initial_program",
            status="ok" if node.is_valid else "eval_failed",
            seq=seq,
            node_id=node.id,
            score=score,
            sample_order=sample_order,
            evaluate_time=eval_time,
        )
        return node

    def _draw_program(
            self,
            prompt: str,
            *,
            stage: str,
            iteration: int | None,
            seq: int,
            operator: str,
            action: str | None = None,
    ) -> _GeneratedProgram | None:
        sample_order = self._tot_sample_nums + 1
        try:
            start = time.time()
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

        generated = self._parse_program_response(response)
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
        if generated is None:
            log_event(
                self,
                event="sample_rejected",
                status="program_parse_failed",
                stage=stage,
                operator=operator,
                sample_order=sample_order,
                iteration=iteration,
                seq=seq,
                counts_budget=False,
            )
        return generated

    def _evaluate_program(
            self,
            program: Program,
            *,
            idea: str,
            operator: str,
    ) -> tuple[float | None, float, int]:
        if not self._has_budget():
            return None, 0.0, self._tot_sample_nums
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

        log_event(
            self,
            event="program_evaluated",
            status="ok" if score is not None else "eval_failed",
            operator=operator,
            sample_order=sample_order,
            score=score,
            evaluate_time=eval_time,
            counts_budget=True,
        )
        return score, eval_time, sample_order

    def _parse_program_response(self, response: str) -> _GeneratedProgram | None:
        idea = _extract_idea(response) or _extract_boxed_text(response) or "Generated program"
        code = _extract_first_code_block(response) or response
        program = self._code_to_program(code)
        if program is None:
            return None
        return _GeneratedProgram(idea=idea, program=program)

    def _code_to_program(self, code: str) -> Program | None:
        code = code.strip()
        if not code:
            return None

        parsed = TextFunctionProgramConverter.text_to_program(code)
        if parsed is not None and len(parsed.functions) == 1:
            function = parsed.functions[0]
            if function.name == self._function_to_evolve_name:
                return parsed
            return TextFunctionProgramConverter.function_to_program(function, self._template_program)

        return SampleTrimmer.sample_to_program(code, self._template_program)

    def _parse_actions(self, response: str, *, expected_count: int) -> list[str]:
        actions: list[str] = []
        for line in response.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(
                r"^(?:[-*]\s*)?(?:\d+[\.\)]\s*)?(?:Action\s*\d*\s*:\s*)?(?P<action>.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if match is None:
                continue
            action = match.group("action").strip()
            if action and not action.startswith(("#", "`")):
                actions.append(action)
        return actions[:expected_count]

    def _sample_trajectory(self, iteration: int, max_iterations: int) -> Trajectory:
        if self._sampling_strategy == "trajectory_ucb":
            return sample_trajectory(
                library=self.library,
                graph=self.graph,
                iteration=iteration,
                max_iterations=max_iterations,
                top_k=self._top_k,
                temperature=self._temperature,
                w_end=self._w_end,
                w_path=self._w_path,
                w_consistency=self._w_consistency,
                w_downside=self._w_downside,
                discount=self._discount,
                positive_threshold=self._positive_threshold,
                c0=self._c0,
                maximize=self._maximize,
            )
        if self._sampling_strategy == "best_node":
            return sample_best_node(library=self.library, graph=self.graph, maximize=self._maximize)
        if self._sampling_strategy == "random":
            return sample_random_trajectory(library=self.library)
        raise ValueError(f"Unknown sampling strategy: {self._sampling_strategy}")

    def _select_base_node(self, trajectory: Trajectory) -> BaseNodeSelection:
        return select_base_node(
            graph=self.graph,
            trajectory=trajectory,
            maximize=self._maximize,
            positive_threshold=self._positive_threshold,
        )

    def _initialization_hint(self, seq: int) -> str:
        if not self._initialization_strategies:
            return _INITIALIZATION_HINTS["constructive"]
        strategy = self._initialization_strategies[seq % len(self._initialization_strategies)]
        return _INITIALIZATION_HINTS.get(strategy, f"Explore a {strategy} program design.")

    def _planned_iterations(self) -> int:
        if self._n_iterations is not None:
            return max(self._n_iterations, 0)
        if self._max_sample_nums is None:
            return 100
        remaining = max(self._max_sample_nums - self._tot_sample_nums, 0)
        return max(1, (remaining + self._actions_per_iteration - 1) // self._actions_per_iteration)

    def _has_budget(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _update_best(self, node: ProgramNode) -> None:
        if not node.is_valid or node.fitness is None:
            return
        if self._best_node is None:
            self._best_node = node
            return
        if self._is_better(node.fitness, self._best_node.fitness):
            self._best_node = node

    def _is_better(self, candidate: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        return candidate > incumbent if self._maximize else candidate < incumbent

    def _result(self) -> TraceAADRunResult:
        nodes = self.graph.nodes()
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=len(nodes),
            n_valid_nodes=sum(1 for node in nodes if node.is_valid),
            n_trajectories=len(self.library.trajectories()),
            n_edges=len(self.graph.edges()),
            n_samples=self._tot_sample_nums,
        )

    @staticmethod
    def _child_event_payload(
            result: tuple[ProgramNode, ImprovementEdge, Trajectory] | None,
            parent_id: int,
            action: str,
    ) -> dict:
        if result is None:
            return {
                "parent_id": parent_id,
                "action": action,
                "valid": False,
                "node_id": None,
                "trajectory_id": None,
            }
        child_node, _edge, trajectory = result
        return {
            "parent_id": parent_id,
            "action": action,
            "valid": child_node.is_valid,
            "node_id": child_node.id,
            "trajectory_id": trajectory.id,
            "fitness": child_node.fitness,
        }


def _extract_idea(response: str) -> str | None:
    match = re.search(r"^\s*Idea\s*:\s*(?P<idea>.+?)\s*$", response, flags=re.IGNORECASE | re.MULTILINE)
    if match is None:
        return None
    return match.group("idea").strip()


def _extract_boxed_text(response: str) -> str | None:
    match = re.search(r"boxed\s*\{(?P<idea>[^{}]+)\}", response, flags=re.IGNORECASE)
    if match is None:
        match = re.search(r"\\boxed\s*\{(?P<idea>[^{}]+)\}", response, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.group("idea").strip()


def _extract_first_code_block(response: str) -> str | None:
    match = re.search(r"```(?:python|py)?\s*(?P<code>.*?)```", response, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group("code").strip()
