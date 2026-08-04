"""TraceAADV4：以完整算法改进轨迹驱动的单父代语义搜索。"""

from __future__ import annotations

import copy
import hashlib
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
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from ..traceaad_artifacts import TraceAADArtifacts
from .._observability import (
    close_llm,
    init_observability,
    is_search_aborted,
    record_sample_failure,
    reset_sample_failures,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .context import build_action_prompt, trajectory_history
from .derivation_graph import DerivationGraph
from .operators import DEFAULT_OPERATORS, Operator, classify_outcome, directed_delta
from .prompt import build_code_prompt, build_initial_prompt
from .schema import EvalResult, ProgramNode, Trajectory
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    sample_trajectory,
    score_active_trajectories,
    select_diverse_trajectories,
    softmax_scores,
    weighted_sample_without_replacement,
)


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


class TraceAADV4:
    """TraceAAD V4: trajectory-as-history single-parent search.

    ``max_active_trajectories`` is the post-management population size ``M``.
    New children stay active until the pool reaches ``2 * M``, then survival
    management contracts back to ``M``.
    """

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: TraceAADArtifacts | None = None,
        max_sample_nums: Optional[int] = 100,
        *,
        n_init: int = 30,
        actions_per_iteration: int = 2,
        max_trajectory_length: int = 8,
        max_active_trajectories: int = 30,
        softmax_temperature: float = 0.2,
        maximize: bool = True,
        value_weights: ValueWeights | None = None,
        operators: tuple[Operator, ...] = DEFAULT_OPERATORS,
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
        if softmax_temperature <= 0:
            raise ValueError("softmax_temperature must be positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if not operators:
            raise ValueError("at least one operator is required")

        self._llm = llm
        self._evaluation = evaluation
        self._artifacts = profiler
        self._profiler = profiler
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = n_init
        self._actions_per_iteration = actions_per_iteration
        self._max_active_trajectories = int(max_active_trajectories)
        self._management_threshold = 2 * self._max_active_trajectories
        self._elite_count = max(2, math.ceil(0.1 * self._max_active_trajectories))
        self._diversity_count = max(2, math.ceil(0.1 * self._max_active_trajectories))
        self._softmax_temperature = float(softmax_temperature)
        self._maximize = maximize
        self._value_weights = value_weights or ValueWeights()
        self._operators = operators
        self._debug_mode = debug_mode
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_sample = -1
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAADV4 requires an evaluation template with exactly one evolvable function."
            )
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._best_node: ProgramNode | None = None
        self._best_trajectory_id: int | None = None
        self._best_node_sample_order: int | None = None
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
            self._record_decision(
                "checkpoint_loaded",
                checkpoint=str(checkpoint),
                sample_order=self._tot_sample_nums,
                next_attempt_id=self._next_attempt_id,
            )

    def run(self) -> TraceAADRunResult:
        stalled_attempts = 0
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
                    self._record_decision(
                        "search_stopped", status="no_active_trajectory"
                    )
                    self._log_progress("search_stopped no_active_trajectory")
                    break
                samples_before = self._tot_sample_nums
                self._run_iteration(attempt_id)
                if self._tot_sample_nums == samples_before:
                    stalled_attempts += 1
                    if stalled_attempts >= self._max_stalled_iterations:
                        self._record_decision(
                            "search_stopped",
                            status="stalled_generation",
                            attempt_id=attempt_id,
                        )
                        self._log_progress(
                            f"search_stopped stalled_generation attempt={attempt_id}"
                        )
                        break
                else:
                    stalled_attempts = 0
                self._maybe_manage_population()
                attempt_id += 1
                self._next_attempt_id = attempt_id
                self._save_checkpoint_if_due()

            result = self._result()
            status = "aborted" if is_search_aborted(self) else "finished"
            return result
        except Exception as exc:
            error = {
                "error_type": type(exc).__name__,
                "error": str(exc)[:1000],
            }
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
                    best_node_id=(
                        None if result.best_node is None else result.best_node.id
                    ),
                    best_score=(
                        None if result.best_node is None else result.best_node.fitness
                    ),
                    best_sample_order=self._best_node_sample_order,
                    method_sample_count=self._tot_sample_nums,
                    n_total_nodes=result.n_total_nodes,
                    n_valid_nodes=result.n_valid_nodes,
                    n_edges=result.n_edges,
                    n_trajectories=result.n_trajectories,
                    search_aborted=is_search_aborted(self),
                    **error,
                )
                self._artifacts.finish()
            close_llm(self._llm)

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._tot_sample_nums - self._last_checkpoint_sample
            >= self._checkpoint_interval
        ):
            save_checkpoint(self)

    def _initialize(self) -> None:
        stalled_draws = 0
        draw_seq = 0
        while (
            self._tot_sample_nums < self._n_init
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=self._init_diversity_hint(self._tot_sample_nums),
            )
            generated = self._draw_program(
                prompt, stage="init", iteration=None, seq=draw_seq, operator="init"
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
            trajectory = self._memory.create_initial(node_id=node.id)
            self._update_best(node, trajectory_id=trajectory.id, operator="init")
            self._record_decision(
                "trajectory_created",
                stage="init",
                node_id=node.id,
                trajectory_id=trajectory.id,
                sample_order=self._tot_sample_nums,
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
        listed = "; ".join(f"'{idea[:100]}'" for idea in ideas)
        return f"Use a clearly different algorithmic idea from: {listed}."

    def _run_iteration(self, attempt_id: int) -> None:
        selected = sample_trajectory(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
        )
        operator = random.choice(self._operators)
        anchor_id = self._select_anchor(selected)
        self._record_decision(
            "operator_selected",
            attempt_id=attempt_id,
            operator=operator.name,
            trajectory_id=selected.id,
            anchor_id=anchor_id,
        )
        prompt = build_action_prompt(
            graph=self._graph,
            trajectory=selected,
            base_node_id=anchor_id,
            base_reason="endpoint_or_best",
            operator_name=operator.name,
            operator_constraint=operator.constraint,
            task_description=self._task_description_str,
            template_function=self._function_to_evolve,
            action_count=self._actions_per_iteration,
            maximize=self._maximize,
            max_steps=self._memory.max_trajectory_length,
        )
        actions = self._generate_actions(prompt, attempt_id)
        for seq, action in enumerate(actions):
            if not self._has_budget() or is_search_aborted(self):
                break
            base_node = self._graph.get_node(anchor_id)
            code_prompt = build_code_prompt(
                current_node=base_node,
                action=action,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                history=trajectory_history(
                    self._graph,
                    selected,
                    base_node_id=anchor_id,
                    max_steps=self._memory.max_trajectory_length,
                ),
                operator_constraint=operator.constraint,
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
            evaluated = self._evaluate(
                generated.program, idea=generated.idea, operator=operator.name
            )
            if evaluated is None or evaluated.fitness is None:
                continue
            child = self._add_node(generated, evaluated)
            delta = directed_delta(base_node.fitness, child.fitness, self._maximize)
            outcome = classify_outcome(delta, self._value_weights.positive_threshold)
            edge = self._graph.add_edge(
                parent_id=anchor_id,
                child_id=child.id,
                action=action,
                operator=operator.name,
                delta=delta,
                outcome=outcome,
                iteration=attempt_id,
            )
            child_trajectory = self._memory.branch_from(
                trajectory_id=selected.id,
                base_node_id=anchor_id,
                child_id=child.id,
                edge_id=edge.id,
            )
            if self._artifacts is not None:
                self._artifacts.record_edge(
                    edge_id=edge.id,
                    parent_id=anchor_id,
                    child_id=child.id,
                    sample_order=self._tot_sample_nums,
                    iteration=attempt_id,
                    seq=seq,
                    operator=operator.name,
                    action=action,
                    trajectory_id=child_trajectory.id,
                )
            self._update_best(
                child,
                trajectory_id=child_trajectory.id,
                iteration=attempt_id,
                operator=operator.name,
            )
        self._memory.record_visit(selected.id)

    def _select_anchor(self, trajectory: Trajectory) -> int:
        endpoint = trajectory.endpoint_id
        best = endpoint
        best_fitness = self._graph.get_node(endpoint).fitness
        for node_id in trajectory.node_ids:
            fitness = self._graph.get_node(node_id).fitness
            if fitness is None:
                continue
            if best_fitness is None or _is_better(
                fitness, best_fitness, self._maximize
            ):
                best, best_fitness = node_id, fitness
        return random.choice((endpoint, best))

    def _add_node(
        self, generated: _GeneratedProgram, evaluated: EvalResult
    ) -> ProgramNode:
        return self._graph.add_node(
            code=str(generated.program), idea=generated.idea, fitness=evaluated.fitness
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
        active = self._memory.active()
        if len(active) < self._management_threshold:
            return
        ranked = list(self._score_active_pool())
        if len(ranked) <= self._max_active_trajectories:
            return
        elite_count = min(self._elite_count, self._max_active_trajectories, len(ranked))
        elites = ranked[:elite_count]
        if self._best_trajectory_id is not None:
            best_route = next(
                (
                    trajectory
                    for trajectory in ranked
                    if trajectory.id == self._best_trajectory_id
                ),
                None,
            )
            if best_route is not None and best_route.id not in {t.id for t in elites}:
                elites = [best_route, *elites[: max(0, elite_count - 1)]]
        diversity_count = min(
            self._diversity_count,
            max(0, self._max_active_trajectories - elite_count),
            max(0, len(ranked) - elite_count),
        )
        diverse = select_diverse_trajectories(
            candidates=tuple(
                trajectory
                for trajectory in ranked
                if trajectory.id not in {t.id for t in elites}
            ),
            graph=self._graph,
            count=diversity_count,
            reference=tuple(elites),
        )
        diverse_ids = {trajectory.id for trajectory in diverse}
        remaining = [
            trajectory
            for trajectory in ranked
            if trajectory.id not in {t.id for t in elites}
            and trajectory.id not in diverse_ids
        ]
        sample_count = self._max_active_trajectories - len(elites) - diversity_count
        scores = [float(trajectory.scalar_value or 0.0) for trajectory in remaining]
        sampled = weighted_sample_without_replacement(
            remaining,
            softmax_scores(scores, self._softmax_temperature),
            sample_count,
        )
        keep_ids = {
            trajectory.id for trajectory in elites + list(diverse) + list(sampled)
        }
        archived = 0
        for trajectory in ranked:
            if trajectory.id not in keep_ids:
                self._memory.archive(trajectory.id)
                archived += 1
        self._record_decision(
            "population_managed",
            management_threshold=self._management_threshold,
            before=len(ranked),
            after=len(self._memory.active()),
            elite_count=elite_count,
            diversity_count=diversity_count,
            archived=archived,
        )

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
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage="action",
                operator="semantic",
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=0,
                sample_time=sample_time,
                n_actions=len(actions),
                response=response,
                status="ok" if actions else "parse_failed",
            )
        return actions

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
        generated = _parse_program_response(
            response, self._function_to_evolve.name
        )
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage=stage,
                operator=operator,
                sample_order=sample_order,
                iteration=iteration,
                seq=seq,
                sample_time=sample_time,
                program_parse_success=generated is not None,
                response=response,
                status="ok" if generated is not None else "parse_failed",
            )
        return generated

    def _evaluate(
        self, program: Program, *, idea: str, operator: str
    ) -> EvalResult | None:
        if not self._has_budget():
            return None
        result, eval_time = self._evaluator.evaluate_program_record_time(program)
        self._tot_sample_nums += 1
        score = result.fitness if isinstance(result, EvalResult) else result
        program_text = str(program)
        if self._artifacts is not None:
            self._artifacts.record_candidate(
                sample_order=self._tot_sample_nums,
                score=None if score is None else float(score),
                operator=operator,
                program=program_text,
                idea=idea,
                code_hash=_code_hash(program_text),
                program_loc=_nonempty_loc(program_text),
                evaluate_time=eval_time,
                status="ok" if score is not None else "eval_failed",
            )
        return None if score is None else EvalResult(fitness=float(score))

    def _update_best(
        self,
        node: ProgramNode,
        *,
        trajectory_id: int | None = None,
        iteration: int | None = None,
        operator: str,
    ) -> None:
        if node.fitness is None:
            return
        if self._best_node is not None and not _is_better(
            node.fitness, self._best_node.fitness, self._maximize
        ):
            return
        previous = self._best_node
        self._best_node = node
        self._best_node_sample_order = self._tot_sample_nums
        if trajectory_id is not None:
            self._best_trajectory_id = trajectory_id
        self._record_decision(
            "best_updated",
            sample_order=self._tot_sample_nums,
            node_id=node.id,
            iteration=iteration,
            operator=operator,
            previous_best_node_id=None if previous is None else previous.id,
        )

    def _record_decision(self, event: str, **payload) -> None:
        if self._artifacts is not None:
            self._artifacts.record_decision(event, **payload)

    def _log_progress(self, message: str) -> None:
        if self._artifacts is not None:
            self._artifacts.log_progress(message)

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


def _code_hash(code: str) -> str:
    text = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _nonempty_loc(code: str) -> int:
    return sum(bool(line.strip()) for line in code.splitlines())


def _is_better(candidate: float, incumbent: float, maximize: bool) -> bool:
    return candidate > incumbent if maximize else candidate < incumbent


def _parse_program_response(
    response: str, function_name: str
) -> _GeneratedProgram | None:
    idea = _extract_idea(response)
    code = _extract_first_code_block(response)
    if idea is None or code is None:
        return None
    parsed = TextFunctionProgramConverter.text_to_program(code)
    if parsed is None or len(parsed.functions) != 1:
        return None
    if parsed.functions[0].name != function_name:
        return None
    return _GeneratedProgram(idea=idea, program=parsed)


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
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>.+?)\s*$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else match.group("idea").strip()


def _extract_first_code_block(response: str) -> str | None:
    match = re.search(
        r"```(?:python|py)?\s*(?P<code>.*?)```",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return None if match is None else match.group("code").strip()


__all__ = ["TraceAADV4", "TraceAADRunResult"]
