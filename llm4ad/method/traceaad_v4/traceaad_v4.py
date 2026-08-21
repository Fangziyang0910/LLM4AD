"""TraceAADV4：以完整算法改进轨迹驱动的单父代语义搜索。"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ...base import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from .artifacts import RunArtifacts
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

ERROR_MAX_CHARS = 360


def _one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3] + "..."


@dataclass(frozen=True, slots=True)
class _GeneratedProgram:
    idea: str
    code: str


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
        artifacts: RunArtifacts | None = None,
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
        max_stalled_iterations: int = 20,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        resume_from: str | Path | None = None,
    ) -> None:
        if (
            n_init < 0
            or actions_per_iteration <= 0
            or max_active_trajectories <= 0
            or softmax_temperature <= 0
            or checkpoint_interval <= 0
            or not operators
        ):
            raise ValueError("TraceAAD V4 search parameters are out of range")

        self._llm = llm
        self._evaluation = evaluation
        self._artifacts = artifacts
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
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_sample = -1

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAADV4 requires an evaluation template with exactly one evolvable function."
            )
        self._template_program = template
        self._function_to_evolve: Function = template.functions[0]
        self._evaluator = SecureEvaluator(evaluation)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._best_node: ProgramNode | None = None
        self._best_trajectory_id: int | None = None
        self._best_node_sample_order: int | None = None
        self._tot_sample_nums = 0
        self._next_attempt_id = 0
        self._initialization_complete = False
        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

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
            while self._has_budget():
                if not self._memory.active():
                    break
                samples_before = self._tot_sample_nums
                self._run_iteration(attempt_id)
                if self._tot_sample_nums == samples_before:
                    stalled_attempts += 1
                    if stalled_attempts >= self._max_stalled_iterations:
                        break
                else:
                    stalled_attempts = 0
                self._maybe_manage_population()
                attempt_id += 1
                self._next_attempt_id = attempt_id
                self._save_checkpoint_if_due()

            result = self._result()
            status = "finished"
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
                    num_samples=self._tot_sample_nums,
                    n_total_nodes=result.n_total_nodes,
                    n_valid_nodes=result.n_valid_nodes,
                    n_edges=result.n_edges,
                    n_trajectories=result.n_trajectories,
                    **error,
                )
            self._llm.close()

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._tot_sample_nums - self._last_checkpoint_sample
            >= self._checkpoint_interval
        ):
            save_checkpoint(self)

    def _initialize(self) -> None:
        draw_seq = 0
        while self._tot_sample_nums < self._n_init and self._has_budget():
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=self._init_diversity_hint(self._tot_sample_nums),
            )
            generated = self._draw_program(
                prompt, stage="init", iteration=None, seq=draw_seq, operator="init"
            )
            draw_seq += 1
            evaluated = self._evaluate(
                generated.code, idea=generated.idea, operator="init"
            )
            if evaluated is None:
                continue
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
            if not self._has_budget():
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
            )
            evaluated = self._evaluate(
                generated.code, idea=generated.idea, operator=operator.name
            )
            if evaluated is None:
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
            code=generated.code, idea=generated.idea, fitness=evaluated.fitness
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
        response = self._llm.draw_sample(prompt)
        actions = _parse_actions(response, expected_count=self._actions_per_iteration)
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage="action",
                operator="semantic",
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=0,
                n_actions=len(actions),
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
    ) -> _GeneratedProgram:
        response = self._llm.draw_sample(prompt)
        parsed = _parse_program_response(response)
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage=stage,
                operator=operator,
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=seq,
                status="ok",
            )
        return parsed

    def _evaluate(
        self, code: str, *, idea: str, operator: str
    ) -> EvalResult | None:
        if not self._has_budget():
            return None
        outcome = self._evaluator.evaluate_program_with_details(code)
        self._tot_sample_nums += 1
        if outcome.failure_kind == "prepare_error":
            raise RuntimeError(
                _one_line(
                    outcome.error
                    or "evaluator preparation failed without an error message",
                    ERROR_MAX_CHARS,
                )
            )
        score = getattr(outcome.result, "fitness", outcome.result)
        try:
            fitness = float(score)
        except (TypeError, ValueError, OverflowError):
            fitness = math.nan
        if not math.isfinite(fitness):
            fitness = None
        if self._artifacts is not None:
            payload = {
                "sample_order": self._tot_sample_nums,
                "score": fitness,
                "operator": operator,
                "program": code,
                "idea": idea,
                "code_hash": _code_hash(code),
                "program_loc": _nonempty_loc(code),
                "status": (
                    "ok"
                    if fitness is not None
                    else (outcome.failure_kind or "invalid_result")
                ),
            }
            if fitness is None:
                payload.update(
                    {
                        "error_type": outcome.error_type,
                        "error": outcome.error,
                    }
                )
            self._artifacts.record_candidate(**payload)
        return None if fitness is None else EvalResult(fitness=fitness)

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


def _parse_program_response(response: str) -> _GeneratedProgram:
    """Lenient extraction: last fenced block, else text after Code:, else the response."""
    text = str(response)
    first_fence = text.find("```")
    blocks = tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )
    if blocks:
        idea = _extract_idea(text[:first_fence])
        return _GeneratedProgram(idea=idea or "", code=blocks[-1])

    code_marker = re.search(r"^\s*Code\s*:\s*", text, re.IGNORECASE | re.MULTILINE)
    if code_marker is not None:
        idea = _extract_idea(text[: code_marker.start()])
        return _GeneratedProgram(idea=idea or "", code=text[code_marker.end() :].strip())
    idea = _extract_idea(text)
    return _GeneratedProgram(idea=idea or "", code=text.strip())


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
        r"^\s*Idea\s*:\s*(?P<idea>\S[^\r\n]*)$",
        response,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else " ".join(match.group("idea").split())


__all__ = ["TraceAADV4", "TraceAADRunResult"]
