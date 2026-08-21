"""Independent implementation of the complete TraceAAD v5 mechanism."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from ...base import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .complexity import code_change_ratio, code_hash, nonempty_loc
from .context import build_action_prompt, trajectory_history
from .derivation_graph import DerivationGraph
from .operators import (
    DEFAULT_OPERATORS,
    Operator,
    classify_outcome,
)
from .prompt import (
    build_code_prompt,
    build_initial_prompt,
    parse_actions,
    parse_program_response,
)
from .schema import OperatorName, ProgramNode, Trajectory
from .trajectory_memory import TrajectoryMemory
from .value import (
    ValueWeights,
    compact_best_node,
    directed_delta,
    is_program_better,
    reference_sampling_distribution,
    score_active_trajectories,
    select_diverse_trajectories,
    trajectory_sampling_distribution,
    weighted_choice,
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
class _PromptContext:
    prompt: str
    token_count: int
    primary_history: str
    reference_history: str


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_trajectories: int
    n_edges: int
    n_samples: int


class TraceAADV5:
    """Trajectory search with local histories and direct trajectory references."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
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
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
        action_max_tokens: int = 1024,
        random_seed: int | None = None,
        max_stalled_iterations: int = 20,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        resume_from: str | Path | None = None,
    ) -> None:
        if (
            n_init < 0
            or actions_per_iteration <= 0
            or max_active_trajectories <= 0
            or max_trajectory_length < 1
            or softmax_temperature <= 0
            or checkpoint_interval <= 0
            or action_max_tokens <= 0
            or not operators
        ):
            raise ValueError("TraceAAD V5 search parameters are out of range")
        self._llm = llm
        self._evaluation = evaluation
        self._artifacts = artifacts
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
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_sample = -1
        self._rng = random.Random(random_seed)

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD v5 requires exactly one evolvable template function."
            )
        self._template_program = template
        self._function_to_evolve: Function = template.functions[0]
        self._evaluator = SecureEvaluator(evaluation)

        self._graph = DerivationGraph()
        self._memory = TrajectoryMemory(max_trajectory_length=max_trajectory_length)
        self._operators = tuple(operator_type() for operator_type in operators)
        self._action_max_tokens = int(action_max_tokens)
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
                # Only mark init done when the target population exists.
                # Abort/timeout during init must keep this False so resume can retry.
                if len(self._memory.trajectories()) >= self._n_init:
                    self._initialization_complete = True
                save_checkpoint(self)
            while self._has_budget():
                if not self._memory.active():
                    break
                before = self._tot_sample_nums
                self._run_iteration(attempt_id)
                stop_for_stall = False
                if self._tot_sample_nums == before:
                    stalled_attempts += 1
                    if stalled_attempts >= self._max_stalled_iterations:
                        stop_for_stall = True
                else:
                    stalled_attempts = 0
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

    def _initialize(self) -> None:
        draw_seq = 0
        while (
            len(self._memory.trajectories()) < self._n_init
            and self._has_budget()
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
            evaluated = self._evaluate(
                generated.code,
                idea=generated.idea,
                operator="init",
            )
            if evaluated is None:
                continue
            node = self._graph.add_node(
                code=generated.code,
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
            self._record_decision(
                "trajectory_created",
                stage="init",
                node_id=node.id,
                trajectory_id=route.id,
                sample_order=self._tot_sample_nums,
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
        selected = self._select_trajectory(attempt_id)
        anchor_id, anchor_role = self._select_anchor(selected)
        reference_candidates = tuple(
            route for route in self._memory.active() if route.id != selected.id
        )
        eligible = [
            operator
            for operator in self._operators
            if operator.name not in (OperatorName.SYNTHESIZE, OperatorName.TRANSFER)
            or reference_candidates
        ]
        if not eligible:
            self._memory.record_visit(selected.id)
            self._record_decision(
                "operator_selected",
                status="no_eligible_operator",
                attempt_id=attempt_id,
                trajectory_id=selected.id,
            )
            return
        operator = self._rng.choice(eligible)
        reference_route: Trajectory | None = None
        reference_node: ProgramNode | None = None
        if operator.name in (OperatorName.SYNTHESIZE, OperatorName.TRANSFER):
            reference_distribution = reference_sampling_distribution(
                candidates=reference_candidates,
                temperature=self._softmax_temperature,
            )
            reference_route = weighted_choice(
                [route for route, _ in reference_distribution],
                [probability for _, probability in reference_distribution],
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
        self._record_decision(
            "operator_selected",
            attempt_id=attempt_id,
            operator=operator.name,
            trajectory_id=selected.id,
            anchor_id=anchor_id,
            anchor_role=anchor_role,
            reference_trajectory_id=(
                None if reference_route is None else reference_route.id
            ),
            reference_program_id=(
                None if reference_node is None else reference_node.id
            ),
        )
        actions = self._generate_actions(
            context,
            operator=operator,
            iteration=attempt_id,
        )
        base_node = self._graph.get_node(anchor_id)
        route_best_before = compact_best_node(selected, self._graph, self._maximize)
        for seq, action in enumerate(actions):
            if not self._has_budget():
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
            generated = self._draw_program(
                code_prompt,
                stage="code",
                iteration=attempt_id,
                seq=seq,
                operator=operator.name,
            )
            old_global = self._best_node
            evaluated = self._evaluate(
                generated.code,
                idea=generated.idea,
                operator=operator.name,
            )
            if evaluated is None:
                continue
            child = self._graph.add_node(
                code=generated.code,
                idea=generated.idea,
                fitness=evaluated,
            )
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
                    anchor_role=anchor_role,
                    primary_trajectory_id=selected.id,
                    reference_trajectory_id=(
                        None if reference_route is None else reference_route.id
                    ),
                    reference_program_id=(
                        None if reference_node is None else reference_node.id
                    ),
                    trajectory_id=child_route.id,
                )
        self._memory.record_visit(selected.id)

    def _select_trajectory(self, attempt_id: int) -> Trajectory:
        distribution = trajectory_sampling_distribution(
            memory=self._memory,
            graph=self._graph,
            maximize=self._maximize,
            w=self._value_weights,
            temperature=self._softmax_temperature,
        )
        routes = [item[0] for item in distribution]
        probs = [item[2] for item in distribution]
        selected = weighted_choice(routes, probs, self._rng)
        self._record_decision(
            "trajectory_selected",
            attempt_id=attempt_id,
            trajectory_id=selected.id,
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
    ) -> _PromptContext:
        primary_history = trajectory_history(
            self._graph,
            selected,
            base_node_id=anchor_id,
            max_steps=self._memory.max_trajectory_length,
        )
        reference_history = (
            None
            if reference_route is None or reference_node is None
            else trajectory_history(
                self._graph,
                reference_route,
                base_node_id=reference_node.id,
                max_steps=self._memory.max_trajectory_length,
            )
        )
        prompt = build_action_prompt(
            base_node=self._graph.get_node(anchor_id),
            primary_history=primary_history,
            operator_constraint=operator.prompt_constraint,
            task_description=self._task_description_str,
            template_function=self._function_to_evolve,
            action_count=self._actions_per_iteration,
            maximize=self._maximize,
            reference_node=reference_node,
            reference_history=reference_history,
        )
        return _PromptContext(
            prompt=prompt,
            token_count=self._count_tokens(prompt),
            primary_history=primary_history.text,
            reference_history=(
                "" if reference_history is None else reference_history.text
            ),
        )

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

    def _generate_actions(
        self,
        context: _PromptContext,
        *,
        operator: Operator,
        iteration: int,
    ) -> list[str]:
        response = self._llm.draw_sample(
            context.prompt, max_tokens=self._action_max_tokens
        )
        actions, _ = parse_actions(
            response,
            expected_count=self._actions_per_iteration,
        )
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage="action",
                operator=operator.name,
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=0,
                prompt_tokens=context.token_count,
                response_tokens=self._count_tokens(response),
                token_count_mode=self._token_count_mode,
                n_actions=len(actions),
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
    ) -> None:
        update, reason = self._best_update_decision(node, self._best_node)
        if not update:
            return
        self._best_node = node
        self._best_node_sample_order = self._tot_sample_nums
        if trajectory_id is not None:
            self._best_trajectory_id = trajectory_id
        self._record_decision(
            "best_updated",
            sample_order=self._tot_sample_nums,
            node_id=node.id,
            reason=reason,
            iteration=iteration,
            operator=operator,
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
        keep_ids = {route.id for route in elites}
        keep_ids.update(route.id for route in diverse)
        keep_ids.update(route.id for route in sampled)
        if self._best_trajectory_id is not None:
            keep_ids.add(self._best_trajectory_id)
        archived_ids = []
        for route in ranked:
            if route.id in keep_ids:
                continue
            self._memory.archive(route.id)
            archived_ids.append(route.id)
        self._record_decision(
            "population_managed",
            kept_ids=sorted(keep_ids),
            archived_ids=archived_ids,
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
    ) -> _GeneratedProgram:
        response = self._llm.draw_sample(prompt)
        parsed = parse_program_response(response)
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage=stage,
                operator=str(operator),
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=seq,
                prompt_tokens=self._count_tokens(prompt),
                response_tokens=self._count_tokens(response),
                token_count_mode=self._token_count_mode,
                status="ok",
            )
        return _GeneratedProgram(
            idea=parsed.declared_idea or "",
            code=parsed.code,
        )

    def _evaluate(
        self,
        code: str,
        *,
        idea: str,
        operator: str | OperatorName,
    ) -> float | None:
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
                "operator": str(operator),
                "program": code,
                "idea": idea,
                "code_hash": code_hash(code),
                "program_loc": nonempty_loc(code),
                "status": "ok" if fitness is not None else (outcome.failure_kind or "invalid_result"),
            }
            if fitness is None:
                payload.update(
                    {
                        "error_type": outcome.error_type,
                        "error": outcome.error,
                    }
                )
            self._artifacts.record_candidate(**payload)
        return fitness

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._tot_sample_nums - self._last_checkpoint_sample
            >= self._checkpoint_interval
        ):
            save_checkpoint(self)

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


__all__ = ["TraceAADRunResult", "TraceAADV5"]
