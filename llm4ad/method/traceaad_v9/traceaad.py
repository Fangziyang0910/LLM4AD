"""TraceAAD V9: trajectory-guided program search on a complete MCTS tree."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
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
from .complexity import code_hash, nonempty_loc
from .context import (
    build_code_prompt,
    formation_history,
    node_history,
)
from .operators import DEFAULT_OPERATORS, DUAL_OPERATORS, Operator
from .prompt import build_initial_prompt, parse_program_response
from .schema import OperatorName, ProgramNode
from .tree import SearchTree, is_node_better
from .value import (
    reference_candidates,
    sample_reference,
    select_expansion_node,
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
    prompt_tokens: int
    current_edge_ids: tuple[int, ...]
    direct_child_edge_ids: tuple[int, ...]
    reference_edge_ids: tuple[int, ...]
    reference_branch_id: int | None
    reference_node: ProgramNode | None


@dataclass(frozen=True, slots=True)
class _EvaluatedCandidate:
    seq: int
    generated: _GeneratedProgram
    fitness: float | None
    sample_order: int


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: ProgramNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_root_children: int
    n_edges: int
    n_samples: int
    n_batches: int


class TraceAADV9:
    """Complete-tree search whose generation is conditioned on improvement history."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
        max_sample_nums: int | None = 1000,
        *,
        n_init: int = 10,
        offspring_per_iteration: int = 2,
        ancestor_history_limit: int = 8,
        direct_child_limit: int = 8,
        direct_child_top_count: int = 4,
        reference_temperature: float = 0.2,
        exploration_constant: float = 0.1,
        expansion_prior_weight: float = 1.0,
        maximize: bool = True,
        operators: tuple[type[Operator], ...] = DEFAULT_OPERATORS,
        code_max_tokens: int = 8192,
        context_token_limit: int | None = None,
        random_seed: int | None = None,
        max_stalled_iterations: int = 20,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 10,
        resume_from: str | Path | None = None,
    ) -> None:
        if max_sample_nums is not None and (
            isinstance(max_sample_nums, bool)
            or not isinstance(max_sample_nums, int)
            or max_sample_nums <= 0
        ):
            raise ValueError(
                "max_sample_nums must be a positive integer or None"
            )
        if (
            n_init < 0
            or offspring_per_iteration <= 0
            or ancestor_history_limit <= 0
            or direct_child_limit <= 0
            or direct_child_top_count <= 0
            or reference_temperature <= 0
            or exploration_constant < 0
            or expansion_prior_weight <= 0
            or code_max_tokens <= 0
            or checkpoint_interval <= 0
        ):
            raise ValueError("TraceAAD V9 search parameters are out of range")
        if context_token_limit is None or context_token_limit <= 0:
            raise ValueError("context_token_limit must be explicitly positive")

        self._llm = llm
        self._evaluation = evaluation
        self._artifacts = artifacts
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = int(n_init)
        self._offspring_per_iteration = int(offspring_per_iteration)
        self._ancestor_history_limit = int(ancestor_history_limit)
        self._direct_child_limit = int(direct_child_limit)
        self._direct_child_top_count = int(direct_child_top_count)
        self._reference_temperature = float(reference_temperature)
        self._exploration_constant = float(exploration_constant)
        self._expansion_prior_weight = float(expansion_prior_weight)
        self._maximize = bool(maximize)
        self._code_max_tokens = int(code_max_tokens)
        self._context_token_limit = int(context_token_limit)
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_batch = -1

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD V9 requires exactly one evolvable template function"
            )
        self._template_program = template
        self._function_to_evolve: Function = template.functions[0]
        self._evaluator = SecureEvaluator(evaluation)

        self._tree = SearchTree()
        self._operators = tuple(operator_type() for operator_type in operators)
        if not self._operators:
            raise ValueError("at least one TraceAAD V9 operator is required")
        if not any(operator.name not in DUAL_OPERATORS for operator in self._operators):
            raise ValueError("TraceAAD V9 requires at least one single-track operator")

        self._best_node: ProgramNode | None = None
        self._best_node_sample_order: int | None = None
        self._tot_sample_nums = 0
        self._next_attempt_id = 0
        self._batch_count = 0
        self._stalled_iterations = 0
        self._initialization_complete = False
        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    def run(self) -> TraceAADRunResult:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        result: TraceAADRunResult | None = None
        try:
            if not self._initialization_complete:
                self._initialize()
                self._initialization_complete = True
                save_checkpoint(self)
            while self._has_budget():
                if not self._tree.root.child_ids:
                    status = "stalled"
                    stop_reason = "empty_tree"
                    break
                nodes_before = len(self._tree.nodes())
                self._run_iteration(self._next_attempt_id)
                self._next_attempt_id += 1
                if len(self._tree.nodes()) == nodes_before:
                    self._stalled_iterations += 1
                else:
                    self._stalled_iterations = 0
                if self._stalled_iterations >= self._max_stalled_iterations:
                    status = "stalled"
                    stop_reason = "stalled_generation"
                    break
                self._save_checkpoint_if_due()
            result = self._result()
            if status == "error":
                status = "finished"
                if not self._has_budget():
                    stop_reason = "budget_exhausted"
            return result
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            raise
        finally:
            if result is None:
                result = self._result()
            save_checkpoint(self)
            if self._artifacts is not None:
                self._artifacts.write_summary(
                    status=status,
                    best_node_id=None
                    if result.best_node is None
                    else result.best_node.id,
                    best_score=None
                    if result.best_node is None
                    else result.best_node.fitness,
                    best_sample_order=self._best_node_sample_order,
                    num_samples=self._tot_sample_nums,
                    n_total_nodes=result.n_total_nodes,
                    n_valid_nodes=result.n_valid_nodes,
                    n_root_children=result.n_root_children,
                    n_edges=result.n_edges,
                    n_batches=result.n_batches,
                    initialization_target=self._n_init,
                    initialization_actual=result.n_root_children,
                    stop_reason=stop_reason,
                    **error,
                )
            self._llm.close()

    def _initialize(self) -> None:
        draw_seq = 0
        while (
            len(self._tree.root.child_ids) < self._n_init
            and self._has_budget()
        ):
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                maximize=self._maximize,
                diversity_hint=self._init_diversity_hint(
                    len(self._tree.root.child_ids)
                ),
            )
            if self._count_tokens(prompt) > self._context_token_limit:
                break
            generated = self._draw_program(
                prompt,
                stage="init",
                iteration=None,
                seq=draw_seq,
                operator="init",
                max_tokens=self._code_max_tokens,
            )
            draw_seq += 1
            fitness, sample_order = self._evaluate_detailed(
                generated.code,
                idea=generated.idea,
                operator="init",
            )
            if fitness is None:
                continue
            node = self._tree.add_initial(
                code=generated.code,
                idea=generated.idea,
                fitness=fitness,
                maximize=self._maximize,
                creation_order=sample_order,
            )
            self._update_best_from_tree(sample_order=sample_order)
            self._record_decision(
                "initial_node_created",
                node_id=node.id,
                sample_order=sample_order,
            )

    def _init_diversity_hint(self, slot: int) -> str:
        if slot == 0:
            return "Provide a simple, complete, and valid algorithm."
        ideas = [node.idea.strip() for node in self._tree.nodes() if node.idea.strip()][
            -6:
        ]
        if not ideas:
            return "Use a clearly different algorithmic idea from a trivial baseline."
        return "Use an idea clearly different from: " + "; ".join(
            f"'{idea[:100]}'" for idea in ideas
        )

    def _run_iteration(self, attempt_id: int) -> None:
        selected_node_id = select_expansion_node(
            self._tree,
            rng=self._rng,
            total_budget=self._max_sample_nums,
            used_budget=self._tot_sample_nums,
            exploration_constant=self._exploration_constant,
            expansion_prior_weight=self._expansion_prior_weight,
        )
        base_node = self._tree.get_node(selected_node_id)
        candidates = reference_candidates(self._tree, base_node.id)
        eligible = [
            operator
            for operator in self._operators
            if operator.name not in DUAL_OPERATORS or candidates
        ]
        operator = self._rng.choice(eligible)
        reference_branch_id: int | None = None
        reference_node: ProgramNode | None = None
        if operator.name in DUAL_OPERATORS:
            reference = sample_reference(
                self._tree,
                base_node.id,
                temperature=self._reference_temperature,
                rng=self._rng,
            )
            if reference is not None:
                reference_branch_id, reference_node = reference

        offspring_count = self._offspring_per_iteration
        if self._max_sample_nums is not None:
            offspring_count = min(
                offspring_count,
                self._max_sample_nums - self._tot_sample_nums,
            )
        if offspring_count <= 0:
            return
        contexts = [
            self._build_code_context(
                base_node=base_node,
                operator=operator,
                candidate_index=seq,
                candidate_count=offspring_count,
                reference_branch_id=reference_branch_id,
                reference_node=reference_node,
            )
            for seq in range(offspring_count)
        ]
        if any(context is None for context in contexts) and operator.name in DUAL_OPERATORS:
            operator = self._rng.choice(
                [item for item in self._operators if item.name not in DUAL_OPERATORS]
            )
            reference_branch_id = None
            reference_node = None
            contexts = [
                self._build_code_context(
                    base_node=base_node,
                    operator=operator,
                    candidate_index=seq,
                    candidate_count=offspring_count,
                    reference_branch_id=None,
                    reference_node=None,
                )
                for seq in range(offspring_count)
            ]
        if any(context is None for context in contexts):
            return
        code_contexts = [context for context in contexts if context is not None]
        context = code_contexts[0]

        self._tree.record_batch_visit(base_node.id)
        self._batch_count += 1
        batch_id = self._batch_count
        self._record_decision(
            "node_selected",
            attempt_id=attempt_id,
            batch_id=batch_id,
            selected_node_id=base_node.id,
            operator=operator.name,
            reference_node_id=None if reference_node is None else reference_node.id,
            requested_children=offspring_count,
            current_formation_edge_ids=context.current_edge_ids,
            direct_child_edge_ids=context.direct_child_edge_ids,
            reference_formation_edge_ids=context.reference_edge_ids,
        )

        global_best_before = self._best_node
        evaluated: list[_EvaluatedCandidate] = []
        for seq, code_context in enumerate(code_contexts):
            if not self._has_budget():
                break
            generated = self._draw_program(
                code_context.prompt,
                stage="direct_code",
                iteration=attempt_id,
                seq=seq,
                operator=operator.name,
                max_tokens=self._code_max_tokens,
                prompt_tokens=code_context.prompt_tokens,
            )
            fitness, sample_order = self._evaluate_detailed(
                generated.code,
                idea=generated.idea,
                operator=operator.name,
                batch_id=batch_id,
            )
            evaluated.append(_EvaluatedCandidate(seq, generated, fitness, sample_order))

        valid = [item for item in evaluated if item.fitness is not None]
        winner_index = self._batch_global_winner(valid, global_best_before)
        for valid_index, item in enumerate(valid):
            is_winner = valid_index == winner_index
            reason = None
            if is_winner:
                reason = self._best_update_reason_values(
                    float(item.fitness),
                    nonempty_loc(item.generated.code),
                    global_best_before,
                )
            child, edge = self._tree.add_child(
                parent_id=base_node.id,
                code=item.generated.code,
                idea=item.generated.idea,
                fitness=float(item.fitness),
                maximize=self._maximize,
                creation_order=item.sample_order,
                operator=operator.name,
                reference_node_id=None
                if context.reference_node is None
                else context.reference_node.id,
                reference_root_branch_id=context.reference_branch_id,
                global_best_directed_fitness=None
                if global_best_before is None
                else global_best_before.directed_fitness,
                new_global_best=is_winner,
                global_best_update_reason=reason,
                iteration=attempt_id,
                batch_id=batch_id,
                sibling_seq=item.seq,
                sample_order=item.sample_order,
            )
            if self._artifacts is not None:
                self._artifacts.record_edge(**asdict(edge))
        if valid:
            winner_sample = (
                None if winner_index is None else valid[winner_index].sample_order
            )
            self._update_best_from_tree(sample_order=winner_sample)

    def _build_code_context(
        self,
        *,
        base_node: ProgramNode,
        operator: Operator,
        candidate_index: int,
        candidate_count: int,
        reference_branch_id: int | None,
        reference_node: ProgramNode | None,
    ) -> _PromptContext | None:
        dual = reference_node is not None
        variants: list[tuple[int, int]] = []
        if dual:
            variants.extend(
                (self._direct_child_limit, ref_limit)
                for ref_limit in range(self._ancestor_history_limit, -1, -1)
            )
            variants.extend(
                (branch_limit, 0)
                for branch_limit in range(self._direct_child_limit - 1, -1, -1)
            )
        else:
            variants.extend(
                (branch_limit, 0)
                for branch_limit in range(self._direct_child_limit, -1, -1)
            )
        for branch_limit, reference_limit in variants:
            current = node_history(
                self._tree,
                base_node.id,
                ancestor_limit=self._ancestor_history_limit,
                direct_child_limit=branch_limit,
                direct_child_top_count=min(self._direct_child_top_count, branch_limit),
            )
            reference = (
                None
                if reference_node is None
                else formation_history(
                    self._tree,
                    reference_node.id,
                    max_edges=reference_limit,
                )
            )
            prompt = build_code_prompt(
                current_node=base_node,
                current_history=current.text,
                operator_constraint=operator.prompt_constraint,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                maximize=self._maximize,
                candidate_index=candidate_index,
                candidate_count=candidate_count,
                reference_node=reference_node,
                reference_history="" if reference is None else reference.text,
            )
            prompt_tokens = self._count_tokens(prompt)
            if prompt_tokens <= self._context_token_limit:
                return _PromptContext(
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    current_edge_ids=current.formation_edge_ids,
                    direct_child_edge_ids=current.direct_child_edge_ids,
                    reference_edge_ids=()
                    if reference is None
                    else reference.formation_edge_ids,
                    reference_branch_id=reference_branch_id,
                    reference_node=reference_node,
                )
        return None

    def _draw_program(
        self,
        prompt: str,
        *,
        stage: str,
        iteration: int | None,
        seq: int,
        operator: str | OperatorName,
        max_tokens: int,
        prompt_tokens: int | None = None,
    ) -> _GeneratedProgram:
        response = self._llm.draw_sample(prompt, max_tokens=max_tokens)
        parsed = parse_program_response(response)
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage=stage,
                operator=str(operator),
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=seq,
                prompt_tokens=self._count_tokens(prompt)
                if prompt_tokens is None
                else prompt_tokens,
                response_tokens=self._count_tokens(response),
                status="ok",
            )
        return _GeneratedProgram(parsed.declared_idea or "", parsed.code)

    def _evaluate_detailed(
        self,
        code: str,
        *,
        idea: str,
        operator: str | OperatorName,
        batch_id: int | None = None,
    ) -> tuple[float | None, int]:
        if not self._has_budget():
            return None, self._tot_sample_nums
        outcome = self._evaluator.evaluate_program_with_details(code)
        self._tot_sample_nums += 1
        sample_order = self._tot_sample_nums
        if outcome.failure_kind == "prepare_error":
            raise RuntimeError(
                _one_line(
                    outcome.error
                    or "evaluator preparation failed without an error message",
                    ERROR_MAX_CHARS,
                )
            )
        score = getattr(outcome.result, "fitness", outcome.result)
        valid_score: float | None
        try:
            candidate = float(score)
        except (TypeError, ValueError, OverflowError):
            candidate = math.nan
        valid_score = candidate if math.isfinite(candidate) else None
        if self._artifacts is not None:
            payload = {
                "sample_order": sample_order,
                "score": valid_score,
                "operator": str(operator),
                "program": code,
                "idea": idea,
                "code_hash": code_hash(code),
                "batch_id": batch_id,
                "status": (
                    "ok"
                    if valid_score is not None
                    else (outcome.failure_kind or "invalid_result")
                ),
            }
            if valid_score is None:
                payload.update(
                    {
                        "error_type": outcome.error_type,
                        "error": outcome.error,
                    }
                )
            self._artifacts.record_candidate(**payload)
        return valid_score, sample_order

    def _batch_global_winner(
        self,
        valid: list[_EvaluatedCandidate],
        incumbent: ProgramNode | None,
    ) -> int | None:
        winner: int | None = None
        next_id = self._tree._next_node_id
        for index, item in enumerate(valid):
            directed = float(item.fitness) if self._maximize else -float(item.fitness)
            candidate_key = (
                directed,
                -nonempty_loc(item.generated.code),
                -(next_id + index),
            )
            if incumbent is not None:
                incumbent_key = (
                    incumbent.directed_fitness,
                    -incumbent.program_loc,
                    -incumbent.id,
                )
                if candidate_key <= incumbent_key:
                    continue
            if winner is None:
                winner = index
                continue
            previous = valid[winner]
            previous_directed = (
                float(previous.fitness) if self._maximize else -float(previous.fitness)
            )
            previous_key = (
                previous_directed,
                -nonempty_loc(previous.generated.code),
                -(next_id + winner),
            )
            if candidate_key > previous_key:
                winner = index
        return winner

    def _best_update_reason_values(
        self,
        fitness: float,
        program_loc: int,
        incumbent: ProgramNode | None,
    ) -> str:
        if incumbent is None:
            return "strict_fitness"
        directed = fitness if self._maximize else -fitness
        if directed > incumbent.directed_fitness:
            return "strict_fitness"
        if (
            directed == incumbent.directed_fitness
            and program_loc < incumbent.program_loc
        ):
            return "tie_shorter"
        raise AssertionError("candidate is not a global-best update")

    def _update_best_from_tree(self, *, sample_order: int | None) -> None:
        best_id = self._tree.root.subtree_best_node_id
        best = None if best_id is None else self._tree.get_node(best_id)
        if best is None or not is_node_better(best, self._best_node):
            return
        self._best_node = best
        self._best_node_sample_order = sample_order

    def _count_tokens(self, prompt: str) -> int:
        counter = getattr(self._llm, "count_tokens", None)
        if callable(counter):
            return int(counter(prompt))
        return len(prompt.encode("utf-8"))

    def _save_checkpoint_if_due(self) -> None:
        if (
            self._checkpoint_dir is not None
            and self._batch_count - self._last_checkpoint_batch
            >= self._checkpoint_interval
        ):
            save_checkpoint(self)

    def _record_decision(self, event: str, **payload) -> None:
        if self._artifacts is not None:
            self._artifacts.record_decision(event, **payload)

    def _has_budget(self) -> bool:
        return (
            self._max_sample_nums is None
            or self._tot_sample_nums < self._max_sample_nums
        )

    def _result(self) -> TraceAADRunResult:
        nodes = self._tree.nodes()
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=len(nodes),
            n_valid_nodes=len(nodes),
            n_root_children=len(self._tree.root.child_ids),
            n_edges=len(self._tree.edges()),
            n_samples=self._tot_sample_nums,
            n_batches=self._batch_count,
        )


__all__ = ["TraceAADRunResult", "TraceAADV9"]
