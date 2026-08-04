"""TraceAAD V8: trajectory-guided program search on a complete MCTS tree."""

from __future__ import annotations

import copy
import hashlib
import json
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
from .._observability import (
    close_llm,
    init_observability,
    is_search_aborted,
    record_sample_failure,
    reset_sample_failures,
)
from ..traceaad_artifacts import TraceAADArtifacts
from .checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from .complexity import code_hash, nonempty_loc
from .context import (
    build_action_prompt,
    build_code_prompt,
    formation_history,
    node_history,
)
from .operators import DEFAULT_OPERATORS, DUAL_OPERATORS, Operator
from .prompt import build_initial_prompt, parse_actions, parse_program_response
from .schema import OperatorName, PROTOCOL_ID, ProgramNode
from .tree import SearchTree, is_node_better
from .value import (
    reference_candidates,
    sample_reference,
    select_expansion_node,
    widening_capacity,
)


def _stable_identity_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _stable_identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_identity_value(item) for item in value]
    if all(hasattr(value, name) for name in ("shape", "dtype", "tobytes")):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
            "array_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    return repr(value)


def _configuration_digest(obj) -> str:
    payload = {
        key: _stable_identity_value(value)
        for key, value in sorted(vars(obj).items())
        if not key.startswith("_")
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _GeneratedProgram:
    idea: str
    program: Program
    sample_time: float


@dataclass(frozen=True, slots=True)
class _PromptContext:
    current_node_id: int
    prompt: str
    prompt_tokens: int
    current_history: str
    reference_history: str
    current_edge_ids: tuple[int, ...]
    direct_child_edge_ids: tuple[int, ...]
    reference_edge_ids: tuple[int, ...]
    reference_branch_id: int | None
    reference_node: ProgramNode | None

    @property
    def used_dual(self) -> bool:
        return self.reference_node is not None


@dataclass(frozen=True, slots=True)
class _EvaluatedCandidate:
    action: str
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


class TraceAADV8:
    """Complete-tree search whose generation is conditioned on improvement history."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: TraceAADArtifacts | None = None,
        max_sample_nums: int | None = 100,
        *,
        n_init: int = 30,
        actions_per_iteration: int = 2,
        ancestor_history_limit: int = 8,
        direct_child_limit: int = 8,
        direct_child_top_count: int = 4,
        reference_temperature: float = 0.2,
        exploration_constant: float = 0.5,
        widening_alpha: float = 0.5,
        maximize: bool = True,
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
        if ancestor_history_limit <= 0:
            raise ValueError("ancestor_history_limit must be positive")
        if direct_child_limit <= 0 or direct_child_top_count <= 0:
            raise ValueError("direct-child history limits must be positive")
        if direct_child_top_count > direct_child_limit:
            raise ValueError("direct_child_top_count cannot exceed direct_child_limit")
        if reference_temperature <= 0:
            raise ValueError("reference_temperature must be positive")
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if not 0 < widening_alpha <= 1:
            raise ValueError("widening_alpha must be in (0, 1]")
        if action_max_tokens <= 0 or code_max_tokens <= 0:
            raise ValueError("output token limits must be positive")
        if context_token_limit is None or context_token_limit <= 0:
            raise ValueError("context_token_limit must be explicitly positive")
        if checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")

        self._llm = llm
        self._evaluation = evaluation
        self._evaluation_configuration_sha256 = _configuration_digest(evaluation)
        self._artifacts = profiler
        self._profiler = profiler
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = int(n_init)
        self._actions_per_iteration = int(actions_per_iteration)
        self._ancestor_history_limit = int(ancestor_history_limit)
        self._direct_child_limit = int(direct_child_limit)
        self._direct_child_top_count = int(direct_child_top_count)
        self._reference_temperature = float(reference_temperature)
        self._exploration_constant = float(exploration_constant)
        self._widening_alpha = float(widening_alpha)
        self._maximize = bool(maximize)
        self._action_max_tokens = int(action_max_tokens)
        self._code_max_tokens = int(code_max_tokens)
        self._context_token_limit = int(context_token_limit)
        self._random_seed = random_seed
        self._rng = random.Random(random_seed)
        self._debug_mode = bool(debug_mode)
        self._max_stalled_iterations = max(1, int(max_stalled_iterations))
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._checkpoint_interval = int(checkpoint_interval)
        self._last_checkpoint_batch = -1
        llm.debug_mode = debug_mode

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError(
                "TraceAAD V8 requires exactly one evolvable template function"
            )
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)

        self._tree = SearchTree()
        self._operators = tuple(operator_type() for operator_type in operators)
        if not self._operators:
            raise ValueError("at least one TraceAAD V8 operator is required")
        if not any(operator.name not in DUAL_OPERATORS for operator in self._operators):
            raise ValueError("TraceAAD V8 requires at least one single-track operator")

        self._best_node: ProgramNode | None = None
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
            self._record_decision(
                "checkpoint_loaded",
                checkpoint=str(checkpoint),
                sample_order=self._tot_sample_nums,
                next_attempt_id=self._next_attempt_id,
            )

    def runtime_identity(self) -> dict[str, str | None]:
        def setting(obj, name: str) -> str | None:
            value = getattr(obj, name, None)
            return None if value is None else repr(value)

        return {
            "task_description_sha256": hashlib.sha256(
                self._task_description_str.encode("utf-8")
            ).hexdigest(),
            "template_program_sha256": hashlib.sha256(
                str(self._template_program).encode("utf-8")
            ).hexdigest(),
            "evaluation_type": f"{type(self._evaluation).__module__}.{type(self._evaluation).__qualname__}",
            "evaluation_configuration_sha256": self._evaluation_configuration_sha256,
            "evaluation_random_seed": setting(self._evaluation, "random_seed"),
            "evaluation_timeout_seconds": setting(self._evaluation, "timeout_seconds"),
            "evaluation_safe_evaluate": setting(self._evaluation, "safe_evaluate"),
            "evaluation_use_numba": setting(self._evaluation, "use_numba_accelerate"),
            "llm_type": f"{type(self._llm).__module__}.{type(self._llm).__qualname__}",
            "llm_model": None
            if getattr(self._llm, "model", None) is None
            else str(self._llm.model),
            "llm_base_url": None
            if getattr(self._llm, "base_url", None) is None
            else str(self._llm.base_url),
            "llm_max_tokens": setting(self._llm, "max_tokens"),
            "llm_temperature": setting(self._llm, "temperature"),
            "llm_enable_thinking": setting(self._llm, "enable_thinking"),
        }

    def search_configuration(self) -> dict:
        return {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "max_sample_nums": self._max_sample_nums,
            "n_init": self._n_init,
            "actions_per_iteration": self._actions_per_iteration,
            "ancestor_history_limit": self._ancestor_history_limit,
            "direct_child_limit": self._direct_child_limit,
            "direct_child_top_count": self._direct_child_top_count,
            "reference_temperature": self._reference_temperature,
            "exploration_constant": self._exploration_constant,
            "widening_alpha": self._widening_alpha,
            "maximize": self._maximize,
            "operators": [str(operator.name) for operator in self._operators],
            "action_max_tokens": self._action_max_tokens,
            "code_max_tokens": self._code_max_tokens,
            "context_token_limit": self._context_token_limit,
            "max_consecutive_sample_failures": self._max_consecutive_sample_failures,
            "max_stalled_iterations": self._max_stalled_iterations,
            "checkpoint_interval": self._checkpoint_interval,
            "random_seed": self._random_seed,
        }

    def run(self) -> TraceAADRunResult:
        status = "error"
        error: dict[str, str] = {}
        result: TraceAADRunResult | None = None
        try:
            if not self._initialization_complete:
                self._initialize()
                self._initialization_complete = True
                save_checkpoint(self)
            while self._has_budget() and not is_search_aborted(self):
                if not self._tree.root.child_ids:
                    self._record_decision("search_stopped", status="empty_tree")
                    break
                nodes_before = len(self._tree.nodes())
                self._run_iteration(self._next_attempt_id)
                self._next_attempt_id += 1
                if len(self._tree.nodes()) == nodes_before:
                    self._stalled_iterations += 1
                else:
                    self._stalled_iterations = 0
                if self._stalled_iterations >= self._max_stalled_iterations:
                    self._record_decision(
                        "search_stopped",
                        status="stalled_generation",
                        attempt_id=self._next_attempt_id - 1,
                    )
                    break
                self._save_checkpoint_if_due()
            result = self._result()
            status = "aborted" if is_search_aborted(self) else "finished"
            return result
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
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
                    best_node_id=None
                    if result.best_node is None
                    else result.best_node.id,
                    best_score=None
                    if result.best_node is None
                    else result.best_node.fitness,
                    best_sample_order=self._best_node_sample_order,
                    method_sample_count=self._tot_sample_nums,
                    n_total_nodes=result.n_total_nodes,
                    n_valid_nodes=result.n_valid_nodes,
                    n_root_children=result.n_root_children,
                    n_edges=result.n_edges,
                    n_batches=result.n_batches,
                    initialization_target=self._n_init,
                    initialization_actual=result.n_root_children,
                    search_aborted=is_search_aborted(self),
                    **error,
                )
                self._artifacts.finish()
            close_llm(self._llm)

    def _initialize(self) -> None:
        stalled_draws = 0
        draw_seq = 0
        while (
            len(self._tree.root.child_ids) < self._n_init
            and self._has_budget()
            and not is_search_aborted(self)
        ):
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                diversity_hint=self._init_diversity_hint(
                    len(self._tree.root.child_ids)
                ),
            )
            if self._count_tokens(prompt) > self._context_token_limit:
                self._record_decision("context_overflow", stage="init")
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
            if generated is None:
                stalled_draws += 1
                if stalled_draws >= self._max_stalled_iterations:
                    break
                continue
            fitness, sample_order = self._evaluate_detailed(
                generated.program,
                idea=generated.idea,
                operator="init",
                sample_time=generated.sample_time,
            )
            if fitness is None:
                continue
            stalled_draws = 0
            node = self._tree.add_initial(
                code=str(generated.program),
                idea=generated.idea,
                fitness=fitness,
                maximize=self._maximize,
                creation_order=sample_order,
            )
            self._update_best_from_tree(
                sample_order=sample_order, iteration=None, operator="init"
            )
            self._record_decision(
                "initial_node_created",
                node_id=node.id,
                sample_order=sample_order,
                root_child_count=len(self._tree.root.child_ids),
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
        selection = select_expansion_node(
            self._tree,
            rng=self._rng,
            total_budget=self._max_sample_nums,
            used_budget=self._tot_sample_nums,
            exploration_constant=self._exploration_constant,
            actions_per_iteration=self._actions_per_iteration,
            widening_alpha=self._widening_alpha,
        )
        base_node = self._tree.get_node(selection.selected_node_id)
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

        predicted_visits = base_node.visit_count + 1
        capacity = widening_capacity(
            predicted_visits,
            actions_per_iteration=self._actions_per_iteration,
            alpha=self._widening_alpha,
        )
        action_count = min(
            self._actions_per_iteration,
            max(0, capacity - len(base_node.child_ids)),
        )
        if action_count <= 0:
            raise AssertionError("selected V8 node has no progressive-widening slot")
        context = self._build_action_context(
            base_node=base_node,
            operator=operator,
            action_count=action_count,
            reference_branch_id=reference_branch_id,
            reference_node=reference_node,
        )
        if context is None and operator.name in DUAL_OPERATORS:
            previous = operator.name
            operator = self._rng.choice(
                [item for item in self._operators if item.name not in DUAL_OPERATORS]
            )
            reference_branch_id = None
            reference_node = None
            context = self._build_action_context(
                base_node=base_node,
                operator=operator,
                action_count=action_count,
                reference_branch_id=None,
                reference_node=None,
            )
            self._record_decision(
                "operator_fallback",
                attempt_id=attempt_id,
                from_operator=previous,
                to_operator=operator.name,
                reason="dual_context_overflow",
            )
        if context is None:
            self._record_decision(
                "context_overflow",
                stage="action",
                attempt_id=attempt_id,
                node_id=base_node.id,
            )
            return

        path = self._tree.record_batch_visit(base_node.id)
        self._batch_count += 1
        batch_id = self._batch_count
        self._record_decision(
            "node_selected",
            attempt_id=attempt_id,
            batch_id=batch_id,
            selected_path=path,
            selected_node_id=base_node.id,
            selection_steps=[
                {
                    "parent_id": step.parent_id,
                    "child_id": step.child_id,
                    "z": step.normalized_value,
                    "g": step.subtree_value,
                    "n": step.visit_count,
                    "uct": step.uct,
                }
                for step in selection.steps
            ],
            widening_capacity=capacity,
            child_count=len(base_node.child_ids),
            child_slots=action_count,
            operator=operator.name,
            reference_root_branch_id=reference_branch_id,
            reference_node_id=None if reference_node is None else reference_node.id,
            current_formation_edge_ids=context.current_edge_ids,
            direct_child_edge_ids=context.direct_child_edge_ids,
            reference_formation_edge_ids=context.reference_edge_ids,
            batch_visit=True,
        )
        actions = self._generate_actions(
            context,
            operator=operator,
            iteration=attempt_id,
            expected_count=action_count,
        )
        code_requests = self._prepare_code_requests(
            actions=actions,
            base_node=base_node,
            context=context,
        )
        while any(item[3] > self._context_token_limit for item in code_requests):
            overflow = max(
                item[3] - self._context_token_limit for item in code_requests
            )
            narrower_limit = min(
                context.prompt_tokens - 1,
                self._context_token_limit - overflow,
            )
            narrower = self._build_action_context(
                base_node=base_node,
                operator=operator,
                action_count=action_count,
                reference_branch_id=context.reference_branch_id,
                reference_node=context.reference_node,
                prompt_token_limit=narrower_limit,
            )
            if narrower is None and context.used_dual:
                previous = operator.name
                operator = self._rng.choice(
                    [
                        item
                        for item in self._operators
                        if item.name not in DUAL_OPERATORS
                    ]
                )
                narrower = self._build_action_context(
                    base_node=base_node,
                    operator=operator,
                    action_count=action_count,
                    reference_branch_id=None,
                    reference_node=None,
                )
                self._record_decision(
                    "operator_fallback",
                    attempt_id=attempt_id,
                    from_operator=previous,
                    to_operator=operator.name,
                    reason="dual_code_context_overflow_after_shrink",
                )
            if narrower is None:
                self._record_decision(
                    "context_overflow",
                    stage="code_after_context_shrink",
                    attempt_id=attempt_id,
                    node_id=base_node.id,
                )
                return
            context = narrower
            self._record_decision(
                "context_shrunk",
                attempt_id=attempt_id,
                operator=operator.name,
                prompt_tokens=context.prompt_tokens,
                current_formation_edge_ids=context.current_edge_ids,
                direct_child_edge_ids=context.direct_child_edge_ids,
                reference_formation_edge_ids=context.reference_edge_ids,
            )
            actions = self._generate_actions(
                context,
                operator=operator,
                iteration=attempt_id,
                expected_count=action_count,
            )
            code_requests = self._prepare_code_requests(
                actions=actions,
                base_node=base_node,
                context=context,
            )

        global_best_before = self._best_node
        evaluated: list[_EvaluatedCandidate] = []
        for seq, action, prompt, prompt_tokens in code_requests:
            if not self._has_budget() or is_search_aborted(self):
                break
            if prompt_tokens > self._context_token_limit:
                self._record_decision(
                    "context_overflow",
                    stage="code",
                    attempt_id=attempt_id,
                    seq=seq,
                    prompt_tokens=prompt_tokens,
                )
                continue
            generated = self._draw_program(
                prompt,
                stage="code",
                iteration=attempt_id,
                seq=seq,
                operator=operator.name,
                action=action,
                max_tokens=self._code_max_tokens,
                prompt_tokens=prompt_tokens,
                parent_node_id=base_node.id,
                batch_id=batch_id,
                reference_node_id=(
                    None
                    if context.reference_node is None
                    else context.reference_node.id
                ),
                reference_root_branch_id=context.reference_branch_id,
            )
            if generated is None:
                continue
            fitness, sample_order = self._evaluate_detailed(
                generated.program,
                idea=generated.idea,
                operator=operator.name,
                sample_time=generated.sample_time,
                parent_node_id=base_node.id,
                action=action,
                iteration=attempt_id,
                batch_id=batch_id,
                sibling_seq=seq,
                reference_node_id=(
                    None
                    if context.reference_node is None
                    else context.reference_node.id
                ),
                reference_root_branch_id=context.reference_branch_id,
            )
            evaluated.append(
                _EvaluatedCandidate(action, seq, generated, fitness, sample_order)
            )

        valid = [item for item in evaluated if item.fitness is not None]
        winner_index = self._batch_global_winner(valid, global_best_before)
        for valid_index, item in enumerate(valid):
            is_winner = valid_index == winner_index
            reason = None
            if is_winner:
                reason = self._best_update_reason_values(
                    float(item.fitness),
                    nonempty_loc(str(item.generated.program)),
                    global_best_before,
                )
            child, edge, backup_changes = self._tree.add_child(
                parent_id=base_node.id,
                code=str(item.generated.program),
                idea=item.generated.idea,
                fitness=float(item.fitness),
                maximize=self._maximize,
                creation_order=item.sample_order,
                operator=operator.name,
                action=item.action,
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
                self._artifacts.record_edge(
                    **{**edge.__dict__}
                    if hasattr(edge, "__dict__")
                    else {
                        "edge_id": edge.id,
                        "parent_id": edge.parent_id,
                        "child_id": edge.child_id,
                        "sample_order": edge.sample_order,
                        "iteration": edge.iteration,
                        "batch_id": edge.batch_id,
                        "seq": edge.sibling_seq,
                        "operator": edge.operator,
                        "action": edge.action,
                        "implemented_idea": edge.implemented_idea,
                        "reference_program_id": edge.reference_node_id,
                        "reference_root_branch_id": edge.reference_root_branch_id,
                        "delta_parent": edge.delta_parent,
                        "delta_global_best": edge.delta_global_best,
                        "outcome": edge.outcome,
                        "delta_loc": edge.delta_loc,
                        "code_change_ratio": edge.code_change_ratio,
                        "new_global_best": edge.new_global_best,
                        "global_best_update_reason": edge.global_best_update_reason,
                    }
                )
            self._record_decision(
                "subtree_backup",
                batch_id=batch_id,
                child_id=child.id,
                changes=backup_changes,
            )
        if valid:
            winner_sample = (
                None if winner_index is None else valid[winner_index].sample_order
            )
            self._update_best_from_tree(
                sample_order=winner_sample,
                iteration=attempt_id,
                operator=operator.name,
            )

    def _build_action_context(
        self,
        *,
        base_node: ProgramNode,
        operator: Operator,
        action_count: int,
        reference_branch_id: int | None,
        reference_node: ProgramNode | None,
        prompt_token_limit: int | None = None,
    ) -> _PromptContext | None:
        token_limit = (
            self._context_token_limit
            if prompt_token_limit is None
            else min(self._context_token_limit, prompt_token_limit)
        )
        if token_limit <= 0:
            return None
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
            prompt = build_action_prompt(
                current_node=base_node,
                current_history=current.text,
                operator_constraint=operator.prompt_constraint,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                action_count=action_count,
                maximize=self._maximize,
                reference_node=reference_node,
                reference_history="" if reference is None else reference.text,
            )
            prompt_tokens = self._count_tokens(prompt)
            if prompt_tokens <= token_limit:
                return _PromptContext(
                    current_node_id=base_node.id,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    current_history=current.text,
                    reference_history="" if reference is None else reference.text,
                    current_edge_ids=current.formation_edge_ids,
                    direct_child_edge_ids=current.direct_child_edge_ids,
                    reference_edge_ids=()
                    if reference is None
                    else reference.formation_edge_ids,
                    reference_branch_id=reference_branch_id,
                    reference_node=reference_node,
                )
        return None

    def _prepare_code_requests(
        self,
        *,
        actions: list[str],
        base_node: ProgramNode,
        context: _PromptContext,
    ) -> list[tuple[int, str, str, int]]:
        requests = []
        for seq, action in enumerate(actions):
            prompt = build_code_prompt(
                current_node=base_node,
                current_history=context.current_history,
                action=action,
                task_description=self._task_description_str,
                template_function=self._function_to_evolve,
                reference_node=context.reference_node,
                reference_history=context.reference_history,
            )
            requests.append((seq, action, prompt, self._count_tokens(prompt)))
        return requests

    def _generate_actions(
        self,
        context: _PromptContext,
        *,
        operator: Operator,
        iteration: int,
        expected_count: int,
    ) -> list[str]:
        start = time.time()
        try:
            response = self._llm.draw_sample(
                context.prompt,
                max_tokens=self._action_max_tokens,
            )
            sample_time = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="action",
                operator=operator.name,
                sample_order=self._tot_sample_nums + 1,
                counts_budget=False,
                iteration=iteration,
                parent_node_id=context.current_node_id,
                reference_node_id=(
                    None
                    if context.reference_node is None
                    else context.reference_node.id
                ),
                reference_root_branch_id=context.reference_branch_id,
            )
            self._record_decision(
                "action_generation_failed",
                iteration=iteration,
                operator=operator.name,
                parent_node_id=context.current_node_id,
                reference_node_id=(
                    None
                    if context.reference_node is None
                    else context.reference_node.id
                ),
                reference_root_branch_id=context.reference_branch_id,
                failure_kind="transport",
            )
            return []
        actions, errors = parse_actions(response, expected_count=expected_count)
        self._record_decision(
            "actions_generated",
            iteration=iteration,
            operator=operator.name,
            parent_node_id=context.current_node_id,
            actions=actions,
            parse_errors=errors,
            current_formation_edge_ids=context.current_edge_ids,
            direct_child_edge_ids=context.direct_child_edge_ids,
            reference_formation_edge_ids=context.reference_edge_ids,
            reference_node_id=(
                None if context.reference_node is None else context.reference_node.id
            ),
            reference_root_branch_id=context.reference_branch_id,
        )
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage="action",
                operator=operator.name,
                sample_order=self._tot_sample_nums + 1,
                iteration=iteration,
                seq=0,
                sample_time=sample_time,
                prompt_tokens=context.prompt_tokens,
                response_tokens=self._count_tokens(response),
                token_count_mode=self._token_count_mode,
                n_actions=len(actions),
                parse_errors=errors,
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
        operator: str | OperatorName,
        max_tokens: int,
        action: str | None = None,
        prompt_tokens: int | None = None,
        parent_node_id: int | None = None,
        batch_id: int | None = None,
        reference_node_id: int | None = None,
        reference_root_branch_id: int | None = None,
    ) -> _GeneratedProgram | None:
        sample_order = self._tot_sample_nums + 1
        start = time.time()
        try:
            response = self._llm.draw_sample(prompt, max_tokens=max_tokens)
            elapsed = time.time() - start
            reset_sample_failures(self)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage=stage,
                operator=str(operator),
                sample_order=sample_order,
                counts_budget=False,
                iteration=iteration,
                seq=seq,
                action=action,
                parent_node_id=parent_node_id,
                batch_id=batch_id,
                reference_node_id=reference_node_id,
                reference_root_branch_id=reference_root_branch_id,
            )
            self._record_decision(
                "code_generation_failed",
                failure_kind="transport",
                iteration=iteration,
                seq=seq,
                operator=str(operator),
                action=action,
                parent_node_id=parent_node_id,
                batch_id=batch_id,
                reference_node_id=reference_node_id,
                reference_root_branch_id=reference_root_branch_id,
            )
            return None
        parsed = parse_program_response(
            response,
            self._template_program,
            self._function_to_evolve.name,
        )
        if self._artifacts is not None:
            self._artifacts.record_llm_call(
                stage=stage,
                operator=str(operator),
                sample_order=sample_order,
                iteration=iteration,
                seq=seq,
                action=action,
                sample_time=elapsed,
                prompt_tokens=self._count_tokens(prompt)
                if prompt_tokens is None
                else prompt_tokens,
                response_tokens=self._count_tokens(response),
                token_count_mode=self._token_count_mode,
                program_parse_success=parsed is not None,
                response=response,
                status="ok" if parsed is not None else "parse_failed",
            )
        if parsed is None:
            self._record_decision(
                "code_generation_failed",
                failure_kind="parse",
                iteration=iteration,
                seq=seq,
                operator=str(operator),
                action=action,
                parent_node_id=parent_node_id,
                batch_id=batch_id,
                reference_node_id=reference_node_id,
                reference_root_branch_id=reference_root_branch_id,
            )
            return None
        return _GeneratedProgram(parsed.idea, parsed.program, elapsed)

    def _evaluate_detailed(
        self,
        program: Program,
        *,
        idea: str,
        operator: str | OperatorName,
        sample_time: float,
        parent_node_id: int | None = None,
        action: str | None = None,
        iteration: int | None = None,
        batch_id: int | None = None,
        sibling_seq: int | None = None,
        reference_node_id: int | None = None,
        reference_root_branch_id: int | None = None,
    ) -> tuple[float | None, int]:
        if not self._has_budget():
            return None, self._tot_sample_nums
        outcome, eval_time = self._evaluator.evaluate_program_record_time_with_details(
            program
        )
        self._tot_sample_nums += 1
        sample_order = self._tot_sample_nums
        result = outcome.result
        score = getattr(result, "fitness", result)
        valid_score = None
        if score is not None:
            try:
                candidate = float(score)
            except (TypeError, ValueError, OverflowError):
                candidate = math.nan
            if math.isfinite(candidate):
                valid_score = candidate
        program_text = str(program)
        if self._artifacts is not None:
            payload = {
                "sample_order": sample_order,
                "score": valid_score,
                "operator": str(operator),
                "program": program_text,
                "idea": idea,
                "code_hash": code_hash(program_text),
                "program_loc": nonempty_loc(program_text),
                "evaluate_time": eval_time,
                "sample_time": sample_time,
                "parent_node_id": parent_node_id,
                "action": action,
                "iteration": iteration,
                "batch_id": batch_id,
                "sibling_seq": sibling_seq,
                "reference_node_id": reference_node_id,
                "reference_root_branch_id": reference_root_branch_id,
                "status": "ok" if valid_score is not None else "eval_failed",
            }
            if valid_score is None:
                payload.update(
                    {
                        "failure_kind": outcome.failure_kind or "invalid_result",
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
                -nonempty_loc(str(item.generated.program)),
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
                -nonempty_loc(str(previous.generated.program)),
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

    def _update_best_from_tree(
        self,
        *,
        sample_order: int | None,
        iteration: int | None,
        operator: str | OperatorName,
    ) -> None:
        best_id = self._tree.root.subtree_best_node_id
        best = None if best_id is None else self._tree.get_node(best_id)
        if best is None or not is_node_better(best, self._best_node):
            return
        old = self._best_node
        self._best_node = best
        self._best_node_sample_order = sample_order
        reason = self._best_update_reason_values(best.fitness, best.program_loc, old)
        self._record_decision(
            "best_updated",
            sample_order=sample_order,
            node_id=best.id,
            reason=reason,
            iteration=iteration,
            operator=operator,
        )

    def _count_tokens(self, prompt: str) -> int:
        counter = getattr(self._llm, "count_tokens", None)
        if callable(counter):
            return int(counter(prompt))
        return len(prompt.encode("utf-8"))

    @property
    def _token_count_mode(self) -> str:
        mode = getattr(self._llm, "token_count_mode", None)
        if isinstance(mode, str):
            return mode
        return (
            "llm_count_tokens"
            if callable(getattr(self._llm, "count_tokens", None))
            else "utf8_byte_upper_bound"
        )

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


__all__ = ["TraceAADRunResult", "TraceAADV8"]
