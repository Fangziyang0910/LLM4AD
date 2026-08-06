"""Independent, compact TraceAAD V8.3 search implementation."""

from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any

from ...base import Evaluation, Function, LLM, Program, SecureEvaluator, TextFunctionProgramConverter
from ..traceaad_artifacts import TraceAADArtifacts
from .context import LocalContext, build_local_context
from .operators import DEFAULT_OPERATORS, Operator
from .prompt import (
    ParsedCall1,
    build_description_prompt,
    build_initial_prompt,
    build_search_prompt,
    parse_call1,
    parse_description,
)
from .schema import AlgorithmRecord, OperatorName, SelectionResult, TreeNode
from .tree import SearchTree
from .value import (
    node_fitness,
    reference_candidates,
    sample_reference,
    select_expansion_node,
)


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    """The small per-expansion record needed to inspect a V8.3 run."""

    attempt: int
    phase: str
    selected_node_id: int | None = None
    selection_path: tuple[int, ...] = ()
    operator: str | None = None
    reference_node_id: int | None = None
    context: str = ""
    call1_prompt: str = ""
    call1_response: str | None = None
    description_prompt: str | None = None
    description_response: str | None = None
    status: str = "unknown"
    selection_steps: tuple[Any, ...] = ()
    global_best_updated: bool = False
    global_best_fitness: float | None = None
    failure_reason: str | None = None
    fitness: float | None = None
    evaluation_time: float | None = None
    node_id: int | None = None


@dataclass(frozen=True, slots=True)
class TraceAADRunResult:
    best_node: TreeNode | None
    n_total_nodes: int
    n_valid_nodes: int
    n_root_children: int
    n_edges: int
    n_samples: int
    n_attempts: int
    initialization_complete: bool
    stop_reason: str


@dataclass(frozen=True, slots=True)
class _Generated:
    idea: str
    program: Program
    description: str
    call1_prompt: str
    call1_response: str
    description_prompt: str
    description_response: str
    llm_time: float


class TraceAADV8_3:
    """TraceAAD V8.3: trajectory-aware MCTS over a single-parent tree.

    The class intentionally keeps the search state in :class:`SearchTree` and
    recomputes all selection values from that tree.  It does not maintain a
    population, operator credit, or a second structural parent.
    """

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: TraceAADArtifacts | None = None,
        max_sample_nums: int | None = 100,
        *,
        n_init: int = 10,
        max_depth: int = 10,
        widening_alpha: float = 0.5,
        exploration_constant: float = 0.1,
        beta: float = 1.0,
        rho: float = 0.25,
        kappa: float = 0.1,
        trajectory_window: int = 8,
        reference_temperature: float = 1.0,
        maximize: bool = True,
        context_token_limit: int | None = None,
        retry_count: int = 3,
        random_seed: int | None = None,
        operators: tuple[Operator, ...] = DEFAULT_OPERATORS,
        max_consecutive_failures: int = 20,
        # Existing experiment callers use this spelling.  It is only an alias
        # for the V8.3 stopping condition, not a second failure mechanism.
        max_consecutive_sample_failures: int | None = None,
        debug_mode: bool = False,
    ) -> None:
        if n_init < 0:
            raise ValueError("n_init must be non-negative")
        if max_sample_nums is not None and max_sample_nums < 0:
            raise ValueError("max_sample_nums must be non-negative or None")
        if max_depth < 2:
            raise ValueError("max_depth must be at least two")
        if not 0 < widening_alpha <= 1:
            raise ValueError("widening_alpha must be in (0, 1]")
        if exploration_constant < 0:
            raise ValueError("exploration_constant must be non-negative")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not 0 <= rho <= 1:
            raise ValueError("rho must be between zero and one")
        if kappa < 0:
            raise ValueError("kappa must be non-negative")
        if trajectory_window <= 0:
            raise ValueError("trajectory_window must be positive")
        if reference_temperature <= 0:
            raise ValueError("reference_temperature must be positive")
        if context_token_limit is not None and context_token_limit <= 0:
            raise ValueError("context_token_limit must be positive when provided")
        if retry_count < 0:
            raise ValueError("retry_count must be non-negative")
        if not operators:
            raise ValueError("at least one V8.3 operator is required")
        limit = (
            max_consecutive_sample_failures
            if max_consecutive_sample_failures is not None
            else max_consecutive_failures
        )
        if limit <= 0:
            raise ValueError("max_consecutive_failures must be positive")

        template = (
            copy.deepcopy(evaluation.template_program)
            if isinstance(evaluation.template_program, Program)
            else TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V8.3 requires exactly one evolvable template function")

        self._llm = llm
        self._evaluation = evaluation
        self._artifacts = profiler
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)
        self._template_program = template
        self._function_to_evolve: Function = copy.deepcopy(template.functions[0])
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._n_init = int(n_init)
        self._max_depth = int(max_depth)
        self._widening_alpha = float(widening_alpha)
        self._exploration_constant = float(exploration_constant)
        self._beta = float(beta)
        self._rho = float(rho)
        self._kappa = float(kappa)
        self._trajectory_window = int(trajectory_window)
        self._reference_temperature = float(reference_temperature)
        self._maximize = bool(maximize)
        self._context_token_limit = context_token_limit
        self._retry_count = int(retry_count)
        self._rng = random.Random(random_seed)
        self._operators = tuple(operators)
        self._max_consecutive_failures = int(limit)
        self._debug_mode = bool(debug_mode)
        self._tree = SearchTree()
        self._best_node: TreeNode | None = None
        self._tot_sample_nums = 0
        self._next_attempt = 0
        self._consecutive_failures = 0
        self._initialization_complete = False
        self._stop_reason = "not_started"
        self._attempt_records: list[AttemptRecord] = []
        self._generation_trace: dict[str, str | None] = {}
        # Keep the base LLM contract untouched.  This flag is used by the
        # platform's implementations but is harmless for scripted LLMs.
        self._llm.debug_mode = debug_mode

    @property
    def tree(self) -> SearchTree:
        return self._tree

    @property
    def best_node(self) -> TreeNode | None:
        return self._best_node

    @property
    def attempts(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._attempt_records)

    @property
    def evaluator_calls(self) -> int:
        return self._tot_sample_nums

    def run(self) -> TraceAADRunResult:
        """Initialize and search until budget or the V8.3 stop condition."""
        try:
            if not self._initialization_complete:
                self._initialize()
            if len(self._tree.root.child_ids) < self._n_init:
                if self._stop_reason == "not_started":
                    self._stop_reason = "initialization_incomplete"
                return self._result()
            self._initialization_complete = True
            while (
                self._tree.root.child_ids
                and self._has_budget()
                and self._consecutive_failures < self._max_consecutive_failures
            ):
                self._search_attempt()
            if self._consecutive_failures >= self._max_consecutive_failures:
                self._stop_reason = "consecutive_failures"
            elif not self._has_budget():
                self._stop_reason = "budget_exhausted"
            else:
                self._stop_reason = "finished"
            return self._result()
        finally:
            if self._artifacts is not None:
                self._artifacts.write_summary(
                    status="finished",
                    best_score=None if self._best_node is None else self._best_node.algorithm.fitness,
                    method_sample_count=self._tot_sample_nums,
                    n_total_nodes=len(self._tree.nodes()),
                    n_valid_nodes=len(self._tree.nodes()),
                    n_root_children=len(self._tree.root.child_ids),
                    n_edges=len(self._tree.edges()),
                    initialization_complete=self._initialization_complete,
                    stop_reason=self._stop_reason,
                )
                self._artifacts.finish()
            close = getattr(self._llm, "close", None)
            if callable(close):
                close()

    def _initialize(self) -> None:
        failures = 0
        while (
            len(self._tree.root.child_ids) < self._n_init
            and self._has_budget()
            and failures < self._max_consecutive_failures
        ):
            slot = len(self._tree.root.child_ids)
            references: tuple[tuple[str, str], ...]
            if slot == 0:
                operator = "i1"
                references = ()
            elif slot == 1:
                operator = "e1"
                references = self._initial_references(1)
            else:
                operator = "e1"
                references = self._initial_references(2)
            prompt = build_initial_prompt(
                task_description=self._task_description_str,
                target_function=self._function_to_evolve,
                operator=operator,
                references=references,
                maximize=self._maximize,
            )
            context = "initialization"
            result = self._generate(prompt, phase="init", operator=operator)
            if result is None:
                failures += 1
                trace = self._generation_trace
                self._record_attempt(
                    phase="init",
                    selection_path=(),
                    operator=operator,
                    reference_node_id=None,
                    context=context,
                    call1_prompt=prompt,
                    call1_response=trace.get("call1_response"),
                    description_prompt=trace.get("description_prompt"),
                    description_response=trace.get("description_response"),
                    status="generation_failed",
                    failure_reason="call_failed_or_parse_failed",
                )
                continue
            fitness, evaluation_time, reason = self._evaluate(result.program)
            if fitness is None:
                failures += 1
                self._record_attempt(
                    phase="init",
                    selection_path=(),
                    operator=operator,
                    reference_node_id=None,
                    context=context,
                    call1_prompt=result.call1_prompt,
                    call1_response=result.call1_response,
                    description_prompt=result.description_prompt,
                    description_response=result.description_response,
                    status="eval_failed",
                    failure_reason=reason,
                    evaluation_time=evaluation_time,
                )
                continue
            node = self._tree.add_initial(
                AlgorithmRecord(
                    design_idea=result.idea,
                    code=str(result.program),
                    description=result.description,
                    fitness=fitness,
                    evaluation_time=evaluation_time,
                ),
                creation_order=self._tot_sample_nums,
            )
            best_updated = self._update_best(node)
            failures = 0
            self._record_attempt(
                phase="init",
                selection_path=(),
                operator=operator,
                reference_node_id=None,
                context=context,
                call1_prompt=result.call1_prompt,
                call1_response=result.call1_response,
                description_prompt=result.description_prompt,
                description_response=result.description_response,
                status="node_created",
                fitness=fitness,
                evaluation_time=evaluation_time,
                node_id=node.id,
                global_best_updated=best_updated,
                global_best_fitness=self._best_node.algorithm.fitness,
            )
        self._initialization_complete = len(self._tree.root.child_ids) >= self._n_init
        if not self._initialization_complete:
            if not self._has_budget():
                self._stop_reason = "budget_exhausted_during_initialization"
            elif failures >= self._max_consecutive_failures:
                self._stop_reason = "consecutive_failures_during_initialization"

    def _search_attempt(self) -> None:
        selection = select_expansion_node(
            self._tree,
            maximize=self._maximize,
            total_budget=self._max_sample_nums,
            used_budget=self._tot_sample_nums,
            exploration_constant=self._exploration_constant,
            beta=self._beta,
            rho=self._rho,
            kappa=self._kappa,
            window=self._trajectory_window,
            rng=self._rng,
            max_depth=self._max_depth,
            widening_alpha=self._widening_alpha,
        )
        node = self._tree.get_node(selection.node_id)
        path = self._tree.record_selection(node.id)
        candidates = reference_candidates(self._tree, node.id)
        operators = tuple(
            operator
            for operator in self._operators
            if operator.name != OperatorName.CROSSOVER or candidates
        )
        if not operators:
            raise RuntimeError("no valid V8.3 operator is available")
        operator = self._rng.choice(operators)
        reference_node = (
            sample_reference(
                candidates,
                maximize=self._maximize,
                temperature=self._reference_temperature,
                rng=self._rng,
            )
            if operator.name == OperatorName.CROSSOVER
            else None
        )
        context = self._build_context(node, reference_node, operator)
        if context is None:
            self._failed_attempt(
                selection,
                path,
                node,
                operator,
                reference_node,
                status="context_failed",
                failure_reason="mandatory_context_exceeds_limit",
            )
            return
        result = self._generate(context[0], phase="search", operator=operator.name)
        if result is None:
            trace = self._generation_trace
            self._failed_attempt(
                selection,
                path,
                node,
                operator,
                reference_node,
                status="generation_failed",
                failure_reason="call_failed_or_parse_failed",
                context=context[1],
                call1_prompt=context[0],
                call1_response=trace.get("call1_response"),
                description_prompt=trace.get("description_prompt"),
                description_response=trace.get("description_response"),
            )
            return
        fitness, evaluation_time, reason = self._evaluate(result.program)
        if fitness is None:
            self._failed_attempt(
                selection,
                path,
                node,
                operator,
                reference_node,
                status="eval_failed",
                failure_reason=reason,
                context=context[1],
                call1_prompt=result.call1_prompt,
                call1_response=result.call1_response,
                description_prompt=result.description_prompt,
                description_response=result.description_response,
                evaluation_time=evaluation_time,
            )
            return
        child, edge = self._tree.add_child(
            parent_id=node.id,
            algorithm=AlgorithmRecord(
                design_idea=result.idea,
                code=str(result.program),
                description=result.description,
                fitness=fitness,
                evaluation_time=evaluation_time,
            ),
            operator=operator.name,
            reference_node_id=None if reference_node is None else reference_node.id,
            creation_order=self._tot_sample_nums,
        )
        # Freeze the direct edge outcome before any later descendants or
        # global-rank changes can alter the expansion arm's historical credit.
        edge.parent_quality = node_fitness(self._tree, node, self._maximize)
        edge.child_quality = node_fitness(self._tree, child, self._maximize)
        best_updated = self._update_best(child)
        self._consecutive_failures = 0
        self._record_attempt(
            selection=selection,
            phase="search",
            selected_node_id=node.id,
            selection_path=path,
            operator=operator.name,
            reference_node_id=None if reference_node is None else reference_node.id,
            context=context[1],
            call1_prompt=result.call1_prompt,
            call1_response=result.call1_response,
            description_prompt=result.description_prompt,
            description_response=result.description_response,
            status="node_created",
            fitness=fitness,
            evaluation_time=evaluation_time,
            node_id=child.id,
            global_best_updated=best_updated,
            global_best_fitness=self._best_node.algorithm.fitness,
        )
        if self._artifacts is not None:
            self._artifacts.record_edge(
                parent_id=edge.parent_id,
                child_id=edge.child_id,
                operator=str(edge.operator),
                reference_node_id=edge.reference_node_id,
                sample_order=self._tot_sample_nums,
            )

    def _failed_attempt(
        self,
        selection: SelectionResult,
        path: tuple[int, ...],
        node: TreeNode,
        operator: Operator,
        reference_node: TreeNode | None,
        *,
        status: str,
        failure_reason: str,
        context: str = "",
        call1_prompt: str = "",
        call1_response: str | None = None,
        description_prompt: str | None = None,
        description_response: str | None = None,
        evaluation_time: float | None = None,
    ) -> None:
        self._consecutive_failures += 1
        self._record_attempt(
            selection=selection,
            phase="search",
            selected_node_id=node.id,
            selection_path=path,
            operator=operator.name,
            reference_node_id=None if reference_node is None else reference_node.id,
            context=context,
            call1_prompt=call1_prompt,
            call1_response=call1_response,
            description_prompt=description_prompt,
            description_response=description_response,
            status=status,
            failure_reason=failure_reason,
            evaluation_time=evaluation_time,
        )

    def _generate(self, prompt: str, *, phase: str, operator: str | OperatorName) -> _Generated | None:
        started = time.monotonic()
        self._generation_trace = {
            "call1_prompt": prompt,
            "call1_response": None,
            "description_prompt": None,
            "description_response": None,
        }
        call1_response: str | None = None
        parsed: ParsedCall1 | None = None
        for retry in range(self._retry_count + 1):
            try:
                call1_response = str(self._llm.draw_sample(prompt))
            except Exception as exc:
                self._record_llm_call(phase, operator, prompt, None, "request_failed", exc)
                continue
            self._generation_trace["call1_response"] = call1_response
            parsed = parse_call1(call1_response, self._template_program, self._function_to_evolve.name)
            self._record_llm_call(
                phase,
                operator,
                prompt,
                call1_response,
                "ok" if parsed is not None else "parse_failed",
                None,
            )
            if parsed is not None:
                break
        if parsed is None or call1_response is None:
            return None

        description_prompt = build_description_prompt(
            task_description=self._task_description_str,
            design_idea=parsed.design_idea,
            code=str(parsed.program),
        )
        self._generation_trace["description_prompt"] = description_prompt
        description_response: str | None = None
        description: str | None = None
        for _retry in range(self._retry_count + 1):
            try:
                description_response = str(self._llm.draw_sample(description_prompt))
            except Exception as exc:
                self._record_llm_call(phase, "description", description_prompt, None, "request_failed", exc)
                continue
            self._generation_trace["description_response"] = description_response
            description = parse_description(description_response)
            self._record_llm_call(
                phase,
                "description",
                description_prompt,
                description_response,
                "ok" if description is not None else "parse_failed",
                None,
            )
            if description is not None:
                break
        if description is None or description_response is None:
            return None
        return _Generated(
            idea=parsed.design_idea,
            program=parsed.program,
            description=description,
            call1_prompt=prompt,
            call1_response=call1_response,
            description_prompt=description_prompt,
            description_response=description_response,
            llm_time=time.monotonic() - started,
        )

    def _evaluate(self, program: Program) -> tuple[float | None, float, str | None]:
        if not self._has_budget():
            return None, 0.0, "evaluator_budget_exhausted"
        outcome, evaluation_time = self._evaluator.evaluate_program_record_time_with_details(program)
        # An evaluator invocation counts immediately, including a failed result.
        self._tot_sample_nums += 1
        result = outcome.result
        raw = getattr(result, "fitness", result)
        try:
            fitness = float(raw)
        except (TypeError, ValueError):
            fitness = None
        if fitness is None or not math.isfinite(fitness):
            reason = outcome.failure_kind or "non_finite_fitness"
            if self._artifacts is not None:
                self._artifacts.record_candidate(
                    sample_order=self._tot_sample_nums,
                    score=None,
                    status="eval_failed",
                    evaluate_time=evaluation_time,
                    failure_kind=reason,
                )
            return None, evaluation_time, reason
        if self._artifacts is not None:
            self._artifacts.record_candidate(
                sample_order=self._tot_sample_nums,
                score=fitness,
                status="ok",
                evaluate_time=evaluation_time,
                program=str(program),
            )
        return fitness, evaluation_time, None

    def _build_context(
        self,
        node: TreeNode,
        reference: TreeNode | None,
        operator: Operator,
    ) -> tuple[str, str] | None:
        # Try the full baseline first, then remove older history and the least
        # important direct branches until the mandatory current code fits.
        # Keep at least the newest formation step and the best direct branch
        # whenever any non-mandatory context can fit.
        variants = [(formation, branches) for formation in (3, 2, 1) for branches in (3, 2, 1)]
        variants.extend((0, branches) for branches in (3, 2, 1, 0))
        for formation_limit, branch_limit in variants:
                local: LocalContext = build_local_context(
                    self._tree,
                    node,
                    maximize=self._maximize,
                    max_formation_edges=formation_limit,
                    max_direct_children=branch_limit,
                )
                reference_payload = None
                if reference is not None:
                    reference_payload = (
                        reference.algorithm.design_idea,
                        reference.algorithm.description,
                        reference.algorithm.code,
                        reference.algorithm.fitness,
                    )
                prompt = build_search_prompt(
                    task_description=self._task_description_str,
                    current_code=node.algorithm.code,
                    current_description=node.algorithm.description,
                    current_fitness=node.algorithm.fitness,
                    history=local.text,
                    operator_name=str(operator.name),
                    operator_instruction=operator.instruction,
                    target_function=self._function_to_evolve,
                    reference=reference_payload,
                    maximize=self._maximize,
                )
                if self._context_token_limit is None or self._count_tokens(prompt) <= self._context_token_limit:
                    return prompt, local.text
        return None

    def _initial_references(self, count: int) -> tuple[tuple[str, str], ...]:
        nodes = list(self._tree.nodes())
        if count > len(nodes):
            count = len(nodes)
        chosen = self._rng.sample(nodes, count)
        return tuple((node.algorithm.design_idea, node.algorithm.code) for node in chosen)

    def _update_best(self, node: TreeNode) -> bool:
        if self._best_node is None:
            self._best_node = node
            return True
        current = self._best_node.algorithm.fitness
        candidate = node.algorithm.fitness
        if (self._maximize and candidate > current) or (not self._maximize and candidate < current):
            self._best_node = node
            return True
        return False

    def _has_budget(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _count_tokens(self, prompt: str) -> int:
        counter = getattr(self._llm, "count_tokens", None)
        if callable(counter):
            return int(counter(prompt))
        return len(prompt.encode("utf-8"))

    def _record_llm_call(
        self,
        phase: str,
        operator: str | OperatorName,
        prompt: str,
        response: str | None,
        status: str,
        exc: Exception | None,
    ) -> None:
        if self._artifacts is None:
            return
        self._artifacts.record_llm_call(
            stage=phase,
            operator=str(operator),
            sample_order=self._tot_sample_nums + 1,
            prompt_tokens=self._count_tokens(prompt),
            response_tokens=None if response is None else self._count_tokens(response),
            status=status,
            response=response,
            error=None if exc is None else str(exc),
        )

    def _record_attempt(self, *, selection: SelectionResult | None = None, **values: Any) -> None:
        self._next_attempt += 1
        if selection is not None and "selection_steps" not in values:
            values["selection_steps"] = selection.steps
        record = AttemptRecord(attempt=self._next_attempt, **values)
        self._attempt_records.append(record)
        if self._artifacts is not None:
            self._artifacts.record_decision(
                "attempt",
                attempt=record.attempt,
                phase=record.phase,
                selected_node_id=record.selected_node_id,
                selection_path=record.selection_path,
                operator=record.operator,
                reference_node_id=record.reference_node_id,
                status=record.status,
                failure_reason=record.failure_reason,
                fitness=record.fitness,
                node_id=record.node_id,
                selection_steps=[] if selection is None else [{
                    "decision_node_id": step.decision_node_id,
                    "option": step.option,
                    "target_node_id": step.target_node_id,
                    "score": step.score,
                    "quality": step.quality,
                    "exploration": step.exploration,
                } for step in selection.steps],
                global_best_updated=record.global_best_updated,
                global_best_fitness=record.global_best_fitness,
            )

    def _result(self) -> TraceAADRunResult:
        count = len(self._tree.nodes())
        return TraceAADRunResult(
            best_node=self._best_node,
            n_total_nodes=count,
            n_valid_nodes=count,
            n_root_children=len(self._tree.root.child_ids),
            n_edges=len(self._tree.edges()),
            n_samples=self._tot_sample_nums,
            n_attempts=len(self._attempt_records),
            initialization_complete=self._initialization_complete,
            stop_reason=self._stop_reason,
        )


# Compatibility spelling for callers that use ``V83`` rather than ``V8_3``.
TraceAADV83 = TraceAADV8_3


__all__ = ["AttemptRecord", "TraceAADRunResult", "TraceAADV8_3", "TraceAADV83"]
