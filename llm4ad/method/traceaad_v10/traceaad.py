"""TraceAAD V10 core search: joint design opportunity allocation.

Each primary slot is one design experiment ``a = (s, o, r)``.  Before the
slot is consumed, the archive is screened to ``K_s`` starts, the opportunity
set is built (Develop / Pivot / Transfer / Restart / SemanticRepair), a
valuation critic names the plausibly optimal subset, coverage allocates
within that subset, and only then does conditioned generation run.  Threads
record each direction's declared idea and budget use, yielding the delayed
payoff statistics ``G_h`` over opening actions.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .critic import (
    build_critic_prompt,
    critic_char_budget,
    edge_outcome,
    fallback_result,
    format_fitness,
    parse_critic_response,
)
from .opportunity import (
    build_opportunities,
    coverage_tuple,
    mid_rank,
    operator_observations,
    screen_shortlist,
    select_by_coverage,
    start_use_counts,
)
from .prompt import (
    build_generation_prompt,
    build_repair_prompt,
    build_root_prompt,
    extract_idea,
    format_failure,
    parse_candidate_response,
)
from .schema import (
    FORMATION_WINDOW,
    G_HORIZONS,
    INITIAL_ROOT_COUNT,
    MAX_REPAIRS,
    OPENING_OPERATORS,
    OPERATORS,
    RESTART,
    RESTART_CARDS,
    SCREEN_SIZE,
    REFERENCE_COUNT,
    COMPETITIVE_SET_SIZE,
    AttemptRecord,
    Pending,
    ProgramNode,
    Thread,
)

ERROR_MAX_CHARS = 500
BEST_AT_MARKS = (100, 250, 500, 750, 1000)


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    code: str
    fitness: float | None
    outcome: str
    node_id: int | None
    created_thread: int | None
    error: str | None
    error_type: str | None
    attempt: int


class TraceAADV10:
    """Trajectory-aware joint design opportunity allocation."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
        budget: int = 1000,
        *,
        n_roots: int = INITIAL_ROOT_COUNT,
        max_tokens: int = 8192,
        seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        task_key: str | None = None,
        context_token_limit: int = 32768,
    ) -> None:
        if budget <= 0 or n_roots <= 0 or max_tokens <= 0 or n_roots > budget:
            raise ValueError("budget, n_roots, and max_tokens must be positive; n_roots <= budget")
        if context_token_limit <= max_tokens:
            raise ValueError("context_token_limit must exceed max_tokens")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V10 requires one evolvable template function")
        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation)
        self._log = artifacts
        self._task = evaluation.task_description
        self._task_key = task_key or "unknown"
        self._function: Function = template.functions[0]
        self._budget = int(budget)
        self._n_roots = int(n_roots)
        self._max_tokens = int(max_tokens)
        self._context_token_limit = int(context_token_limit)
        self._critic_char_budget = critic_char_budget(self._context_token_limit, self._max_tokens)
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

        self._nodes: dict[int, ProgramNode] = {}
        self._threads: dict[int, Thread] = {}
        self._attempts: list[AttemptRecord] = []
        self._global_memory: list[int] = []
        self._pending: Pending | None = None
        self._slot_best: list[float | None] = []
        self._n_eval = 0  # primary evaluator slots
        self._n_calls = 0  # includes bounded repair evaluations
        self._round_index = 0
        self._gen_llm_calls = 0
        self._critic_llm_calls = 0
        self._critic_invalid = 0
        self._repair_llm_calls = 0
        self._repair_eval_calls = 0
        self._draw_index = 0
        self._llm_tokens = {
            "generation_prompt": 0,
            "generation_completion": 0,
            "critic_prompt": 0,
            "critic_completion": 0,
            "repair_prompt": 0,
            "repair_completion": 0,
        }
        self._best_node_id: int | None = None
        self._rng = random.Random(seed)
        self._next_node = 1
        self._next_thread = 1
        self._token_accounting_failures = 0

        if resume_from is not None:
            load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = Path(resume_from).parent
            self._next_ids()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    @property
    def best(self) -> ProgramNode | None:
        return None if self._best_node_id is None else self._nodes[self._best_node_id]

    def run(self) -> None:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            if self._pending is not None:
                outcome = self._settle_pending()
                self._finish_attempt(outcome)
            while len(self._root_nodes()) < self._n_roots and self._has_budget():
                self._make_root()
            if len(self._root_nodes()) < self._n_roots:
                status = "initialization_failure"
                stop_reason = "evaluator_budget_exhausted_during_initialization"
                return
            while self._has_budget():
                before = self._n_eval
                self._step()
                if self._n_eval == before:
                    raise RuntimeError("V10 step made no primary evaluator progress")
            status = "finished"
            stop_reason = "evaluator_budget_exhausted"
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            raise
        finally:
            save_checkpoint(self)
            if self._log is not None:
                best = self.best
                self._log.write_summary(
                    status=status,
                    stop_reason=stop_reason,
                    best_node_id=None if best is None else best.id,
                    best_score=None if best is None else best.fitness,
                    task=self._task_key,
                    budget_slots=self._n_eval,
                    evaluator_call_count=self._n_calls,
                    generation_llm_calls=self._gen_llm_calls,
                    critic_llm_calls=self._critic_llm_calls,
                    critic_invalid=self._critic_invalid,
                    repair_llm_calls=self._repair_llm_calls,
                    repair_evaluator_calls=self._repair_eval_calls,
                    **{f"{key}_tokens": value for key, value in self._llm_tokens.items()},
                    n_nodes=len(self._nodes),
                    n_threads=len(self._threads),
                    n_roots=len(self._root_nodes()),
                    mechanism="trajectory_aware_joint_design_opportunity_allocation",
                    allocation="critic_competitive_set_plus_lexicographic_coverage",
                    critic_schedule="once_per_primary_slot_after_initialization",
                    token_accounting=self._token_accounting_mode(),
                    token_accounting_failures=self._token_accounting_failures,
                    constants={
                        "K_s": SCREEN_SIZE,
                        "K_d": REFERENCE_COUNT,
                        "K_c": COMPETITIVE_SET_SIZE,
                        "H_tau": FORMATION_WINDOW,
                        "H_G": list(G_HORIZONS),
                        "N_root": self._n_roots,
                        "N_card": RESTART_CARDS,
                        "max_repairs": MAX_REPAIRS,
                    },
                    operator_counts=self._operator_counts(),
                    operator_observations=operator_observations(self._attempts, self._threads),
                    best_at=self._best_at(),
                    **error,
                )
                self._log.finish()

    def _root_nodes(self) -> list[ProgramNode]:
        return [node for node in self._nodes.values() if node.parent_id is None]

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _next_ids(self) -> None:
        self._next_node = max(self._nodes, default=0) + 1
        self._next_thread = max(self._threads, default=0) + 1

    def _operator_counts(self) -> dict[str, int]:
        counts = {operator: 0 for operator in OPERATORS}
        for attempt in self._attempts:
            if attempt.operator in counts:
                counts[attempt.operator] += 1
        return counts

    def _best_at(self) -> dict[int, float]:
        marks: dict[int, float] = {}
        for mark in BEST_AT_MARKS:
            if len(self._slot_best) >= mark:
                window = self._slot_best[:mark]
                values = [value for value in window if value is not None]
                if values:
                    marks[mark] = max(values)
        return marks

    # ------------------------------------------------------------------
    # One design experiment per primary slot
    # ------------------------------------------------------------------

    def _step(self) -> None:
        slot = self._reserve_slot()
        self._round_index += 1
        shortlist = screen_shortlist(
            self._nodes, self._attempts, self._best_node_id, rng=self._rng
        )
        opportunities, references = build_opportunities(self._nodes, shortlist)
        self._log_round_start(shortlist, references, opportunities)
        critic = self._run_critic(slot, shortlist, opportunities, references)
        chosen = select_by_coverage(critic.entries, self._attempts, self._nodes, self._rng)
        self._log_allocation(critic, chosen)
        self._generate(chosen, slot)
        outcome = self._settle_pending()
        self._finish_attempt(outcome)

    def _make_root(self) -> None:
        prompt = build_root_prompt(
            task_description=self._task, template_function=self._function
        )
        slot = self._reserve_slot()
        response = self._draw("generation", prompt)
        root_idea = extract_idea(response) or "initial algorithm"
        if self._log is not None:
            self._log.record_decision(
                {
                    "stage": "root",
                    "round": 0,
                    "slot": slot,
                    "prompt": prompt,
                    "response": response,
                    "idea": root_idea,
                }
            )
        self._pending = Pending(
            prompt=prompt,
            response=response,
            operator="root",
            idea=root_idea,
            slot=slot,
            round_index=0,
            start_id=None,
            reference_id=None,
            base_code="",
            start_fitness=None,
            q_origin=None,
            semantic_mismatch=None,
            opportunity_id="O0",
            critic_rank=None,
        )
        save_checkpoint(self)
        outcome = self._settle_pending()
        self._finish_attempt(outcome)

    def _run_critic(self, slot, shortlist, opportunities, references):
        prompt_data = build_critic_prompt(
            task_description=self._task,
            slot=slot,
            remaining_budget=self._budget - self._n_eval + 1,
            primary_evaluations=self._n_eval - 1,
            best_node=self.best,
            nodes=self._nodes,
            threads=self._threads,
            attempts=self._attempts,
            shortlist=shortlist,
            opportunities=opportunities,
            references=references,
            char_budget=self._critic_char_budget,
        )
        result = None
        response = ""
        for _ in range(2):
            response = self._draw("critic", prompt_data.prompt)
            result = parse_critic_response(response, opportunities, prompt_data.valid_labels)
            if result is not None:
                break
        if result is None:
            self._critic_invalid += 1
            result = fallback_result(self._nodes, shortlist, opportunities)
        if self._log is not None:
            self._log.record_decision(
                {
                    "stage": "critic",
                    "round": self._round_index,
                    "slot": slot,
                    "prompt": prompt_data.prompt,
                    "prompt_chars": len(prompt_data.prompt),
                    "prompt_char_budget": prompt_data.char_budget,
                    "prompt_code_clipped": prompt_data.clipped,
                    "dropped_reference_codes": list(prompt_data.dropped_reference_codes),
                    "response": response,
                    "invalid": result.invalid,
                    "competitive_set": [
                        {
                            "opportunity_id": entry.opportunity.opportunity_id,
                            "operator": entry.opportunity.operator,
                            "start_id": entry.opportunity.start_id,
                            "reference_id": entry.opportunity.reference_id,
                            "rank": entry.rank,
                            "reason": entry.reason,
                            "evidence_refs": list(entry.evidence_refs),
                            "expected_payoff_horizon": entry.payoff_horizon,
                            "semantic_mismatch": entry.semantic_mismatch,
                        }
                        for entry in result.entries
                    ],
                    "not_applicable": [
                        {"opportunity_id": oid, "reason": reason}
                        for oid, reason in result.not_applicable
                    ],
                }
            )
        return result

    def _generate(self, chosen, slot: int) -> None:
        opportunity = chosen.opportunity
        if opportunity.operator == RESTART:
            cards = self._restart_cards()
            prompt = build_generation_prompt(
                task_description=self._task,
                operator=opportunity.operator,
                nodes=self._nodes,
                start_id=None,
                restart_cards=cards,
                template_function=self._function,
            )
            best = self.best
            q_origin = None if best is None else best.fitness
            start_fitness = None
            base_code = ""
        else:
            start = self._nodes[opportunity.start_id]
            prompt = build_generation_prompt(
                task_description=self._task,
                operator=opportunity.operator,
                nodes=self._nodes,
                start_id=opportunity.start_id,
                reference_id=opportunity.reference_id,
                semantic_mismatch=(
                    chosen.semantic_mismatch
                    if opportunity.operator == "semantic_repair"
                    else None
                ),
            )
            q_origin = None
            start_fitness = start.fitness
            base_code = start.code
        response = self._draw("generation", prompt)
        idea = extract_idea(response) or f"{opportunity.operator} attempt at slot {slot}"
        if self._log is not None:
            self._log.record_decision(
                {
                    "stage": "generation",
                    "round": self._round_index,
                    "slot": slot,
                    "opportunity_id": opportunity.opportunity_id,
                    "operator": opportunity.operator,
                    "start_id": opportunity.start_id,
                    "reference_id": opportunity.reference_id,
                    "critic_rank": chosen.rank,
                    "prompt": prompt,
                    "response": response,
                    "idea": idea,
                }
            )
        self._pending = Pending(
            prompt=prompt,
            response=response,
            operator=opportunity.operator,
            idea=idea,
            slot=slot,
            round_index=self._round_index,
            start_id=opportunity.start_id,
            reference_id=opportunity.reference_id,
            base_code=base_code,
            start_fitness=start_fitness,
            q_origin=q_origin,
            semantic_mismatch=chosen.semantic_mismatch,
            opportunity_id=opportunity.opportunity_id,
            critic_rank=chosen.rank,
        )
        save_checkpoint(self)

    def _restart_cards(self) -> list[str]:
        cards: list[str] = []
        for node_id in self._global_memory[-RESTART_CARDS:]:
            node = self._nodes.get(node_id)
            if node is None or node.parent_id is None:
                continue
            parent = self._nodes.get(node.parent_id)
            if parent is None:
                continue
            cards.append(
                f"Idea: {node.idea or 'unavailable'}; result: improve; "
                f"fitness {format_fitness(parent.fitness)} -> {format_fitness(node.fitness)}"
            )
        return cards

    # ------------------------------------------------------------------
    # Candidate settlement and bounded execution repair
    # ------------------------------------------------------------------

    def _reserve_slot(self) -> int:
        if not self._has_budget():
            raise RuntimeError("primary evaluator budget exhausted")
        self._n_eval += 1
        return self._n_eval

    def _draw(self, kind: str, prompt: str) -> str:
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if self._seed is not None:
            kwargs["seed"] = self._seed + self._draw_index
        self._draw_index += 1
        if kind == "critic":
            self._critic_llm_calls += 1
        elif kind == "repair":
            self._repair_llm_calls += 1
        else:
            self._gen_llm_calls += 1
        response = str(self._llm.draw_sample(prompt, **kwargs))
        self._count_tokens(kind, prompt, response)
        return response

    def _count_tokens(self, kind: str, prompt: str, response: str) -> None:
        # Token accounting is telemetry, not search: a failing tokenizer
        # endpoint disables the counters visibly instead of killing the run.
        if self._token_accounting_failures:
            return
        count_prompt = getattr(self._llm, "count_prompt_tokens", None)
        count_text = getattr(self._llm, "count_tokens", None)
        if not callable(count_prompt) or not callable(count_text):
            return
        try:
            prompt_tokens = int(count_prompt(prompt))
            completion_tokens = int(count_text(response))
        except Exception as exc:  # noqa: BLE001 - telemetry must not stop search
            self._token_accounting_failures += 1
            if self._log is not None:
                self._log.record_event(
                    "token_accounting_disabled",
                    round=self._round_index,
                    error_type=type(exc).__name__,
                    error=str(exc)[:500],
                )
            return
        self._llm_tokens[f"{kind}_prompt"] += prompt_tokens
        self._llm_tokens[f"{kind}_completion"] += completion_tokens

    def _token_accounting_mode(self) -> str:
        if self._token_accounting_failures:
            return "disabled_after_failure"
        count_prompt = getattr(self._llm, "count_prompt_tokens", None)
        count_text = getattr(self._llm, "count_tokens", None)
        if callable(count_prompt) and callable(count_text):
            return "llm_count_tokens"
        return "unavailable"

    def _settle_pending(self) -> CandidateOutcome:
        pending = self._pending
        if pending is None:
            raise RuntimeError("cannot settle without a pending candidate")
        current = pending
        for attempt in range(current.attempt, MAX_REPAIRS + 2):
            parsed = parse_candidate_response(current.response)
            started = time.perf_counter()
            result = self._evaluator.evaluate_program_with_details(parsed.code)
            elapsed = time.perf_counter() - started
            self._n_calls += 1
            if attempt > 1:
                self._repair_eval_calls += 1

            failure_kind = getattr(result, "failure_kind", None)
            raw = getattr(result, "result", None)
            raw_fitness = getattr(raw, "fitness", raw)
            try:
                fitness = float(raw_fitness)
            except (TypeError, ValueError, OverflowError):
                fitness = math.nan

            if failure_kind is None and math.isfinite(fitness):
                if any(node.code == parsed.code for node in self._nodes.values()):
                    final = CandidateOutcome(
                        code=parsed.code,
                        fitness=fitness,
                        outcome="duplicate",
                        node_id=None,
                        created_thread=None,
                        error=None,
                        error_type=None,
                        attempt=attempt,
                    )
                    self._consume_start_thread_slot(current, joined_fitness=None)
                    self._record_eval(
                        current,
                        elapsed=elapsed,
                        outcome_name=final.outcome,
                        fitness=fitness,
                        node_id=None,
                        created_thread=None,
                        attempt=attempt,
                        error_type=None,
                        error=None,
                    )
                    return final
                node, created_thread = self._register_valid_child(current, parsed.code, fitness)
                final = CandidateOutcome(
                    code=parsed.code,
                    fitness=fitness,
                    outcome=self._attempt_outcome(current, fitness),
                    node_id=node.id,
                    created_thread=created_thread,
                    error=None,
                    error_type=None,
                    attempt=attempt,
                )
                self._record_eval(
                    current,
                    elapsed=elapsed,
                    outcome_name=final.outcome,
                    fitness=fitness,
                    node_id=node.id,
                    created_thread=created_thread,
                    attempt=attempt,
                    error_type=None,
                    error=None,
                )
                return final

            if failure_kind is not None:
                outcome_name = "timeout" if failure_kind == "timeout" else "invalid"
                error_type = getattr(result, "error_type", None)
                message = _short_error(
                    getattr(result, "error", None) or f"evaluation failed: {failure_kind}"
                )
                repair_error = format_failure(
                    error_type, message, getattr(result, "traceback", None)
                )
            else:
                outcome_name, error_type = "invalid", "invalid_result"
                message = _short_error(
                    getattr(result, "error", None)
                    or f"evaluator returned invalid fitness: {raw_fitness!r}"
                )
                repair_error = message
            self._record_eval(
                current,
                elapsed=elapsed,
                outcome_name=outcome_name,
                fitness=None,
                node_id=None,
                created_thread=None,
                attempt=attempt,
                error_type=error_type,
                error=message,
            )
            if attempt <= MAX_REPAIRS:
                repair_prompt = build_repair_prompt(
                    task_description=self._task,
                    idea=current.idea,
                    base_code=current.base_code,
                    failed_code=parsed.code,
                    error=repair_error,
                )
                current = replace(
                    current,
                    prompt=repair_prompt,
                    response=self._draw("repair", repair_prompt),
                    attempt=attempt + 1,
                )
                self._pending = current
                save_checkpoint(self)
                continue
            final = CandidateOutcome(
                code=parsed.code,
                fitness=None,
                outcome=outcome_name,
                node_id=None,
                created_thread=None,
                error=message,
                error_type=error_type,
                attempt=attempt,
            )
            self._consume_start_thread_slot(current, joined_fitness=None)
            return final
        raise AssertionError("unreachable candidate settlement")

    @staticmethod
    def _attempt_outcome(pending: Pending, fitness: float) -> str:
        if pending.start_id is not None and pending.start_fitness is not None:
            return edge_outcome(pending.start_fitness, fitness)
        if pending.operator == RESTART and pending.q_origin is not None:
            # A restart is judged against the incumbent at its creation.
            return edge_outcome(pending.q_origin, fitness)
        if pending.operator == "root":
            return "new_root"
        return "plateau"

    def _register_valid_child(
        self, pending: Pending, code: str, fitness: float
    ) -> tuple[ProgramNode, int | None]:
        if pending.operator == "root":
            thread_id = self._next_thread
            node = self._new_node(code, fitness, None, thread_id, pending.idea, pending.slot)
            self._next_thread += 1
            thread = Thread(
                id=thread_id,
                origin_action="init",
                origin_idea=pending.idea,
                origin_slot=pending.slot,
                created_node_id=node.id,
                q_origin=None,
                opportunities_used=1,
                best_history=[fitness],
            )
            self._threads[thread.id] = thread
            return node, thread.id

        if pending.operator == RESTART:
            # No start state and no parent: the thread asks whether a fresh
            # hypothesis can beat the incumbent frozen at allocation.
            thread_id = self._next_thread
            node = self._new_node(code, fitness, None, thread_id, pending.idea, pending.slot)
            self._next_thread += 1
            thread = Thread(
                id=thread_id,
                origin_action=RESTART,
                origin_idea=pending.idea,
                origin_slot=pending.slot,
                created_node_id=node.id,
                q_origin=pending.q_origin,
                opportunities_used=1,
                best_history=[fitness],
            )
            self._threads[thread.id] = thread
            return node, thread.id

        start = self._nodes[pending.start_id]
        if pending.operator in OPENING_OPERATORS:
            self._consume_start_thread_slot(pending, joined_fitness=None)
            thread_id = self._next_thread
            node = self._new_node(
                code, fitness, start.id, thread_id, pending.idea, pending.slot
            )
            self._next_thread += 1
            thread = Thread(
                id=thread_id,
                origin_action=pending.operator,
                origin_idea=pending.idea,
                origin_slot=pending.slot,
                created_node_id=node.id,
                q_origin=pending.start_fitness,
                opportunities_used=1,
                best_history=[fitness],
            )
            self._threads[thread.id] = thread
            return node, thread.id

        self._consume_start_thread_slot(pending, joined_fitness=fitness)
        node = self._new_node(
            code, fitness, start.id, start.thread_id, pending.idea, pending.slot
        )
        return node, None

    def _consume_start_thread_slot(self, pending: Pending, *, joined_fitness: float | None) -> None:
        """The start's thread consumed one primary slot, success or failure.

        Opening actions send their valid child to a new thread, so the start's
        thread keeps its best quality unchanged; continuation children join
        and can raise it.
        """
        if pending.start_id is None:
            return
        thread = self._threads[self._nodes[pending.start_id].thread_id]
        thread.opportunities_used += 1
        best = thread.best_fitness
        if joined_fitness is not None and joined_fitness > best:
            best = joined_fitness
        thread.best_history.append(best)

    def _new_node(
        self,
        code: str,
        fitness: float,
        parent_id: int | None,
        thread_id: int,
        idea: str | None,
        slot: int,
    ) -> ProgramNode:
        node = ProgramNode(
            id=self._next_node,
            code=code,
            fitness=fitness,
            parent_id=parent_id,
            thread_id=thread_id,
            idea=idea,
            slot=slot,
        )
        self._next_node += 1
        self._nodes[node.id] = node
        if self._best_node_id is None or fitness > self._nodes[self._best_node_id].fitness:
            self._best_node_id = node.id
            if self._log is not None:
                self._log.record_best(
                    code=code,
                    fitness=fitness,
                    slot=slot,
                    node_id=node.id,
                    idea=idea,
                    thread_id=thread_id,
                )
        if parent_id is not None and parent_id in self._nodes:
            parent = self._nodes[parent_id]
            if fitness > parent.fitness and node.id not in self._global_memory:
                self._global_memory.append(node.id)
                self._global_memory = self._global_memory[-64:]
                if self._log is not None:
                    self._log.record_memory(
                        {
                            "node_id": node.id,
                            "parent_id": parent_id,
                            "fitness_before": parent.fitness,
                            "fitness_after": fitness,
                            "idea": idea,
                            "slot": slot,
                        }
                    )
        return node

    def _finish_attempt(self, outcome: CandidateOutcome) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("cannot finish an attempt without a pending candidate")
        thread_of_start = (
            None
            if pending.start_id is None
            else self._nodes[pending.start_id].thread_id
        )
        self._attempts.append(
            AttemptRecord(
                slot=pending.slot,
                round_index=pending.round_index,
                operator=pending.operator,
                idea=pending.idea,
                outcome=outcome.outcome,
                start_id=pending.start_id,
                start_fitness=pending.start_fitness,
                child_id=outcome.node_id,
                child_fitness=outcome.fitness,
                thread_of_start=thread_of_start,
                created_thread=outcome.created_thread,
                reference_id=pending.reference_id,
                q_origin=pending.q_origin,
                error=outcome.error,
            )
        )
        best = self.best
        self._slot_best.append(None if best is None else best.fitness)
        if self._log is not None:
            self._log.record_event(
                "settle",
                round=pending.round_index,
                slot=pending.slot,
                operator=pending.operator,
                opportunity_id=pending.opportunity_id,
                critic_rank=pending.critic_rank,
                start_id=pending.start_id,
                reference_id=pending.reference_id,
                outcome=outcome.outcome,
                fitness=outcome.fitness,
                node_id=outcome.node_id,
                created_thread=outcome.created_thread,
                thread_of_start=thread_of_start,
                attempt=outcome.attempt,
                error=outcome.error,
            )
            if thread_of_start is not None:
                self._log_thread(self._threads[thread_of_start])
            if outcome.created_thread is not None:
                self._log_thread(self._threads[outcome.created_thread])
        self._pending = None
        save_checkpoint(self)

    def _record_eval(
        self,
        pending: Pending,
        *,
        elapsed: float,
        outcome_name: str,
        fitness: float | None,
        node_id: int | None,
        created_thread: int | None,
        attempt: int,
        error_type: str | None,
        error: str | None,
    ) -> None:
        if self._log is None:
            return
        self._log.record_evaluation(
            [
                pending.slot,
                pending.round_index,
                pending.operator,
                pending.idea,
                "" if pending.start_id is None else pending.start_id,
                "" if pending.reference_id is None else pending.reference_id,
                "" if node_id is None else node_id,
                outcome_name,
                "" if fitness is None else fitness,
                "" if pending.start_fitness is None else pending.start_fitness,
                "" if pending.q_origin is None else pending.q_origin,
                "" if created_thread is None else created_thread,
                attempt,
                "initial" if attempt == 1 else "repair",
                f"{elapsed:.6f}",
                error_type or "",
                error or "",
            ]
        )

    # ------------------------------------------------------------------
    # Mechanism event logging
    # ------------------------------------------------------------------

    def _log_round_start(self, shortlist, references, opportunities) -> None:
        if self._log is None:
            return
        values = [node.fitness for node in self._nodes.values()]
        uses = start_use_counts(self._attempts)
        self._log.record_event(
            "round_start",
            round=self._round_index,
            slot=self._n_eval,
            shortlist=[
                {
                    "id": node_id,
                    "fitness": self._nodes[node_id].fitness,
                    "mid_rank": mid_rank(self._nodes[node_id].fitness, values),
                    "start_uses": uses.get(node_id, 0),
                    "thread_id": self._nodes[node_id].thread_id,
                }
                for node_id in shortlist
            ],
            references={str(key): value for key, value in references.items()},
            opportunities=[
                {
                    "opportunity_id": item.opportunity_id,
                    "operator": item.operator,
                    "start_id": item.start_id,
                    "reference_id": item.reference_id,
                }
                for item in opportunities
            ],
        )

    def _log_allocation(self, critic, chosen) -> None:
        if self._log is None:
            return
        self._log.record_event(
            "allocation",
            round=self._round_index,
            slot=self._n_eval,
            invalid=critic.invalid,
            competitive_set=[
                {
                    "opportunity_id": entry.opportunity.opportunity_id,
                    "operator": entry.opportunity.operator,
                    "start_id": entry.opportunity.start_id,
                    "reference_id": entry.opportunity.reference_id,
                    "rank": entry.rank,
                    "reason": entry.reason,
                    "coverage": list(
                        coverage_tuple(entry.opportunity, self._attempts, self._nodes)
                    ),
                }
                for entry in critic.entries
            ],
            not_applicable=[oid for oid, _ in critic.not_applicable],
            chosen={
                "opportunity_id": chosen.opportunity.opportunity_id,
                "operator": chosen.opportunity.operator,
                "start_id": chosen.opportunity.start_id,
                "reference_id": chosen.opportunity.reference_id,
                "rank": chosen.rank,
                "coverage": list(
                    coverage_tuple(chosen.opportunity, self._attempts, self._nodes)
                ),
            },
        )

    def _log_thread(self, thread: Thread) -> None:
        if self._log is None:
            return
        payload = {
            "id": thread.id,
            "origin_action": thread.origin_action,
            "origin_idea": thread.origin_idea,
            "origin_slot": thread.origin_slot,
            "created_node_id": thread.created_node_id,
            "q_origin": thread.q_origin,
            "opportunities_used": thread.opportunities_used,
            "best_history": list(thread.best_history),
        }
        for horizon in G_HORIZONS:
            payload[f"G{horizon}"] = thread.g_value(horizon)
        self._log.record_thread(payload)


def _short_error(message: Any) -> str:
    text = " ".join(str(message).split())
    return text if len(text) <= ERROR_MAX_CHARS else text[: ERROR_MAX_CHARS - 3] + "..."


__all__ = ["TraceAADV10"]
