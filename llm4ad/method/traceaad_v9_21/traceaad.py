"""TraceAAD V9.21 core search.

The unit of search is an idea hypothesis, not an individual piece of code.
For each selected hypothesis the method independently proposes one
continuation and one branch idea.  Each idea is implemented twice from the
same frozen snapshot, so evaluator feedback can distinguish an idea from a
single lucky or failed implementation.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .history import edge_outcome, render_formation_path, render_ledger, render_public_card
from .prompt import (
    build_idea_prompt,
    build_realization_prompt,
    build_repair_prompt,
    build_root_prompt,
    format_failure,
    parse_candidate_response,
    parse_idea_response,
)
from .schema import (
    BATCH_SIZE,
    INITIAL_ROOT_COUNT,
    MAX_REPAIRS,
    Pending,
    ProgramNode,
    Hypothesis,
    Realization,
    Attempt,
    RESPONSE_CLIP,
    REALIZATIONS_PER_IDEA,
)

ERROR_MAX_CHARS = 500


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    code: str
    fitness: float | None
    outcome: str
    response: float
    node_id: int | None
    error: str | None
    error_type: str | None
    attempt: int


class TraceAADV921:
    """Hypothesis search with a frozen four-candidate atomic experiment."""

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
    ) -> None:
        if budget <= 0 or n_roots <= 0 or max_tokens <= 0 or n_roots > budget:
            raise ValueError("budget, n_roots, and max_tokens must be positive; n_roots <= budget")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.21 requires one evolvable template function")
        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation)
        self._log = artifacts
        self._task = evaluation.task_description
        self._task_key = task_key or "unknown"
        self._function: Function = template.functions[0]
        self._budget = int(budget)
        self._n_roots = int(n_roots)
        self._max_tokens = int(max_tokens)
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)

        self._nodes: dict[int, ProgramNode] = {}
        self._hypotheses: dict[int, Hypothesis] = {}
        self._realizations: list[Realization] = []
        self._attempts: list[Attempt] = []
        self._global_memory: list[int] = []
        self._pending: Pending | None = None
        self._batch_context: dict[str, Any] | None = None
        self._n_eval = 0  # primary evaluator slots
        self._n_calls = 0  # includes bounded repair evaluations
        self._llm_calls = 0
        self._repair_llm_calls = 0
        self._repair_eval_calls = 0
        self._batch_index = 0
        self._root_center: float | None = None
        self._root_scale: float | None = None
        self._best_node_id: int | None = None
        self._rng = random.Random(seed)
        self._next_node = 1
        self._next_hypothesis = 1
        self._next_realization = 1

        if resume_from is not None:
            load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = Path(resume_from).parent
            self._next_ids()

    @property
    def best(self) -> ProgramNode | None:
        return None if self._best_node_id is None else self._nodes[self._best_node_id]

    @property
    def _outcome_counts(self) -> Counter[str]:
        return Counter(item.outcome for item in self._attempts)

    def run(self) -> None:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            if self._pending is not None:
                self._apply_outcome(*self._settle_pending())
            if self._batch_context is not None:
                self._resume_batch_context()

            while len(self._root_hypotheses()) < self._n_roots and self._has_budget():
                self._make_root()
            if len(self._root_hypotheses()) < self._n_roots:
                status = "initialization_failure"
                stop_reason = "evaluator_budget_exhausted_during_initialization"
                return
            if self._root_center is None:
                self._freeze_root_scale()

            while self._has_budget():
                before = self._n_eval
                self._ordinary_batch()
                if self._n_eval == before:
                    raise RuntimeError("V9.21 batch made no primary evaluator progress")
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
                    llm_call_count=self._llm_calls,
                    repair_llm_calls=self._repair_llm_calls,
                    repair_evaluator_calls=self._repair_eval_calls,
                    n_nodes=len(self._nodes),
                    n_hypotheses=len(self._hypotheses),
                    n_roots=len(self._root_hypotheses()),
                    mechanism="hypothesis_search_paired_two_ideas_two_realizations",
                    allocation="one_step_hypothesis_ucb",
                    online_behavesim=False,
                    realizations_per_idea=REALIZATIONS_PER_IDEA,
                    batch_size=BATCH_SIZE,
                    action_counts={"continue": self._count_proposals("continue"), "branch": self._count_proposals("branch")},
                    outcome_counts=dict(sorted(self._outcome_counts.items())),
                    **error,
                )
                self._log.finish()

    # ------------------------------------------------------------------
    # Search state and allocation
    # ------------------------------------------------------------------

    def _root_hypotheses(self) -> list[Hypothesis]:
        return [h for h in self._hypotheses.values() if h.parent_hypothesis_id is None]

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _next_ids(self) -> None:
        self._next_node = max(self._nodes, default=0) + 1
        self._next_hypothesis = max(self._hypotheses, default=0) + 1
        self._next_realization = max((item.id for item in self._realizations), default=0) + 1

    def _new_node(
        self,
        *,
        code: str,
        fitness: float,
        parent_id: int | None,
        hypothesis_id: int | None,
        idea: str | None,
        role: str,
        slot: int,
    ) -> ProgramNode:
        node = ProgramNode(
            id=self._next_node,
            code=code,
            fitness=fitness,
            parent_id=parent_id,
            hypothesis_id=hypothesis_id,
            idea=idea,
            role=role,
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
                    hypothesis_id=hypothesis_id,
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

    def _freeze_root_scale(self) -> None:
        values = [node.fitness for node in self._nodes.values() if node.parent_id is None]
        if not values:
            self._root_center, self._root_scale = 0.0, 1.0
            return
        center = statistics.median(values)
        mad = statistics.median([abs(value - center) for value in values])
        spread = max(values) - min(values)
        self._root_center = float(center)
        self._root_scale = float(1.4826 * mad if mad > 1e-12 else spread if spread > 1e-12 else 1.0)

    def _normalized_quality(self, fitness: float) -> float:
        center = 0.0 if self._root_center is None else self._root_center
        scale = 1.0 if self._root_scale is None else self._root_scale
        return max(-8.0, min(8.0, (fitness - center) / scale))

    def _select_hypothesis(self) -> tuple[Hypothesis, list[dict[str, float | int]]]:
        hypotheses = list(self._hypotheses.values())
        if not hypotheses:
            raise RuntimeError("cannot select a hypothesis before initialization")
        # Standard one-step UCB: quality and response share the frozen,
        # dimensionless root scale; no hand-tuned feature weights.
        snapshot = []
        for h in hypotheses:
            quality = self._normalized_quality(self._nodes[h.scaffold_node_id].fitness)
            uncertainty = math.sqrt(math.log(self._batch_index + 2.0) / (h.trials + 1.0))
            snapshot.append(
                {
                    "id": h.id,
                    "scaffold_quality": quality,
                    "response_mean": h.response_mean,
                    "trials": h.trials,
                    "uncertainty": uncertainty,
                    "opportunity": quality + h.response_mean + uncertainty,
                }
            )
        maximum = max(float(item["opportunity"]) for item in snapshot)
        ties = [h for h, item in zip(hypotheses, snapshot) if abs(float(item["opportunity"]) - maximum) <= 1e-12]
        chosen = self._rng.choice(ties)
        return chosen, snapshot

    # ------------------------------------------------------------------
    # Idea and realization experiment
    # ------------------------------------------------------------------

    def _make_root(self) -> None:
        prompt = build_root_prompt(task_description=self._task, template_function=self._function)
        slot = self._reserve_slot()
        response = self._draw(prompt)
        root_idea = parse_candidate_response(response).idea or "initial algorithm"
        self._pending = Pending(
            prompt=prompt,
            response=response,
            parent_id=None,
            hypothesis_id=None,
            proposal="root",
            idea=root_idea,
            base_code="",
            batch=0,
            slot=slot,
            base_scaffold_fitness=0.0,
            base_parent_fitness=0.0,
        )
        save_checkpoint(self)
        outcome, _ = self._settle_pending(role="root")
        if outcome.node_id is None or outcome.fitness is None:
            return
        hypothesis = Hypothesis(
            id=self._next_hypothesis,
            entry_idea=root_idea,
            source_node_id=outcome.node_id,
            scaffold_node_id=outcome.node_id,
            working_node_id=outcome.node_id,
            parent_hypothesis_id=None,
            donor_node_id=None,
            created_batch=0,
        )
        self._next_hypothesis += 1
        self._hypotheses[hypothesis.id] = hypothesis
        self._nodes[outcome.node_id].hypothesis_id = hypothesis.id
        if self._log is not None:
            self._log.record_hypothesis(self._hypothesis_payload(hypothesis))

    def _ordinary_batch(self) -> None:
        if not self._has_budget():
            return
        self._batch_index += 1
        hypothesis, snapshot = self._select_hypothesis()
        scaffold = self._nodes[hypothesis.scaffold_node_id]
        working = (
            self._nodes[hypothesis.working_node_id]
            if hypothesis.working_node_id is not None
            else scaffold
        )
        card_id = self._select_public_card(hypothesis)
        card = render_public_card(self._nodes, card_id)
        if self._log is not None:
            self._log.record_event(
                "batch_start",
                batch=self._batch_index,
                hypothesis_id=hypothesis.id,
                scaffold_node_id=scaffold.id,
                working_node_id=working.id,
                public_card_node_id=card_id,
                snapshot=snapshot,
            )

        # Both proposals use the same frozen parent state.  The continuation
        # tests the current idea; the branch tests a new idea with one public
        # measured transition.  Neither proposal sees the other response.
        self._run_proposal(
            hypothesis=hypothesis,
            proposal="continue",
            scaffold=scaffold,
            working=working,
            card=None,
        )
        if self._has_budget():
            self._run_proposal(
                hypothesis=hypothesis,
                proposal="branch",
                scaffold=scaffold,
                working=working,
                card=card,
                card_id=card_id,
            )
        self._batch_context = None
        save_checkpoint(self)

    def _run_proposal(
        self,
        *,
        hypothesis: Hypothesis,
        proposal: str,
        scaffold: ProgramNode,
        working: ProgramNode,
        card: str | None,
        card_id: int | None = None,
    ) -> None:
        idea_prompt = build_idea_prompt(
            task_description=self._task,
            base_code=scaffold.code,
            base_fitness=scaffold.fitness,
            working_code=working.code,
            working_fitness=working.fitness,
            entry_idea=hypothesis.entry_idea if proposal == "continue" else None,
            formation_history=render_formation_path(self._nodes, scaffold.id),
            ledger=render_ledger(hypothesis, self._realizations),
            proposal=proposal,
            public_card=card,
        )
        idea_response = self._draw(idea_prompt)
        idea = parse_idea_response(idea_response)
        if idea is None:
            if proposal == "continue":
                idea = hypothesis.entry_idea
            else:
                raise ValueError("V9.21 branch idea response did not contain an idea")
        if self._log is not None:
            self._log.record_decision(
                {
                    "stage": "idea",
                    "batch": self._batch_index,
                    "proposal": proposal,
                    "hypothesis_id": hypothesis.id,
                    "public_card_node_id": card_id,
                    "prompt": idea_prompt,
                    "response": idea_response,
                    "idea": idea,
                }
            )

        target_hypothesis = hypothesis
        if proposal == "branch":
            target_hypothesis = self._create_branch_hypothesis(
                source=hypothesis,
                scaffold=scaffold,
                idea=idea,
                card_id=card_id,
            )
        base_parent = working if proposal == "continue" and working.id != scaffold.id else scaffold
        realization_prompt = build_realization_prompt(
            task_description=self._task,
            idea=idea,
            base_code=scaffold.code,
            base_fitness=scaffold.fitness,
            working_code=working.code,
            working_fitness=working.fitness,
            formation_history=render_formation_path(self._nodes, scaffold.id),
            ledger=render_ledger(target_hypothesis, self._realizations),
            proposal=proposal,
            public_card=card,
        )
        expected = min(REALIZATIONS_PER_IDEA, self._budget - self._n_eval)
        if expected <= 0:
            return
        self._batch_context = {
            "batch": self._batch_index,
            "proposal": proposal,
            "hypothesis_id": target_hypothesis.id,
            "idea": idea,
            "base_parent_id": base_parent.id,
            "scaffold_id": scaffold.id,
            "scaffold_fitness": scaffold.fitness,
            "base_parent_fitness": base_parent.fitness,
            "base_code": base_parent.code,
            "prompt": realization_prompt,
            "card_id": card_id,
            "remaining": expected,
        }
        self._resume_batch_context()

    def _resume_batch_context(self) -> None:
        context = self._batch_context
        if context is None:
            return
        while int(context["remaining"]) > 0 and self._has_budget():
            slot = self._reserve_slot()
            response = self._draw(str(context["prompt"]))
            if self._log is not None:
                self._log.record_decision(
                    {
                        "stage": "realization",
                        "batch": context["batch"],
                        "proposal": context["proposal"],
                        "hypothesis_id": context["hypothesis_id"],
                        "idea": context["idea"],
                        "slot": slot,
                        "public_card_node_id": context.get("card_id"),
                        "prompt": context["prompt"],
                        "response": response,
                    }
                )
            self._pending = Pending(
                prompt=str(context["prompt"]),
                response=response,
                parent_id=int(context["base_parent_id"]),
                hypothesis_id=int(context["hypothesis_id"]),
                proposal=str(context["proposal"]),
                idea=str(context["idea"]),
                base_code=str(context["base_code"]),
                batch=int(context["batch"]),
                slot=slot,
                base_scaffold_fitness=float(context["scaffold_fitness"]),
                base_parent_fitness=float(context["base_parent_fitness"]),
            )
            save_checkpoint(self)
            self._apply_outcome(*self._settle_pending())
            context["remaining"] = int(context["remaining"]) - 1
            save_checkpoint(self)
        self._batch_context = None

    def _create_branch_hypothesis(
        self,
        *,
        source: Hypothesis,
        scaffold: ProgramNode,
        idea: str,
        card_id: int | None,
    ) -> Hypothesis:
        hypothesis = Hypothesis(
            id=self._next_hypothesis,
            entry_idea=idea,
            source_node_id=scaffold.id,
            scaffold_node_id=scaffold.id,
            working_node_id=None,
            parent_hypothesis_id=source.id,
            donor_node_id=card_id,
            created_batch=self._batch_index,
        )
        self._next_hypothesis += 1
        self._hypotheses[hypothesis.id] = hypothesis
        if self._log is not None:
            self._log.record_hypothesis(self._hypothesis_payload(hypothesis))
        return hypothesis

    def _apply_outcome(self, outcome: CandidateOutcome, pending: Pending) -> None:
        hypothesis_id = pending.hypothesis_id
        if hypothesis_id is None:
            return
        hypothesis = self._hypotheses[hypothesis_id]
        source_scaffold = self._nodes[hypothesis.scaffold_node_id]
        realization = Realization(
            id=self._next_realization,
            hypothesis_id=hypothesis_id,
            idea=pending.idea,
            parent_id=pending.parent_id,
            slot=pending.slot,
            outcome=outcome.outcome,
            fitness=outcome.fitness,
            response=outcome.response,
            node_id=outcome.node_id,
            error=outcome.error,
            attempt=outcome.attempt,
        )
        self._next_realization += 1
        self._realizations.append(realization)
        hypothesis.realization_ids.append(realization.id)
        hypothesis.responses.append(outcome.response)
        hypothesis.last_batch = pending.batch
        if outcome.node_id is not None and outcome.fitness is not None:
            node = self._nodes[outcome.node_id]
            working = (
                self._nodes[hypothesis.working_node_id]
                if hypothesis.working_node_id is not None
                else None
            )
            if working is None or node.fitness > working.fitness:
                hypothesis.working_node_id = node.id
            if node.fitness > source_scaffold.fitness:
                hypothesis.scaffold_node_id = node.id
        if self._log is not None:
            self._log.record_event(
                "realization_settled",
                batch=pending.batch,
                slot=pending.slot,
                proposal=pending.proposal,
                hypothesis_id=hypothesis_id,
                idea=pending.idea,
                outcome=outcome.outcome,
                fitness=outcome.fitness,
                response=outcome.response,
                node_id=outcome.node_id,
            )
            self._log.record_hypothesis(self._hypothesis_payload(hypothesis))

    # ------------------------------------------------------------------
    # Candidate evaluation and bounded recovery
    # ------------------------------------------------------------------

    def _reserve_slot(self) -> int:
        if not self._has_budget():
            raise RuntimeError("primary evaluator budget exhausted")
        self._n_eval += 1
        return self._n_eval

    def _draw(self, prompt: str) -> str:
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if self._seed is not None:
            kwargs["seed"] = self._seed + self._llm_calls
        self._llm_calls += 1
        return str(self._llm.draw_sample(prompt, **kwargs))

    def _settle_pending(self, *, role: str = "realization") -> tuple[CandidateOutcome, Pending]:
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
                    final = CandidateOutcome(parsed.code, fitness, "duplicate", -RESPONSE_CLIP, None, None, None, attempt)
                    self._record_eval(current, role, final.outcome, fitness, final.response, elapsed, None, None, attempt, None)
                    self._finish_attempt(current, final)
                    return final, current
                node = self._new_node(
                    code=parsed.code,
                    fitness=fitness,
                    parent_id=current.parent_id,
                    hypothesis_id=current.hypothesis_id,
                    idea=current.idea,
                    role=role,
                    slot=current.slot,
                )
                response_value = self._response(fitness, current.base_scaffold_fitness)
                outcome_name = edge_outcome(current.base_parent_fitness, fitness)
                final = CandidateOutcome(parsed.code, fitness, outcome_name, response_value, node.id, None, None, attempt)
                self._record_eval(current, role, outcome_name, fitness, response_value, elapsed, None, None, attempt, node.id)
                self._finish_attempt(current, final)
                return final, current

            if failure_kind is not None:
                outcome_name = "timeout" if failure_kind == "timeout" else "invalid"
                error_type = getattr(result, "error_type", None)
                message = _short_error(getattr(result, "error", None) or f"evaluation failed: {failure_kind}")
                repair_error = format_failure(error_type, message, getattr(result, "traceback", None))
            else:
                outcome_name, error_type = "invalid", "invalid_result"
                message = _short_error(getattr(result, "error", None) or f"evaluator returned invalid fitness: {raw_fitness!r}")
                repair_error = message
            self._record_eval(current, role, outcome_name, None, -RESPONSE_CLIP, elapsed, error_type, message, attempt, None)
            if attempt <= MAX_REPAIRS:
                repair_prompt = build_repair_prompt(
                    task_description=self._task,
                    idea=current.idea,
                    base_code=current.base_code,
                    failed_code=parsed.code,
                    error=repair_error,
                )
                self._repair_llm_calls += 1
                current = replace(current, prompt=repair_prompt, response=self._draw(repair_prompt), attempt=attempt + 1)
                self._pending = current
                save_checkpoint(self)
                continue
            final = CandidateOutcome(parsed.code, None, outcome_name, -RESPONSE_CLIP, None, message, error_type, attempt)
            self._finish_attempt(current, final)
            return final, current
        raise AssertionError("unreachable candidate settlement")

    def _response(self, fitness: float, scaffold_fitness: float) -> float:
        scale = 1.0 if self._root_scale is None else self._root_scale
        return max(-RESPONSE_CLIP, min(RESPONSE_CLIP, (fitness - scaffold_fitness) / scale))

    def _record_eval(
        self,
        pending: Pending,
        role: str,
        outcome: str,
        fitness: float | None,
        response: float,
        elapsed: float,
        error_type: str | None,
        error: str | None,
        attempt: int,
        node_id: int | None,
    ) -> None:
        if self._log is None:
            return
        self._log.record_evaluation(
            [
                pending.slot,
                pending.batch,
                "" if pending.hypothesis_id is None else pending.hypothesis_id,
                pending.proposal,
                pending.idea,
                "" if pending.parent_id is None else pending.parent_id,
                "" if node_id is None else node_id,
                outcome,
                "" if fitness is None else fitness,
                pending.base_parent_fitness,
                pending.base_scaffold_fitness,
                response,
                attempt,
                "initial" if attempt == 1 else "repair",
                f"{elapsed:.6f}",
                error_type or "",
                error or "",
            ]
        )

    def _finish_attempt(self, pending: Pending, outcome: CandidateOutcome) -> None:
        self._attempts.append(
            Attempt(
                slot=pending.slot,
                batch=pending.batch,
                hypothesis_id=pending.hypothesis_id,
                proposal=pending.proposal,
                idea=pending.idea,
                parent_id=pending.parent_id,
                node_id=outcome.node_id,
                outcome=outcome.outcome,
                fitness=outcome.fitness,
                response=outcome.response,
                error=outcome.error,
            )
        )
        self._pending = None
        save_checkpoint(self)

    # ------------------------------------------------------------------
    # Public evidence and reporting helpers
    # ------------------------------------------------------------------

    def _select_public_card(self, hypothesis: Hypothesis) -> int | None:
        if not self._global_memory:
            return None
        ancestors = set()
        current = self._nodes.get(hypothesis.scaffold_node_id)
        while current is not None:
            ancestors.add(current.id)
            current = self._nodes.get(current.parent_id) if current.parent_id is not None else None
        candidates = [node_id for node_id in self._global_memory if node_id not in ancestors]
        if not candidates:
            return None
        # A card is evidence, not a score.  Choose among the latest measured
        # improvements so one frozen global best cannot anchor every batch.
        return self._rng.choice(candidates[-8:])

    def _hypothesis_payload(self, h: Hypothesis) -> dict[str, Any]:
        scaffold = self._nodes[h.scaffold_node_id]
        working = None if h.working_node_id is None else self._nodes[h.working_node_id]
        return {
            "id": h.id,
            "entry_idea": h.entry_idea,
            "source_node_id": h.source_node_id,
            "scaffold_node_id": h.scaffold_node_id,
            "working_node_id": h.working_node_id,
            "parent_hypothesis_id": h.parent_hypothesis_id,
            "donor_node_id": h.donor_node_id,
            "created_batch": h.created_batch,
            "trials": h.trials,
            "response_mean": h.response_mean,
            "scaffold_fitness": scaffold.fitness,
            "working_fitness": None if working is None else working.fitness,
        }

    def _count_proposals(self, proposal: str) -> int:
        return sum(1 for item in self._attempts if item.proposal == proposal)


def _short_error(message: Any) -> str:
    text = " ".join(str(message).split())
    return text if len(text) <= ERROR_MAX_CHARS else text[: ERROR_MAX_CHARS - 3] + "..."


__all__ = ["TraceAADV921"]
