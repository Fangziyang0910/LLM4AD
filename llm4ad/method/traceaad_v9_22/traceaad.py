"""TraceAAD V9.22 core search.

The unit of search is an idea hypothesis, not an individual piece of code.
For each selected hypothesis the method chooses proposal actions with a
global action-UCB and independently realizes each chosen idea twice from the
same frozen snapshot.  Scaffold and working implementations receive separate
credit, while quality is calibrated by current scaffold ranks rather than a
saturating root scale.
"""

from __future__ import annotations

import math
import random
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
    IDEAS_PER_BATCH,
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
    response_working: float
    response_scaffold: float
    improved_working: bool
    improved_scaffold: bool
    usable: bool
    node_id: int | None
    error: str | None
    error_type: str | None
    attempt: int


class TraceAADV922:
    """Rank-calibrated hypothesis search with dual-baseline credit."""

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
            raise ValueError("TraceAAD V9.22 requires one evolvable template function")
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
        self._action_stats: dict[str, dict[str, float]] = {
            "continue": {
                "trials": 0.0,
                "usable": 0.0,
                "credits_sum": 0.0,
                "working_improvements": 0.0,
                "response_working_sum": 0.0,
                "invalid": 0.0,
                "timeout": 0.0,
                "duplicate": 0.0,
            },
            "branch": {
                "trials": 0.0,
                "usable": 0.0,
                "credits_sum": 0.0,
                "working_improvements": 0.0,
                "response_working_sum": 0.0,
                "invalid": 0.0,
                "timeout": 0.0,
                "duplicate": 0.0,
            },
        }
        self._quality_rank_count = 0
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
            while self._has_budget():
                before = self._n_eval
                self._ordinary_batch()
                if self._n_eval == before:
                    raise RuntimeError("V9.22 batch made no primary evaluator progress")
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
                    mechanism="rank_calibrated_dual_baseline_hypothesis_search",
                    allocation="hypothesis_ucb_plus_action_ucb",
                    online_behavesim=False,
                    ideas_per_batch=IDEAS_PER_BATCH,
                    realizations_per_idea=REALIZATIONS_PER_IDEA,
                    batch_size=BATCH_SIZE,
                    action_counts={"continue": self._count_proposals("continue"), "branch": self._count_proposals("branch")},
                    action_stats=self._action_stats,
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

    def _quality_rank(self, fitness: float, values: list[float]) -> float:
        """Return a tie-aware empirical percentile for higher-is-better fitness."""
        if len(values) <= 1:
            return 0.5
        less = sum(value < fitness for value in values)
        equal = sum(value == fitness for value in values)
        rank = less + (equal - 1) / 2.0
        return max(0.0, min(1.0, rank / (len(values) - 1)))

    def _scaffold_quality_values(self) -> list[float]:
        """Return one quality value per distinct current hypothesis scaffold.

        Descendant nodes are evidence for formation and realization, not extra
        quality mass.  Deduplicating shared scaffolds also prevents many branch
        hypotheses that currently return to the same scaffold from changing the
        calibration of every other hypothesis.
        """
        values: list[float] = []
        seen: set[int] = set()
        for hypothesis in self._hypotheses.values():
            node_id = hypothesis.scaffold_node_id
            if node_id in seen:
                continue
            node = self._nodes.get(node_id)
            if node is None:
                raise RuntimeError(f"hypothesis {hypothesis.id} references missing scaffold {node_id}")
            seen.add(node_id)
            values.append(node.fitness)
        return values

    def _quality_ranks(self) -> dict[int, float]:
        """Rank all known nodes against the current hypothesis scaffold layer.

        The returned map includes non-scaffold nodes because working
        implementations need the same reference scale when an opportunity is
        selected.  They do not contribute values to that scale.
        """
        values = self._scaffold_quality_values()
        self._quality_rank_count = len(values)
        return {node.id: self._quality_rank(node.fitness, values) for node in self._nodes.values()}

    @staticmethod
    def _ucb(values: list[float], batch_index: int) -> tuple[float, float, float]:
        """Hoeffding UCB for bounded implementation credits."""
        n = len(values)
        credits = [max(0.0, min(1.0, (value + 1.0) / 2.0)) for value in values]
        mean = sum(credits) / n if n else 0.5
        radius = math.sqrt(math.log(batch_index + 2.0) / (2.0 * (n + 1.0)))
        return mean, radius, min(1.0, mean + radius)

    def _progress_ucb(self, hypothesis: Hypothesis) -> tuple[float, float, float]:
        return self._ucb(hypothesis.response_working_values, self._batch_index)

    def _action_ucb(self, proposal: str) -> tuple[float, float, float]:
        stats = self._action_stats[proposal]
        n = stats["trials"]
        # Action selection asks whether this proposal reliably produces a
        # usable next working implementation.  Use a Beta-Bernoulli estimate
        # of strict working improvements; continuous rank response remains a
        # separate diagnostic and hypothesis-progress signal.
        posterior_mean = (stats["working_improvements"] + 1.0) / (n + 2.0)
        radius = math.sqrt(math.log(self._batch_index + 2.0) / (2.0 * (n + 1.0)))
        return posterior_mean, radius, min(1.0, posterior_mean + radius)

    def _select_proposals(self, hypothesis: Hypothesis) -> tuple[list[str], dict[str, dict[str, float]]]:
        action_snapshot: dict[str, dict[str, float]] = {}
        virtual_trials = {
            proposal: self._action_stats[proposal]["trials"]
            for proposal in ("continue", "branch")
        }
        virtual_successes = {
            proposal: self._action_stats[proposal]["working_improvements"]
            for proposal in ("continue", "branch")
        }
        for proposal in ("continue", "branch"):
            mean, radius, upper = self._action_ucb(proposal)
            action_snapshot[proposal] = {"mean": mean, "radius": radius, "upper": upper}
        proposals: list[str] = []
        for _ in range(IDEAS_PER_BATCH):
            untried = [proposal for proposal in ("continue", "branch") if virtual_trials[proposal] == 0.0]
            if untried:
                # Observe each action once before allowing concentration.
                chosen = self._rng.choice(untried)
            else:
                virtual_upper: dict[str, float] = {}
                for proposal in ("continue", "branch"):
                    trials = virtual_trials[proposal]
                    mean = (virtual_successes[proposal] + 1.0) / (trials + 2.0)
                    radius = math.sqrt(math.log(self._batch_index + 2.0) / (2.0 * (trials + 1.0)))
                    virtual_upper[proposal] = min(1.0, mean + radius)
                maximum = max(virtual_upper.values())
                choices = [proposal for proposal, upper in virtual_upper.items() if abs(upper - maximum) <= 1e-12]
                chosen = self._rng.choice(choices)
            proposals.append(chosen)
            virtual_trials[chosen] += float(REALIZATIONS_PER_IDEA)
        return proposals, action_snapshot

    def _select_hypothesis(self) -> tuple[Hypothesis, list[dict[str, float | int]]]:
        hypotheses = list(self._hypotheses.values())
        if not hypotheses:
            raise RuntimeError("cannot select a hypothesis before initialization")
        quality_ranks = self._quality_ranks()
        snapshot = []
        for h in hypotheses:
            scaffold = self._nodes[h.scaffold_node_id]
            working = scaffold if h.working_node_id is None else self._nodes[h.working_node_id]
            quality = quality_ranks[scaffold.id]
            working_quality = quality_ranks[working.id]
            progress_mean, uncertainty, progress_ucb = self._progress_ucb(h)
            # The response UCB is a bounded upper estimate of the next rank
            # change.  The opportunity is the better of the already-usable
            # scaffold and the one-step working implementation upper bound.
            # This keeps an unfinished repair line alive without rewarding a
            # hypothesis merely because it has many descendants.
            progress_upper_delta = 2.0 * progress_ucb - 1.0
            # Keep the confidence bonus un-clipped at the opportunity-score
            # level.  Clipping here would make every untried hypothesis reach
            # the same score of one and erase scaffold-quality ordering.
            working_upper = working_quality + progress_upper_delta
            opportunity = max(quality, working_upper)
            snapshot.append(
                {
                    "id": h.id,
                    "scaffold_quality_rank": quality,
                    "working_quality_rank": working_quality,
                    "working_upper_score": working_upper,
                    "progress_upper_delta": progress_upper_delta,
                    "progress_mean": progress_mean,
                    "progress_ucb": progress_ucb,
                    "trials": h.trials,
                    "uncertainty": uncertainty,
                    "opportunity": opportunity,
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
            source_hypothesis_id=None,
            proposal="root",
            idea=root_idea,
            base_code="",
            batch=0,
            slot=slot,
            base_scaffold_fitness=0.0,
            base_parent_fitness=0.0,
            base_working_fitness=0.0,
            quality_reference=[],
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
        formation_history = render_formation_path(self._nodes, scaffold.id)
        ledger = render_ledger(hypothesis, self._realizations)
        proposals, action_snapshot = self._select_proposals(hypothesis)
        card_id = self._select_public_card(hypothesis) if "branch" in proposals else None
        card = render_public_card(self._nodes, card_id)
        quality_reference = self._scaffold_quality_values()
        if self._log is not None:
            self._log.record_event(
                "batch_start",
                batch=self._batch_index,
                hypothesis_id=hypothesis.id,
                scaffold_node_id=scaffold.id,
                working_node_id=working.id,
                public_card_node_id=card_id,
                snapshot=snapshot,
                action_snapshot=action_snapshot,
                proposal_plan=proposals,
            )

        # The proposal choices and all context strings are frozen at batch
        # start.  A first realization cannot rewrite the second prompt.
        for proposal in proposals:
            if not self._has_budget():
                break
            self._run_proposal(
                hypothesis=hypothesis,
                proposal=proposal,
                scaffold=scaffold,
                working=working,
                card=card if proposal == "branch" else None,
                card_id=card_id,
                formation_history=formation_history,
                source_ledger=ledger,
                quality_reference=quality_reference,
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
        formation_history: str | None = None,
        source_ledger: str | None = None,
        quality_reference: list[float] | None = None,
    ) -> None:
        formation_history = formation_history or render_formation_path(self._nodes, scaffold.id)
        source_ledger = source_ledger or render_ledger(hypothesis, self._realizations)
        if quality_reference is None:
            quality_reference = self._scaffold_quality_values()
        idea_prompt = build_idea_prompt(
            task_description=self._task,
            base_code=scaffold.code,
            base_fitness=scaffold.fitness,
            working_code=working.code if proposal == "continue" else None,
            working_fitness=working.fitness if proposal == "continue" else None,
            entry_idea=hypothesis.entry_idea if proposal == "continue" else None,
            formation_history=formation_history,
            ledger=source_ledger,
            proposal=proposal,
            public_card=card,
        )
        idea_response = self._draw(idea_prompt)
        idea = parse_idea_response(idea_response)
        if idea is None:
            if proposal == "continue":
                idea = hypothesis.entry_idea
            else:
                raise ValueError("V9.22 branch idea response did not contain an idea")
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
        # A branch is deliberately generated from the stable scaffold and
        # does not receive the source working implementation.  Its working
        # baseline is therefore the scaffold itself; otherwise a branch would
        # receive credit for repairing code it was never shown.
        proposal_working_fitness = working.fitness if proposal == "continue" else scaffold.fitness
        realization_prompt = build_realization_prompt(
            task_description=self._task,
            idea=idea,
            base_code=scaffold.code,
            base_fitness=scaffold.fitness,
            working_code=working.code if proposal == "continue" else None,
            working_fitness=working.fitness if proposal == "continue" else None,
            formation_history=formation_history,
            ledger=source_ledger,
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
            "source_hypothesis_id": hypothesis.id,
            "idea": idea,
            "base_parent_id": base_parent.id,
            "scaffold_id": scaffold.id,
            "scaffold_fitness": scaffold.fitness,
            "base_parent_fitness": base_parent.fitness,
            "base_working_fitness": proposal_working_fitness,
            "quality_reference": quality_reference,
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
            # Mark this sibling as consumed before persisting the pending
            # response.  If the process stops after that checkpoint, resume
            # settles the same response without issuing it a second time.
            context["remaining"] = int(context["remaining"]) - 1
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
                source_hypothesis_id=int(context["source_hypothesis_id"]),
                proposal=str(context["proposal"]),
                idea=str(context["idea"]),
                base_code=str(context["base_code"]),
                batch=int(context["batch"]),
                slot=slot,
                base_scaffold_fitness=float(context["scaffold_fitness"]),
                base_parent_fitness=float(context["base_parent_fitness"]),
                base_working_fitness=float(context["base_working_fitness"]),
                quality_reference=[float(value) for value in context["quality_reference"]],
            )
            save_checkpoint(self)
            self._apply_outcome(*self._settle_pending())
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
        scaffold = self._nodes[hypothesis.scaffold_node_id]
        working = (
            self._nodes[hypothesis.working_node_id]
            if hypothesis.working_node_id is not None
            else scaffold
        )
        realization = Realization(
            id=self._next_realization,
            hypothesis_id=hypothesis_id,
            idea=pending.idea,
            parent_id=pending.parent_id,
            slot=pending.slot,
            proposal=pending.proposal,
            outcome=outcome.outcome,
            fitness=outcome.fitness,
            response=outcome.response,
            response_working=outcome.response_working,
            response_scaffold=outcome.response_scaffold,
            improved_working=outcome.improved_working,
            improved_scaffold=outcome.improved_scaffold,
            usable=outcome.usable,
            node_id=outcome.node_id,
            error=outcome.error,
            attempt=outcome.attempt,
        )
        self._next_realization += 1
        self._realizations.append(realization)
        hypothesis.realization_ids.append(realization.id)
        hypothesis.response_working_values.append(outcome.response_working)
        hypothesis.response_scaffold_values.append(outcome.response_scaffold)
        hypothesis.usable_trials += int(outcome.usable)
        hypothesis.working_improvements += int(outcome.improved_working)
        hypothesis.scaffold_improvements += int(outcome.improved_scaffold)
        action_owner = self._hypotheses.get(pending.source_hypothesis_id or hypothesis_id, hypothesis)
        for action_hypothesis in {id(hypothesis): hypothesis, id(action_owner): action_owner}.values():
            action_hypothesis.action_trials[pending.proposal] = (
                action_hypothesis.action_trials.get(pending.proposal, 0) + 1
            )
            action_hypothesis.action_usable_trials[pending.proposal] = (
                action_hypothesis.action_usable_trials.get(pending.proposal, 0) + int(outcome.usable)
            )
            action_hypothesis.action_improvements[pending.proposal] = (
                action_hypothesis.action_improvements.get(pending.proposal, 0)
                + int(outcome.improved_working)
            )
        hypothesis.last_batch = pending.batch
        stats = self._action_stats[pending.proposal]
        stats["trials"] += 1.0
        stats["usable"] += float(outcome.usable)
        stats["credits_sum"] += max(0.0, min(1.0, (outcome.response_working + 1.0) / 2.0))
        stats["working_improvements"] += float(outcome.improved_working)
        stats["response_working_sum"] += outcome.response_working
        if outcome.outcome in {"invalid", "invalid_result"}:
            stats["invalid"] += 1.0
        elif outcome.outcome == "timeout":
            stats["timeout"] += 1.0
        elif outcome.outcome == "duplicate":
            stats["duplicate"] += 1.0
        if outcome.node_id is not None and outcome.fitness is not None:
            node = self._nodes[outcome.node_id]
            if hypothesis.working_node_id is None or node.fitness > working.fitness:
                hypothesis.working_node_id = node.id
            if node.fitness > scaffold.fitness:
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
                response_working=outcome.response_working,
                response_scaffold=outcome.response_scaffold,
                usable=outcome.usable,
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
                    final = CandidateOutcome(
                        code=parsed.code,
                        fitness=fitness,
                        outcome="duplicate",
                        response=-RESPONSE_CLIP,
                        response_working=-RESPONSE_CLIP,
                        response_scaffold=-RESPONSE_CLIP,
                        improved_working=False,
                        improved_scaffold=False,
                        usable=False,
                        node_id=None,
                        error=None,
                        error_type=None,
                        attempt=attempt,
                    )
                    self._record_eval(
                        current,
                        role,
                        final.outcome,
                        fitness,
                        final.response,
                        final.response_working,
                        final.response_scaffold,
                        final.usable,
                        elapsed,
                        None,
                        None,
                        attempt,
                        None,
                    )
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
                response_working = self._rank_delta(
                    fitness,
                    current.base_working_fitness,
                    current.quality_reference,
                )
                response_scaffold = self._rank_delta(
                    fitness,
                    current.base_scaffold_fitness,
                    current.quality_reference,
                )
                outcome_name = edge_outcome(current.base_parent_fitness, fitness)
                final = CandidateOutcome(
                    code=parsed.code,
                    fitness=fitness,
                    outcome=outcome_name,
                    response=response_working,
                    response_working=response_working,
                    response_scaffold=response_scaffold,
                    improved_working=fitness > current.base_working_fitness,
                    improved_scaffold=fitness > current.base_scaffold_fitness,
                    usable=True,
                    node_id=node.id,
                    error=None,
                    error_type=None,
                    attempt=attempt,
                )
                self._record_eval(
                    current,
                    role,
                    outcome_name,
                    fitness,
                    response_working,
                    response_working,
                    response_scaffold,
                    True,
                    elapsed,
                    None,
                    None,
                    attempt,
                    node.id,
                )
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
            self._record_eval(
                current,
                role,
                outcome_name,
                None,
                -RESPONSE_CLIP,
                -RESPONSE_CLIP,
                -RESPONSE_CLIP,
                False,
                elapsed,
                error_type,
                message,
                attempt,
                None,
            )
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
            final = CandidateOutcome(
                code=parsed.code,
                fitness=None,
                outcome=outcome_name,
                response=-RESPONSE_CLIP,
                response_working=-RESPONSE_CLIP,
                response_scaffold=-RESPONSE_CLIP,
                improved_working=False,
                improved_scaffold=False,
                usable=False,
                node_id=None,
                error=message,
                error_type=error_type,
                attempt=attempt,
            )
            self._finish_attempt(current, final)
            return final, current
        raise AssertionError("unreachable candidate settlement")

    @staticmethod
    def _rank_value(fitness: float, values: list[float]) -> float:
        if len(values) <= 1:
            return 0.5
        less = sum(value < fitness for value in values)
        equal = sum(value == fitness for value in values)
        rank = less + (equal - 1) / 2.0
        return max(0.0, min(1.0, rank / (len(values) - 1)))

    @classmethod
    def _rank_delta(cls, fitness: float, reference: float, values: list[float]) -> float:
        if not values:
            return 0.0
        # Rank candidate and reference in the same augmented sample.  Ranking
        # against the baseline alone would map both a current maximum and a
        # new strict maximum to 1.0, erasing breakthrough credit.
        augmented = [*values, fitness, reference]
        delta = cls._rank_value(fitness, augmented) - cls._rank_value(reference, augmented)
        return max(-RESPONSE_CLIP, min(RESPONSE_CLIP, delta))

    def _record_eval(
        self,
        pending: Pending,
        role: str,
        outcome: str,
        fitness: float | None,
        response: float,
        response_working: float,
        response_scaffold: float,
        usable: bool,
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
                pending.base_working_fitness,
                pending.base_scaffold_fitness,
                response,
                response_working,
                response_scaffold,
                usable,
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
                source_hypothesis_id=pending.source_hypothesis_id,
                proposal=pending.proposal,
                idea=pending.idea,
                parent_id=pending.parent_id,
                node_id=outcome.node_id,
                outcome=outcome.outcome,
                fitness=outcome.fitness,
                response=outcome.response,
                response_working=outcome.response_working,
                response_scaffold=outcome.response_scaffold,
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
        related_hypotheses = {hypothesis.id}
        parent_id = hypothesis.parent_hypothesis_id
        while parent_id is not None:
            related_hypotheses.add(parent_id)
            parent = self._hypotheses.get(parent_id)
            parent_id = None if parent is None else parent.parent_hypothesis_id
        candidates = [
            node_id
            for node_id in self._global_memory
            if self._nodes.get(node_id) is not None
            and self._nodes[node_id].hypothesis_id not in related_hypotheses
        ]
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
            "response_working_mean": h.response_working_mean,
            "response_scaffold_mean": h.response_scaffold_mean,
            "usable_trials": h.usable_trials,
            "working_improvements": h.working_improvements,
            "scaffold_improvements": h.scaffold_improvements,
            "action_trials": h.action_trials,
            "action_usable_trials": h.action_usable_trials,
            "action_improvements": h.action_improvements,
            "scaffold_fitness": scaffold.fitness,
            "working_fitness": None if working is None else working.fitness,
        }

    def _count_proposals(self, proposal: str) -> int:
        return sum(1 for item in self._attempts if item.proposal == proposal)


def _short_error(message: Any) -> str:
    text = " ".join(str(message).split())
    return text if len(text) <= ERROR_MAX_CHARS else text[: ERROR_MAX_CHARS - 3] + "..."


__all__ = ["TraceAADV922"]
