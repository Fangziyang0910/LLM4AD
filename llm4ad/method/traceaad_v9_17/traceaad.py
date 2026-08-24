"""TraceAAD V9.17 competition-gated hypothesis discovery and development."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from pathlib import Path
from typing import Any

from ...base import Evaluation, Function, LLM, SecureEvaluator, TextFunctionProgramConverter
from .artifacts import RunArtifacts
from .checkpoint import load_checkpoint, save_checkpoint
from .history import parent_path, render_path
from .prompt import (
    build_generation_prompt,
    build_repair_prompt,
    build_root_prompt,
    format_failure_feedback,
    parse_program_response,
    preflight_code,
)
from .schema import (
    BLOCK_HORIZON,
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    BlockKind,
    BlockState,
    GenerationState,
    Hypothesis,
    HypothesisStatus,
    Intent,
    Pending,
    Phase,
)
from .selection import competition_line, rank_hypotheses, select_refine_parent
from .source import code_diff
from .tree import Tree, VIRTUAL_ROOT_ID


class TraceAADV917:
    """Run the V9.17 fixed-block hypothesis competition protocol."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        artifacts: RunArtifacts | None = None,
        budget: int = 1000,
        *,
        n_roots: int = INITIAL_ROOT_COUNT,
        maximize: bool = True,
        max_tokens: int = 8192,
        max_history: int = MAX_HISTORY_EVENTS,
        seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        error_retries: int = 2,
        error_handling: bool = True,
        adaptive_sweeps: bool = True,
        fork_from_initialization: bool = False,
    ) -> None:
        if min(budget, n_roots, max_tokens, max_history) <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if error_retries < 0:
            raise ValueError("error_retries must be non-negative")
        template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.17 requires one evolvable template function")

        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation)
        self._log = artifacts
        self._task = evaluation.task_description
        self._function: Function = template.functions[0]
        self._budget = budget
        self._n_roots = n_roots
        self._active_capacity = n_roots
        self._max_tokens = max_tokens
        self._max_history = max_history
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        self._error_retries = error_retries
        self._error_handling = error_handling
        self._adaptive_sweeps = adaptive_sweeps
        self._allow_scheduler_fork = fork_from_initialization

        self._tree = Tree(maximize=maximize)
        self._hypotheses: dict[int, Hypothesis] = {}
        self._active_ids: list[int] = []
        self._reserve_ids: list[int] = []
        self._phase = Phase.ROOTS
        self._generation: GenerationState | None = None
        self._pending: Pending | None = None

        self._n_eval = 0
        self._n_llm_calls = 0
        self._repair_llm_calls = 0
        self._n_calls = 0
        self._root_slots = 0
        self._refine_slots = 0
        self._explore_slots = 0
        self._next_hypothesis_id = 1
        self._next_block_id = 1

        self._initial_order: list[int] = []
        self._initial_cursor = 0
        self._bootstrap_deltas: list[float] = []
        self._s_r = 0.0
        self._s_r_frozen = False

        self._cycle = 0
        self._sweep = 0
        self._eligible_ids: list[int] = []
        self._sweep_order: list[int] = []
        self._sweep_cursor = 0
        self._successful_ids: list[int] = []
        self._active_block: BlockState | None = None
        self._terminal_after_block = False

        self._discovery_attempted = False
        self._discovery_source_id: int | None = None
        self._discovery_candidate_hypothesis_id: int | None = None
        self._maturing_hypothesis_id: int | None = None
        self._discovery_attempts = 0
        self._valid_discoveries = 0
        self._block_counts = {kind.value: 0 for kind in BlockKind}

        self._attempt_elapsed = 0.0
        self._preflight_error: str | None = None
        self._candidate_hash = ""

        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self):
        return self._tree.best()

    @property
    def primary_slots_remaining(self) -> int:
        return max(0, self._budget - self._n_eval)

    def run(self) -> None:
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            if self._generation is not None or self._pending is not None:
                self._continue_generation()
            while self._has_budget():
                self._advance()
            status = "finished"
            stop_reason = (
                "budget_exhausted_before_initial_roots"
                if len(self._tree.root_algorithms()) < self._n_roots
                else "primary_budget_exhausted"
            )
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
                    best_algorithm_id=None if best is None else best.id,
                    best_score=None if best is None else best.fitness,
                    budget_slots=self._n_eval,
                    llm_call_count=self._n_llm_calls,
                    repair_llm_calls=self._repair_llm_calls,
                    evaluator_call_count=self._n_calls,
                    repair_evaluator_calls=self._n_calls - self._n_eval,
                    root_slots=self._root_slots,
                    refine_slots=self._refine_slots,
                    explore_slots=self._explore_slots,
                    n_algorithms=len(self._tree.valid_algorithms()),
                    n_roots=len(self._tree.root_algorithms()),
                    n_hypotheses=len(self._hypotheses),
                    active_hypotheses=self._active_ids,
                    reserve_hypotheses=self._reserve_ids,
                    phase=self._phase.value,
                    scheduler=(
                        "adaptive_gain_continuation"
                        if self._adaptive_sweeps
                        else "fixed_cycle"
                    ),
                    cycle=self._cycle,
                    sweep=self._sweep,
                    s_r=self._s_r,
                    discovery_attempts=self._discovery_attempts,
                    valid_discoveries=self._valid_discoveries,
                    block_counts=self._block_counts,
                    has_pending=self._pending is not None,
                    has_generation=self._generation is not None,
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _advance(self) -> None:
        if self._generation is not None or self._pending is not None:
            self._continue_generation()
            return
        if self._phase is Phase.ROOTS:
            self._advance_roots()
        elif self._phase is Phase.INITIAL_MATURATION:
            self._advance_initial_maturation()
        elif self._phase is Phase.DEVELOPMENT:
            self._advance_development()
        elif self._phase is Phase.DISCOVERY:
            self._advance_discovery()
        elif self._phase is Phase.MATURATION:
            self._advance_maturation()
        elif self._phase is Phase.TERMINAL:
            self._advance_terminal()
        else:  # pragma: no cover - exhaustive guard
            raise RuntimeError(f"unsupported V9.17 phase: {self._phase}")

    def _advance_roots(self) -> None:
        roots = self._tree.root_algorithms()
        if len(roots) >= self._n_roots:
            self._initial_order = [item.hypothesis_id for item in roots]
            if any(item is None for item in self._initial_order):
                raise RuntimeError("valid root is missing its hypothesis")
            self._initial_order = [int(item) for item in self._initial_order]
            self._phase = Phase.INITIAL_MATURATION
            self._event("initial_roots_complete", hypothesis_ids=self._initial_order)
            save_checkpoint(self)
            return
        self._start_generation(
            build_root_prompt(
                task_description=self._task,
                template_function=self._function,
                maximize=self._tree.maximize,
                error_handling=self._error_handling,
            ),
            parent_id=VIRTUAL_ROOT_ID,
            intent=None,
            mode="root",
            hypothesis_id=None,
        )

    def _advance_initial_maturation(self) -> None:
        if self._active_block is not None:
            self._run_block_step()
            return
        if self._initial_cursor >= len(self._initial_order):
            self._s_r = (
                float(statistics.median(self._bootstrap_deltas))
                if self._bootstrap_deltas
                else 0.0
            )
            self._s_r_frozen = True
            for hypothesis_id in self._initial_order:
                self._hypotheses[hypothesis_id].status = HypothesisStatus.ACTIVE
            self._active_ids = self._rank_ids(self._initial_order)
            self._event(
                "initial_maturation_complete",
                s_r=self._s_r,
                valid_refine_edges=len(self._bootstrap_deltas),
                active_ids=self._active_ids,
            )
            self._cycle = 1
            self._begin_development()
            return
        target = min(BLOCK_HORIZON, self.primary_slots_remaining)
        if target <= 0:
            return
        hypothesis_id = self._initial_order[self._initial_cursor]
        self._begin_block(
            hypothesis_id,
            BlockKind.INITIAL_MATURATION,
            target,
            terminal_after=target < BLOCK_HORIZON,
        )
        self._run_block_step()

    def _begin_development(self) -> None:
        self._phase = Phase.DEVELOPMENT
        self._eligible_ids = self._rank_ids(self._active_ids)
        self._sweep = 1
        self._sweep_order = []
        self._sweep_cursor = 0
        self._successful_ids = []
        save_checkpoint(self)

    def _advance_development(self) -> None:
        if self._active_block is not None:
            self._run_block_step()
            return
        if self._sweep_order and self._sweep_cursor >= len(self._sweep_order):
            self._event(
                "sweep_finish",
                cycle=self._cycle,
                sweep=self._sweep,
                eligible_ids=self._sweep_order,
                successful_ids=self._successful_ids,
            )
            if self._adaptive_sweeps and self._successful_ids:
                self._eligible_ids = self._rank_ids(self._successful_ids)
                self._sweep += 1
                self._sweep_order = []
                self._sweep_cursor = 0
                self._successful_ids = []
            else:
                self._eligible_ids = []
                self._sweep_order = []
                self._sweep_cursor = 0
                self._phase = Phase.DISCOVERY
            save_checkpoint(self)
            return
        if not self._sweep_order:
            if not self._eligible_ids:
                self._phase = Phase.DISCOVERY
                save_checkpoint(self)
                return
            self._sweep_order = self._rank_ids(self._eligible_ids)
            self._sweep_cursor = 0
            self._successful_ids = []
            self._event(
                "sweep_start",
                cycle=self._cycle,
                sweep=self._sweep,
                frozen_order=self._sweep_order,
            )
            save_checkpoint(self)
        hypothesis_id = self._sweep_order[self._sweep_cursor]
        remaining = self.primary_slots_remaining
        if remaining < BLOCK_HORIZON:
            self._phase = Phase.TERMINAL
            self._begin_block(
                hypothesis_id,
                BlockKind.TERMINAL,
                remaining,
                terminal_after=True,
            )
        else:
            self._begin_block(
                hypothesis_id, BlockKind.DEVELOPMENT, BLOCK_HORIZON
            )
        self._run_block_step()

    def _advance_discovery(self) -> None:
        if self._discovery_attempted:
            self._finish_discovery_transition()
            return
        remaining = self.primary_slots_remaining
        if remaining < 1 + BLOCK_HORIZON:
            self._phase = Phase.TERMINAL
            hypothesis_id = self._highest_active_id()
            self._begin_block(
                hypothesis_id,
                BlockKind.TERMINAL,
                remaining,
                terminal_after=True,
            )
            self._run_block_step()
            return
        source_id = self._highest_active_id()
        source = self._hypotheses[source_id]
        parent = self._tree.get_algorithm(source.frontier_node_id)
        self._discovery_attempted = True
        self._discovery_source_id = source_id
        self._discovery_candidate_hypothesis_id = None
        self._discovery_attempts += 1
        self._prepare_generation(
            self._prompt(parent, Intent.EXPLORE),
            parent_id=parent.id,
            intent=Intent.EXPLORE,
            mode="discovery",
            hypothesis_id=None,
        )
        self._event(
            "discovery_start",
            cycle=self._cycle,
            source_hypothesis_id=source_id,
            source_frontier_id=parent.id,
            source_quality=source.best_quality,
            competition_line=self._competition_line(),
        )
        self._continue_generation()
        self._finish_discovery_transition()

    def _finish_discovery_transition(self) -> None:
        if self._generation is not None or self._pending is not None:
            return
        candidate_id = self._discovery_candidate_hypothesis_id
        if candidate_id is None:
            self._event(
                "discovery_finish",
                cycle=self._cycle,
                source_hypothesis_id=self._discovery_source_id,
                status="invalid",
            )
            self._reset_discovery()
            self._cycle += 1
            self._begin_development()
            return
        self._maturing_hypothesis_id = candidate_id
        candidate = self._hypotheses[candidate_id]
        self._event(
            "discovery_finish",
            cycle=self._cycle,
            source_hypothesis_id=self._discovery_source_id,
            status="valid",
            candidate_hypothesis_id=candidate_id,
            birth_quality=candidate.best_quality,
        )
        self._phase = Phase.MATURATION
        save_checkpoint(self)

    def _advance_maturation(self) -> None:
        if self._active_block is not None:
            self._run_block_step()
            return
        hypothesis_id = self._maturing_hypothesis_id
        if hypothesis_id is None:
            raise RuntimeError("maturation phase has no maturing hypothesis")
        if self.primary_slots_remaining < BLOCK_HORIZON:
            raise RuntimeError("discovery did not reserve a complete maturation block")
        self._begin_block(hypothesis_id, BlockKind.MATURATION, BLOCK_HORIZON)
        self._run_block_step()

    def _advance_terminal(self) -> None:
        if self._active_block is not None:
            self._run_block_step()
            return
        remaining = self.primary_slots_remaining
        if remaining <= 0:
            return
        if not self._active_ids:
            raise RuntimeError("terminal refinement requires an active hypothesis")
        self._begin_block(
            self._highest_active_id(),
            BlockKind.TERMINAL,
            remaining,
            terminal_after=True,
        )
        self._run_block_step()

    def _begin_block(
        self,
        hypothesis_id: int,
        kind: BlockKind,
        target_steps: int,
        *,
        terminal_after: bool = False,
    ) -> None:
        if self._active_block is not None:
            raise RuntimeError("cannot begin a second V9.17 block")
        if target_steps <= 0:
            raise ValueError("block target must be positive")
        hypothesis = self._hypotheses[hypothesis_id]
        block = BlockState(
            id=self._next_block_id,
            hypothesis_id=hypothesis_id,
            kind=kind,
            q_before=hypothesis.best_quality,
            target_steps=target_steps,
        )
        self._next_block_id += 1
        self._active_block = block
        self._terminal_after_block = terminal_after
        self._event(
            "block_start",
            block_id=block.id,
            block_kind=block.kind.value,
            hypothesis_id=hypothesis_id,
            q_before=block.q_before,
            target_steps=target_steps,
            cycle=self._cycle,
            sweep=self._sweep,
        )
        save_checkpoint(self)

    def _run_block_step(self) -> None:
        block = self._active_block
        if block is None:
            raise RuntimeError("no V9.17 block is active")
        if block.completed_steps >= block.target_steps:
            self._finish_block()
            return
        parent = select_refine_parent(
            self._tree, block.hypothesis_id, scale=self._s_r
        )
        step = block.completed_steps + 1
        block.selected_parent_ids.append(parent.id)
        self._start_generation(
            self._prompt(parent, Intent.REFINE),
            parent_id=parent.id,
            intent=Intent.REFINE,
            mode=block.kind.value,
            hypothesis_id=block.hypothesis_id,
            block_id=block.id,
            block_step=step,
        )
        if (
            self._active_block is not None
            and self._active_block.completed_steps >= self._active_block.target_steps
        ):
            self._finish_block()

    def _finish_block(self) -> None:
        block = self._active_block
        if block is None:
            raise RuntimeError("no V9.17 block is active")
        hypothesis = self._hypotheses[block.hypothesis_id]
        q_after = hypothesis.best_quality
        gain = q_after - block.q_before
        if gain < -1e-12:
            raise RuntimeError("hypothesis frontier quality decreased")
        gain = max(0.0, gain)
        hypothesis.last_block_gain = gain
        self._block_counts[block.kind.value] += 1
        self._event(
            "block_finish",
            block_id=block.id,
            block_kind=block.kind.value,
            hypothesis_id=block.hypothesis_id,
            q_before=block.q_before,
            q_after=q_after,
            gain=gain,
            completed_steps=block.completed_steps,
            valid_results=block.valid_results,
            selected_parent_ids=block.selected_parent_ids,
            cycle=self._cycle,
            sweep=self._sweep,
        )
        kind = block.kind
        hypothesis_id = block.hypothesis_id
        terminal_after = self._terminal_after_block
        self._active_block = None
        self._terminal_after_block = False

        if kind is BlockKind.INITIAL_MATURATION:
            self._initial_cursor += 1
        elif kind is BlockKind.DEVELOPMENT:
            if gain > 0:
                self._successful_ids.append(hypothesis_id)
            self._sweep_cursor += 1
        elif kind is BlockKind.MATURATION:
            self._compete(hypothesis_id)
            return
        if terminal_after or kind is BlockKind.TERMINAL:
            self._phase = Phase.TERMINAL
        save_checkpoint(self)

    def _compete(self, candidate_id: int) -> None:
        candidate = self._hypotheses[candidate_id]
        origin = self._tree.get_algorithm(candidate.origin_node_id)
        candidate_ids = [*self._active_ids, candidate_id]
        ranked = self._rank_ids(candidate_ids)
        kept = ranked[: self._active_capacity]
        displaced = ranked[self._active_capacity :]
        for hypothesis_id in kept:
            self._hypotheses[hypothesis_id].status = HypothesisStatus.ACTIVE
        for hypothesis_id in displaced:
            self._hypotheses[hypothesis_id].status = HypothesisStatus.RESERVE
            if hypothesis_id not in self._reserve_ids:
                self._reserve_ids.append(hypothesis_id)
        self._reserve_ids = [
            item for item in self._reserve_ids if item not in set(kept)
        ]
        self._active_ids = kept
        self._event(
            "competition",
            cycle=self._cycle,
            candidate_hypothesis_id=candidate_id,
            birth_quality=self._tree.quality(origin),
            matured_quality=candidate.best_quality,
            maturation_gain=candidate.last_block_gain,
            ranking=[
                {
                    "hypothesis_id": item,
                    "quality": self._hypotheses[item].best_quality,
                    "status": self._hypotheses[item].status.value,
                }
                for item in ranked
            ],
            active_ids=self._active_ids,
            reserve_ids=self._reserve_ids,
            competition_line=self._competition_line(),
        )
        self._reset_discovery()
        self._cycle += 1
        self._begin_development()

    def _reset_discovery(self) -> None:
        self._discovery_attempted = False
        self._discovery_source_id = None
        self._discovery_candidate_hypothesis_id = None
        self._maturing_hypothesis_id = None

    def _start_generation(
        self,
        prompt: str,
        *,
        parent_id: int,
        intent: Intent | None,
        mode: str,
        hypothesis_id: int | None,
        block_id: int | None = None,
        block_step: int | None = None,
    ):
        self._prepare_generation(
            prompt,
            parent_id=parent_id,
            intent=intent,
            mode=mode,
            hypothesis_id=hypothesis_id,
            block_id=block_id,
            block_step=block_step,
        )
        return self._continue_generation()

    def _prepare_generation(
        self,
        prompt: str,
        *,
        parent_id: int,
        intent: Intent | None,
        mode: str,
        hypothesis_id: int | None,
        block_id: int | None = None,
        block_step: int | None = None,
    ) -> None:
        if self._generation is not None or self._pending is not None:
            raise RuntimeError("another V9.17 generation transaction is active")
        self._generation = GenerationState(
            prompt=prompt,
            parent_id=parent_id,
            intent=None if intent is None else intent.value,
            mode=mode,
            hypothesis_id=hypothesis_id,
            block_id=block_id,
            block_step=block_step,
        )
        save_checkpoint(self)

    def _continue_generation(self):
        child = None
        while self._generation is not None:
            if self._pending is None:
                state = self._generation
                kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
                if self._seed is not None:
                    kwargs["seed"] = self._seed + self._n_eval + state.attempt
                response = self._llm.draw_sample(state.prompt, **kwargs)
                self._n_llm_calls += 1
                if state.attempt > 1:
                    self._repair_llm_calls += 1
                self._pending = Pending(
                    response=response,
                    attempt=state.attempt,
                    attempt_kind="initial" if state.attempt == 1 else "repair",
                )
                save_checkpoint(self)
            child = self._process_pending()
        return child

    def _process_pending(self):
        pending = self._pending
        state = self._generation
        if pending is None or state is None:
            raise RuntimeError("no pending V9.17 candidate to process")
        if pending.attempt != state.attempt:
            raise RuntimeError("pending attempt does not match generation state")

        started = time.perf_counter()
        parsed = parse_program_response(pending.response)
        self._candidate_hash = hashlib.sha256(parsed.code.encode()).hexdigest()[:16]
        self._preflight_error = preflight_code(parsed.code, self._function.name)
        result = self._evaluator.evaluate_program_with_details(parsed.code)
        self._attempt_elapsed = time.perf_counter() - started
        self._n_calls += 1
        if not state.primary_charged:
            self._charge_primary(state)

        raw_fitness = getattr(result.result, "fitness", result.result)
        try:
            fitness = float(raw_fitness)
        except (TypeError, ValueError, OverflowError):
            fitness = math.nan
        if not math.isfinite(fitness):
            status = result.failure_kind or "invalid_result"
            message = result.error or f"evaluator returned invalid fitness: {raw_fitness!r}"
            return self._reject_pending(
                status=status,
                error=message,
                error_type=result.error_type,
                code=parsed.code,
                traceback=result.traceback,
            )

        child = self._accept_pending(parsed.declared_idea, parsed.code, fitness)
        self._record_evaluation(
            state,
            pending,
            child_id=child.id,
            fitness=fitness,
            status="ok",
        )
        if self.best is child and self._log is not None:
            self._log.record_best(
                code=parsed.code,
                fitness=fitness,
                eval_count=self._n_eval,
                child_id=child.id,
            )
        self._pending = None
        self._generation = None
        save_checkpoint(self)
        return child

    def _charge_primary(self, state: GenerationState) -> None:
        if self._n_eval >= self._budget:
            raise RuntimeError("primary evaluator budget is exhausted")
        state.primary_charged = True
        self._n_eval += 1
        if state.intent is None:
            self._root_slots += 1
            return
        parent = self._tree.get_algorithm(state.parent_id)
        parent.count += 1
        if state.intent == Intent.REFINE.value:
            self._refine_slots += 1
            parent.refine_count += 1
            if state.hypothesis_id is None:
                raise RuntimeError("Refine slot has no hypothesis")
            self._hypotheses[state.hypothesis_id].primary_slots += 1
            block = self._active_block
            if block is None or block.id != state.block_id:
                raise RuntimeError("Refine slot is not attached to its active block")
            block.completed_steps += 1
        elif state.intent == Intent.EXPLORE.value:
            self._explore_slots += 1
            parent.explore_count += 1
        else:
            raise RuntimeError(f"unknown generation intent: {state.intent}")

    def _accept_pending(self, idea: str | None, code: str, fitness: float):
        state = self._generation
        if state is None:
            raise RuntimeError("accepted candidate has no generation state")
        if state.parent_id == VIRTUAL_ROOT_ID:
            hypothesis_id = self._allocate_hypothesis_id()
            source_hypothesis_id = None
            parent_code = ""
            result_label = "initial"
        else:
            parent = self._tree.get_algorithm(state.parent_id)
            if parent.code is None or parent.hypothesis_id is None:
                raise RuntimeError("candidate parent is incomplete")
            parent_code = parent.code
            if state.intent == Intent.EXPLORE.value:
                hypothesis_id = self._allocate_hypothesis_id()
                source_hypothesis_id = parent.hypothesis_id
            elif state.intent == Intent.REFINE.value:
                hypothesis_id = int(state.hypothesis_id)
                source_hypothesis_id = None
                if hypothesis_id != parent.hypothesis_id:
                    raise RuntimeError("Refine child changed hypothesis")
            else:
                raise RuntimeError("non-root candidate has no intent")
            parent_quality = self._tree.quality(parent)
            quality = fitness if self._tree.maximize else -fitness
            result_label = (
                "improved"
                if quality > parent_quality
                else "worse"
                if quality < parent_quality
                else "equal"
            )

        diff, added, removed = (
            ("", 0, 0) if state.parent_id == VIRTUAL_ROOT_ID else code_diff(parent_code, code)
        )
        child = self._tree.add_algorithm(
            code=code,
            fitness=fitness,
            parent_id=state.parent_id,
            hypothesis_id=hypothesis_id,
            idea=idea,
            diff=diff,
            added=added,
            removed=removed,
            result=result_label,
            created_by=state.intent,
        )
        child_quality = self._tree.quality(child)
        if state.parent_id == VIRTUAL_ROOT_ID or state.intent == Intent.EXPLORE.value:
            hypothesis = Hypothesis(
                id=hypothesis_id,
                origin_node_id=child.id,
                source_hypothesis_id=source_hypothesis_id,
                status=HypothesisStatus.MATURING,
                frontier_node_id=child.id,
                best_quality=child_quality,
            )
            self._hypotheses[hypothesis_id] = hypothesis
            if state.intent == Intent.EXPLORE.value:
                self._discovery_candidate_hypothesis_id = hypothesis_id
                self._maturing_hypothesis_id = hypothesis_id
                self._valid_discoveries += 1
        else:
            hypothesis = self._hypotheses[hypothesis_id]
            if child_quality > hypothesis.best_quality:
                hypothesis.best_quality = child_quality
                hypothesis.frontier_node_id = child.id

        block = self._active_block
        if state.intent == Intent.REFINE.value and block is not None:
            if block.id != state.block_id:
                raise RuntimeError("valid Refine result belongs to another block")
            block.valid_results += 1
            if block.kind is BlockKind.INITIAL_MATURATION:
                parent = self._tree.get_algorithm(state.parent_id)
                self._bootstrap_deltas.append(
                    abs(self._tree.quality(child) - self._tree.quality(parent))
                )
        return child

    def _reject_pending(
        self,
        *,
        status: str,
        error: str,
        error_type: str | None,
        code: str,
        traceback: str | None,
    ):
        pending = self._pending
        state = self._generation
        if pending is None or state is None:
            raise RuntimeError("rejected candidate has no generation state")
        feedback = format_failure_feedback(
            error_type=error_type,
            error=error,
            traceback=traceback,
            preflight_error=self._preflight_error,
        )
        state.failed_code = code
        state.failure_feedback = feedback
        self._record_evaluation(
            state,
            pending,
            child_id=None,
            fitness=None,
            status=status,
            error=feedback,
            error_type=error_type,
        )
        self._pending = None
        if pending.attempt <= self._error_retries:
            parent_code = None
            if state.parent_id != VIRTUAL_ROOT_ID:
                parent_code = self._tree.get_algorithm(state.parent_id).code
            state.prompt = build_repair_prompt(
                task_description=self._task,
                parent_code=parent_code,
                failed_code=code,
                error=feedback,
                intent=None if state.intent is None else Intent(state.intent),
                maximize=self._tree.maximize,
                reliability=self._error_handling,
            )
            state.attempt += 1
        else:
            self._generation = None
        save_checkpoint(self)
        return None

    def _prompt(self, algorithm, intent: Intent) -> str:
        if algorithm.code is None or algorithm.fitness is None:
            raise RuntimeError("cannot prompt from an incomplete algorithm")
        path = parent_path(self._tree, algorithm.id, max_events=self._max_history)
        return build_generation_prompt(
            task_description=self._task,
            code=algorithm.code,
            fitness=algorithm.fitness,
            history_text=render_path(self._tree, path),
            intent=intent,
            maximize=self._tree.maximize,
            error_handling=self._error_handling,
        )

    def _record_evaluation(
        self,
        state: GenerationState,
        pending: Pending,
        *,
        child_id: int | None,
        fitness: float | None,
        status: str,
        error: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if self._log is None:
            return
        block = self._active_block
        hypothesis_id = state.hypothesis_id
        if child_id is not None:
            hypothesis_id = self._tree.get_algorithm(child_id).hypothesis_id
        self._log.record_evaluation(
            eval_count=self._n_eval,
            llm_call_count=self._n_llm_calls,
            evaluator_call_count=self._n_calls,
            parent_id=state.parent_id,
            child_id="" if child_id is None else child_id,
            intent=state.intent or "",
            mode=state.mode,
            phase=self._phase.value,
            hypothesis_id="" if hypothesis_id is None else hypothesis_id,
            cycle=self._cycle,
            sweep=self._sweep,
            block_id="" if state.block_id is None else state.block_id,
            block_kind="" if block is None else block.kind.value,
            block_step="" if state.block_step is None else state.block_step,
            status=status,
            fitness="" if fitness is None else fitness,
            error=error or "",
            attempt=pending.attempt,
            attempt_kind=pending.attempt_kind,
            elapsed_seconds=f"{self._attempt_elapsed:.6f}",
            preflight_error=self._preflight_error or "",
            candidate_hash=self._candidate_hash,
            error_type=error_type or "",
            s_r=self._s_r,
            active_ids=self._active_ids,
            competition_line=(
                "" if self._competition_line() is None else self._competition_line()
            ),
        )

    def _allocate_hypothesis_id(self) -> int:
        hypothesis_id = self._next_hypothesis_id
        self._next_hypothesis_id += 1
        return hypothesis_id

    def _rank_ids(self, hypothesis_ids: list[int]) -> list[int]:
        unique = dict.fromkeys(hypothesis_ids)
        return [item.id for item in rank_hypotheses(self._hypotheses[key] for key in unique)]

    def _highest_active_id(self) -> int:
        if not self._active_ids:
            raise RuntimeError("V9.17 has no active hypothesis")
        return self._rank_ids(self._active_ids)[0]

    def _competition_line(self) -> float | None:
        if len(self._active_ids) < self._active_capacity:
            return None
        return competition_line(self._hypotheses[item] for item in self._active_ids)

    def _event(self, event: str, **payload: object) -> None:
        if self._log is not None:
            self._log.record_event(event, **payload)

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget

    def _assert_invariants(self) -> None:
        if self._n_eval != self._root_slots + self._refine_slots + self._explore_slots:
            raise RuntimeError("primary slot accounting is inconsistent")
        if self._n_calls < self._n_eval:
            raise RuntimeError("evaluator calls cannot be fewer than primary slots")
        if self._repair_llm_calls > self._n_llm_calls:
            raise RuntimeError("repair LLM calls exceed all LLM calls")
        if len(self._active_ids) > self._active_capacity:
            raise RuntimeError("active hypothesis capacity was exceeded")
        if len(set(self._active_ids)) != len(self._active_ids):
            raise RuntimeError("active hypothesis IDs are not unique")
        if set(self._active_ids) & set(self._reserve_ids):
            raise RuntimeError("a hypothesis is both active and reserve")
        for hypothesis_id in self._active_ids:
            if self._hypotheses[hypothesis_id].status is not HypothesisStatus.ACTIVE:
                raise RuntimeError("active set and hypothesis status disagree")
        for hypothesis_id in self._reserve_ids:
            if self._hypotheses[hypothesis_id].status is not HypothesisStatus.RESERVE:
                raise RuntimeError("reserve set and hypothesis status disagree")
        for algorithm in self._tree.valid_algorithms():
            if algorithm.hypothesis_id not in self._hypotheses:
                raise RuntimeError("valid algorithm has no known hypothesis")
            if algorithm.parent_id not in {None, VIRTUAL_ROOT_ID}:
                parent = self._tree.get_algorithm(int(algorithm.parent_id))
                if algorithm.created_by == Intent.REFINE.value:
                    if algorithm.hypothesis_id != parent.hypothesis_id:
                        raise RuntimeError("Refine edge crossed a hypothesis boundary")
                elif algorithm.created_by == Intent.EXPLORE.value:
                    if algorithm.hypothesis_id == parent.hypothesis_id:
                        raise RuntimeError("Explore edge did not create a hypothesis")
        for hypothesis in self._hypotheses.values():
            algorithms = self._tree.hypothesis_algorithms(hypothesis.id)
            if not algorithms:
                raise RuntimeError("hypothesis has no valid algorithms")
            frontier = max(algorithms, key=lambda item: (self._tree.quality(item), -item.id))
            if frontier.id != hypothesis.frontier_node_id:
                raise RuntimeError("cached hypothesis frontier is inconsistent")
            if not math.isclose(
                self._tree.quality(frontier), hypothesis.best_quality, abs_tol=1e-12
            ):
                raise RuntimeError("cached hypothesis quality is inconsistent")
        if self._pending is not None:
            if self._generation is None:
                raise RuntimeError("pending response has no generation transaction")
            if self._pending.attempt != self._generation.attempt:
                raise RuntimeError("pending and generation attempts disagree")
        if not self._adaptive_sweeps and self._sweep > 1:
            raise RuntimeError("FixedCycle cannot enter a continuation sweep")


__all__ = ["TraceAADV917"]
