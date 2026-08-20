"""TraceAAD V9.14: Single Tree evolutionary algorithm search."""

from __future__ import annotations

import copy
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any

from ...base import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from .artifacts import RunArtifacts
from .checkpoint import CHECKPOINT_VERSION, load_checkpoint, save_checkpoint
from .history import drop_oldest, one_line, parent_path, render_path
from .prompt import (
    ProgramResponseError,
    build_generation_prompt,
    build_root_prompt,
    parse_program_response,
)
from .schema import (
    INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS,
    PROTOCOL_ID,
    REFINE_PROBABILITY,
    Algorithm,
    Intent,
    Outcome,
    Pending,
)
from .selection import select
from .source import code_diff
from .tree import Tree, is_better

TRANSPORT_RETRIES = 3
ERROR_MAX_CHARS = 360


def draw_intent(seed: int | None, iteration: int) -> Intent:
    """确定性伪随机抽取生成意图 (70% Refine / 30% Explore)，断点恢复完全幂等。"""
    token = "none" if seed is None else str(seed)
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}:intent:{token}:{iteration}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return Intent.REFINE if value < REFINE_PROBABILITY else Intent.EXPLORE


class TraceAADV914:
    """TraceAAD V9.14 算法搜索主流程控制器。"""

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
        context_limit: int | None = None,
        max_history: int = MAX_HISTORY_EVENTS,
        seed: int | None = 0,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
        debug_mode: bool = False,
    ) -> None:
        if budget <= 0 or n_roots <= 0 or max_tokens <= 0 or max_history <= 0:
            raise ValueError("budget, n_roots, max_tokens, and max_history must be positive")
        if context_limit is None or context_limit <= 0:
            raise ValueError("context_limit must be explicitly positive")
        if (
            evaluation.use_numba_accelerate
            or evaluation.use_protected_div
            or evaluation.random_seed is not None
        ):
            raise ValueError("V9.14 requires candidate code to be executed unchanged")

        template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if template is None or len(template.functions) != 1:
            raise ValueError("TraceAAD V9.14 requires one evolvable template function")

        self._llm = llm
        self._log = artifacts
        self._task = evaluation.task_description
        self._template = template
        self._function: Function = copy.deepcopy(template.functions[0])
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)
        self._budget = budget
        self._n_roots = n_roots
        self._maximize = maximize
        self._max_tokens = max_tokens
        self._context_limit = context_limit
        self._max_history = max_history
        self._seed = seed
        self._checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
        llm.debug_mode = debug_mode

        # 核心单根算法树与状态
        self._tree = Tree(maximize=self._maximize)
        self._pending: Pending | None = None
        self._n_candidates = 0
        self._n_eval = 0
        self._iteration = 0
        self._initialization_complete = False
        self._bootstrapped: set[int] = set()
        self._bootstrap_deltas: list[float] = []
        self._s: float | None = None
        self._best_id: int | None = None

        if resume_from is not None:
            checkpoint = load_checkpoint(self, resume_from)
            if self._checkpoint_dir is None:
                self._checkpoint_dir = checkpoint.parent

    @property
    def best(self) -> Algorithm | None:
        """获取当前全局最优算法节点。"""
        return None if self._best_id is None else self._tree.get_algorithm(self._best_id)

    def search_configuration(self) -> dict[str, Any]:
        """返回搜索超参配置字典。"""
        return {
            "protocol_id": PROTOCOL_ID,
            "checkpoint_schema_version": CHECKPOINT_VERSION,
            "budget": self._budget,
            "n_roots": self._n_roots,
            "max_history": self._max_history,
            "maximize": self._maximize,
            "max_tokens": self._max_tokens,
            "context_limit": self._context_limit,
            "seed": self._seed,
            "refine_probability": REFINE_PROBABILITY,
            "explore_probability": 1.0 - REFINE_PROBABILITY,
        }

    def run(self) -> None:
        """执行完整搜索生命周期。"""
        status = "error"
        stop_reason: str | None = None
        error: dict[str, str] = {}
        try:
            # 1. 优先恢复未完成的 pending 请求
            if self._pending is not None:
                self._process_pending()

            # 2. 初始化阶段: 生成 8 个根算法并完成 Refine Bootstrap
            if not self._initialization_complete:
                self._initialize()

            if not self._initialization_complete:
                status = "initialization_failure"
                stop_reason = "evaluator_budget_exhausted_during_initialization"
                return

            # 3. 正式搜索主循环
            while self._has_budget():
                assert self._s is not None
                choice = select(self._tree, self._s)
                intent = draw_intent(self._seed, self._iteration)

                # 生成新候选并立即评价落盘
                self._generate(
                    self._prompt(choice.algorithm_id, intent),
                    parent_id=choice.algorithm_id,
                    stage="search",
                    iteration=self._iteration,
                    intent=intent.value,
                )

            status = "finished"
            stop_reason = "evaluator_budget_exhausted"
        except Exception as exc:
            error = {"error_type": type(exc).__name__, "error": str(exc)[:1000]}
            if self._log is not None:
                self._log.record_error("run", exc)
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
                    method_sample_count=self._n_candidates,
                    evaluator_call_count=self._n_eval,
                    n_algorithms=len(self._tree.valid_algorithms()),
                    n_branches=len(self._tree.branch_ids),
                    n_iterations=self._iteration,
                    initialization_complete=self._initialization_complete,
                    bootstrapped=sorted(self._bootstrapped),
                    bootstrap_deltas=self._bootstrap_deltas,
                    s=self._s,
                    pending_order=None if self._pending is None else self._pending.order,
                    **error,
                )
                self._log.finish()
            self._llm.close()

    def _initialize(self) -> None:
        """初始化阶段：生成 8 个初始主算法并各执行一次 Bootstrap Refine。"""
        # 1. 从虚拟根节点生成 8 个初始算法
        while len(self._tree.branch_ids) < self._n_roots and self._has_budget():
            prompt = build_root_prompt(
                task_description=self._task,
                template_function=self._function,
                maximize=self._maximize,
            )
            if not self._fits(prompt):
                raise RuntimeError("root prompt plus output bound exceeds context limit")
            self._generate(
                prompt,
                parent_id=self._tree.virtual_root_id,
                stage="root_generation",
                iteration=None,
                intent=None,
            )

        # 2. 对每个初始算法执行 1 次 Refine Bootstrap
        for root_id in tuple(self._tree.branch_ids):
            if not self._has_budget():
                break
            if root_id in self._bootstrapped:
                continue
            self._generate(
                self._prompt(root_id, Intent.REFINE),
                parent_id=root_id,
                stage="bootstrap",
                iteration=None,
                intent=Intent.REFINE.value,
            )

        complete = (
            len(self._tree.branch_ids) == self._n_roots
            and self._bootstrapped == set(self._tree.branch_ids)
        )
        if not complete:
            save_checkpoint(self)
            return

        # 3. 估计单步尺度因子 s
        self._s = (
            float(statistics.median(self._bootstrap_deltas))
            if self._bootstrap_deltas
            else 0.0
        )
        self._initialization_complete = True
        save_checkpoint(self)

    def _prompt(self, algorithm_id: int, intent: Intent) -> str:
        """构建包含当前算法代码、父代形成历史与生成意图的 Prompt。"""
        algo = self._tree.get_algorithm(algorithm_id)
        assert algo.code is not None and algo.fitness is not None

        selected = parent_path(self._tree, algorithm_id, max_events=self._max_history)
        shown = selected
        while True:
            prompt = build_generation_prompt(
                task_description=self._task,
                code=algo.code,
                fitness=algo.fitness,
                history_text=render_path(self._tree, shown),
                intent=intent,
                maximize=self._maximize,
            )
            if self._fits(prompt):
                return prompt
            if not shown:
                raise RuntimeError(
                    "task, current code, and output budget exceed context "
                    "even with no history events"
                )
            shown = drop_oldest(shown)

    def _generate(
        self,
        prompt: str,
        *,
        parent_id: int | None,
        stage: str,
        iteration: int | None,
        intent: str | None,
    ) -> Algorithm | None:
        """调用 LLM 生成代码响应，写入 pending 状态并立即流转评价。"""
        if self._pending is not None:
            raise RuntimeError("cannot request a candidate while one is pending")
        generation_seed = (
            None if self._seed is None else self._seed + self._n_candidates + 1
        )
        response = self._draw(prompt, generation_seed)
        self._n_candidates += 1
        if parent_id is not None and parent_id != self._tree.virtual_root_id:
            self._tree.get_algorithm(parent_id).count += 1

        self._pending = Pending(
            order=self._n_candidates,
            parent_id=parent_id,
            stage=stage,
            iteration=iteration,
            intent=intent,
            response=response,
        )
        save_checkpoint(self)
        return self._process_pending()

    def _draw(self, prompt: str, seed: int | None) -> str:
        """向模型请求文本补全，具备传输失败重试机制。"""
        last_error: Exception | None = None
        kwargs: dict[str, Any] = {"max_tokens": self._max_tokens}
        if seed is not None:
            kwargs["seed"] = seed
        for attempt in range(TRANSPORT_RETRIES + 1):
            try:
                return self._llm.draw_sample(prompt, **kwargs)
            except Exception as exc:
                last_error = exc
                save_checkpoint(self)
                if self._log is not None:
                    self._log.record_llm_call(
                        status="transport",
                        transport_attempt=attempt + 1,
                        error=f"{type(exc).__name__}: {exc}",
                    )
        raise RuntimeError("model transport retry limit exhausted") from last_error

    def _process_pending(self) -> Algorithm | None:
        """解析 pending 的响应并真实调用评价器。"""
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending candidate to process")

        parent = (
            None
            if pending.parent_id is None or pending.parent_id == self._tree.virtual_root_id
            else self._tree.get_algorithm(pending.parent_id)
        )

        if pending.stage == "bootstrap" and pending.parent_id is not None:
            self._bootstrapped.add(pending.parent_id)

        # 1. 解析 Idea + Code
        try:
            parsed = parse_program_response(
                pending.response, self._template, self._function.name
            )
        except ProgramResponseError as exc:
            return self._fail_pending(
                pending=pending,
                parent=parent,
                status="parse_failed",
                error=one_line(str(exc), ERROR_MAX_CHARS),
            )

        idea = parsed.declared_idea
        code = str(parsed.program)

        # 2. 计算与父代码的 diff
        diff: str | None = None
        added = 0
        removed = 0
        if parent is not None and parent.code is not None:
            diff, added, removed = code_diff(parent.code, code)

        # 3. 真实运行评价器
        outcome, _elapsed = self._evaluator.evaluate_program_record_time_with_details(
            code
        )
        self._n_eval += 1

        if outcome.failure_kind == "prepare_error":
            raise RuntimeError(
                one_line(
                    outcome.error
                    or "evaluator preparation failed without an error message",
                    ERROR_MAX_CHARS,
                )
            )

        score = getattr(outcome.result, "fitness", outcome.result)
        try:
            parsed_fitness = float(score)
        except (TypeError, ValueError, OverflowError):
            parsed_fitness = math.nan

        # 4. 评价失败：直接丢弃并记日志
        if not math.isfinite(parsed_fitness):
            return self._fail_pending(
                pending=pending,
                parent=parent,
                status="eval_failed",
                error=one_line(
                    outcome.error
                    or f"evaluator returned non-finite fitness: {score!r}",
                    ERROR_MAX_CHARS,
                ),
            )

        # 5. 评价成功：创建新的 Algorithm 节点
        q = parsed_fitness if self._maximize else -parsed_fitness
        dq = None if parent is None or parent.q is None else q - parent.q
        outcome_val = _outcome(parent is not None, dq=dq)

        child = self._tree.add_algorithm(
            code=code,
            fitness=parsed_fitness,
            parent_id=None if parent is None else parent.id,
            intent=pending.intent,
            idea=idea,
            diff=diff,
            added=added,
            removed=removed,
            dq=dq,
            outcome=outcome_val,
            stage=pending.stage,
            iteration=pending.iteration,
        )
        kind = "root_new" if parent is None else "new"

        # 统计 Bootstrap 阶段的单步变化幅度
        if pending.stage == "bootstrap" and dq is not None:
            self._bootstrap_deltas.append(abs(dq))

        if pending.stage == "search" and pending.iteration is not None:
            self._iteration = max(self._iteration, pending.iteration + 1)

        # 检查是否更新全局最优
        is_new_best = self._update_best(child)
        if is_new_best and self._log is not None:
            assert child.code is not None and child.fitness is not None
            self._log.record_best(
                code=child.code,
                fitness=child.fitness,
                eval_count=self._n_eval,
                iteration=pending.iteration,
                order=pending.order,
                algorithm_id=child.id,
            )

        self._pending = None
        save_checkpoint(self)
        if self._log is not None:
            best_fit = None if self.best is None else self.best.fitness
            self._log.record_candidate(
                order=pending.order,
                stage=pending.stage,
                iteration=pending.iteration,
                parent_id=pending.parent_id,
                child_id=child.id,
                intent=pending.intent,
                kind=kind,
                outcome=outcome_val,
                status="ok",
                parent_fitness=None if parent is None else parent.fitness,
                child_fitness=child.fitness,
                dq=dq,
                error=None,
                eval_count=self._n_eval,
                branch_id=self._tree.branch_id_of(child.id),
                best_fitness=best_fit,
                is_new_best=is_new_best,
                budget=self._budget,
            )
        return child

    def _fail_pending(
        self,
        *,
        pending: Pending,
        parent: Algorithm | None,
        status: str,
        error: str,
    ) -> None:
        """候选失败时直接丢弃，不污染树，仅存盘与记录日志。"""
        self._pending = None
        save_checkpoint(self)
        if self._log is not None:
            branch_id = None if parent is None else self._tree.branch_id_of(parent.id)
            best_fit = None if self.best is None else self.best.fitness
            self._log.record_candidate(
                order=pending.order,
                stage=pending.stage,
                iteration=pending.iteration,
                parent_id=pending.parent_id,
                child_id=None,
                intent=pending.intent,
                kind="invalid",
                outcome=None,
                status=status,
                parent_fitness=None if parent is None else parent.fitness,
                child_fitness=None,
                dq=None,
                error=error,
                eval_count=self._n_eval,
                branch_id=branch_id,
                best_fitness=best_fit,
                is_new_best=False,
                budget=self._budget,
            )
        return None

    def _update_best(self, algorithm: Algorithm | None) -> bool:
        """检查并更新全局最优解。"""
        if algorithm is None:
            return False
        if not is_better(algorithm, self.best):
            return False
        self._best_id = algorithm.id
        return True

    def _fits(self, prompt: str) -> bool:
        return self._tokens(prompt) + self._max_tokens <= self._context_limit

    def _tokens(self, text: str) -> int:
        for name in ("count_prompt_tokens", "count_tokens"):
            counter = getattr(self._llm, name, None)
            if callable(counter):
                return int(counter(text))
        raise RuntimeError("model tokenizer is unavailable")

    def _has_budget(self) -> bool:
        return self._n_eval < self._budget


def _outcome(has_parent: bool, *, dq: float | None) -> Outcome | None:
    """根据质量变化计算定性结果标签。"""
    if not has_parent or dq is None:
        return None
    if dq > 0:
        return Outcome.IMPROVE
    if dq < 0:
        return Outcome.REGRESS
    return Outcome.PLATEAU


__all__ = [
    "TraceAADV914",
    "draw_intent",
]
