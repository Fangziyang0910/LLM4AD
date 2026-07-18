"""Operator Portfolio —— opportunity-aware discounted UCB + bounded context。

候选 = 结构性可行性 trigger；选择 = softmax(z/T) 与全局 exploration mixture。
第一版暂时保留 late novelty probability cap。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .operators import Operator, OperatorContext
from .operator_signals import OperatorPreview
from .schema import OperatorName


@dataclass
class OperatorStats:
    attempt_count: int = 0
    eligible_count: int = 0
    discounted_mass: float = 0.0
    ema_utility: float | None = None
    ema_downside: float | None = None
    ema_valid: float | None = None
    ema_novel: float | None = None
    ema_cost: float | None = None
    ema_global_best: float | None = None
    ema_near_record: float | None = None

    def mean_utility(self) -> float:
        return self.ema_utility if self.ema_utility is not None else 0.0

    def mean_downside(self) -> float:
        return self.ema_downside if self.ema_downside is not None else 0.0

    def valid_rate(self) -> float:
        return self.ema_valid if self.ema_valid is not None else 0.5

    def novel_rate(self) -> float:
        return self.ema_novel if self.ema_novel is not None else 0.5

    def mean_cost(self) -> float:
        return self.ema_cost if self.ema_cost is not None else 0.0

    def global_best_rate(self) -> float:
        return self.ema_global_best if self.ema_global_best is not None else 0.0

    def near_record_rate(self) -> float:
        return self.ema_near_record if self.ema_near_record is not None else 0.0

    # Backward-compatible aliases used by older logs/tests.
    def mean_gain(self) -> float:
        return self.mean_utility()

    def regress_rate(self) -> float:
        return self.mean_downside()


@dataclass(frozen=True)
class PortfolioWeights:
    lambda_downside: float = 0.5
    lambda_cost: float = 0.05
    cost_scale: float = 120.0
    ucb_c: float = 0.5
    ucb_n0: float = 1.0
    context_bound: float = 0.2
    epsilon_init: float = 0.15
    epsilon_end: float = 0.05
    temperature_init: float = 1.0
    temperature_end: float = 0.5
    temperature_floor: float = 0.5
    ema_decay: float = 0.8
    global_best_bonus: float = 0.5
    near_record_bonus: float = 0.25
    near_record_tolerance: float = 0.10
    late_novelty_max_probability: float = 0.2
    prior_pseudo_count: float = 2.0
    prior_mean: float = 0.05
    score_clip: float = 2.0
    # Legacy aliases retained so older configs/tests still construct.
    alpha: float = 1.0
    beta_v: float = 0.0
    beta_n: float = 0.0
    delta_r: Optional[float] = None
    delta_c: Optional[float] = None
    min_probability: float = 0.0


@dataclass(frozen=True)
class SelectionDecision:
    operator: Operator
    probabilities: dict[str, float]
    scores: dict[str, float]
    components: dict[str, dict[str, float]]
    eligible: tuple[str, ...]
    eligible_counts: dict[str, int]
    attempt_counts: dict[str, int]
    previews: dict[str, OperatorPreview] = field(default_factory=dict)
    temperature: float = 1.0
    epsilon: float = 0.0


def signed_utility(delta_norm: float) -> float:
    return math.tanh(delta_norm)


def aggregate_batch_utility(utilities: list[float]) -> float:
    if not utilities:
        return -1.0
    return 0.5 * max(utilities) + 0.5 * (sum(utilities) / len(utilities))


class OperatorPortfolio:
    def __init__(
        self,
        operators: tuple[Operator, ...],
        weights: PortfolioWeights,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.operators = operators
        self.weights = weights
        self._rng = rng
        self.stats: dict[str, OperatorStats] = {op.name: OperatorStats() for op in operators}
        self._updated_iterations: set[tuple[str, int]] = set()

    def candidates(
        self,
        ctx: OperatorContext | None,
        *,
        previews: dict[str, OperatorPreview] | None = None,
    ) -> list[Operator]:
        if previews is not None:
            cands = [op for op in self.operators if previews.get(op.name) and previews[op.name].eligible]
        elif ctx is None:
            cands = list(self.operators)
        else:
            cands = [op for op in self.operators if op.trigger(ctx)]
        if not cands:
            cands = [op for op in self.operators if op.name == OperatorName.ENDPOINT]
        return cands

    def _phase(self, iteration: int, max_iter: int) -> int:
        frac = iteration / max(max_iter, 1)
        return 0 if frac < 0.33 else (1 if frac < 0.66 else 2)

    def _temperature(self, iteration: int, max_iter: int) -> float:
        frac = min(1.0, iteration / max(max_iter, 1))
        scheduled = self.weights.temperature_init + (
            self.weights.temperature_end - self.weights.temperature_init
        ) * frac
        return max(self.weights.temperature_floor, scheduled)

    def _epsilon(self, iteration: int, max_iter: int, stagnation: int = 0) -> float:
        frac = min(1.0, iteration / max(max_iter, 1))
        scheduled = self.weights.epsilon_init + (
            self.weights.epsilon_end - self.weights.epsilon_init
        ) * frac
        stagnation_frac = min(1.0, max(0, stagnation) / 12.0)
        return min(1.0, scheduled + 0.10 * stagnation_frac)

    def _clip_component(self, value: float) -> float:
        bound = max(1e-6, self.weights.score_clip)
        return max(-bound, min(bound, value))

    def _ucb(self, op_name: str) -> float:
        total_eligible = sum(s.eligible_count for s in self.stats.values())
        effective_attempts = self.stats[op_name].discounted_mass
        return self.weights.ucb_c * math.sqrt(
            math.log(1.0 + total_eligible)
            / (1.0 + effective_attempts + self.weights.ucb_n0)
        )

    def _prior(self, op_name: str) -> float:
        # A small neutral prior keeps newly eligible arms from being completely cold.
        n0 = max(0.0, self.weights.prior_pseudo_count)
        if n0 <= 0.0:
            return 0.0
        effective_attempts = self.stats[op_name].discounted_mass
        return self.weights.prior_mean * n0 / (n0 + effective_attempts)

    def _score_components(
        self,
        op: Operator,
        *,
        context_bonus: float = 0.0,
    ) -> dict[str, float]:
        s = self.stats[op.name]
        mu = self._clip_component(self.weights.alpha * s.mean_utility())
        downside = self._clip_component(s.mean_downside())
        cost = self._clip_component(
            math.tanh(max(0.0, s.mean_cost()) / max(self.weights.cost_scale, 1e-6))
        )
        lambda_d = self.weights.lambda_downside
        lambda_c = self.weights.lambda_cost
        if self.weights.delta_r is not None:
            lambda_d = self.weights.delta_r
        if self.weights.delta_c is not None:
            lambda_c = self.weights.delta_c
        gb = self.weights.global_best_bonus * s.global_best_rate()
        nr = self.weights.near_record_bonus * s.near_record_rate()
        ctx = max(
            -self.weights.context_bound,
            min(self.weights.context_bound, context_bonus),
        )
        prior = self._prior(op.name)
        ucb = self._ucb(op.name)
        score = mu - lambda_d * downside - lambda_c * cost + gb + nr + ctx + prior + ucb
        return {
            "mu": mu,
            "downside": downside,
            "cost": cost,
            "global_best": gb,
            "near_record": nr,
            "context": ctx,
            "prior": prior,
            "ucb": ucb,
            "score": score,
        }

    def mark_eligible(self, eligible_names: list[str] | tuple[str, ...]) -> None:
        for name in eligible_names:
            if name in self.stats:
                self.stats[name].eligible_count += 1

    def probabilities(
        self,
        *,
        ctx: OperatorContext | None,
        iteration: int,
        max_iter: int,
        previews: dict[str, OperatorPreview] | None = None,
        mark_eligibility: bool = False,
        stagnation: int = 0,
    ) -> dict[str, float]:
        decision = self.evaluate(
            ctx=ctx,
            iteration=iteration,
            max_iter=max_iter,
            previews=previews,
            mark_eligibility=mark_eligibility,
            stagnation=stagnation,
        )
        return decision.probabilities

    def evaluate(
        self,
        *,
        ctx: OperatorContext | None,
        iteration: int,
        max_iter: int,
        previews: dict[str, OperatorPreview] | None = None,
        mark_eligibility: bool = False,
        stagnation: int = 0,
    ) -> SelectionDecision:
        cands = self.candidates(ctx, previews=previews)
        eligible_names = tuple(op.name for op in cands)
        if mark_eligibility:
            self.mark_eligible(eligible_names)

        components: dict[str, dict[str, float]] = {}
        scores: dict[str, float] = {}
        for op in cands:
            bonus = 0.0
            if previews is not None and op.name in previews:
                bonus = previews[op.name].context_bonus
            comps = self._score_components(op, context_bonus=bonus)
            components[op.name] = comps
            scores[op.name] = comps["score"]

        temperature = self._temperature(iteration, max_iter)
        epsilon = max(
            0.0,
            min(1.0, self._epsilon(iteration, max_iter, stagnation)),
        )
        if len(cands) == 1:
            probabilities = {cands[0].name: 1.0}
        else:
            values = [scores[op.name] for op in cands]
            mx = max(values)
            exps = [math.exp((v - mx) / max(temperature, 1e-6)) for v in values]
            total = sum(exps)
            softmax = [e / total for e in exps]
            # Legacy min_probability still available as extra floor if explicitly set > 0.
            floor = min(max(self.weights.min_probability, 0.0), 1.0 / len(cands))
            if floor > 0.0:
                mixed = [floor + (1.0 - floor * len(cands)) * p for p in softmax]
            else:
                mixed = [
                    (1.0 - epsilon) * p + epsilon / len(cands) for p in softmax
                ]
            probabilities = {op.name: p for op, p in zip(cands, mixed)}

            phase = self._phase(iteration, max_iter)
            if phase == 2 and OperatorName.NOVELTY in probabilities and len(cands) > 1:
                cap = min(1.0, max(0.0, self.weights.late_novelty_max_probability))
                novelty_p = probabilities[OperatorName.NOVELTY]
                if novelty_p > cap:
                    probabilities[OperatorName.NOVELTY] = cap
                    other_total = sum(
                        p for name, p in probabilities.items() if name != OperatorName.NOVELTY
                    )
                    if other_total > 0:
                        scale = (1.0 - cap) / other_total
                        probabilities = {
                            name: (p * scale if name != OperatorName.NOVELTY else p)
                            for name, p in probabilities.items()
                        }

        # Placeholder operator; choose() replaces with sampled one.
        return SelectionDecision(
            operator=cands[0],
            probabilities=probabilities,
            scores=scores,
            components=components,
            eligible=eligible_names,
            eligible_counts={name: self.stats[name].eligible_count for name in self.stats},
            attempt_counts={name: self.stats[name].attempt_count for name in self.stats},
            previews=previews or {},
            temperature=temperature,
            epsilon=epsilon,
        )

    def choose(
        self,
        *,
        ctx: OperatorContext,
        iteration: int,
        max_iter: int,
        previews: dict[str, OperatorPreview] | None = None,
        stagnation: int = 0,
    ) -> SelectionDecision:
        decision = self.evaluate(
            ctx=ctx,
            iteration=iteration,
            max_iter=max_iter,
            previews=previews,
            mark_eligibility=True,
            stagnation=stagnation,
        )
        cands = [op for op in self.operators if op.name in decision.probabilities]
        if len(cands) == 1:
            selected = cands[0]
        else:
            random_source = self._rng if self._rng is not None else random
            r = random_source.random()
            cum = 0.0
            selected = cands[-1]
            for op in cands:
                cum += decision.probabilities[op.name]
                if r <= cum:
                    selected = op
                    break
        return SelectionDecision(
            operator=selected,
            probabilities=decision.probabilities,
            scores=decision.scores,
            components=decision.components,
            eligible=decision.eligible,
            eligible_counts=decision.eligible_counts,
            attempt_counts=decision.attempt_counts,
            previews=decision.previews,
            temperature=decision.temperature,
            epsilon=decision.epsilon,
        )

    def _record(
        self,
        *,
        op: Operator,
        utility: float,
        downside: float,
        valid: bool,
        novel: bool,
        cost: float,
        global_best: bool,
        near_record: bool,
    ) -> None:
        s = self.stats[op.name]
        s.attempt_count += 1
        decay = min(1.0, max(0.0, self.weights.ema_decay))
        s.discounted_mass = decay * s.discounted_mass + 1.0
        s.ema_utility = self._ema(s.ema_utility, utility, decay)
        s.ema_downside = self._ema(s.ema_downside, downside, decay)
        s.ema_valid = self._ema(s.ema_valid, float(valid), decay)
        s.ema_novel = self._ema(s.ema_novel, float(novel), decay)
        s.ema_cost = self._ema(s.ema_cost, cost, decay)
        s.ema_global_best = self._ema(s.ema_global_best, float(global_best), decay)
        s.ema_near_record = self._ema(
            s.ema_near_record,
            float(near_record or global_best),
            decay,
        )

    @staticmethod
    def _ema(previous: float | None, observation: float, decay: float) -> float:
        if previous is None:
            return observation
        return decay * previous + (1.0 - decay) * observation

    def update_batch(
        self,
        *,
        op: Operator,
        iteration: int,
        batch_utility: float | None = None,
        normalized_reward: float | None = None,
        downside: float = 0.0,
        best_valid: bool,
        best_novel: bool,
        best_regress: bool | None = None,
        total_cost: float,
        global_best: bool = False,
        near_record: bool = False,
    ) -> None:
        """Record exactly one comparable outcome for an operator attempt."""
        key = (op.name, iteration)
        if key in self._updated_iterations:
            raise ValueError(f"operator {op.name!r} already updated for iteration {iteration}")
        self._updated_iterations.add(key)
        if batch_utility is None:
            if normalized_reward is None:
                raise ValueError("batch_utility or normalized_reward is required")
            utility = max(-1.0, min(1.0, normalized_reward))
        else:
            utility = max(-1.0, min(1.0, batch_utility))
        if best_regress is not None and downside <= 0.0 and best_regress:
            downside = max(0.0, -utility)
        self._record(
            op=op,
            utility=utility,
            downside=max(0.0, min(1.0, downside)),
            valid=best_valid,
            novel=best_novel,
            cost=max(0.0, total_cost),
            global_best=global_best,
            near_record=near_record,
        )

    def snapshot(self) -> dict[str, dict]:
        return {
            name: {
                "n_calls": s.attempt_count,
                "attempt_count": s.attempt_count,
                "eligible_count": s.eligible_count,
                "discounted_mass": s.discounted_mass,
                "mean_gain": s.mean_utility(),
                "mean_utility": s.mean_utility(),
                "mean_downside": s.mean_downside(),
                "valid_rate": s.valid_rate(),
                "novel_rate": s.novel_rate(),
                "regress_rate": s.mean_downside(),
                "mean_cost": s.mean_cost(),
                "global_best_rate": s.global_best_rate(),
                "near_record_rate": s.near_record_rate(),
            }
            for name, s in self.stats.items()
        }
