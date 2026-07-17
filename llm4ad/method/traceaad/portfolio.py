"""Operator Portfolio —— bandit + 阶段感知。

候选 = trigger 通过的算子；在候选内用 softmax(operator_value / temperature) 采样。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .operators import Operator, OperatorContext
from .schema import OperatorName


@dataclass
class OperatorStats:
    n_calls: int = 0
    ema_gain: float | None = None
    ema_valid: float | None = None
    ema_novel: float | None = None
    ema_regress: float | None = None
    ema_cost: float | None = None
    ema_global_best: float | None = None
    ema_near_record: float | None = None

    def mean_gain(self) -> float:
        return self.ema_gain if self.ema_gain is not None else 0.0

    def valid_rate(self) -> float:
        return self.ema_valid if self.ema_valid is not None else 0.5

    def novel_rate(self) -> float:
        return self.ema_novel if self.ema_novel is not None else 0.5

    def regress_rate(self) -> float:
        return self.ema_regress if self.ema_regress is not None else 0.0

    def mean_cost(self) -> float:
        return self.ema_cost if self.ema_cost is not None else 0.0

    def global_best_rate(self) -> float:
        return self.ema_global_best if self.ema_global_best is not None else 0.0

    def near_record_rate(self) -> float:
        return self.ema_near_record if self.ema_near_record is not None else 0.0


@dataclass(frozen=True)
class PortfolioWeights:
    alpha: float = 1.0
    beta_v: float = 0.5
    beta_n: float = 0.3
    delta_r: float = 0.5
    delta_c: float = 0.05
    cost_scale: float = 120.0
    temperature_init: float = 1.0
    temperature_end: float = 0.5
    temperature_floor: float = 0.5
    ema_decay: float = 0.8
    global_best_bonus: float = 0.75
    near_record_bonus: float = 0.25
    near_record_tolerance: float = 0.10
    min_probability: float = 0.05
    late_novelty_max_probability: float = 0.2


_ROLE_PHASE_BONUS: dict[str, tuple[float, float, float]] = {
    "explore": (0.2, 0.1, 0.05),
    "recombine": (0.25, 0.4, 0.15),
    "path_correct": (0.25, 0.25, 0.25),
    "exploit": (0.2, 0.35, 0.5),
    "simplify": (0.05, 0.2, 0.4),
}


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

    def candidates(self, ctx: OperatorContext) -> list[Operator]:
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

    def _value(self, op: Operator, phase: int) -> float:
        s = self.stats[op.name]
        gain = math.tanh(s.mean_gain())
        normalized_cost = math.tanh(
            max(0.0, s.mean_cost()) / max(self.weights.cost_scale, 1e-6)
        )
        bonus = _ROLE_PHASE_BONUS.get(op.role, (0.1, 0.1, 0.1))[phase]
        return (
            self.weights.alpha * gain
            + self.weights.beta_v * s.valid_rate()
            + self.weights.beta_n * s.novel_rate()
            - self.weights.delta_r * s.regress_rate()
            - self.weights.delta_c * normalized_cost
            + self.weights.global_best_bonus * s.global_best_rate()
            + self.weights.near_record_bonus * s.near_record_rate()
            + bonus
        )

    def probabilities(
        self, *, ctx: OperatorContext | None, iteration: int, max_iter: int
    ) -> dict[str, float]:
        cands = self.candidates(ctx)
        if len(cands) == 1:
            return {cands[0].name: 1.0}
        phase = self._phase(iteration, max_iter)
        temperature = self._temperature(iteration, max_iter)
        values = [self._value(op, phase) for op in cands]
        mx = max(values)
        exps = [math.exp((v - mx) / max(temperature, 1e-6)) for v in values]
        total = sum(exps)
        raw = [e / total for e in exps]
        floor = min(max(self.weights.min_probability, 0.0), 1.0 / len(cands))
        probabilities = [floor + (1.0 - floor * len(cands)) * p for p in raw]
        if phase == 2:
            novelty_index = next(
                (i for i, op in enumerate(cands) if op.name == OperatorName.NOVELTY),
                None,
            )
            if novelty_index is not None and len(cands) > 1:
                cap = min(1.0, max(floor, self.weights.late_novelty_max_probability))
                if probabilities[novelty_index] > cap:
                    probabilities[novelty_index] = cap
                    other_total = sum(
                        p for i, p in enumerate(probabilities) if i != novelty_index
                    )
                    scale = (1.0 - cap) / other_total
                    probabilities = [
                        p * scale if i != novelty_index else p
                        for i, p in enumerate(probabilities)
                    ]
        return {op.name: p for op, p in zip(cands, probabilities)}

    def choose(self, *, ctx: OperatorContext, iteration: int, max_iter: int) -> Operator:
        cands = self.candidates(ctx)
        if len(cands) == 1:
            return cands[0]
        probabilities = self.probabilities(ctx=ctx, iteration=iteration, max_iter=max_iter)
        random_source = self._rng if self._rng is not None else random
        r = random_source.random()
        cum = 0.0
        for op in cands:
            cum += probabilities[op.name]
            if r <= cum:
                return op
        return cands[-1]

    def _record(
        self,
        *,
        op: Operator,
        gain: float,
        valid: bool,
        novel: bool,
        regress: bool,
        cost: float,
        global_best: bool,
        near_record: bool,
    ) -> None:
        s = self.stats[op.name]
        s.n_calls += 1
        decay = min(1.0, max(0.0, self.weights.ema_decay))
        s.ema_gain = self._ema(s.ema_gain, gain, decay)
        s.ema_valid = self._ema(s.ema_valid, float(valid), decay)
        s.ema_novel = self._ema(s.ema_novel, float(novel), decay)
        s.ema_regress = self._ema(s.ema_regress, float(regress), decay)
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
        normalized_reward: float,
        best_valid: bool,
        best_novel: bool,
        best_regress: bool,
        total_cost: float,
        global_best: bool = False,
        near_record: bool = False,
    ) -> None:
        """Record exactly one comparable outcome for an operator iteration."""
        key = (op.name, iteration)
        if key in self._updated_iterations:
            raise ValueError(f"operator {op.name!r} already updated for iteration {iteration}")
        self._updated_iterations.add(key)
        self._record(
            op=op,
            gain=max(-1.0, min(1.0, normalized_reward)),
            valid=best_valid,
            novel=best_novel,
            regress=best_regress,
            cost=max(0.0, total_cost),
            global_best=global_best,
            near_record=near_record,
        )

    def snapshot(self) -> dict[str, dict]:
        return {
            name: {
                "n_calls": s.n_calls, "mean_gain": s.mean_gain(),
                "valid_rate": s.valid_rate(), "novel_rate": s.novel_rate(),
                "regress_rate": s.regress_rate(), "mean_cost": s.mean_cost(),
                "global_best_rate": s.global_best_rate(),
                "near_record_rate": s.near_record_rate(),
            }
            for name, s in self.stats.items()
        }
