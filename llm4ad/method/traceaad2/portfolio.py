"""Operator Portfolio —— bandit + 阶段感知（design §5/§7）。

候选 = trigger 通过的算子；在候选内用 softmax(operator_value/τ) 采样。
value = α·gain + βv·valid + βn·novel − δr·regress − δc·cost + role 阶段 bonus。
τ 与 role-bonus 都随搜索阶段变化（早期偏 explore/recombine，晚期偏 exploit/simplify）。
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
    sum_gain: float = 0.0
    n_valid: int = 0
    n_novel: int = 0
    n_regress: int = 0
    sum_cost: float = 0.0

    def mean_gain(self) -> float:
        return self.sum_gain / self.n_calls if self.n_calls else 0.0

    def valid_rate(self) -> float:
        return self.n_valid / self.n_calls if self.n_calls else 0.5

    def novel_rate(self) -> float:
        return self.n_novel / self.n_calls if self.n_calls else 0.5

    def regress_rate(self) -> float:
        return self.n_regress / self.n_calls if self.n_calls else 0.0

    def mean_cost(self) -> float:
        return self.sum_cost / self.n_calls if self.n_calls else 0.0


@dataclass(frozen=True)
class PortfolioWeights:
    alpha: float = 1.0       # gain
    beta_v: float = 0.5      # valid rate
    beta_n: float = 0.3      # novelty
    delta_r: float = 0.5     # regression penalty
    delta_c: float = 0.05    # cost penalty
    tau_init: float = 1.0
    tau_end: float = 0.3


# role -> (early, mid, late) bonus（explore 早期从 0.5 降到 0.2，避免 novelty 靠 role-bonus 主导）
_ROLE_PHASE_BONUS: dict[str, tuple[float, float, float]] = {
    "explore": (0.2, 0.1, 0.05),
    "recombine": (0.25, 0.4, 0.15),
    "generalize": (0.15, 0.3, 0.25),
    "path_correct": (0.25, 0.25, 0.25),
    "exploit": (0.2, 0.35, 0.5),
    "simplify": (0.05, 0.2, 0.4),
    "abstract": (0.1, 0.1, 0.1),
}


class OperatorPortfolio:
    def __init__(self, operators: tuple[Operator, ...], weights: PortfolioWeights) -> None:
        self.operators = operators
        self.weights = weights
        self.stats: dict[str, OperatorStats] = {op.name: OperatorStats() for op in operators}

    def candidates(self, ctx: OperatorContext) -> list[Operator]:
        cands = [op for op in self.operators if op.trigger(ctx)]
        if not cands:
            cands = [op for op in self.operators if op.name == OperatorName.ENDPOINT]
        return cands

    def _phase(self, iteration: int, max_iter: int) -> int:
        frac = iteration / max(max_iter, 1)
        return 0 if frac < 0.33 else (1 if frac < 0.66 else 2)

    def _tau(self, iteration: int, max_iter: int) -> float:
        frac = min(1.0, iteration / max(max_iter, 1))
        return self.weights.tau_init + (self.weights.tau_end - self.weights.tau_init) * frac

    def _value(self, op: Operator, phase: int) -> float:
        s = self.stats[op.name]
        gain = math.tanh(s.mean_gain())  # 归一化到 [-1,1]
        bonus = _ROLE_PHASE_BONUS.get(op.role, (0.1, 0.1, 0.1))[phase]
        return (
            self.weights.alpha * gain
            + self.weights.beta_v * s.valid_rate()
            + self.weights.beta_n * s.novel_rate()
            - self.weights.delta_r * s.regress_rate()
            - self.weights.delta_c * math.tanh(s.mean_cost())
            + bonus
        )

    def choose(self, *, ctx: OperatorContext, iteration: int, max_iter: int) -> Operator:
        cands = self.candidates(ctx)
        if len(cands) == 1:
            return cands[0]
        phase = self._phase(iteration, max_iter)
        tau = self._tau(iteration, max_iter)
        values = [self._value(op, phase) for op in cands]
        mx = max(values)
        exps = [math.exp((v - mx) / max(tau, 1e-6)) for v in values]
        total = sum(exps)
        r = random.random()
        cum = 0.0
        for op, e in zip(cands, exps):
            cum += e / total
            if r <= cum:
                return op
        return cands[-1]

    def update(self, *, op: Operator, gain: float, valid: bool, novel: bool,
               regress: bool, cost: float) -> None:
        s = self.stats[op.name]
        s.n_calls += 1
        s.sum_gain += gain
        if valid:
            s.n_valid += 1
        if novel:
            s.n_novel += 1
        if regress:
            s.n_regress += 1
        s.sum_cost += cost

    def snapshot(self) -> dict[str, dict]:
        return {
            name: {
                "n_calls": s.n_calls, "mean_gain": s.mean_gain(),
                "valid_rate": s.valid_rate(), "novel_rate": s.novel_rate(),
                "regress_rate": s.regress_rate(), "mean_cost": s.mean_cost(),
            }
            for name, s in self.stats.items()
        }
