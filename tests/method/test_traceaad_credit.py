from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm4ad.method.traceaad.portfolio import OperatorPortfolio, PortfolioWeights
from llm4ad.method.traceaad.schema import OperatorName


@dataclass
class _StubOperator:
    name: str
    role: str = "exploit"

    def trigger(self, ctx) -> bool:
        return True


def test_portfolio_records_one_batch_outcome_per_operator_iteration() -> None:
    op = _StubOperator("test_operator")
    portfolio = OperatorPortfolio((op,), PortfolioWeights())

    portfolio.update_batch(
        op=op,
        iteration=7,
        normalized_reward=0.4,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=123.0,
        global_best=True,
    )

    stats = portfolio.snapshot()[op.name]
    assert stats["n_calls"] == 1
    assert stats["mean_gain"] == pytest.approx(0.4)
    assert stats["valid_rate"] == pytest.approx(1.0)
    assert stats["novel_rate"] == pytest.approx(0.0)
    assert stats["regress_rate"] == pytest.approx(0.0)
    assert stats["mean_cost"] == pytest.approx(123.0)
    assert stats["global_best_rate"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="already updated"):
        portfolio.update_batch(
            op=op,
            iteration=7,
            normalized_reward=-1.0,
            best_valid=False,
            best_novel=False,
            best_regress=True,
            total_cost=999.0,
        )


def test_portfolio_credit_follows_recent_outcomes_instead_of_lifetime_mean() -> None:
    op = _StubOperator("test_operator")
    portfolio = OperatorPortfolio((op,), PortfolioWeights())

    for iteration in range(8):
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            normalized_reward=-1.0,
            best_valid=True,
            best_novel=False,
            best_regress=True,
            total_cost=10.0,
        )
    for iteration in range(8, 16):
        portfolio.update_batch(
            op=op,
            iteration=iteration,
            normalized_reward=1.0,
            best_valid=True,
            best_novel=True,
            best_regress=False,
            total_cost=10.0,
        )

    assert portfolio.snapshot()[op.name]["mean_gain"] > 0.5


def test_global_best_bonus_increases_an_operators_selection_probability() -> None:
    record_op = _StubOperator("record_operator")
    ordinary_op = _StubOperator("ordinary_operator")
    weights = PortfolioWeights(
        alpha=0.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=1.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((record_op, ordinary_op), weights)
    common = dict(
        iteration=1,
        normalized_reward=0.0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=10.0,
    )
    portfolio.update_batch(op=record_op, global_best=True, **common)
    portfolio.update_batch(op=ordinary_op, global_best=False, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=1, max_iter=10)

    assert probabilities[record_op.name] > probabilities[ordinary_op.name]


def test_near_record_bonus_increases_operator_probability() -> None:
    near = _StubOperator("near")
    ordinary = _StubOperator("ordinary")
    weights = PortfolioWeights(
        alpha=0.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        near_record_bonus=1.0,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((near, ordinary), weights)
    common = dict(
        iteration=1,
        normalized_reward=0.0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=0.0,
        global_best=False,
    )
    portfolio.update_batch(op=near, near_record=True, **common)
    portfolio.update_batch(op=ordinary, near_record=False, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=1, max_iter=10)

    assert probabilities[near.name] > probabilities[ordinary.name]


def test_late_phase_caps_novelty_without_removing_softmax_exploration() -> None:
    novelty = _StubOperator(OperatorName.NOVELTY)
    exploit = _StubOperator(OperatorName.ENDPOINT)
    weights = PortfolioWeights(
        alpha=10.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        min_probability=0.05,
        late_novelty_max_probability=0.2,
    )
    portfolio = OperatorPortfolio((novelty, exploit), weights)
    common = dict(
        iteration=0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=0.0,
    )
    portfolio.update_batch(op=novelty, normalized_reward=1.0, **common)
    portfolio.update_batch(op=exploit, normalized_reward=-1.0, **common)

    early = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)
    late = portfolio.probabilities(ctx=None, iteration=90, max_iter=100)

    assert min(early.values()) >= 0.05
    assert late[OperatorName.NOVELTY] == pytest.approx(0.2)


def test_temperature_floor_prevents_late_softmax_from_freezing() -> None:
    better = _StubOperator("better")
    worse = _StubOperator("worse")
    weights = PortfolioWeights(
        alpha=1.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=0.0,
        global_best_bonus=0.0,
        temperature_init=0.0,
        temperature_end=0.0,
        temperature_floor=0.5,
        min_probability=0.0,
    )
    portfolio = OperatorPortfolio((better, worse), weights)
    common = dict(
        iteration=0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
        total_cost=0.0,
    )
    portfolio.update_batch(op=better, normalized_reward=1.0, **common)
    portfolio.update_batch(op=worse, normalized_reward=-1.0, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=100, max_iter=100)

    assert probabilities[worse.name] > 0.01


def test_actual_batch_cost_remains_comparable_in_portfolio_selection() -> None:
    fast = _StubOperator("fast")
    slow = _StubOperator("slow")
    weights = PortfolioWeights(
        alpha=0.0,
        beta_v=0.0,
        beta_n=0.0,
        delta_r=0.0,
        delta_c=1.0,
        global_best_bonus=0.0,
        min_probability=0.0,
        cost_scale=120.0,
    )
    portfolio = OperatorPortfolio((fast, slow), weights)
    common = dict(
        iteration=0,
        normalized_reward=0.0,
        best_valid=True,
        best_novel=False,
        best_regress=False,
    )
    portfolio.update_batch(op=fast, total_cost=30.0, **common)
    portfolio.update_batch(op=slow, total_cost=300.0, **common)

    probabilities = portfolio.probabilities(ctx=None, iteration=0, max_iter=100)

    assert probabilities[fast.name] > probabilities[slow.name]
